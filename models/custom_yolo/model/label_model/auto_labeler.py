import json
import threading

import cv2
import torch
from PyQt5.QtCore import pyqtSignal
from ultralytics import YOLO
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    BitsAndBytesConfig
)
"""
    for loop 내부에 라벨 생성 코드의 들여쓰기를 수정하여 각 객체에 대해 LLM 모델을 적용
    라벨 결과 저장 시, 매개변수 순서를 labels, output_path로 일치시킴
    라벨 출력 시, 딕셔너리 키 이름을 "coordinates"로 통일
"""


class AutoLabeler:
    log_signal = pyqtSignal(str)

    def __init__(self, config: dict):
        """
        :param config: Dictionary loaded from settings.json
        """
        # Extract device from config

        device_str = config.get("labeling", {}).get("device", "cpu")
        if device_str == 'cuda' and not torch.cuda.is_available():
            self.log_signal.emit("CUDA 사용 불가능, CPU로 전환")
            device_str = 'cpu'
        self.device = device_str

        # LLM Model name from config
        self.llm_model_name = config.get("llm_model", "microsoft/phi-2")
        self.detector_model_path = config.get("detector_model_path", "yolov11n.pt")

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4"
        )

        self.llm_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            device_map="auto",
            quantization_config=self.quant_config,
            trust_remote_code=True
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/phi-2",
            padding_side="left"
        )

        self.detector = None  # 지연 초기화
        self.class_map = {}
        class_info = config.get("class_info", {})
        for label_str, label_id in class_info.items():
            if label_str.isdigit():
                # many selections
                pass
            else:
                # only 2 selections
                pass

        inv_map = {}
        for k, v in class_info.items():
            inv_map[v] = int(k)
        self.class_map = inv_map

    def initialize_detector(self, model_path=None):
        if model_path is None:
            model_path = "yolov8n-seg.pt"
        self.detector = YOLO(self.detector_model_path)

    def detect_objects(self, image_path):
        """
        이미지에서 객체를 감지하고, 감지된 객체의 좌표를 반환합니다.
        :param image_path: 이미지 파일 경로
        :return: 감지된 객체의 좌표 리스트
        """
        # 이미지에서 객체 감지
        results = self.detector.predict(
            image_path,
            imgsz=640,
            conf=0.5,
            device=self.device,
            workers=0  # 멀티프로세싱 비활성화
        )

        # 첫 번째 이미지의 결과 // 여러 이미지일 경우 리스트로 반환
        boxes = results[0].boxes.xywhn.cpu().numpy()  # x, y, w, h, class
        if len(boxes) == 0:
            print("no objects detected")
        return boxes

    def generate_label_for_box(self, img, box):
        x_center, y_center, width, height = box
        x1 = int((x_center - width / 2) * img.shape[1])
        y1 = int((y_center - height / 2) * img.shape[0])
        x2 = int((x_center + width / 2) * img.shape[1])
        y2 = int((y_center + height / 2) * img.shape[0])
        crop_img = img[y1:y2, x1:x2]

        try:
            prompt = self._create_prompt(crop_img)
            response = self._query_llm(prompt)
            label = self._parse_response(response)
            return {
                'coordinates': box.tolist(),
                'label': label,
                'class_id': self.class_map.get(label, 3)
            }
        except Exception as e:
            print(f"Error generating label for box {box}: {e}")
            return {
                'coordinates': box.tolist(),
                'label': 'etc',
                'class_id': self.class_map.get('etc', 3)
            }

    def generate_labels(self, image_path, boxes):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        labels = []
        if len(boxes) == 0:
            return labels
        for box in boxes:
            result = self.generate_label_for_box(img, box)
            labels.append(result)
        return labels


    def _create_prompt(self, image):
        """
        Convert cropped image to a short hex string for the LLM prompt.
        """
        _, buffer = cv2.imencode('.jpg', image)
        hex_snippet = buffer.tobytes().hex()[:200]
        prompt = f"""
        [이미지 데이터 헥사: {hex_snippet}]
        다음 이미지 영역을 분석하고 적절한 라벨을 선택하세요:
        - question: 문제 내용 영역
        - answer: 정답 영역
        - figure: 그래프/삽화
        - q_num: 문제 번호
        - q_type: 문제 유형
        - etc: 기타 영역

        반드시 JSON 형식으로 응답: {{"label": "선택한_라벨"}}
        """
        return prompt

    def _query_llm(self, prompt):
        try:
            model_device = next(self.llm_model.parameters()).device
            # 각 tensor를 모델의 디바이스로 이동
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model_device) for k, v in inputs.items()}

            with torch.inference_mode():
                outputs = self.llm_model.generate(
                    **inputs,
                    do_sample=False,
                    max_new_tokens=100,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                return self._query_llm(prompt)  # 재시도
            raise

    def _parse_response(self, response):
        """
        Look for a JSON snippet { "label": "some_label" }
        """
        if not response:
            return 'etc'
        try:
            # response 에 json 형식 체크
            if '{' not in response or '}' not in response:
                return 'etc'
            # json 형식으로 변환 후 label 값 추출
            json_part = response.split('{', 1)[1].split('}', 1)[0] + '}'
            data = json.loads("{" + json_part.strip("{}") + "}")
            return data.get('label', 'etc')
        except Exception as e:
            print(f"Error parsing response: {e}")
            return 'etc'

    def save_label(self, labels, output_path):
        with open(output_path, 'w', encoding='utf-8') as f:
            for label in labels:
                coords_str = ' '.join(map(str, label['coordinates']))
                f.write(f"{label['class_id']} {coords_str}\n")



