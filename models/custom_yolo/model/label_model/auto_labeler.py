import json
import threading

import cv2
import torch
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from concurrent.futures import ThreadPoolExecutor

"""
    for loop 내부에 라벨 생성 코드의 들여쓰기를 수정하여 각 객체에 대해 LLM 모델을 적용
    라벨 결과 저장 시, 매개변수 순서를 labels, output_path로 일치시킴
    라벨 출력 시, 딕셔너리 키 이름을 "coordinates"로 통일
"""

class AutoLabeler:
    def __init__(self, device='cpu'):
        # object detection model
        self.device = 'cpu'
        self.detector = None # 지연 초기화
        self.class_map = {
            'question': 0,
            'answer': 1,
            'figure': 2,
            'etc': 3,
            'q_num': 4,
            'q_type': 5
        }
        self.llm_lock = threading.Lock()
        self.default_model = "microsoft/phi-4"

        # Local LLM model
        self.llm_model = AutoModelForCausalLM.from_pretrained(
            self.default_model,
            #device_map="cuda" if torch.cuda.is_available() else "cpu",
            device_map=self.device,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            token=True,
            trust_remote_code=True
        )
        # self.llm_model = AutoModel.from_pretrained(self.default_model)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.default_model,
            #device_map="cuda" if torch.cuda.is_available() else "cpu",
            device_map=self.device,
            torch_dtype=torch.float16,
            load_in_4bit=True,
            token=True,
            trust_remote_code=True
        )

    def initialize_detector(self, model_path='yolov8n-seg.pt'):
        self.detector = YOLO(model_path)

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

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(self.generate_label_for_box, img, box) for box in boxes]
            labels = [f.result() for f in futures]
            return labels

    def _create_prompt(self, image):
        _, buffer = cv2.imencode('.jpg', image)
        return f"""
        [이미지 데이터 헥사: {buffer.tobytes().hex()[:200]}]
        다음 이미지 영역을 분석하고 적절한 라벨을 선택하세요:
        - question: 문제 내용 영역
        - answer: 정답 영역
        - figure: 그래프/삽화
        - q_num: 문제 번호
        - q_type: 문제 유형
        - etc: 기타 영역
        
        반드시 JSON 형식으로 응답: {{"label": "선택한_라벨"}}
        """

    def _query_llm(self, prompt):
        try:
            with self.llm_lock:
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                outputs = self.llm_model.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                print(f"LLM Response : {response}")
                return response
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                torch.cuda.empty_cache()
                return self._query_llm(prompt)  # 재시도
            raise

    def _parse_response(self, response):
        if not response:
            return 'etc'
        try:
            # response 에 json 형식 체크
            if '{' not in response or '}' not in response:
                return 'etc'
            # json 형식으로 변환 후 label 값 추출
            json_part = response.split('{')[1].split('}')[0] + '}'
            data = json.loads('{' + json_part)
            return data.get('label', 'etc')
        except Exception as e:
            print(f"Error parsing response: {e}")
            return 'etc'

    def save_labels(self, labels, output_path):
        """
        라벨을 파일로 저장합니다.
        :param labels: 라벨 리스트
        :param output_path: 출력 파일 경로
        """
        with open(output_path, 'w') as f:
            for label in labels:
                line = f"{label['class_id']} {' '.join(map(str, label['coordinates']))}\n"
                f.write(line)
