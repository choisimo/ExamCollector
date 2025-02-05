import json
import argparse
import os
import threading
import tkinter as tk
from collections import defaultdict
from tkinter import ttk, filedialog, messagebox
from trash.AI.classifier import ProblemClassifier
from trash.comp_image.image_parser import ImageProcessor
from trash.comp_image.docu_parser import DocumentParser


class DocumentProcessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = self.load_config()
        self.processor = None
        self.setup_ui()

        # 컴포넌트 초기화
        self.image_processor = ImageProcessor(
            self.config.get("paths", {}).get("image_dir", "output_images")
        )
        self.document_parser = DocumentParser(self.image_processor)
        self.classifier = ProblemClassifier(
            self.config.get("ai", {}).get("system_prompt")
        )

    def load_config(self):
        try:
            with open('settings.json', 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            return {}

    def setup_ui(self):
        """GUI 컴포넌트 초기화"""
        # 프레임 생성
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # 파일 선택 버튼
        self.btn_select = ttk.Button(
            self.main_frame,
            text="파일 선택",
            command=self.select_file
        )
        self.btn_select.pack(pady=10)

        # 진행 상태 표시기
        self.progress = ttk.Progressbar(
            self.main_frame,
            mode='indeterminate'
        )
        self.progress.pack(fill=tk.X, padx=10)

        # 로그 출력 영역
        self.log_area = tk.Text(
            self.main_frame,
            height=15,
            state=tk.DISABLED
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    def log(self, message):
        self.root.after(0, self._update_log, message)


    def _update_log(self, message):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, message + "\n")
        self.log_area.config(state=tk.DISABLED)
        self.log_area.see(tk.END)

    def select_file(self):
        file_types = [
            ('문서 파일', '*.docx;*.pptx;*.hwp'),
            ('모든 파일', '*.*')
        ]
        path = filedialog.askopenfilename(filetypes=file_types)
        if path:
            self.process_file(path)

    def start_processing(self):
        self.processor.process_file()

    def open_config(self):
        try:
            os.startfile('settings.json')
        except Exception as e:
            self.log(f"Error opening config file: {e}")

    def process_file(self, path: str):
        """파일 처리 시작"""
        if not path:
            return

        self.btn_select.config(state=tk.DISABLED)
        self.progress.start()

        def worker():
            global text
            try:
                self.log(f"[처리 시작] {os.path.basename(path)}")

                # 문서 파싱
                ext = os.path.splitext(path)[1].lower()
                if ext == '.docx':
                    parsed = self.document_parser.parse_docx(path)
                elif ext == '.pptx':
                    parsed = self.document_parser.parse_ppt(path)
                elif ext == '.hwp':
                    parsed = self.document_parser.parse_hwp(path)
                else:
                    raise ValueError("지원하지 않는 파일 형식")


                # 이미지 OCR 처리
                self.log("[이미지 OCR 처리 시작]")
                ocr_results = []
                for img in os.listdir(self.image_processor.output_dir):
                    if img.lower().endswith(('.png', '.jpg', 'jpeg')):
                        text = self.image_processor.ocr_process(
                            os.path.join(self.image_processor.output_dir, img)
                        )
                    if text:
                        ocr_results.append({
                            "image": img,
                            "text": text
                        })

                # AI 문제 분류
                self.log("[문제 분류 시작]")
                classified = []
                for item in parsed['contents']:
                    if item['type'] == 'text':
                        classification = self.classifier.classify(item['content'])
                        classified.append({
                            **item,
                            "classification": classification
                        })

                        # 결과 저장
                        output_path = os.path.join(
                            self.config.get("paths", {}).get("output_dir", "./output/" + os.path.basename(path)),
                            "result.json"
                        )
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                "metadata": parsed['metadata'],
                                "contents": classified,
                                "ocr_results": ocr_results
                            }, f, indent=2, ensure_ascii=False)

                        self.log(f"[처리 완료] 결과 저장 위치: {output_path}")
                        messagebox.showinfo("완료", "파일 처리가 성공적으로 완료되었습니다.")

            except Exception as e:
                 self.log(f"[오류 발생] {str(e)}")
                 messagebox.showerror("오류", str(e))
            finally:
                self.progress.stop()
                self.btn_select.config(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()


class MainProcessor:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.image_processor = ImageProcessor()
        self.document_parser = DocumentParser(self.image_processor)
        self.classifier = ProblemClassifier()

    def process(self, file_path):
        parsed = self.parse_file(file_path)
        ocr_results = self.process_images(parsed['images'])
        classified = self.classify_contents(parsed['contents'])

        output = {
            "metadata": parsed['metadata'],
            "problems": [],
            "statistics": {
                "total_problems": 0,
                "by_difficulty": defaultdict(int),
                "by_topic": defaultdict(int)
            }
        }

        problem_counter = 1
        for item in parsed['contents']:
            if item['type'] == 'problem':
                problem_data = self._build_problem_data(item, ocr_results)
                output['problems'].append(problem_data)
                output['statistics']['total_problems'] += 1
                output['statistics']['by_difficulty'][problem_data['difficulty']] += 1
                for topic in problem_data['topics']:
                    output['statistics']['by_topic'][topic] += 1

        return output

    def _build_problem_data(self, item, ocr_results):
        return {
            "id": f"P-{item['position']}",
            "number": item['number'],
            "text": "\n".join(item['content']),
            "images": [
                {
                    "path": img_path,
                    "ocr_text": next(
                        (ocr['text'] for ocr in ocr_results
                         if ocr['path'] == img_path), ""
                    )
                } for img_path in item['images']
            ],
            "classification": item['classification']
        }

    # ---------------------------
    # main.py
    # ---------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = DocumentProcessorApp(root)
    root.mainloop()
