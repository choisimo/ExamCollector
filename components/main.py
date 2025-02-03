import json
import argparse
import os
import threading
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from components.AI.classifier import ProblemClassifier
from components.comp_image.extractor import ImageProcessor
from components.comp_image.parser import DocumentParser


class DocumentProcessorApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.config = self.load_config()
        self.processor = None
        self.setup_ui()

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
        self.progress.pack(fill=tk.X, padx=20)

        # 로그 출력 영역
        self.log_area = tk.Text(
            self.main_frame,
            height=15,
            state=tk.DISABLED
        )
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

    # In DocumentProcessorApp
    def log_message(self, msg: str):
        self.root.after(0, self._actual_log_message, msg)

    def _actual_log_message(self, msg):
        self.log_area.config(state=tk.NORMAL)
        self.log_area.insert(tk.END, f"{msg}\n")
        self.log_area.config(state=tk.DISABLED)

    def select_file(self):
        """파일 선택 다이얼로그"""
        file_types = [
            ('Word 문서', '*.docx'),
            ('PowerPoint 문서', '*.pptx'),
            ('hwp 문서', '*.hwp'),
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
            try:
                processor = MainProcessor(self.config)
                result = processor.process(path)

                output_path = os.path.join(
                    self.config['output_dir'],
                    'result.json'
                )
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(result, f, indent=2)

                self.log_message(f"결과 저장 완료: {output_path}")
                messagebox.showinfo("완료", "처리 성공!")

            except Exception as e:
                self.log_message(f"오류: {str(e)}")
                messagebox.showerror("오류", str(e))
            finally:
                self.progress.stop()
                self.btn_select.config(state=tk.NORMAL)

        threading.Thread(target=worker, daemon=True).start()


class MainProcessor:
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.document_parser = DocumentParser(self.image_processor)
        self.classifier = ProblemClassifier()

    def process_docx(self, file_path):
        """
        :param file_path: DOCX, PPTX file path
        :return:
            - metadata: file metadata
            - contents: text contents
            - images: image information
        """

        if file_path.endswith(".docx"):
            return self.document_parser.parse_docx(file_path)
        elif file_path.endswith(".pptx"):
            return self.document_parser.parse_ppt(file_path)
        elif file_path.endswith(".hwp"):
            return self.document_parser.parse_hwp(file_path)
        else:
            raise Exception("Unsupported file format")

        processed = []

        for item in parsed['contents']:
            if imte['type'] == 'text':
                classified = self.classifier.classify(item['content'])
                processed.append({
                    "type": "text",
                    "content": item['content'],
                    "classification": classified
                })

        return {
            "metadata": parsed['metadata'],
            "contents": processed,
            "images": parsed['images'],
            "directory": self.image_processor.output_dir
        }


    # ---------------------------
    # main.py
    # ---------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("800x600")
    app = DocumentProcessorApp(root)
    root.mainloop()
