import os
from datetime import time
from docx import Document
from pptx import Presentation
import json
import argparse
from docx import Document
from pptx import Presentation
import pytesseract
from PIL import Image
import ollama

from components.AI.classifier import ProblemClassifier
from components.comp_image.extractor import ImageProcessor
from components.comp_image.parser import DocumentParser


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
        else:
            raise Exception("Unsupported file format")

        processed = []
        for item in parsed['content']:
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
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--file", required=True, help="File path to process")
    arg_parser.add_argument('-o', '--output', default='output.json', help='출력 JSON 파일 경로')
    args = arg_parser.parse_args()

    processor = MainProcessor()
    try:
        result = processor.process_file(args.input_file)
    except Exception as e:
        print(f"Error processing file: {e}")
        exit(1)

    try:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Output written to {args.output} \n total_problems: {len(result['problems'])}\n total images: {result['images']} ")
    except Exception as e:
        print(f"Error writing output: {e}")
        exit(1)