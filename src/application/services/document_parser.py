import io
from fastapi import UploadFile
from typing import Optional
from docx import Document as DocxDocument
from PyPDF2 import PdfReader

class DocumentParserService:
    @staticmethod
    async def parse(upload_file: UploadFile) -> str:
        content = await upload_file.read()
        filename = upload_file.filename.lower()
        if filename.endswith('.pdf'):
            reader = PdfReader(io.BytesIO(content))
            text = ''
            for page in reader.pages:
                page_text = page.extract_text() or ''
                text += page_text
            return text
        elif filename.endswith('.docx'):
            doc = DocxDocument(io.BytesIO(content))
            return '\n'.join([p.text for p in doc.paragraphs])
        elif filename.endswith('.txt'):
            return content.decode('utf-8', errors='ignore')
        elif filename.endswith('.hwp'):
            # HWP parsing requires additional library; stub implementation
            try:
                import hwp5
                doc = hwp5.HWP5File(io.BytesIO(content))
                return getattr(doc, 'body_text', '')
            except ImportError:
                return ''
        else:
            raise ValueError('Unsupported file type')
