import os
import tempfile  # 올바른 임포트로 수정
from fastapi import UploadFile
from typing import List, Optional
from models.custom_yolo.core.services.convert_document import DocuConverter

class DocumentConverterService:
    """
    문서 파일을 JPEG 이미지로 변환하는 서비스.
    
    이 서비스는 DocuConverter를 활용하여 PDF, DOCX 등의 문서를 
    JPEG 이미지 시리즈로 변환합니다. OCR이나 이미지 분석 등의 
    추가 처리 작업을 위한 준비 단계로 사용됩니다.
    """
    
    def __init__(self, converter_options: Optional[dict] = None):
        """
        문서 변환 서비스를 초기화합니다.
        
        Args:
            converter_options: 변환기의 선택적 설정 (DPI, 품질 설정 등)
        """
        # Optionally, load settings for converter
        # e.g., poppler path, dpi
        self.converter = DocuConverter(**(converter_options or {}))

    async def convert(self, upload_file: UploadFile) -> List[bytes]:
        """
        업로드된 문서를 JPEG 이미지 바이트 목록으로 변환합니다.
        
        이 메서드는 업로드된 파일을 임시 저장한 후, DocuConverter를 사용하여
        하나 이상의 JPEG 이미지로 변환하고, 이를 바이트 배열 목록으로 반환합니다.
        
        Args:
            upload_file: FastAPI에서 제공하는 업로드된 파일 객체
            
        Returns:
            JPEG 이미지를 표현하는 바이트 배열의 목록
            
        Raises:
            ValueError: 파일을 변환할 수 없는 경우
            IOError: 파일 작업 중 문제가 발생한 경우
        """
        content = await upload_file.read()
        # Save upload to a temp file
        suffix = os.path.splitext(upload_file.filename)[1]
        
        tmp_path = ""
        jpg_paths = []  # 예외 처리를 위해 기본값으로 초기화
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
                
            # Perform conversion
            jpg_paths = self.converter.convert([tmp_path])
            images: List[bytes] = []
            
            for path in jpg_paths:
                with open(path, 'rb') as f:
                    images.append(f.read())
                    
            return images
            
        except Exception as e:
            # 구체적인 예외 처리 추가 가능
            raise ValueError(f"문서 변환 실패: {str(e)}") from e
            
        finally:
            # 임시 파일 정리
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
                
            # 변환된 jpg 파일 정리
            for path in jpg_paths:
                if os.path.exists(path):
                    os.remove(path)
