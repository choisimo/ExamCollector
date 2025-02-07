import os
import atexit
import comtypes.client
import win32com.client as win32
from pdf2image import convert_from_path
from pathlib import Path


# 배치 변환 함수 (여러 PDF 파일을 동시에 변환)
from concurrent.futures import ThreadPoolExecutor, as_completed

from models.custom_yolo.common.error_handling.app_logger import conversion_wrapper
from models.custom_yolo.core.services.db_service import DBService



class DocuConverter:
    def __init__(self):
        # COM 객체는 인스턴스 변수로 관리
        self.word_app = comtypes.client.CreateObject('Word.Application')
        self.word_app.Visible = False
        self.hwp_app = None
        atexit.register(self.cleanup_com)

    @conversion_wrapper
    def cleanup_com(self):
        try:
            if self.word_app:
                self.word_app.Quit()
            if self.hwp_app:
                self.hwp_app.Quit()
        except Exception as e:
            print(f"COM 객체 정리 오류: {e}")

    @staticmethod
    def _sanitize_path(path: str) -> str:
        # 절대 경로로 변환하고, Windows와 Linux 모두에서 사용할 수 있도록 함.
        return os.path.abspath(path)

    # --------------------------------------------------------------------------
    # PDF -> JPG
    # --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def pdf_to_jpg(pdf_path: str, output_dir: str = None, dpi: int = 300, poppler_path: str = None) -> list:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        # DB 값 가져오기
        db_service = DBService()
        row = db_service.get_document_converter()  # 예: id=1 레코드

        # 1) output_dir이 지정되지 않았다면 DB나 pdf_path 폴더를 fallback
        if not output_dir:
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            if row and row.get("output_dir"):
                output_dir = os.path.join(row["output_dir"], base_name)
            else:
                output_dir = os.path.join(os.path.dirname(pdf_path), base_name)
        os.makedirs(output_dir, exist_ok=True)

        # 2) poppler_path가 지정되지 않았다면 DB -> fallback
        if not poppler_path:
            if row and row.get("poppler_path"):
                poppler_path = row["poppler_path"]
            else:
                # 예시 기본 경로
                print("Poppler 경로가 지정되지 않았습니다. 기본 경로 사용.")
                poppler_path = r"C:\Program Files\poppler-21.03.0\Library\bin"

        poppler_path = os.path.normpath(poppler_path)

        # pdfinfo.exe 존재 여부 검사
        if not os.path.exists(os.path.join(poppler_path, "pdfinfo.exe")):
            raise RuntimeError(f"Poppler 유틸리티(pdfinfo.exe)가 발견되지 않았습니다: {poppler_path}")

        # 3) pdf2image 변환
        try:
            images = convert_from_path(
                pdf_path=pdf_path,
                dpi=dpi,
                output_folder=output_dir,
                fmt='jpg',  # 여기서는 'jpg' 고정
                thread_count=4,  # 여기서는 4 고정
                poppler_path=poppler_path
            )
        except Exception as e:
            raise RuntimeError(f"PDF 변환 실패: {str(e)}")

        # 각 이미지 객체는 filename 속성을 가짐
        filenames = [img.filename for img in images if hasattr(img, 'filename')]
        if not filenames:
            raise RuntimeError("PDF 변환 실패: 변환된 JPG 파일이 없습니다.")
        return filenames

    # --------------------------------------------------------------------------
    # Word -> PDF
    # --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def word_to_pdf(word_path: str, output_path: str = None, poppler_path=None) -> str:
        if not os.path.exists(word_path):
            raise FileNotFoundError(f"Word 파일을 찾을 수 없습니다: {word_path}")

        # Windows 형식의 경로로 출력 PDF 경로 생성
        output_path = output_path or str(Path(word_path).with_suffix('.pdf'))
        word_path = DocuConverter._sanitize_path(word_path)

        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        try:
            doc = word.Documents.Open(word_path)
            doc.SaveAs(output_path, FileFormat=17)
            doc.Close()
        except Exception as e:
            print(f"Error converting Word to PDF: {e}")
            raise
        finally:
            word.Quit()
        return output_path

    # --------------------------------------------------------------------------
    # HWP -> PDF
    # --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def hwp_to_pdf(hwp_path: str, output_path: str = None) -> str:
        if not os.path.exists(hwp_path):
            raise FileNotFoundError(f"HWP 파일을 찾을 수 없습니다: {hwp_path}")

        hwp_path = DocuConverter._sanitize_path(hwp_path)
        output_path = output_path or str(Path(hwp_path).with_suffix('.pdf'))

        hwp = win32.gencache.EnsureDispatch('HWPFrame.HwpObject')
        hwp.RegisterModule('FilePathCheckDLL', 'SecurityModule')
        hwp.Open(hwp_path)
        try:
            # HWP 버전 또는 환경에 따라 Format 지정
            if hasattr(hwp.HParameterSet.HFileOpenSave, 'Format'):
                hwp.HParameterSet.HFileOpenSave.Format = "PDF"
            else:
                hwp.HParameterSet.HFileOpenSave.Format = "PDF (16)"
            hwp.HAction.Execute("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
        except Exception as e:
            raise RuntimeError(f"HWP 변환 오류: {str(e)}") from e
        finally:
            hwp.Quit()
        return output_path

    # --------------------------------------------------------------------------
    # Word -> JPG (Word->PDF->JPG)
    # --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def word_to_jpg(word_path: str, output_dir: str = None, dpi: int = 300, poppler_path=None) -> list:
        """Word -> PDF -> JPG 변환"""
        pdf_path = DocuConverter.word_to_pdf(word_path)
        return DocuConverter.pdf_to_jpg(pdf_path, output_dir, dpi, poppler_path)

    # --------------------------------------------------------------------------
    # HWP -> JPG (HWP->PDF->JPG)
    # --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def hwp_to_jpg(hwp_path: str, output_dir: str = None, dpi: int = 300, poppler_path=None) -> list:
        """HWP -> PDF -> JPG 변환"""
        pdf_path = DocuConverter.hwp_to_pdf(hwp_path)
        return DocuConverter.pdf_to_jpg(pdf_path, output_dir, dpi, poppler_path)

# --------------------------------------------------------------------------
# 배치 변환 함수 (여러 PDF를 동시에 JPG 변환)
# --------------------------------------------------------------------------
    @staticmethod
    @conversion_wrapper
    def batch_convert(pdf_list, max_workers=4):
        converter = DocuConverter()
        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(converter.pdf_to_jpg, pdf): pdf
                for pdf in pdf_list
            }
            for future in as_completed(futures):
                pdf_file = futures[future]
                try:
                    res = future.result()
                    # res는 변환된 파일명 리스트
                    results.extend(res)
                except Exception as e:
                    print(f"{futures[future]} 변환 실패: {e}")
        return results
