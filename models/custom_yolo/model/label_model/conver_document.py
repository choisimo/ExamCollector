import os
import comtypes.client
import win32com.client as win32
from pdf2jpg import pdf2jpg
from pathlib import Path


class DocuConverter:
    @staticmethod
    def pdf_to_jpg(pdf_path: str, output_dir: str = None, dpi: int = 300) -> list:
        """
        PDF 파일을 JPG 이미지로 변환 (여러 페이지 지원)

        :param pdf_path: 변환할 PDF 파일 경로
        :param output_dir: 출력 디렉토리 (기본값: PDF 파일 경로)
        :param dpi: 이미지 해상도
        :return: 생성된 JPG 파일 경로 리스트
        """
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF 파일을 찾을 수 없습니다: {pdf_path}")

        output_dir = output_dir or os.path.dirname(pdf_path)
        result = pdf2jpg.convert_pdf2jpg(pdf_path, output_dir, dpi=dpi, pages="ALL")
        return [os.path.join(output_dir, f) for f in result[0]['output_jpgfiles']]

    @staticmethod
    def word_to_pdf(word_path: str, output_path: str = None) -> str:
        """
        Word 문서를 PDF로 변환

        :param word_path: 변환할 Word 파일 경로
        :param output_path: 출력 PDF 경로 (기본값: 원본 파일명.pdf)
        :return: 생성된 PDF 파일 경로
        """
        if not os.path.exists(word_path):
            raise FileNotFoundError(f"Word 파일을 찾을 수 없습니다: {word_path}")

        word = comtypes.client.CreateObject('Word.Application')
        word.Visible = False
        doc = word.Documents.Open(word_path)

        output_path = output_path or Path(word_path).with_suffix('.pdf').as_posix()
        doc.SaveAs(output_path, FileFormat=17)
        doc.Close()
        word.Quit()
        return output_path

    @staticmethod
    def hwp_to_pdf(hwp_path: str, output_path: str = None) -> str:
        """
        HWP 문서를 PDF로 변환 (한컴오피스 설치 필요)

        :param hwp_path: 변환할 HWP 파일 경로
        :param output_path: 출력 PDF 경로 (기본값: 원본 파일명.pdf)
        :return: 생성된 PDF 파일 경로
        """
        if not os.path.exists(hwp_path):
            raise FileNotFoundError(f"HWP 파일을 찾을 수 없습니다: {hwp_path}")

        hwp = win32.gencache.EnsureDispatch('HWPFrame.HwpObject')
        hwp.RegisterModule('FilePathCheckDLL', 'SecurityModule')
        hwp.Open(hwp_path)

        output_path = output_path or Path(hwp_path).with_suffix('.pdf').as_posix()
        hwp.HAction.GetDefault("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
        hwp.HParameterSet.HFileOpenSave.filename = output_path
        hwp.HParameterSet.HFileOpenSave.Format = "PDF"
        hwp.HAction.Execute("FileSaveAs_S", hwp.HParameterSet.HFileOpenSave.HSet)
        hwp.Quit()
        return output_path

    @staticmethod
    def word_to_jpg(word_path: str, output_dir: str = None, dpi: int = 300) -> list:
        """Word -> PDF -> JPG 변환"""
        pdf_path = DocuConverter.word_to_pdf(word_path)
        return DocuConverter.pdf_to_jpg(pdf_path, output_dir, dpi)

    @staticmethod
    def hwp_to_jpg(hwp_path: str, output_dir: str = None, dpi: int = 300) -> list:
        """HWP -> PDF -> JPG 변환"""
        pdf_path = DocuConverter.hwp_to_pdf(hwp_path)
        return DocuConverter.pdf_to_jpg(pdf_path, output_dir, dpi)
