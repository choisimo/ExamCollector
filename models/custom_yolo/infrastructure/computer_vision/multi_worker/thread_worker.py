# Worker 클래스 정의 (QRunnable 사용)
import os

from PyQt5.QtCore import QRunnable, pyqtSlot, QMetaObject, Qt, Q_ARG

from models.custom_yolo.core.services.convert_document import DocuConverter
from models.custom_yolo.common.observer_pattern.observer_registry import DocumentConversionSignals
from models.custom_yolo.infrastructure.storage.SettingsManager import Settings

from concurrent.futures import ThreadPoolExecutor

_executor = ThreadPoolExecutor(max_workers=1)


class SaveSettingsWorker(QRunnable):
    def __init__(self, settings, save_path):
        super().__init__()
        self.settings = settings
        self.save_path = save_path

    @pyqtSlot()
    def run(self):
        # 파일 I/O는 이 run() 내에서 처리됨
        try:
            # 예시: settings_manager의 update_and_save() 메서드 호출
            Settings().update_and_save(self.settings, self.save_path)
        except Exception as e:
            # 오류 발생 시 로깅 혹은 별도 처리 (메시지 박스는 메인 스레드에서 처리해야 함)
            print("Error saving settings:", e)


class LoadSettingsWorker(QRunnable):
    def __init__(self, file_path, callback):
        super().__init__()
        self.file_path = file_path
        self.callback = callback  # 작업 완료 후 메인 스레드에서 호출할 콜백 함수

    @pyqtSlot()
    def run(self):
        # 파일 I/O는 여기서 처리 (메인 스레드와 분리됨)
        try:
            Settings().reload_settings(self.file_path)
            new_settings = Settings().all_settings
        except Exception as e:
            new_settings = {}
            print("Error loading settings:", e)
        # 메인 스레드에서 UI 업데이트를 수행하도록 invokeMethod 사용
        QMetaObject.invokeMethod(self.callback, "update_ui_fields_from_worker",
                                 Qt.QueuedConnection,
                                 Q_ARG(dict, new_settings))


class DocumentConversionRunnable(QRunnable):
    def __init__(self, doc_path, dpi=300, output_dir=None, poppler_path=None):
        super(DocumentConversionRunnable, self).__init__()
        self.doc_path = doc_path
        self.dpi = dpi
        self.signals = DocumentConversionSignals()

    @pyqtSlot()
    def run(self):
        ext = os.path.splitext(self.doc_path)[1].lower()
        output_dir = os.path.dirname(self.doc_path)
        try:
            self.signals.log.emit(f"문서 변환 시작: {self.doc_path}")
            if ext in [".doc", ".docx"]:
                jpg_files = DocuConverter.word_to_jpg(self.doc_path, output_dir, dpi=self.dpi)
            elif ext == ".hwp":
                jpg_files = DocuConverter.hwp_to_jpg(self.doc_path, output_dir, dpi=self.dpi)
            elif ext == ".pdf":
                jpg_files = DocuConverter.pdf_to_jpg(self.doc_path, output_dir, dpi=self.dpi)
            else:
                raise Exception("지원되지 않는 파일 형식입니다.")
            self.signals.log.emit(f"변환 완료: {len(jpg_files)}개의 JPG 파일 생성됨")
            self.signals.finished.emit(jpg_files)
        except Exception as e:
            self.signals.error.emit(e)
