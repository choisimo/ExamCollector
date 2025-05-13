from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget, QMessageBox
)

from models.custom_yolo.infrastructure.computer_vision.label_model.auto_labeler import AutoLabeler
from models.custom_yolo.infrastructure.computer_vision.label_model.program_gui.document import DocumentConverterTab
from models.custom_yolo.infrastructure.computer_vision.label_model.program_gui.label import LabelingTab
from models.custom_yolo.infrastructure.computer_vision.label_model.program_gui.setting import SettingsTab
from models.custom_yolo.infrastructure.computer_vision.learning_model.training_model import TrainingModel
from models.custom_yolo.common.memory_monitor import (ResourceGraphWidget)
from models.custom_yolo.infrastructure.storage.SettingsManager import Settings
import torch
import win32com.client
import logging

# configure logging for GUI
logging.basicConfig(level=logging.INFO)

class MainWindow(QMainWindow):
    """시험지 자동 라벨링 및 학습 시스템 메인 윈도우"""
    def __init__(self):
        super().__init__()
        self.settings = Settings().all_settings
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("시험지 자동 라벨링 및 학습 시스템")
        self.resize(1200, 800)
        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # 1) Construct an AutoLabeler with the current settings
        auto_labeler = AutoLabeler()
        model_path = self.settings.get("detector_model_path", None)
        try:
            auto_labeler.initialize_detector(model_path)
            logging.info(f"Detector initialized with model: {model_path}")
        except Exception as e:
            logging.error(f"Failed to initialize detector: {e}")
            QMessageBox.warning(self, "Detector Error", f"모델 초기화 중 오류가 발생했습니다: {e}")

        # 2) Add Labeling Tab
        labeling_tab = LabelingTab(auto_labeler)
        idx = tabs.addTab(labeling_tab, "Labeling")
        tabs.setTabToolTip(idx, "문제 이미지에 라벨을 자동 생성합니다.")

        # 4) Resource Monitor Tab
        resource_tab = ResourceGraphWidget(interval=1000)
        idx = tabs.addTab(resource_tab, "Resource Monitor")
        tabs.setTabToolTip(idx, "시스템 자원 사용량 실시간 모니터링")

        # Inside MainWindow's init_ui method:
        training_tab = TrainingModel()
        settings_tab = SettingsTab()

        # settings_saved 신호 연결
        settings_tab.settings_updated.connect(training_tab.load_settings_into_ui)
        # resource_updated 신호 연결
        settings_tab.resource_updated.connect(training_tab.update_resources)

        idx = tabs.addTab(training_tab, "Training")
        tabs.setTabToolTip(idx, "모델 학습 설정 및 실행")
        idx = tabs.addTab(settings_tab, "Settings")
        tabs.setTabToolTip(idx, "환경 설정 관리 및 저장")

        docu_converter_tab = DocumentConverterTab()
        idx = tabs.addTab(docu_converter_tab, "Document")
        tabs.setTabToolTip(idx, "문서 변환 및 OCR 처리")

    def closeEvent(self, event):
        # 모든 탭(또는 워커가 있는 객체) 순회하여 작업 스레드 종료
        for i in range(self.centralWidget().count()):
            tab = self.centralWidget().widget(i)
            if hasattr(tab, 'worker') and tab.worker and tab.worker.isRunning():
                try:
                    tab.worker.quit()
                    tab.worker.wait(1000)
                    logging.info(f"Worker on tab {i} quit cleanly.")
                except Exception:
                    tab.worker.terminate()
                    tab.worker.wait(1000)
                    logging.warning(f"Worker on tab {i} forced terminated.")
        # GPU 자원 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # COM 객체 정리
        try:
            win32com.client.Dispatch('Word.Application').Quit()
        except Exception:
            pass
        event.accept()
