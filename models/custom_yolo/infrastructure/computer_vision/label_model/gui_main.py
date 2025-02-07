from PyQt5.QtWidgets import (
    QMainWindow, QTabWidget
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


class MainWindow(QMainWindow):
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
        auto_labeler.initialize_detector(self.settings.get("detector_model_path", None))

        # 2) Add Labeling Tab
        labeling_tab = LabelingTab(auto_labeler)
        tabs.addTab(labeling_tab, "Labeling")

        # 4) Resource Monitor Tab
        resource_tab = ResourceGraphWidget(interval=1000)
        tabs.addTab(resource_tab, "Resource Monitor")

        # Inside MainWindow's init_ui method:
        training_tab = TrainingModel()
        settings_tab = SettingsTab()

        # settings_saved 신호 연결
        settings_tab.settings_updated.connect(training_tab.load_settings_into_ui)
        # resource_updated 신호 연결
        settings_tab.resource_updated.connect(training_tab.update_resources)

        tabs.addTab(training_tab, "Training")
        tabs.addTab(settings_tab, "Settings")

        docu_converter_tab = DocumentConverterTab()
        tabs.addTab(docu_converter_tab, "Document")

    def closeEvent(self, event):
        # 모든 탭(또는 워커가 있는 객체) 순회하여 작업 스레드 종료
        for i in range(self.centralWidget().count()):
            tab = self.centralWidget().widget(i)
            if hasattr(tab, 'worker') and tab.worker and tab.worker.isRunning():
                tab.worker.terminate()
                tab.worker.wait(1000)  # 1초 그레이스 기간
        # GPU 자원 해제
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # COM 객체 정리
        try:
            win32com.client.Dispatch('Word.Application').Quit()
        except Exception:
            pass
        event.accept()


