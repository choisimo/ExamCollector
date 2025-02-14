import os

from PyQt5.QtCore import pyqtSignal, QUrl, QTimer
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QPlainTextEdit, QGroupBox, QLineEdit, QSpinBox, \
    QFileDialog, QFormLayout, QLabel, QHBoxLayout, QMessageBox
from models.custom_yolo.common.device import DeviceChecker
from models.custom_yolo.core.services.db_service import DBService
from models.custom_yolo.infrastructure.computer_vision.learning_model.training_worker import TrainingWorker

# -------------------------------------------------------------
# Resource cache: training resource 초기화를 위한 캐시
_resource_cache = {}


class TrainingModel(QWidget):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, db_service: DBService):
        super().__init__()
        self.db_service = db_service
        self.current_config_id = None  # 현재 활성 설정 ID 추적
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)

        self.model_path_edit = QLineEdit()
        self.data_yaml_edit = QLineEdit()
        self.epochs_spin = QSpinBox()
        self.batch_spin = QSpinBox()

        self.worker = None

        self.init_ui()
        self.load_training_config_into_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 로그 출력
        layout.addWidget(QLabel("Training Log"))
        layout.addWidget(self.log_area)

        # 모델 구성 그룹
        grp_model = QGroupBox("Training Model Configuration")
        f_layout = QFormLayout()

        # YOLO model path
        btn_select_model = QPushButton("Browse")
        btn_select_model.clicked.connect(self.select_model_file)
        row_model = self._make_row(self.model_path_edit, btn_select_model)
        f_layout.addRow("select YOLO model [.pt] : ", row_model)

        # Data YAML path
        btn_select_data = QPushButton("Browse")
        btn_select_data.clicked.connect(self.select_data_file)
        row_data = self._make_row(self.data_yaml_edit, btn_select_data)
        f_layout.addRow("select custom data [.yaml] : ", row_data)

        # epochs
        self.epochs_spin.setRange(1, 1000)
        f_layout.addRow("epochs:", self.epochs_spin)

        # batch
        self.batch_spin.setRange(1, 128)
        f_layout.addRow("batch size:", self.batch_spin)

        grp_model.setLayout(f_layout)
        layout.addWidget(grp_model)

        # 버튼 레이아웃
        btn_layout = QHBoxLayout()
        train_btn = QPushButton("학습 시작")
        train_btn.clicked.connect(self.start_training)
        btn_layout.addWidget(train_btn)

        cancel_btn = QPushButton("학습 중단")
        cancel_btn.clicked.connect(self.cancel_training)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _make_row(self, line_edit, btn):
        """Helper to horizontally place a line-edit + button in a single row."""
        w = QWidget()
        l = QHBoxLayout()
        l.addWidget(line_edit)
        l.addWidget(btn)
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)
        return w

    def load_training_config(self, config: dict):
        """
        DB에서 training_config 정보를 읽어서
        """
        _latest_config = self.db_service.get_training_config()
        if config:
            self.current_config_id = config['id']
            self._update_ui_from_config(config)
        else:
            self._create_default_config()

    def _update_ui_from_config(self, config: dict):
        """UI 요소 업데이트"""
        self.model_path_edit.setText(config.get('YOLO_model', ''))
        self.data_yaml_edit.setText(config.get('data_yaml', ''))
        self.epochs_spin.setValue(config.get('epochs', 50))
        self.batch_spin.setValue(config.get('batch_size', 16))

    def _create_default_config(self):
        """기본 설정 생성"""
        default_data = {
            'epochs': 50,
            'batch_size': 16,
            'device': 'cpu',
            'imgsz': 640,
            'YOLO_model': 'yolov11n.pt',
            'data_yaml': 'custom_data.yaml'
        }
        self.current_config_id = self.db_service.insert_training_config(default_data)

    def update_resources(self, resources):
        """
        메인 창에서 리소스가 변경되었을 때 호출되는 메서드입니다.
        리소스는 다른 탭에서 업데이트된 것을 반영합니다.
        """
        self.current_resources.update(resources)
        self.log_area.appendPlainText(f"Resource updated: {resources}")

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "YOLO Model (*.pt)")
        # change custom model path in db
        if path:
            self.model_path_edit.setText(path)
            self._update_db_config({'YOLO_model': path})

    def _update_db_config(self, new_config):
        if self.
    def select_data_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML files (*.yaml *.yml)")
        if path:
            self.data_yaml_edit.setText(path)
            row = self.db_service.get_training_config()
            if row:
                self.db_service.update_training_config(row['id'], path)
            else:
                self.db_service.insert_training_config(
                    epochs=50, batch_size=16, device="cpu", imgsz=640,
                    YOLO_model="yolov8n.pt", custom_model_path="", data_yaml=path,
                    data_config_id=1, results_config_id=1, model_paths_id=1
                )

    def start_training(self):
        # device checking
        row = self.db_service.get_training_config()
        if row:
            device = row.get("device", "cpu")
            try:
                DeviceChecker().validate_training_device(device)
            except ValueError as e:
                QMessageBox.critical(self, "Device Error", str(e))
                return
        else:
            QMessageBox.critical(self, "Error", "No training configuration found! Please set up training config.")
            return

        model_path = self.model_path_edit.text().strip()
        data_yaml = self.data_yaml_edit.text().strip()
        epochs = self.epochs_spin.value()
        batch = self.batch_spin.value()

        self.worker = TrainingWorker(
            model_path=model_path,
            data_yaml=data_yaml,
            epochs=epochs,
            batch=batch,
            device=device
        )
        self.worker.log_signal.connect(self.log_area.appendPlainText)
        self.worker.progress.connect(lambda p: self.log_area.appendPlainText(f"{p}%"))
        self.worker.finished.connect(self.handle_training_finished)
        self.worker.error.connect(self.handle_training_error)
        self.worker.start()

    def cancel_training(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            QMessageBox.information(self, "Cancelled", "학습 작업이 중단 되었습니다.")
            self.log_area.appendPlainText("학습 중단 요청됨.")

    def handle_training_finished(self, best_model):
        self.log_area.appendPlainText(f"학습 완료! best_model: {best_model}")

    def handle_training_error(self, err_msg):
        self.log_area.appendPlainText(f"학습 오류 발생: {err_msg}")

