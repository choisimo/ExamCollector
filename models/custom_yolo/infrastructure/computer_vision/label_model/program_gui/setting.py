import json
import os

from PyQt5.QtCore import pyqtSignal, QTimer, QThreadPool, pyqtSlot
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QGroupBox, QFormLayout, QLineEdit, QPushButton, QSpinBox, QComboBox, \
    QHBoxLayout, QRadioButton, QLabel, QPlainTextEdit, QFileDialog, QMessageBox

from models.custom_yolo.common.settings.settings_manager import SettingsManager
from models.custom_yolo.core.services.db_service import DBService
from models.custom_yolo.infrastructure.computer_vision.multi_worker.thread_worker import SaveSettingsWorker, \
    LoadSettingsWorker, _executor


class SettingsTab(QWidget):
    def __init__(self):
        super().__init__()
        self.db_service = DBService()
        self.settings_manger = SettingsManager()
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # Training Model Configuration
        grp_train_box = QGroupBox("Training Configuration")
        train_box_layout = QFormLayout()
        ####################################################################################
        # 모델 설정 그룹
        grp_model = QGroupBox("Base Model Configuration")
        model_layout = QFormLayout()
        # detector path
        self.detector_path_edit = QLineEdit(self.settings.get("detector_model_path", ""))
        btn_browse_detector = QPushButton("Browse")
        btn_browse_detector.clicked.connect(lambda: self.browse_file(self.detector_path_edit))
        row_widget1 = self._path_row(self.detector_path_edit, btn_browse_detector)
        model_layout.addRow("YOLO Model :", row_widget1)
        # LLM model
        self.llm_model_edit = QLineEdit(self.settings.get("llm_model", "microsoft/phi-2"))
        model_layout.addRow("LLM Model:", self.llm_model_edit)
        grp_model.setLayout(model_layout)
        layout.addWidget(grp_model)
        ####################################################################################
        # 데이터 설정 그룹
        grp_data = QGroupBox("Learning Data")
        data_layout = QFormLayout()

        # 기존 settings에서 값 가져오기
        data_settings = self.settings.get("data", {})
        data_path = data_settings.get("path", "")
        train_path = data_settings.get("train", "")
        val_path = data_settings.get("val", "")

        # train_path와 val_path에서 data_path 부분 제거
        def get_relative_path(full_path, base_path):
            if base_path and full_path.startswith(base_path):
                rel = full_path[len(base_path):]
                # 만약 경로 구분자가 맨 앞에 있다면 제거
                return rel.lstrip(os.sep)
            return full_path

        relative_train = get_relative_path(train_path, data_path)
        relative_val = get_relative_path(val_path, data_path)

        # UI 위젯 생성 시 적용
        self.path_edit = QLineEdit(data_path)
        self.train_path_edit = QLineEdit(relative_train)
        self.val_path_edit = QLineEdit(relative_val)

        btn_browse_train = QPushButton("Browse")
        btn_browse_train.clicked.connect(lambda: self.browse_dir(self.train_path_edit))
        btn_browse_val = QPushButton("Browse")
        btn_browse_val.clicked.connect(lambda: self.browse_dir(self.val_path_edit))
        btn_browse_data = QPushButton("Browse")
        btn_browse_data.clicked.connect(lambda: self.browse_dir(self.path_edit))

        data_layout.addRow("Data Path:", self._path_row(self.path_edit, btn_browse_data))
        data_layout.addRow("Train Data:", self._path_row(self.train_path_edit, btn_browse_train))
        data_layout.addRow("Val Data:", self._path_row(self.val_path_edit, btn_browse_val))

        grp_data.setLayout(data_layout)
        ####################################################################################
        grp_params = QGroupBox("Parameters")
        training_layout = QFormLayout()

        training_config = self.settings.get("training", {})
        epochs_value = training_config.get("epochs", 50)
        batch_value = training_config.get("batch_size", 16)
        imgsz_value = training_config.get("imgsz", 640)

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 1000)
        self.spin_epochs.setValue(epochs_value)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 128)
        self.spin_batch.setValue(batch_value)

        self.combo_imgsz = QComboBox()
        # just as an example
        self.combo_imgsz.addItems(["320", "640", "1280"])
        # attempt to set the current to whatever is in config
        index_imgsz = self.combo_imgsz.findText(str(imgsz_value))
        if index_imgsz >= 0:
            self.combo_imgsz.setCurrentIndex(index_imgsz)

        training_layout.addRow("Epochs:", self.spin_epochs)
        training_layout.addRow("Batch Size:", self.spin_batch)
        training_layout.addRow("Img Size:", self.combo_imgsz)

        # Device radio
        grp_device = QGroupBox("Device")
        device_str = training_config.get("device", "cpu")
        device_layout = QHBoxLayout()
        self.device_choices = {
            "CUDA (Nvidia)": "cuda",
            "CPU": "cpu",
            "AMD (HIP)": "hip"
        }
        self.rb_cpu = QRadioButton("CPU")
        self.rb_cuda = QRadioButton("CUDA (Nvidia)")
        self.rb_hip = QRadioButton("AMD (HIP)")
        self.device_btns = [self.rb_cpu, self.rb_cuda, self.rb_hip]
        device_layout.addWidget(self.rb_cpu)
        device_layout.addWidget(self.rb_cuda)
        device_layout.addWidget(self.rb_hip)
        grp_device.setLayout(device_layout)
        training_layout.addRow("Compute Device:", grp_device)

        # Check the radio that matches
        if device_str.lower() == "cuda":
            self.rb_cuda.setChecked(True)
        elif device_str.lower() == "hip":
            self.rb_hip.setChecked(True)
        else:
            self.rb_cpu.setChecked(True)

        grp_params.setLayout(training_layout)
        train_box_layout.addWidget(grp_data)
        train_box_layout.addWidget(grp_params)
        train_box_layout.addWidget(grp_device)

        grp_train_box.setLayout(train_box_layout)

        # 최종적으로 main 레이아웃에 추가
        layout.addWidget(grp_train_box)
        ####################################################################################
        # Label Save Mode 설정 (auto 또는 manual 선택)
        grp_label = QGroupBox("Label Save Mode")
        label_layout = QHBoxLayout()
        self.combo_label_save_mode = QComboBox()
        self.combo_label_save_mode.addItems(["auto", "manual"])
        # 기본값 설정: settings["other_settings"]["label_save_mode"]
        other_settings = self.settings.get("other_settings", {})
        label_save_mode = other_settings.get("label_save_mode", "auto")
        index_mode = self.combo_label_save_mode.findText(label_save_mode)
        if index_mode != -1:
            self.combo_label_save_mode.setCurrentIndex(index_mode)
        label_layout.addWidget(QLabel("Label Save Mode:"))
        label_layout.addWidget(self.combo_label_save_mode)
        grp_label.setLayout(label_layout)
        layout.addWidget(grp_label)

        # selected save location
        grp_save = QGroupBox("Save Location")
        save_layout = QFormLayout()

        # Existing save path
        self.save_path_edit = QLineEdit(self.db_service.get_global_setting('settings_path'))  # 기본 경로 표시
        btn_browse_save = QPushButton("Browse")
        btn_browse_save.clicked.connect(self.browse_save_location)
        save_layout.addRow("Settings Path:", self._path_row(self.save_path_edit, btn_browse_save))

        # New row for results path
        self.results_path_edit = QLineEdit(self.settings.get("results", {}).get("path", ""))
        btn_browse_results = QPushButton("Browse")
        btn_browse_results.clicked.connect(lambda: self.browse_dir(self.results_path_edit))
        save_layout.addRow("Results Path:", self._path_row(self.results_path_edit, btn_browse_results))

        grp_save.setLayout(save_layout)
        layout.addWidget(grp_save)

        # Full JSON read-only display (optional)
        grp_json = QGroupBox("Raw JSON (read-only)")
        json_layout = QVBoxLayout()
        self.json_edit = QPlainTextEdit(json.dumps(self.settings, indent=2))
        self.json_edit.setReadOnly(True)
        json_layout.addWidget(self.json_edit)
        grp_json.setLayout(json_layout)
        layout.addWidget(grp_json)
        ####################################################################################
        # Buttons
        # 버튼 영역: 불러오기 / 저장
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("설정 불러오기")
        btn_load.clicked.connect(self.load_from_disk)
        btn_layout.addWidget(btn_load)
        btn_save = QPushButton("저장하기")
        btn_save.clicked.connect(self.save_to_disk)
        btn_layout.addWidget(btn_save)

        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def _path_row(self, line_edit, btn):
        row_widget = QWidget()
        row_layout = QHBoxLayout()
        row_layout.addWidget(line_edit)
        row_layout.addWidget(btn)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_widget.setLayout(row_layout)
        return row_widget

    def browse_file(self, line_edit):
        current_path = line_edit.text()
        initial_path = os.path.dirname(current_path) if current_path else ""
        path, _ = QFileDialog.getOpenFileName(self, "Browse file", initial_path, "All Files (*.*)")
        if path:
            line_edit.setText(path)

    def browse_dir(self, line_edit):
        current_path = line_edit.text()
        initial_dir = current_path if os.path.isdir(current_path) else ""
        d = QFileDialog.getExistingDirectory(self, "Browse directory", initial_dir)
        if d:
            line_edit.setText(d)

    def browse_save_location(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save Settings As...", "", "JSON Files (*.json)")
        if path:
            self.save_path_edit.setText(path)

    def save_to_disk(self):
        training_config = self.db_service.get_training_config()
        if training_config:
            # settings에 저장된 값으로 업데이트
            self.db_service.update_training_config(training_config['id'], {
                "YOLO_model": self.model_path_edit.text(),
                "data_yaml": self.data_yaml_edit.text(),
                "epochs": self.epochs_spin.value(),
                "batch_size": self.batch_spin.value(),
                "device": self.device_choices[self.device_btns[0].text()]
            })

    def update_ui_fields(self):
        """UI 위젯에 현재 settings 값을 적용합니다."""
        # 모델 설정
        self.detector_path_edit.setText(self.settings.get("detector_model_path", ""))
        self.llm_model_edit.setText(self.settings.get("llm_model", ""))
        # 데이터 설정
        data = self.settings.get("data", {})
        self.path_edit.setText(data.get("path", ""))
        self.train_path_edit.setText(data.get("train", ""))
        self.val_path_edit.setText(data.get("val", ""))
        # Training 설정
        training = self.settings.get("training", {})
        self.spin_epochs.setValue(training.get("epochs", 50))
        self.spin_batch.setValue(training.get("batch_size", 16))
        imgsz = training.get("imgsz", 640)
        index_imgsz = self.combo_imgsz.findText(str(imgsz))
        if index_imgsz >= 0:
            self.combo_imgsz.setCurrentIndex(index_imgsz)
        # Device 설정
        device = training.get("device", "cpu").lower()
        for rb in self.device_btns:
            if self.device_choices[rb.text()].lower() == device:
                rb.setChecked(True)
            else:
                rb.setChecked(False)
        # Raw JSON 업데이트
        self.json_edit.setPlainText(json.dumps(self.settings, indent=2))

    def load_from_disk(self):
        # 파일 선택 다이얼로그를 통해 설정 파일 선택 (확인 버튼 누르면 바로 파일 경로 반환)
        file_path, _ = QFileDialog.getOpenFileName(self, "Load Settings", "", "JSON Files (*.json)")
        if file_path:
            # Worker를 생성하여 QThreadPool에 제출
            worker = LoadSettingsWorker(file_path, self)
            QThreadPool.globalInstance().start(worker)

    @pyqtSlot(dict)
    def update_ui_fields_from_worker(self, new_settings):
        # 백그라운드에서 로드한 설정을 UI에 적용 (메인 스레드에서 실행됨)
        self.settings = new_settings
        self.update_ui_fields()
        QMessageBox.information(self, "Loaded", "Settings loaded from file.")
