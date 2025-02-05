import json

import psutil
import torch.cuda
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QPlainTextEdit, QGroupBox, QLineEdit, QRadioButton, \
    QSpinBox, QFileDialog, QFormLayout, QLabel, QHBoxLayout, QMessageBox
from ultralytics import YOLO


class TrainingWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_path, data_yaml, epochs, batch, device):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.device = device

    def run(self):
        try:
            self.log_signal.emit(f"\n[Training] Using model={self.model_path}, data={self.data_yaml}, device={self.device}")
            model = YOLO(self.model_path)

            self.log_signal.emit(f"학습 시작 (장치: {self.device.upper()})")
            model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                batch=self.batch,
                imgsz=640,
                device=self.device,
                verbose=False,
                plots=True
            )

            save_dir = model.trainer.save_dir
            best_model = f"{save_dir}/weights/best.pt"
            self.finished.emit(best_model)

        except Exception as e:
            self.error.emit(f"학습 오류: {str(e)}")


class TrainingModel(QWidget):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, settings: dict, settings_path: str):
        super().__init__()
        self.device_choices = None
        self.settings = settings
        self.settings_path = settings_path

        self.model = None
        self.log_area = QPlainTextEdit()
        self.log_area.setReadOnly(True)

        self.model_path_edit = QLineEdit()
        self.data_yaml_edit = QLineEdit()
        self.epochs_spin = QSpinBox()
        self.batch_spin = QSpinBox()
        self.device_btns = []

        self.init_ui()
        self.load_settings_into_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 로그 출력
        layout.addWidget(QLabel("Training Log:"))
        layout.addWidget(self.log_area)

        # 모델 구성 그룹
        grp_model = QGroupBox("Model Configuration")
        model_layout = QFormLayout()

        # YOLO model path
        btn_select_model = QPushButton("Browse YOLO .pt Model [custom YOLO model]")
        btn_select_model.clicked.connect(self.select_model_file)
        row_model = self._make_row(self.model_path_edit, btn_select_model)
        model_layout.addRow("YOLO Model Path", row_model)

        # Data YAML path
        btn_select_data = QPushButton("Browse Data YAML")
        btn_select_data.clicked.connect(self.select_data_file)
        row_data = self._make_row(self.data_yaml_edit, btn_select_data)
        model_layout.addRow("custom data.yaml : ", row_data)

        grp_model.setLayout(model_layout)
        layout.addWidget(grp_model)

        # 리소스 선택 그룹 (Resource Configuration)
        grp_device = QGroupBox("Resource Configuration")
        device_layout = QVBoxLayout()
        # 미리 정의된 device 매핑
        self.device_choices = {
            "CUDA": "cuda",
            "CPU": "cpu",
            "AMD (ROCm)": "hip"
        }
        self.device_btns = []
        # 라디오 버튼 생성
        for label in self.device_choices:
            rb = QRadioButton(label)
            device_layout.addWidget(rb)
            self.device_btns.append(rb)
        grp_device.setLayout(device_layout)
        layout.addWidget(grp_device)

        # settings.json의 training 섹션에서 device 값을 불러와서 해당 버튼 선택
        training_config = self.settings.get("training", {})
        device_str = training_config.get("device", "cpu").lower()  # 예: "cuda", "cpu", "hip"
        for rb in self.device_btns:
            if self.device_choices[rb.text()].lower() == device_str:
                rb.setChecked(True)
            else:
                rb.setChecked(False)

        # 학습 파라미터 그룹
        grp_params = QGroupBox("Training Parameters")
        params_layout = QFormLayout()

        self.epochs_spin.setRange(1, 5000)
        self.batch_spin.setRange(1, 256)

        params_layout.addRow("Epochs", self.epochs_spin)
        params_layout.addRow("Batch Size", self.batch_spin)

        grp_params.setLayout(params_layout)
        layout.addWidget(grp_params)

        # training 실행
        train_btn = QPushButton("학습 시작")
        train_btn.clicked.connect(self.start_training)
        layout.addWidget(train_btn)

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

    def select_model_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select YOLO model", "", "YOLO Model (*.pt)")
        if path:
            self.model_path_edit.setText(path)
            self.settings.get("training", {})["YOLO_model"] = path

    def select_data_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select data.yaml", "", "YAML files (*.yaml *.yml)")
        if path:
            self.data_yaml_edit.setText(path)
            self.settings.get("training", {})["data_yaml"] = path

    def load_settings_into_ui(self):
        """
        Load training-related fields from self.settings into the UI fields
        so the user sees current JSON values.
        """
        # YOLO .pt
        model_path = self.settings.get("training", {}).get("YOLO_model", "custom_yolo.pt")
        self.model_path_edit.setText(model_path)

        # data.yaml
        data_yaml = self.settings.get("training", {}).get("data_yaml", "custom_data.yaml")
        self.data_yaml_edit.setText(data_yaml)

        # epochs, batch
        train_cfg = self.settings.get("training", {})
        self.epochs_spin.setValue(train_cfg.get("epochs", 50))
        self.batch_spin.setValue(train_cfg.get("batch_size", 16))

        # device
        device_str = self.settings.get("training", {}).get("device", "cpu").lower()
        # match radio
        for rb in self.device_btns:
            # e.g. "CPU", "CUDA", "AMD (HIP)"
            if self.device_choices[rb.text()].lower() == device_str:
                rb.setChecked(True)
            else:
                rb.setChecked(False)

    def current_device_value(self):
        """
        Figure out which device radio is checked and return "cpu", "cuda", or "hip".
        """
        for rb in self.device_btns:
            if rb.isChecked():
                return self.device_choices[rb.text()]
        return "cpu"

    def start_training(self):
        try:
            # 최신 설정 파일 로드
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                current_settings = json.load(f)

            # 동적 데이터 생성
            data_yaml_content = self.generate_data_yaml(current_settings)
            data_yaml_path = self.data_yaml_edit.text() if not None else current_settings['training']['data_yaml']

            # 작업자 스레드 생성 및 실행
            self.worker = TrainingWorker(
                model_path=self.model_path_edit.text(),
                data_yaml=data_yaml_path,
                epochs=self.epochs_spin.value(),
                batch=self.batch_spin.value(),
                device=self.current_device_value()
            )

            self.worker.log_signal.connect(self.log_area.appendPlainText)
            self.worker.finished.connect(self.handle_training_result)
            self.worker.error.connect(self.handle_training_error)
            self.worker.start()

        except Exception as e:
            self.log_area.appendPlainText(f"초기화 오류: {str(e)}")

    def handle_training_result(self, model_path):
        self.log_area.appendPlainText(f"\n학습 완료! 최종 모델: {model_path}")
        # 모델 내보내기 로직 추가

    def handle_training_error(self, message):
        self.log_area.appendPlainText(f"\n오류 발생: {message}")
        QMessageBox.critical(self, "학습 오류", message)

    def save_current_settings_back_to_json(self):
        """
        (Optional) If you want to save updated training config back into settings.json
        from this tab directly.
        Otherwise you can rely on the main SettingsTab to handle saving.
        """
        try:
            # 1) Update self.settings from the UI
            self.settings["detector_model_path"] = self.model_path_edit.text().strip()

            # training sub-dict
            if "training" not in self.settings:
                self.settings["training"] = {}
            self.settings["training"]["data_yaml"] = self.data_yaml_edit.text().strip()
            self.settings["training"]["epochs"] = self.epochs_spin.value()
            self.settings["training"]["batch_size"] = self.batch_spin.value()
            self.settings["training"]["device"] = self.current_device_value()

            # 2) Write to disk
            with open(self.settings_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2)

            self.log_area.appendPlainText("Training settings saved back to settings.json")

        except Exception as ex:
            self.log_area.appendPlainText(f"Failed to save settings: {ex}")

    def generate_data_yaml(self, settings):
        """
        (Optional) If you want to generate a data.yaml file from the UI fields
        and save it to disk.
        """
        data_config = settings.get('data', {})
        class_info = settings.get('class_info', {})
        sorted_classes = sorted(class_info.items(), key=lambda x: int(x[0]))
        class_names = [v for _, v in sorted_classes]

        yaml_contents = f"""
        path: {data_config.get('path', '')}
        train: {data_config.get('train', '')}
        val: {data_config.get('val', '')}
        
        names:
        """
        for idx, name in enumerate(class_names):
            yaml_contents += f"  {idx}: {name}\n"

        return yaml_contents
