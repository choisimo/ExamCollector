import torch.cuda
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QPlainTextEdit, QGroupBox, QLineEdit, QRadioButton, \
    QSpinBox, QFileDialog
from ultralytics import YOLO


class TrainingModel(QWidget):
    def __init__(self):
        super().__init__()
        self.model = None
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 모델 구성 그룹
        grp_model = QGroupBox("Model Configuration")
        model_layout = QVBoxLayout()
        self.btn_model = QPushButton("Custom Yolo Model Load")
        self.model_path = QLineEdit()
        self.btn_model.clicked.connect(self.select_model)
        model_layout.addWidget(self.btn_model)
        model_layout.addWidget(self.model_path)
        grp_model.setLayout(model_layout)
        layout.addWidget(grp_model)

        # 리소스 선택 그룹
        grp_device = QGroupBox("Resource Configuration")
        device_layout = QVBoxLayout()
        self.device_choices = {
            "CUDA": "cuda",
            "CPU": "cpu",
            "AMD (ROCm)": "hip"
        }
        self.device_btns = []
        for text in self.device_choices:
            btn = QRadioButton(text)
            btn.toggled.connect(self.update_device)
            device_layout.addWidget(btn)
            self.device_btns.append(btn)
        self.device_btns[0].setChecked(True)
        grp_device.setLayout(device_layout)
        layout.addWidget(grp_device)

        # 학습 파라미터 그룹
        grp_params = QGroupBox("Training Parameters")
        params_layout = QVBoxLayout()
        self.epochs = QSpinBox()
        self.epochs.setValue(100)
        self.batch_size = QSpinBox()
        self.batch_size.setValue(16)
        self.batch_size.setRange(1, 64)
        params_layout.addRow("Epochs", self.epochs)
        params_layout.addRow("Batch Size", self.batch_size)
        grp_params.setLayout(params_layout)
        layout.addWidget(grp_params)

        # 실행
        self.train_btn = QPushButton("학습 시작")
        self.train_btn.clicked.connect(self.start_training)
        layout.addWidget(self.train_btn)

        # 로그 출력
        self.log_area = QPlainTextEdit()
        layout.addWidget(self.log_area)

        self.setLayout(layout)

    def select_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "YOLO model 선택", "", "YOLO model (*.pt)")
        if path:
            self.model_path.setText(path)

    def update_device(self):
        selected = next(
            (self.device_choices[btn.text()]
             for btn in self.device_btns if btn.isChecked()), "cpu")
        self.log_area.appendPlainText(f"Selected device: {selected}")

    def start_training(self):
        try:
            device = next(
                (self.device_choices[btn.text()]
                for btn in self.device_btns if btn.isChecked()), "cpu")


            model_path = self.model_path.text() if self.model_path.text() else 'yolov8n-seg.pt'
            self.model = YOLO(model_path)

            # training configuration
            self.model.train(
                data='custom_data.yaml',
                imgsz=640,
                epochs=self.epochs.value(),
                batch=self.batch_size.value(),
                device=device,
                plots=True
            )

            # 모델 저장 및 로그 출력
            save_path, _ = QFileDialog.getSaveFileName(
                self, "모델 저장", "", "ONNX model (*.onnx)")
            if save_path:
                self.model.export(format='onnx', imgsz=640, simplify=True, device=device)
                self.log_area.appendPlainText(f"Model saved to {save_path}")

        except Exception as e:
            self.log_area.appendPlainText(f"Error: {e}")
            self.model = None

    def check_is_valid_device(self, selected):
        next((self.device_choices[btn.text()] for btn in self.device_btns if btn.isChecked()), "cpu")

        if selected == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA selected, but not available in your PC")
        if selected == "hip" and not torch.backends.rocm.is_available():
            raise ValueError("ROCm selected, but not availalbe in your PC")