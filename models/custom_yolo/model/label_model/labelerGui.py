import json
import os
import sys
from functools import partial

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QPlainTextEdit, QMessageBox, QScrollArea, QGroupBox, QRadioButton,
    QComboBox, QFormLayout, QSpinBox, QLineEdit, QProgressDialog
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from sympy import true

from auto_labeler import AutoLabeler
from models.custom_yolo.model.learning_model.training_model import TrainingModel
from models.custom_yolo.model.resource.memory_monitor import (ResourceGraphWidget)


class LabelingWorker(QThread):
    error = pyqtSignal(Exception)
    finished = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, auto_labeler, image_path):
        super().__init__()
        self.auto_labeler = auto_labeler
        self.image_path = image_path

    def run(self):
        try:
            boxes = self.auto_labeler.detect_objects(self.image_path)
            labels = []
            # 이미지는 한 번만 읽어오기
            img = cv2.imread(self.image_path)
            for idx, box in enumerate(boxes):
                labels.append(self.auto_labeler.generate_label_for_box(img, box))
                self.progress.emit(int((idx + 1) / len(boxes) * 100))
            self.finished.emit(labels)
        except Exception as e:
            self.error.emit(e)


class LabelingTab(QWidget):
    def __init__(self, auto_labeler):
        super().__init__()
        self.progress_dialog = None
        self.log_edit = QPlainTextEdit()
        self.mem_monitor = None
        self.image_label = None
        self.worker = None
        self.auto_labeler = auto_labeler
        self.image_path = None
        self.labels = None
        self.annotated_image = None
        self.init_ui()

    def init_ui(self):
        try:
            layout = QVBoxLayout()

            # 이미지 선택 버튼
            btn_select = QPushButton("이미지 파일 선택")
            btn_select.clicked.connect(self.select_image)
            layout.addWidget(btn_select)

            # 자동 라벨링 실행 버튼
            btn_run = QPushButton("자동 라벨링 실행")
            btn_run.clicked.connect(self.run_labeling)
            self.log("라벨링을 시작합니다..")
            layout.addWidget(btn_run)

            # 라벨링 결과 미리보기
            self.image_label = QLabel("라벨링 결과 미리보기")
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScaledContents(True)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.image_label)
            layout.addWidget(scroll, stretch=1)

            # log 출력
            self.log_edit.setReadOnly(True)
            layout.addWidget(self.log_edit, stretch=0)

            # 결과 저장 버튼
            btn_save = QPushButton("라벨 결과 저장")
            btn_save.clicked.connect(self.save_labels)
            layout.addWidget(btn_save)

            self.setLayout(layout)
        except Exception as e:
            print("ERROR", e)

    def log(self, message):
        self.log_edit.appendPlainText(message)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.png)")
        if file_path:
            self.image_path = file_path
            self.log(f"선택한 이미지 파일 개수 : {len(file_path)} 개")
        else:
            self.log("이미지 파일 선택 취소")

    def run_labeling(self):
        try:
            if not self.image_path:
                self.log("먼저 이미지 파일을 선택해주세요.")
                return

            self.progress_dialog = QProgressDialog("라벨링 진행 중...", "취소", 0, len(self.image_path), self)

            for idx, image in enumerate(self.image_path):
                self.worker = LabelingWorker(self.auto_labeler, image)
                self.worker.finished.connect(partial(self.handle_single_result, idx))
                self.worker.progress.connect(self.update_progress)
                self.worker.start()

        except Exception as e:
            self.log(f"라벨링 오류: {e}")
            QMessageBox.critical(self, "오류", f"라벨링 오류: {e}")

    def handle_single_result(self, idx, labels):
        self.progress_dialog.setValue(idx+1)
        self.log(f"[{idx}] 번 이미지 라벨링 {labels} 완료")

    def update_progress(self, progress):
        self.log(f"진행률: {progress}%")

    def labeling_finished(self, labels):
        self.labels = labels
        # origin image 로드
        img = cv2.imread(self.image_path)
        if img is None:
            self.log(f"원본 이미지를 찾을 수 없습니다 : {self.image_path}")
            return

        # 예: AutoLabeler.detect_objects()로 얻은 boxes 배열과, labels 배열이 있다고 가정합니다.
        # 만약 self.labels가 각 객체에 대한 dict 형태라면, 여기서 'coordinates'와 'label' 키를 사용합니다.
        # 아래 예제에서는 normalized 좌표를 사용한다고 가정합니다.
        for label_info in self.labels:
            # label_info 예: {'coordinates': [x_center, y_center, w, h], 'label': "question", ...}
            box = label_info['coordinates']  # normalized 좌표
            label_text = label_info['label']

            # 이미지의 크기에 맞게 좌표 변환
            h_img, w_img = img.shape[:2]
            x_center, y_center, w_box, h_box = box
            x_center *= w_img
            y_center *= h_img
            w_box *= w_img
            h_box *= h_img
            x1 = int(x_center - w_box / 2)
            y1 = int(y_center - h_box / 2)
            x2 = int(x_center + w_box / 2)
            y2 = int(y_center + h_box / 2)
            # 사각형 그리기
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            # 라벨 텍스트 표시
            cv2.putText(img, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # BGR 이미지를 RGB로 변환 후 QImage에 표시
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img_rgb.shape
        bytes_per_line = 3 * width
        qt_image = QImage(img_rgb.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qt_image).scaled(self.image_label.size(), Qt.KeepAspectRatio))

    def save_labels(self):
        if self.labels and self.image_path:
            # settings 에서 저장 모두 읽어오기
            setting_file = "settings.json"
            try:
                with open(setting_file, 'r', encoding='utf-8') as f:
                    setting = json.load(f)
                label_save_mode = setting.get("other_settings", {}).get("label_save_mode", "auto")
            except Exception as e:
                self.log(f"설정 파일 로드 실패 {e}")
                label_save_mode = "auto"

            if label_save_mode == "auto":
                # 이미지 파일 확장자를 .txt 으로 변경하여 자동 저장하기
                file_path = self.image_path.rsplit('.', 1)[0] + '.txt'
            else:
                # 수동 모드 : 사용자가 직접 위치 설정하기
                file_path, _ = QFileDialog.getSaveFileName(
                    self, "label 저장", "", "텍스트 파일 (*.txt)")
                if file_path:
                    self.auto_labeler.save_label(self.labels, file_path)
                    QMessageBox.information(self, "저장 완료", f"파일 저장 위치 : \n{file_path}")


class SettingsTab(QWidget):
    def __init__(self, settings: dict, settings_path: str):
        """
        :param settings: The loaded dict from settings.json
        :param settings_path: The file path to reload and save
        """
        super().__init__()
        self.combo_imgsz = None
        self.spin_batch = None
        self.spin_epochs = None
        self.detector_path_edit = None
        self.llm_model_edit = None
        self.val_path_edit = None
        self.train_path_edit = None
        self.path_edit = None
        self.settings = settings
        self.settings_path = settings_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # 모델 설정 그룹
        grp_model = QGroupBox("Model Configuration")
        model_layout = QFormLayout()

        # detector path
        self.detector_path_edit = QLineEdit(self.settings.get("detector_model_path", ""))
        btn_browse_detector = QPushButton("Browse...")
        btn_browse_detector.clicked.connect(lambda: self.browse_file(self.detector_path_edit))
        row_widget1 = self._path_row(self.detector_path_edit, btn_browse_detector)
        model_layout.addRow("Detector Model:", row_widget1)

        # LLM model
        self.llm_model_edit = QLineEdit(self.settings.get("llm_model", "microsoft/phi-2"))
        model_layout.addRow("LLM Model:", self.llm_model_edit)
        grp_model.setLayout(model_layout)
        layout.addWidget(grp_model)

        # 데이터 설정 그룹
        grp_data = QGroupBox("Data Configuration")
        data_layout = QFormLayout()

        self.path_edit = QLineEdit(self.settings.get("data", {}).get("path", ""))
        self.train_path_edit = QLineEdit(self.settings.get("data", {}).get("train", ""))
        self.val_path_edit = QLineEdit((self.settings.get("data", {}).get("val", "")))

        btn_browse_train = QPushButton("Browse...")
        btn_browse_train.clicked.connect(lambda: self.browse_dir(self.train_path_edit))
        btn_browse_val = QPushButton("Browse...")
        btn_browse_val.clicked.connect(lambda: self.browse_dir(self.val_path_edit))

        data_layout.addRow("Train Data:", self._path_row(self.train_path_edit, btn_browse_train))
        data_layout.addRow("Val Data:", self._path_row(self.val_path_edit, btn_browse_val))
        grp_data.setLayout(data_layout)
        layout.addWidget(grp_data)

        # Training
        grp_training = QGroupBox("Training")
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
        self.grp_device = QGroupBox("Device")
        device_str = training_config.get("device", "cpu")
        device_layout = QHBoxLayout()
        self.device_choices = {"CUDA": "cuda", "CPU": "cpu", "AMD (HIP)": "hip"}
        self.rb_cpu = QRadioButton("CPU")
        self.rb_cuda = QRadioButton("CUDA (Nvidia)")
        self.rb_hip = QRadioButton("AMD (HIP)")
        device_layout.addWidget(self.rb_cpu)
        device_layout.addWidget(self.rb_cuda)
        device_layout.addWidget(self.rb_hip)
        self.grp_device.setLayout(device_layout)
        training_layout.addRow("Compute Device:", self.grp_device)

        # Check the radio that matches
        if device_str.lower() == "cuda":
            self.rb_cuda.setChecked(True)
        elif device_str.lower() == "hip":
            self.rb_hip.setChecked(True)
        else:
            self.rb_cpu.setChecked(True)

        grp_training.setLayout(training_layout)
        layout.addWidget(grp_training)

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

        # Full JSON read-only display (optional)
        grp_json = QGroupBox("Raw JSON (read-only)")
        json_layout = QVBoxLayout()
        self.json_edit = QPlainTextEdit(json.dumps(self.settings, indent=2))
        self.json_edit.setReadOnly(True)
        json_layout.addWidget(self.json_edit)
        grp_json.setLayout(json_layout)
        layout.addWidget(grp_json)

        # Buttons
        # 버튼 영역: 불러오기 / 저장
        btn_layout = QHBoxLayout()
        btn_load = QPushButton("불러오기")
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
        path, _ = QFileDialog.getOpenFileName(self, "Browse file", "", "All Files (*.*)")
        if path:
            line_edit.setText(path)

    def browse_dir(self, line_edit):
        d = QFileDialog.getExistingDirectory(self, "Browse directory")
        if d:
            line_edit.setText(d)

    def load_from_disk(self):
        """
        Re-read settings from settings.json and refresh the UI
        """
        if os.path.exists(self.settings_path):
            with open(self.settings_path, 'r', encoding='utf-8') as f:
                self.settings = json.load(f)
            # re-init UI fields
            self.init_ui()
        else:
            QMessageBox.warning(self, "Warning", "No settings.json found.")

    def save_to_disk(self):
        """
        Gather user changes from UI => self.settings => write settings.json
        """
        # 모델 설정 업데이트
        self.settings["detector_model_path"] = self.detector_path_edit.text()
        self.settings["llm_model"] = self.llm_model_edit.text()

        # 데이터 설정 업데이트
        if "data" not in self.settings:
            self.settings["data"] = {}
        self.settings['data'] = {
            'path': self.path_edit.text(),
            'train': self.train_path_edit.text(),
            'val': self.val_path_edit.text()
        }

        # 학습 설정 업데이트
        if "training" not in self.settings:
            self.settings["training"] = {}
        # 학습 설정 업데이트
        self.settings['training'].update({
            'epochs': self.spin_epochs.value(),
            'batch_size': self.spin_batch.value(),
            'imgsz': int(self.combo_imgsz.currentText())
        })

        # Labeling 설정 (예: device)
        if "labeling" not in self.settings:
            self.settings["labeling"] = {}
        self.settings["labeling"]["label_save_mode"] = self.combo_label_save_mode.currentText()

        # gather device from radio
        if self.rb_cuda.isChecked():
            self.settings["training"]["device"] = "cuda"
        elif self.rb_hip.isChecked():
            self.settings["training"]["device"] = "hip"
        else:
            self.settings["training"]["device"] = "cpu"

        # optionally re-generate the raw JSON
        self.json_edit.setPlainText(json.dumps(self.settings, indent=2))

        # write to disk
        with open(self.settings_path, 'w', encoding='utf-8') as f:
            json.dump(self.settings, f, indent=2)
        QMessageBox.information(self, "Saved", "Settings saved to settings.json.")


class MainWindow(QMainWindow):
    def __init__(self, settings: dict, settings_path: str):
        super().__init__()
        self.settings = settings
        self.settings_path = settings_path
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("시험지 자동 라벨링 및 학습 시스템")
        self.resize(1200, 800)

        tabs = QTabWidget()
        self.setCentralWidget(tabs)

        # 1) Construct an AutoLabeler with the current settings
        auto_labeler = AutoLabeler(config=self.settings)
        auto_labeler.initialize_detector(self.settings.get("detector_model_path", None))

        # 2) Add Labeling Tab
        labeling_tab = LabelingTab(auto_labeler)
        tabs.addTab(labeling_tab, "Labeling")

        # 4) Resource Monitor Tab
        resource_tab = ResourceGraphWidget(interval=1000)
        tabs.addTab(resource_tab, "Resource Monitor")

        # 5) Training Tab
        training_tab = TrainingModel(self.settings, self.settings_path)
        tabs.addTab(training_tab, "Training")

        # Add Settings Tab
        settings_tab = SettingsTab(self.settings, self.settings_path)
        tabs.addTab(settings_tab, "Settings")


def main():
    app = QApplication(sys.argv)
    # 1) Load settings from settings.json (or create defaults if missing)
    settings_path = "settings.json"

    if os.path.exists(settings_path):
        with open(settings_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
    else:
        # fallback
        config = {
            {
                "detector_model_path": "C:/WorkSpace/git/ExamCollector/models/custom_yolo/model/label_model/yolo11n.pt",
                "llm_model": "deepseek-ai/Janus-Pro-7B",
                "data": {
                    "path": "C:/WorkSpace/git/ExamCollector/models/custom_yolo/data/",
                    "train": "images/train",
                    "val": "images/val"
                },
                "class_info": {
                    "0": "question",
                    "1": "answer",
                    "2": "figure",
                    "3": "etc",
                    "4": "q_num",
                    "5": "q_type"
                },
                "nc": 6,
                "training": {
                    "epochs": 30,
                    "batch_size": 15,
                    "imgsz": 1280,
                    "device": "cuda",
                    "data_yaml": "./custom_data.yaml"
                },
                "labeling": {
                    "device": "cuda"
                },
                "other_settings": {
                    "use_gpu": "true",
                    "log_level": "INFO",
                    "label_save_mode": "auto"
                }
            }
        }

    window = MainWindow(config, settings_path)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
