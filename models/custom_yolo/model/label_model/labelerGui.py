import json
import os

import cv2
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QTabWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QTextEdit, QPlainTextEdit, QMessageBox, QScrollArea, QGroupBox, QRadioButton
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QThread, pyqtSignal
import torch
from ultralytics import YOLO
from auto_labeler import AutoLabeler
from models.custom_yolo.model.learning_model.training_model import TrainingModel
from models.custom_yolo.model.resource.memory_monitor import MemoryMonitor

class LabelingWorker(QThread):
    error = pyqtSignal(Exception)

    finished = pyqtSignal(list)
    progress = pyqtSignal(int)

    def __init__(self, auto_labeler, image_path):
        super().__init__()
        self.auto_labeler = auto_labeler
        self.image_path = image_path

    def run(self):
        boxes = self.auto_labeler.detect_objects(self.image_path)
        labels = []
        # 이미지는 한 번만 읽어오기
        img = cv2.imread(self.image_path)
        for idx, box in enumerate(boxes):
            labels.append(self.auto_labeler.generate_label_for_box(img, box))
            self.progress.emit(int((idx + 1) / len(boxes) * 100))
        self.finished.emit(labels)


class LabelingTab(QWidget):
    def __init__(self, auto_labeler):
        super().__init__()
        self.mem_monitor = None
        self.log_edit = None
        self.image_label = None
        self.worker = None
        self.auto_labeler = auto_labeler
        self.image_path = None
        self.labels = None
        self.annotated_image = None
        self.init_ui()

    def init_ui(self):
        self.mem_monitor = MemoryMonitor()
        self.mem_monitor.update.connect(self.log)
        self.mem_monitor.start()

        layout = QVBoxLayout()

        # 이미지 선택 버튼
        btn_select = QPushButton("이미지 파일 선택")
        btn_select.clicked.connect(self.select_image)
        layout.addWidget(btn_select)

        # 자동 라벨링 실행 버튼
        btn_run = QPushButton("자동 라벨링 실행")
        btn_run.clicked.connect(self.run_labeling)
        layout.addWidget(btn_run)

        # 라벨링 결과 미리보기
        self.image_label = QLabel("라벨링 결과 미리보기")
        self.image_label.setAlignment(Qt.AlignCenter)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(self.image_label)
        layout.addWidget(scroll, stretch=1)

        # log 출력
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, stretch=0)

        # 결과 저장 버튼
        btn_save = QPushButton("라벨 결과 저장")
        btn_save.clicked.connect(self.save_labels)
        layout.addWidget(btn_save)

        self.setLayout(layout)

    def log(self, message):
        self.log_edit.appendPlainText(message)

    def select_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.png)")
        if file_path:
            self.image_path = file_path
            self.log(f"이미지 파일 선택: {file_path}")
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio))
        else:
            self.log("이미지 파일 선택 취소")

    def run_labeling(self):
        try:
            if not self.image_path:
                raise FileNotFoundError("이미지 파일이 선택되지 않았습니다.")

            self.worker = LabelingWorker(self.auto_labeler, self.image_path)
            self.worker.progress.connect(self.update_progress)
            self.worker.finished.connect(self.labeling_finished)
            self.worker.start()

        except Exception as e:
            self.log(f"라벨링 오류: {e}")
            QMessageBox.critical(self, "오류", f"라벨링 오류: {e}")

    def update_progress(self, progress):
        self.log(f"진행률: {progress}%")

    def labeling_finished(self, labels):
        self.labels = labels

    def save_labels(self):
        if self.labels and self.image_path:
            file_path, _ = QFileDialog.getSaveFileName(
                self, "label 저장", "", "텍스트 파일 (*.txt)")
            if file_path:
                self.auto_labeler.save_labels(self.labels, file_path)
                QMessageBox.information(self, "저장 완료", f"파일 저장 위치 : \n{file_path}")


class SettingsTab(QWidget):
    def __init__(self, settings_path):
        super().__init__()
        self.save_btn = None
        self.load_btn = None
        self.text_edit = None
        self.settings_path = settings_path
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        label = QLabel("프로젝트 설정 파일 (JSON)")
        layout.addWidget(label)

        self.text_edit = QPlainTextEdit()
        layout.addWidget(self.text_edit, stretch=1)

        btn_layout = QHBoxLayout()
        self.load_btn = QPushButton("설정 불러오기")
        self.load_btn.clicked.connect(self.load_settings)
        btn_layout.addWidget(self.load_btn)

        self.save_btn = QPushButton("설정 저장")
        self.save_btn.clicked.connect(self.save_settings)
        btn_layout.addWidget(self.save_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)
        self.load_settings()

        # resource selection section
        grp_resources = QGroupBox("hardware excel")
        resource_layout = QHBoxLayout()
        self.resource_btns = {
            "CUDA": QRadioButton("NVIDIA GPU"),
            "CPU": QRadioButton("CPU Only"),
            "ROCm": QRadioButton("AMD GPU")
        }
        for btn in self.resource_btns.values():
            resource_layout.addWidget(btn)
        grp_resources.setLayout(resource_layout)
        layout.insertWidget(1, grp_resources)

    def load_settings(self):
        if os.path.exists(self.settings_path):
            try:
                with open(self.settings_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                # JSON 파싱 (오류가 발생하면 사용자에게 알림)
                json.loads(content)  # 올바른 JSON 형식인지 확인
                self.text_edit.setPlainText(content)
            except Exception as e:
                QMessageBox.critical(self, "오류", f"설정 파일 읽기 오류: {e}")
        else:
            self.text_edit.setPlainText("설정 파일이 존재하지 않습니다.")

    def save_settings(self):
        content = self.text_edit.toPlainText()
        try:
            # JSON 형식 확인
            json.loads(content)
        except Exception as e:
            QMessageBox.critical(self, "오류", f"JSON 형식 오류: {e}")
            return

        with open(self.settings_path, 'w', encoding='utf-8') as f:
            f.write(content)
        QMessageBox.information(self, "저장 완료", "설정 파일이 저장되었습니다.")


class MainWindow(QMainWindow):
    def __init__(self, auto_labeler):
        super().__init__()
        self.settings_tab = None
        self.labeling_tab = None
        self.tabs = None
        self.model_training = TrainingModel()
        self.setWindowTitle("시험지 자동 라벨링 및 학습 시스템")
        self.resize(1000, 700)
        # AutoLabeler 인스턴스 생성
        self.auto_labeler = auto_labeler
        self.init_ui()

    def init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self.model_training)

        # Labeling 탭
        self.labeling_tab = LabelingTab(self.auto_labeler)
        self.tabs.addTab(self.labeling_tab, "Labeling")

        # Settings 탭
        self.settings_tab = SettingsTab("settings.json")
        self.tabs.addTab(self.settings_tab, "Settings")

    def log(self, message):
        # Labeling 탭의 로그 창에 메시지 추가
        self.labeling_tab.log(message)


def main():
    app = QApplication([])
    print("GPU 사용 가능 여부:", torch.cuda.is_available())

    auto_labeler = AutoLabeler()
    auto_labeler.initialize_detector()

    # memory clean up
    torch.cuda.empty_cache() if torch.cuda.is_available() else ""
    import gc
    gc.collect()

    window = MainWindow(auto_labeler)
    window.show()
    app.exec_()

    # 메모리 해제
    del auto_labeler
    torch.cuda.empty_cache() if torch.cuda.is_available() else ""
    gc.collect()


# 프로그램 종료 시 GPU 메모리 해제
if __name__ == "__main__":
    main()
