from functools import partial

import cv2
from PyQt5.QtCore import QThread, pyqtSignal, QThreadPool, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QWidget, QPlainTextEdit, QVBoxLayout, QPushButton, QLabel, QScrollArea, QFileDialog, \
    QProgressDialog, QMessageBox

from models.custom_yolo.infrastructure.computer_vision.label_model.auto_labeler import AutoLabeler


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
    def __init__(self, auto_labeler: AutoLabeler):
        super().__init__()
        self.auto_labeler: AutoLabeler = auto_labeler
        self.image_paths = []
        self.image_path = None
        self.labels = None

        self.progress_dialog = None
        self.log_edit = QPlainTextEdit()
        self.image_label = None

        self.init_ui()
        self.thread_pool = QThreadPool.globalInstance()
        self.thread_pool.setMaxThreadCount(2)  # 동시 실행 최대 2개로 제한

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
            layout.addWidget(btn_run)

            # 라벨링 결과 미리보기
            self.image_label.setAlignment(Qt.AlignCenter)
            self.image_label.setScaledContents(True)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(self.image_label)
            layout.addWidget(scroll)

            # log 출력
            self.log_edit.setReadOnly(True)
            layout.addWidget(self.log_edit)

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
        file_paths, _ = QFileDialog.getOpenFileNames(self, "이미지 파일 선택", "", "이미지 파일 (*.jpg *.png)")
        if file_paths:
            self.image_paths = file_paths
            self.log(f"선택한 이미지 파일 개수 : {len(file_paths)} 개")
        else:
            self.log("이미지 파일 선택 취소")

    def run_labeling(self):
        if not self.image_paths:
            self.log("먼저 이미지 파일을 선택해주세요.")
            return
        self.progress_dialog = QProgressDialog("라벨링 진행 중...", "취소", 0, len(self.image_paths), self)

        for idx, image_path in enumerate(self.image_paths):
            worker = LabelingWorker(self.auto_labeler, image_path)
            worker.finished.connect(partial(self.handle_single_result, idx))
            worker.progress.connect(self.update_progress)
            self.thread_pool.start(worker)

    def handle_single_result(self, idx, labels):
        self.progress_dialog.setValue(idx + 1)
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
        self.image_label.setPixmap(
            QPixmap.fromImage(qt_image).scaled(
                self.image_label.width(),
                self.image_label.height(),
                Qt.KeepAspectRatio
            )
        )

    def save_labels(self):
        """
        라벨을 TXT로 저장.
        DB에서 label_save_mode(= auto/manual) 여부를 가져올 수도 있지만,
        여기서는 AutoLabeler 내부의 label_save_mode를 활용해도 됨.
        """
        if not self.image_paths:
            self.log("라벨링할 이미지가 없습니다.")
            return

        label_save_mode = self.auto_labeler.label_save_mode

        single_image: bool = True
        # 이미지 for 문으로 돌면서 라벨링
        if len(self.image_paths) == 1:
            self.image_path = self.image_paths[0]
            if not self.labels:
                self.log("라벨링 먼저 실행해주세요.")
                return
        else:
            single_image = False
            for idx, image_path in enumerate(self.image_paths):
                self.image_path = image_path
                self.log(f"[{idx}] 번 이미지 라벨링 중...")
                self.labels = self.auto_labeler.generate_labels(image_path, self.auto_labeler.detect_objects(image_path))
                self.labeling_finished(self.labels)

        if label_save_mode == "auto":
            # 이미지 파일 확장자를 .txt 으로 변경하여 자동 저장하기
            file_name = self.image_path.rsplit('.', 1)[0] + '.txt'
        else:
            # 수동 모드 : 사용자가 직접 위치 설정하기
            file_name, _ = QFileDialog.getSaveFileName(
                self, "label 저장", "", "텍스트 파일 (*.txt)")

        self.auto_labeler.save_label(self.labels, file_name)
        QMessageBox.information(self, "저장 완료", f"라벨 파일: {file_name}")
