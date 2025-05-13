from PyQt5.QtCore import QThread, pyqtSignal
from ultralytics import YOLO


class TrainingWorker(QThread):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)
    log_signal = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, model_path, data_yaml, epochs, batch, device):
        super().__init__()
        self._is_cancelled = None
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.batch = batch
        self.device = device

    def run(self):
        try:
            self._is_cancelled = False
            model = YOLO(self.model_path)
            self.log_signal.emit(f"모델 로드 완료 \n"
                                 f"--------전체 정보-------\n"
                                 f"[모델] {self.model_path}, [데이터] {self.data_yaml}, [연산장치] {self.device}\n"
                                 f"------------------------\n")

            for epoch in range(self.epochs):
                if self._is_cancelled:
                    self.log_signal.emit("학습 중단 signal 수신. 학습 중단 중...")
                    return

                model.train(
                    data=self.data_yaml,
                    epochs=self.epochs,
                    batch=self.batch,
                    imgsz=640,
                    device=self.device,
                    verbose=False,
                    plots=True
                )

                # 학습 진행률 업데이트
                self.log_signal.emit(f"Epoch {epoch+1}/{self.epochs} 진행 중...")
                self.msleep(500)
                self.progress.emit(int((epoch + 1) / self.epochs * 100 ))

            # 학습 완료 후 최적 모델 경로 반환
            save_dir = model.trainer.save_dir
            best_model = f"{save_dir}/weights/best.pt"
            self.finished.emit(best_model)
        except Exception as e:
            self.error.emit(f"학습 오류: {str(e)}")

    def cancel(self):
        self._is_cancelled = True
