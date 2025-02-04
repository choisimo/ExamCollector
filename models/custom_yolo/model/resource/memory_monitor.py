from datetime import time

import psutil
import torch
from PyQt5.QtCore import QThread, pyqtSignal


class MemoryMonitor(QThread):
    update = pyqtSignal(float)

    def run(self):
        while True:
            self.update.emit(f"CPU Usage: {psutil.cpu_percent()}%")
            if torch.cuda.is_available():
                mem = torch.cuda.memory_allocated() // 1024**2
                self.update.emit(f"GPU Memory: {mem / 1024} GB")
            else:
                ram = psutil.virtual_memory().used // 1024**2
                self.update.emit(f"RAM Memory GB : {ram / 1024} GB")

            time.sleep(1)