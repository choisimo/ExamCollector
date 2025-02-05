import time
import GPUtil
import pyqtgraph as pg
import psutil
import torch
from PyQt5.QtCore import QThread, pyqtSignal, QTimer
from PyQt5.QtWidgets import QVBoxLayout, QWidget


class ResourceGraphWidget(QWidget):
    def __init__(self, interval=1000, parent=None):
        """
        interval: 업데이트 간격 (밀리초)
        """
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.plot_widget = pg.PlotWidget(title="리소스 사용량")
        self.layout.addWidget(self.plot_widget)

        # X축 (시간, 초)와 각 데이터(Y축) 저장용 리스트
        self.x_data = []  # 시간 (초 단위)
        self.y_cpu = []   # CPU 사용률 (%)
        self.y_ram = []   # RAM 사용률 (%)
        self.y_gpu = []   # 첫 번째 GPU 사용률 (%)

        self.ptr = 0

        # 각 데이터에 대한 plot curve 생성 (색상과 레이블 지정)
        self.curve_cpu = self.plot_widget.plot(pen=pg.mkPen('y', width=2), name="CPU 사용률")
        self.curve_ram = self.plot_widget.plot(pen=pg.mkPen('g', width=2), name="RAM 사용률")
        self.curve_gpu = self.plot_widget.plot(pen=pg.mkPen('r', width=2), name="GPU 사용률")

        # 타이머를 설정하여 주기적으로 데이터를 업데이트
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(interval)

        # 범례를 추가 (PlotWidget에서 addLegend를 사용)
        self.plot_widget.addLegend()
        self.plot_widget.setLabel('left', '사용률', '%')
        self.plot_widget.setLabel('bottom', '시간', 's')

    def update_data(self):
        # 현재 시간(초) 증가
        self.x_data.append(self.ptr)
        self.ptr += 1

        # CPU 사용률 (psutil.cpu_percent()는 0~100 사이의 값)
        cpu_usage = psutil.cpu_percent(interval=None)
        self.y_cpu.append(cpu_usage)

        # RAM 사용률 (virtual_memory().percent는 사용률 %)
        ram_usage = psutil.virtual_memory().percent
        self.y_ram.append(ram_usage)

        # GPU 사용률 (여기서는 첫 번째 GPU의 load 값을 사용; load는 0~1 사이의 값 -> 100*load)
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu_usage = gpus[0].load * 100  # 첫번째 GPU의 로딩률 (%)
        else:
            gpu_usage = 0
        self.y_gpu.append(gpu_usage)

        # 데이터가 너무 많아지면 마지막 100개 데이터만 유지
        if len(self.x_data) > 100:
            self.x_data = self.x_data[-100:]
            self.y_cpu = self.y_cpu[-100:]
            self.y_ram = self.y_ram[-100:]
            self.y_gpu = self.y_gpu[-100:]

        # 각 curve 업데이트
        self.curve_cpu.setData(self.x_data, self.y_cpu)
        self.curve_ram.setData(self.x_data, self.y_ram)
        self.curve_gpu.setData(self.x_data, self.y_gpu)