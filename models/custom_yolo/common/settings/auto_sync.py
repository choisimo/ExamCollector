from PyQt5.QtCore import QTimer


class SettingSyncronizer:
    def __init__(self, interval=300):
        self.timer = QTimer()
        self.timer.timeout.connect(self.sync)
        self.timer.start(interval * 1000)

    def sync(self):
        """
        DB 와 SettingsManager의 설정을 동기화합니다.
        DB <-> Cache
        """
        pass