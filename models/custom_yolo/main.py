import sys

from PyQt5.QtWidgets import QApplication

from models.custom_yolo.core.services.db_service import DBService
from models.custom_yolo.infrastructure.computer_vision.label_model.gui_main import MainWindow


def main():
    # init) DB 연결 초기화 및 접속 객체 생성
    DBService()
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
