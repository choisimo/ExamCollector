import logging
import sys
import traceback


class AppLogger:
    def __init__(self):
        # 로그 파일 이름, 로그 레벨, 포맷 설정 (파일 매개변수는 filename로 지정)
        logging.basicConfig(
            filename='app_error.log',
            level=logging.DEBUG,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def my_exception_hook(self, exctype, value, tb):
        # 예외 정보를 문자열로 변환
        error_message = "".join(traceback.format_exception(exctype, value, tb))
        # 콘솔에 출력
        print("Uncaught exception:", error_message)
        # 로그 파일에 기록
        logging.error("Uncaught exception:\n%s", error_message)
        # 기본 예외 후크 호출 (필요 시)
        sys.__excepthook__(exctype, value, tb)


# AppLogger 인스턴스 생성 후 전역 예외 후크 설정
logger = AppLogger()
sys.excepthook = logger.my_exception_hook
