import os

from PyQt5.QtCore import QUrl, QThreadPool, pyqtSignal, QObject, QRunnable, pyqtSlot
from PyQt5.QtGui import QDesktopServices
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QPlainTextEdit, QFileDialog, QMessageBox, \
    QHBoxLayout

from models.custom_yolo.core.services.convert_document import DocuConverter
from models.custom_yolo.core.services.db_service import DBService
from models.custom_yolo.infrastructure.computer_vision.multi_worker.thread_worker import DocumentConversionRunnable


# Worker Signals 정의
class DocumentConversionSignals(QObject):
    finished = pyqtSignal(list)   # 변환된 JPG 파일 경로 리스트
    error = pyqtSignal(Exception)
    log = pyqtSignal(str)


class DocumentConverterTab(QWidget):
    def __init__(self):
        super().__init__()
        self.doc_path = []   # 사용자가 선택한 문서 파일(들)
        self.output_dir = None  # UI에서 선택한 출력 폴더
        self.pdf_list = []
        self.thread_pool = QThreadPool.globalInstance()

        # UI 초기화
        self.init_ui()
        # DB에서 document_converter 설정 로드
        self.load_document_converter_settings()

    def init_ui(self):
        layout = QVBoxLayout()

        # 상단: 파일 선택 영역
        file_select_layout = QHBoxLayout()
        btn_select = QPushButton("문서 파일 선택")
        btn_select.clicked.connect(self.select_document)
        file_select_layout.addWidget(btn_select)

        self.file_label = QLabel("선택된 파일 없음")
        file_select_layout.addWidget(self.file_label)
        layout.addLayout(file_select_layout)

        # 출력 폴더 선택 영역
        output_layout = QHBoxLayout()
        btn_select_output = QPushButton("출력 폴더 선택")
        btn_select_output.clicked.connect(self.select_output_dir)
        output_layout.addWidget(btn_select_output)

        self.output_path_edit = QPlainTextEdit()
        self.output_path_edit.setPlaceholderText("출력 폴더 경로 (미지정 시 입력 파일 폴더)")
        self.output_path_edit.setFixedHeight(30)
        output_layout.addWidget(self.output_path_edit)
        layout.addLayout(output_layout)

        # Poppler 경로 설정 영역
        poppler_layout = QHBoxLayout()
        btn_select_poppler = QPushButton("Poppler 경로 선택")
        btn_select_poppler.clicked.connect(self.select_poppler_path)
        poppler_layout.addWidget(btn_select_poppler)

        self.poppler_path_edit = QPlainTextEdit()
        self.poppler_path_edit.setPlaceholderText("Poppler 경로 (save/load)")
        self.poppler_path_edit.setFixedHeight(30)
        poppler_layout.addWidget(self.poppler_path_edit)
        layout.addLayout(poppler_layout)

        # 변환 실행 버튼
        btn_convert = QPushButton("JPG로 변환")
        btn_convert.clicked.connect(self.start_conversion)
        layout.addWidget(btn_convert)

        # 로그/메시지 출력 영역
        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit)

        # 하단: 현재 document_converter 설정 표시 영역
        self.current_settings_label = QLabel("현재 document_converter 설정:")
        layout.addWidget(self.current_settings_label)

        self.setLayout(layout)

    def select_document(self):
        """
        여러 문서파일을 선택할 수 있도록 QFileDialog.getOpenFileNames 사용
        """
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "문서 파일 선택",
            "",
            "문서 파일 (*.doc *.docx *.hwp *.pdf)"
        )
        if file_paths:
            self.pdf_list = file_paths
            self.doc_path = file_paths
            self.file_label.setText(f"선택된 파일: {file_paths}")
            self.log("파일 선택됨: " + ', '.join(file_paths))
        else:
            self.log("사용자 요청에 의해 파일 선택이 취소 되었습니다.")

    def convert_multiple_pdfs(self):
        if not self.pdf_list:
            QMessageBox.warning(self, "경고", "먼저 문서 파일을 선택하세요.")
            return

        # 병렬 변환 실행
        results = DocuConverter().batch_convert(self.pdf_list, max_workers=4)
        if results:
            self.log(f"변환 완료: {len(results)}개의 JPG 파일 생성됨.")
        else:
            self.log("변환 실패 또는 생성된 JPG 파일 없음.")

    def select_output_dir(self):
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "출력 폴더 선택",
            ""
        )
        if dir_path:
            self.output_dir = dir_path
            self.output_path_edit.setPlainText(dir_path)
            self.log("출력 폴더 선택됨: " + dir_path)
        else:
            self.log("출력 폴더 선택 취소")

    def select_poppler_path(self):
        # Poppler의 실행 파일이 있는 폴더를 선택
        dir_path = QFileDialog.getExistingDirectory(
            self,
            "Poppler 경로 선택",
            ""
        )
        if dir_path:
            self.poppler_path_edit.setPlainText(dir_path)
            self.log("Poppler 경로 선택됨: " + dir_path)
        else:
            self.log("Poppler 경로 선택 취소")

    def start_conversion(self):
        if not self.doc_path:
            QMessageBox.warning(self, "경고", "먼저 문서 파일을 선택하세요.")
            return

        # 출력 폴더: 사용자가 지정하지 않으면 입력 파일과 같은 폴더
        output_dir = self.output_path_edit.toPlainText().strip()
        default_dir = None
        if not output_dir:
            if len(self.doc_path) == 1:
                default_dir = os.path.dirname(self.doc_path[0])
            else:
                default_dir = os.getcwd()

        # DB에서 저장된 값이 있는지 조회
        db_service = DBService()
        row = db_service.get_document_converter()  # 예: id=1
        if row:
            fallback_dir = row.get("output_dir", default_dir)
            output_dir = fallback_dir
        else:
            output_dir = default_dir
        self.output_dir = output_dir
        self.output_path_edit.setPlainText(output_dir)

        # Poppler 경로: poppler_path_edit의 값이 있으면 사용, 없으면 settings에서 가져오거나 기본값
        poppler_path = self.poppler_path_edit.toPlainText().strip()
        if not poppler_path:
            poppler_path = row.get("poppler_path", r"C:/Program Files/poppler-21.03.0/Library/bin")

        self.poppler_path_edit.setPlainText(poppler_path)
        
        # DocumentConversionRunnable parameter 전달
        worker = DocumentConversionRunnable(
            doc_path=self.doc_path,
            dpi=row.get("dpi", 300),
            output_dir=output_dir,
            poppler_path=poppler_path
        )
        worker.signals.log.connect(self.log)
        worker.signals.finished.connect(self.conversion_finished)
        worker.signals.error.connect(self.conversion_error)
        self.thread_pool.start(worker)

    def conversion_finished(self, jpg_files):
        # 변환 완료 후 결과 폴더 열기
        db_service = DBService()
        row = db_service.get_document_converter()
        if row.get('open_after_finish') == 1:
            if self.doc_path and len(self.doc_path) == 1:
                output_dir = os.path.dirname(self.doc_path[0])
                QDesktopServices.openUrl(QUrl.fromLocalFile(output_dir))
            else:
                if self.output_dir:
                    QDesktopServices.openUrl(QUrl.fromLocalFile(self.output_dir))
        self.log(f"변환 완료: {len(jpg_files)}개의 JPG 파일 생성됨.")

    def conversion_error(self, e):
        self.log("변환 중 오류 발생: " + str(e))
        QMessageBox.critical(self, "오류", f"문서 변환 중 오류 발생:\n{e}")

    def log(self, message):
        self.log_edit.appendPlainText(message)
        self.update_current_settings_display()

    def load_document_converter_settings(self):
        # settings.json에서 document_converter 항목 읽기
        db_service = DBService()
        row = db_service.get_document_converter()
        if row:
            poppler_path = row.get("poppler_path", "")
            output_dir = row.get("output_dir", "")
        else:
            poppler_path = "C:/Program Files/poppler-21.03.0/Library/bin"
            output_dir = None

        self.poppler_path_edit.setPlainText(poppler_path)
        self.output_path_edit.setPlainText(output_dir)
        self.update_current_settings_display()

    def update_current_settings_display(self):
        """
        하단에 '현재 document_converter 설정' 레이블 업데이트
        """
        current_poppler = self.poppler_path_edit.toPlainText().strip()
        current_output = self.output_path_edit.toPlainText().strip()
        # 혹은 self.output_dir 값도 고려
        if not current_output and self.output_dir:
            current_output = self.output_dir
        display_text = (
            f"현재 Poppler 경로: {current_poppler}\n"
            f"현재 출력(저장) 경로: {current_output if current_output else '미지정'}"
        )
        self.current_settings_label.setText(display_text)

    def save_document_converter_settings(self):
        # UI로부터 poppler_path 얻기
        new_poppler = self.poppler_path_edit.toPlainText().strip()
        new_output = self.output_path_edit.toPlainText().strip()

        db_service = DBService()
        last_id = 0
        row = db_service.get_document_converter()
        if row:
            # 이미 있으면 update
            db_service.update_document_converter(
                id_val=row["id"],
                poppler_path=new_poppler,
                output_dir=new_output
            )
            self.log("document_converter 설정이 업데이트되었습니다.")
        else:
            # 없으면 insert
            new_id = db_service.insert_document_converter(
                poppler_path=new_poppler,
                output_dir=new_output
            )
            self.log(f"새로운 document_converter 설정이 추가되었습니다. (id={new_id})")

        self.update_current_settings_display()
