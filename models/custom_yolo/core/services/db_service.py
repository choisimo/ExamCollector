import json
from contextlib import contextmanager
from typing import Dict, Optional, List

from PyQt5.QtCore import pyqtSignal

from definition import ROOT_DIR
from models.custom_yolo.common.error_handling.db_logger import handle_db_errors
from models.custom_yolo.core.services.crud_helper import CRUDHelper, TableSchema
from models.custom_yolo.infrastructure.storage.SQLite_connector import SQLiteConnector
from models.custom_yolo.resources.sql.db_init_load import DB_INIT_QUERY
from models.custom_yolo.core.domain.schema_definitions import (
    DocumentConverterSchema,
    GlobalSettingsSchema,
    ClassInfoSchema,
    DataConfigSchema,
    ResultsConfigSchema,
    ModelPathsSchema,
    LabelingConfigSchema,
    OtherSettingsSchema,
    TrainingConfigSchema
)


class DBService:
    DB_CONN_SIGNAL = pyqtSignal(str)

    def __init__(self, db_path: str = None):
        self._connector = SQLiteConnector()  # 싱글턴 커넥터
        self._conn = self._connector.connection  # 실제 커넥션 객체
        self.crud = CRUDHelper(self._conn)
        self.DB_CONN_SIGNAL.emit("DB 연결 완료")

    @contextmanager
    def transaction(self):
        """
        transaction context manager
        """
        try:
            yield self._conn.commit()
        except Exception as e:
            self._conn.rollback()
            raise e

    # 1. 공통 유틸리티 메서드
    @handle_db_errors
    def _get_max_id(self, schema) -> int:
        query = f"SELECT MAX({schema.pk}) FROM {schema.name}"
        cur = self._conn.execute(query)
        return cur.fetchone()[0] or 0

    # 2. 문서 변환기 설정
    def insert_document_converter(self, data: Dict) -> int:
        return self.crud.create(DocumentConverterSchema(), data)

    def get_document_converter(self, id_val: int = None) -> Optional[Dict]:
        schema = DocumentConverterSchema()
        if not id_val:
            id_val = self._get_max_id(schema)
        return self.crud.get(schema, id_val)

    def update_document_converter(self, id_val: int, data: Dict) -> None:
        self.crud.update(DocumentConverterSchema(), id_val, data)

    # 3. 글로벌 설정 (Key-Value)
    def insert_or_update_global_setting(self, key_: str, value_: str) -> None:
        schema = GlobalSettingsSchema()
        existing = self.crud.get(schema, key_)
        if existing:
            self.crud.update(schema, key_, {"value": value_})
        else:
            self.crud.create(schema, {"key": key_, "value": value_})

    # 4. 클래스 정보 관리
    def insert_class_info(self, data: Dict) -> int:
        return self.crud.create(ClassInfoSchema(), data)

    def get_all_class_info(self) -> List[Dict]:
        return self.crud.get_all(ClassInfoSchema())

    # 5. 데이터 설정
    def insert_data_config(self, data: Dict) -> int:
        return self.crud.create(DataConfigSchema(), data)

    # 6. 결과 설정
    def insert_results_config(self, data: Dict) -> int:
        return self.crud.create(ResultsConfigSchema(), data)

    # 7. 모델 경로 관리
    def insert_model_paths(self, data: Dict) -> int:
        return self.crud.create(ModelPathsSchema(), data)

    # 8. 라벨링 설정
    def insert_labeling_config(self, data: Dict) -> int:
        return self.crud.create(LabelingConfigSchema(), data)

    # 9. 기타 설정
    def insert_other_settings(self, data: Dict) -> int:
        return self.crud.create(OtherSettingsSchema(), data)

    # 10. 학습 설정
    def insert_training_config(self, data: Dict) -> int:
        return self.crud.create(TrainingConfigSchema(), data)

    def get_training_config(self, id_val: int = None) -> Optional[Dict]:
        schema = TrainingConfigSchema()
        if not id_val:
            id_val = self._get_max_id(schema)
        return self.crud.get(schema, id_val)

    def update_training_config(self, id_val: int, data: Dict) -> None:
        return self.crud.update(TrainingConfigSchema(), id_val, data)

    # 11. 스키마 초기화
    def init_schema(self):
        from models.custom_yolo.resources.sql.db_init_load import DB_INIT_QUERY
        with self._conn:
            self._conn.executescript(DB_INIT_QUERY().get_init_query())

    def close(self):
        self._connector.close()
