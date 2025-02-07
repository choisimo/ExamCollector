import json
from contextlib import contextmanager

from PyQt5.QtCore import pyqtSignal

from definition import ROOT_DIR
from models.custom_yolo.common.error_handling.db_logger import handle_db_errors
from models.custom_yolo.core.services.crud_helper import CRUDHelper
from models.custom_yolo.infrastructure.storage.SQLite_connector import SQLiteConnector
from models.custom_yolo.resources.sql.db_init_load import DB_INIT_QUERY


class DBService:
    DB_CONN_SIGNAL = pyqtSignal(str)
    """
    DBService: 각 테이블에 대한 기본적인 CRUD 메서드를 제공하는 서비스 클래스.

    - init_schema(): 상기의 SCHEMA_SQL을 실행하여 테이블 전부 초기화.
    - insert_XXX(), get_XXX(), update_XXX(), delete_XXX() 형태의 메서드를 필요에 따라 작성.
    - 트랜잭션(ACID) 보장을 위해 with self._conn: 블록 내에서 DML을 수행.
    """

    def __init__(self, db_path: str = None):
        self._connector = SQLiteConnector()  # 싱글턴 커넥터
        self._conn = self._connector.connection  # 실제 커넥션 객체
        # 필요한 경우, 다음과 같이 FK 활성화를 보장할 수도 있음
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

    @handle_db_errors
    def _get_max_id(self, table):
        query = f"SELECT MAX({table.pk} FROM {table.name}"
        cur = self._conn.execute(query)
        return cur.fetchone()[0] or 0

    # -------------------------------
    # 1) 전체 스키마 초기화
    # -------------------------------
    def init_schema(self):
        """
        SCHEMA_SQL을 이용해 모든 테이블을 초기화(DROP 후 CREATE).
        실제 운영 시에는 신중히 사용할 것.
        """
        with self._conn:
            self._conn.executescript(DB_INIT_QUERY().get_init_query())

    # ------------------------------------------------------------------------------
    # 2) document_converter
    # ------------------------------------------------------------------------------
    def insert_document_converter(self, poppler_path: str, output_dir: str) -> int:
        """
        document_converter 테이블에 레코드 추가. ACID 트랜잭션 보장.
        반환: 새 레코드의 id (PRIMARY KEY).
        """
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO document_converter (poppler_path, output_dir)
                VALUES (?, ?)
            """, (poppler_path, output_dir))
            return cur.lastrowid

    def get_document_converter(self, id_val: int = None):
        """
        특정 id로 document_converter 조회.
        반환: dict or None
        """
        # id_val 미지정 시 마지막 레코드 반환
        if id_val is None:
            id_val = self._conn.execute("SELECT MAX(id) FROM document_converter").fetchone()[0]
        cur = self._conn.cursor()
        cur.execute("""
            SELECT * FROM document_converter
            WHERE id = ?
        """, (id_val,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None

    def update_document_converter(self, id_val: int, poppler_path: str, output_dir: str):
        with self._conn:
            self._conn.execute("""
                UPDATE document_converter
                SET poppler_path = ?, output_dir = ?
                WHERE id = ?
            """, (poppler_path, output_dir, id_val))

    def delete_document_converter(self, id_val: int):
        with self._conn:
            self._conn.execute("""
                DELETE FROM document_converter
                WHERE id = ?
            """, (id_val,))

    # ------------------------------------------------------------------------------
    # 3) global_settings (key-value)
    # ------------------------------------------------------------------------------
    def insert_or_update_global_setting(self, key_: str, value_: str):
        """
        global_settings 테이블에 key/value를 upsert.
        """
        with self._conn:
            self._conn.execute("""
                INSERT INTO global_settings (key, value)
                VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value=excluded.value
            """, (key_, value_))

    def get_global_setting(self, key_: str) -> str:
        cur = self._conn.cursor()
        cur.execute("SELECT value FROM global_settings WHERE key = ?", (key_,))
        row = cur.fetchone()
        return row["value"] if row else None

    def delete_global_setting(self, key_: str):
        with self._conn:
            self._conn.execute("""
                DELETE FROM global_settings WHERE key = ?
            """, (key_,))

    # ------------------------------------------------------------------------------
    # 4) class_info
    # ------------------------------------------------------------------------------
    def insert_class_info(self, class_idx: int, class_name: str):
        with self._conn:
            self._conn.execute("""
                INSERT INTO class_info (class_idx, class_name)
                VALUES (?, ?)
            """, (class_idx, class_name))

    def get_class_info(self, class_idx: int):
        cur = self._conn.cursor()
        cur.execute("""
            SELECT class_idx, class_name FROM class_info
            WHERE class_idx = ?
        """, (class_idx,))
        row = cur.fetchone()
        if row:
            return dict(row)
        return None

    def get_all_class_info(self):
        cur = self._conn.cursor()
        cur.execute("SELECT class_idx, class_name FROM class_info ORDER BY class_idx")
        rows = cur.fetchall()
        return [dict(r) for r in rows]

    def update_class_info(self, class_idx: int, class_name: str):
        with self._conn:
            self._conn.execute("""
                UPDATE class_info
                SET class_name = ?
                WHERE class_idx = ?
            """, (class_name, class_idx))

    def delete_class_info(self, class_idx: int):
        with self._conn:
            self._conn.execute("""
                DELETE FROM class_info
                WHERE class_idx = ?
            """, (class_idx,))

    # ------------------------------------------------------------------------------
    # 5) data_config
    # ------------------------------------------------------------------------------
    def insert_data_config(self, base_path: str, train_path: str, val_path: str) -> int:
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO data_config (base_path, train_path, val_path)
                VALUES (?, ?, ?)
            """, (base_path, train_path, val_path))
            return cur.lastrowid

    def get_data_config(self, id_val: int):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM data_config WHERE id = ?", (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    # (update, delete 등 필요 시 추가)

    # ------------------------------------------------------------------------------
    # 6) results_config
    # ------------------------------------------------------------------------------
    def insert_results_config(self, base_path: str, train_path: str, val_path: str) -> int:
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO results_config (base_path, train_path, val_path)
                VALUES (?, ?, ?)
            """, (base_path, train_path, val_path))
            return cur.lastrowid

    def get_results_config(self, id_val: int):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM results_config WHERE id = ?", (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    # (update, delete 등 필요 시 추가)

    # ------------------------------------------------------------------------------
    # 7) model_paths
    # ------------------------------------------------------------------------------
    def insert_model_paths(self, detector_path: str, llm_model: str) -> int:
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO model_paths (detector_model_path, llm_model)
                VALUES (?, ?)
            """, (detector_path, llm_model))
            return cur.lastrowid

    def get_model_paths(self, id_val: int = None):
        cur = self._conn.cursor()
        if id_val is None:
            return self._conn.execute("SELECT MAX(id) FROM model_paths").fetchone()[0]
        cur.execute("SELECT * FROM model_paths WHERE id = ?", (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------------------
    # 8) labeling_config
    # ------------------------------------------------------------------------------
    def insert_labeling_config(self,
                               device: str,
                               label_save_mode: str,
                               llm_max_retries: int,
                               detect_imgsz: int,
                               detect_conf: float,
                               llm_max_new_tokens: int) -> int:
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO labeling_config (
                    device, label_save_mode, llm_max_retries,
                    detect_imgsz, detect_conf, llm_max_new_tokens
                )
                VALUES (?, ?, ?, ?, ?, ?)
            """, (device, label_save_mode, llm_max_retries,
                  detect_imgsz, detect_conf, llm_max_new_tokens))
            return cur.lastrowid

    def update_labeling_config(self, id_val: int, path: str):
        with self._conn:
            cur = self._conn.execute("""
            UPDATE SET labeling_config = ?
             VALUES (? ?)
            """)

    def get_labeling_config(self, id_val: int = None):
        if id_val is None:
            return self._conn.execute("SELECT MAX(id) FROM labeling_config").fetchone()[0]
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM labeling_config WHERE id = ?", (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------------------
    # 9) other_settings
    # ------------------------------------------------------------------------------
    def insert_other_settings(self, use_gpu: int, log_level: str) -> int:
        """
        use_gpu: 1 or 0
        """
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO other_settings (use_gpu, log_level)
                VALUES (?, ?)
            """, (use_gpu, log_level))
            return cur.lastrowid

    def get_other_settings(self, id_val: int):
        cur = self._conn.cursor()
        cur.execute("SELECT * FROM other_settings WHERE id = ?", (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    # ------------------------------------------------------------------------------
    # 10) training_config
    # ------------------------------------------------------------------------------
    def insert_training_config(self,
                               epochs: int,
                               batch_size: int,
                               device: str,
                               imgsz: int,
                               YOLO_model: str,
                               custom_model_path: str,
                               data_yaml: str,
                               data_config_id: int,
                               results_config_id: int,
                               model_paths_id: int) -> int:
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO training_config (
                    epochs, batch_size, device, imgsz,
                    YOLO_model, custom_model_path, data_yaml,
                    data_config_id, results_config_id, model_paths_id
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (epochs, batch_size, device, imgsz,
                  YOLO_model, custom_model_path, data_yaml,
                  data_config_id, results_config_id, model_paths_id))
            return cur.lastrowid

    @handle_db_errors
    def get_training_config(self, id_val: int = None):
        table = self.schemas['training']
        if not id_val:
            id_val = self._get_max_id(table)
        if id_val == 0:
            return None
        return self.crud.get(table, id_val)

    # (update, delete 등 필요 시 추가)
    def update_model_path(self, target_id: int, path: str):
        cur = self._conn.cursor()
        cur.execute("UPDATE custom_model_path SET path = ? WHERE id = ?", (path, target_id))
        row = cur.fetchone()
        return dict(row) if row else None

    def insert_model_path(self, path):
        with self._conn:
            cur = self._conn.execute("""
                INSERT INTO custom_model_path (path)
                VALUES (?)
            """, (path,))
            return cur.lastrowid

    def update_training_config(self, id_val, path):
        with self._conn:
            cur = self._conn.execute(f"""
                UPDATE training_config
                SET data_yaml = ?
                WHERE id = ?
            """, (path, id_val))
        return cur.lastrowid

    # ------------------------------------------------------------------------------
    # 추가적으로, 필요한 쿼리는 아래처럼 자유롭게 구현 가능
    # ------------------------------------------------------------------------------
    def close(self):
        """ 명시적 DB 연결 해제 (옵션) """
        self._connector.close()

