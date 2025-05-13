import json
import os
import sqlite3
import threading
from definition import ROOT_DIR
from models.custom_yolo.core.services.db_service import DBService
from models.custom_yolo.resources.sql.db_init_load import DB_INIT_QUERY


class SQLiteConnector:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if not cls._instance:
                cls._instance = super().__new__(cls)
                cls._instance._init_connection()
        return cls._instance

    def _init_connection(self):
        self.db_path = os.path.join(ROOT_DIR, 'models/custom_yolo/core/domain/settings.sqlite')
        if not os.path.exists(self.db_path):
            DBService().init_schema(DB_INIT_QUERY().get_init_query())
        """데이터베이스 연결 초기화"""
        self.conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")  # 성능 개선

    @property
    def connection(self):
        """활성화된 연결 객체 반환"""
        if not self.conn:
            self._init_connection()
        return self.conn

    def close(self):
        """명시적 연결 종료"""
        if self.conn:
            self.conn.close()
