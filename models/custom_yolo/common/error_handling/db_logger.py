import sqlite3
from sqlite3 import DatabaseError
from functools import wraps


class Database_db_error(Exception):
    pass


def handle_db_errors(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except sqlite3.IntegrityError as e:
            raise DatabaseError(f"무결성 제약 위반: {str(e)}") from e
        except sqlite3.OperationalError as e:
            raise DatabaseError(f"DB 오퍼레이션 오류: {str(e)}") from e
    return wrapper
