import sqlite3
from dataclasses import dataclass
from typing import List, Any, Dict, Optional


@dataclass
class TableSchema:
    name: str
    pk: str
    columns: List[str]


class CRUDHelper:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def _generate_placeholders(self, columns: List[str]) -> str:
        """
        예: columns=['poppler_path','output_dir'] -> '?, ?'
        """
        return ', '.join(['?'] * len(columns))

    def create(self, table: TableSchema, data: Dict[str, Any]) -> int:
        """INSERT into {table.name} (columns...) values (?,...)"""
        # 테이블에 실제 존재하는 컬럼만 추출
        columns = [col for col in data if col in table.columns]
        placeholders = self._generate_placeholders(columns)
        query = f"""
            INSERT INTO {table.name} ({', '.join(columns)})
            VALUES ({placeholders})
        """
        values = [data[col] for col in columns]
        with self.conn:
            cur = self.conn.execute(query, values)
            return cur.lastrowid

    def update(self, table: TableSchema, id_val: Any, data: Dict[str, Any]) -> None:
        """UPDATE {table.name} SET col=?... WHERE pk=?"""
        columns = [col for col in data if col in table.columns]
        set_clause = ', '.join([f"{col} = ?" for col in columns])
        query = f"""
            UPDATE {table.name}
            SET {set_clause}
            WHERE {table.pk} = ?
        """
        values = [data[col] for col in columns] + [id_val]
        with self.conn:
            self.conn.execute(query, values)

    def get(self, table: TableSchema, id_val: Any) -> Optional[Dict[str, Any]]:
        """SELECT * FROM {table.name} WHERE pk=?"""
        query = f"SELECT * FROM {table.name} WHERE {table.pk} = ?"
        cur = self.conn.execute(query, (id_val,))
        row = cur.fetchone()
        return dict(row) if row else None

    def get_all(self, table: TableSchema) -> List[Dict[str, Any]]:
        """SELECT * FROM {table.name}"""
        query = f"SELECT * FROM {table.name}"
        cur = self.conn.execute(query)
        return [dict(r) for r in cur.fetchall()]
