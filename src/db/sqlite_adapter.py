"""SQLite adapter implementation."""

from typing import Any, Dict, List, Optional

import sqlite3

from src.db.types import DBConfig


class SQLiteAdapter:
    db_type = "sqlite"

    def __init__(self, config: DBConfig):
        self.config = config
        if not config.sqlite_path:
            raise ValueError("SQLITE_PATH must be set for SQLite support")
        self.database_name = config.sqlite_path

    def _get_connection(self):
        return sqlite3.connect(self.config.sqlite_path)

    def test_connection(self) -> bool:
        try:
            conn = self._get_connection()
            cur = conn.cursor()
            cur.execute("SELECT 1")
            result = cur.fetchone()
            cur.close()
            conn.close()
            return result[0] == 1
        except Exception:
            return False

    def execute_query(
        self, query: str, params: Optional[tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        conn = None
        cur = None
        try:
            conn = self._get_connection()
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()
            cur.execute(query, params or ())
            return [dict(row) for row in cur.fetchall()]
        except Exception as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        finally:
            if cur:
                cur.close()
            if conn:
                conn.close()

    def get_table_list(self) -> List[Dict[str, Any]]:
        query = """
        SELECT name AS TABLE_NAME, '' AS TABLE_COMMENT
        FROM sqlite_master
        WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
        ORDER BY name
        """
        return self.execute_query(query)

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        rows = self.execute_query(f"PRAGMA table_info('{table_name}')")
        result = []
        for row in rows:
            result.append(
                {
                    "COLUMN_NAME": row.get("name"),
                    "DATA_TYPE": row.get("type"),
                    "CHARACTER_MAXIMUM_LENGTH": None,
                    "IS_NULLABLE": "NO" if row.get("notnull") == 1 else "YES",
                    "COLUMN_DEFAULT": row.get("dflt_value"),
                    "COLUMN_COMMENT": "",
                    "COLUMN_KEY": "PRI" if row.get("pk") == 1 else "",
                }
            )
        return result

    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        query = f"PRAGMA index_list('{table_name}')"
        rows = self.execute_query(query)
        return [
            {
                "INDEX_NAME": row.get("name"),
                "COLUMN_NAME": "",
                "NON_UNIQUE": 0 if row.get("unique") == 1 else 1,
                "INDEX_TYPE": "",
            }
            for row in rows
        ]

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        result: List[Dict[str, Any]] = []
        for table in self.get_table_list():
            table_name = table["TABLE_NAME"]
            rows = self.execute_query(f"PRAGMA foreign_key_list('{table_name}')")
            for row in rows:
                result.append(
                    {
                        "CONSTRAINT_NAME": str(row.get("id")),
                        "TABLE_NAME": table_name,
                        "COLUMN_NAME": row.get("from"),
                        "REFERENCED_TABLE_NAME": row.get("table"),
                        "REFERENCED_COLUMN_NAME": row.get("to"),
                    }
                )
        return result

    def get_create_table_ddl(self, table_name: str) -> str:
        rows = self.execute_query(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,),
        )
        if rows:
            return rows[0].get("sql", "") or ""
        return ""

    def connect_vanna(self, vn: Any) -> None:
        if not hasattr(vn, "connect_to_sqlite"):
            raise RuntimeError("Vanna does not support SQLite in this build")
        vn.connect_to_sqlite(self.config.sqlite_path)
