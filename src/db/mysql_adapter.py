"""MySQL adapter implementation."""

from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool

from src.db.types import DBConfig


class MySQLAdapter:
    db_type = "mysql"

    def __init__(self, config: DBConfig):
        self.config = config
        if not all([config.host, config.user, config.password, config.database]):
            raise ValueError("Missing MySQL connection configuration")

        self.database_name = config.database or ""
        self._pool: Optional[MySQLConnectionPool] = None
        self._pool_config = {
            "pool_name": "askdb_pool",
            "pool_size": 5,
            "pool_reset_session": True,
            "host": config.host,
            "port": config.port or 3306,
            "user": config.user,
            "password": config.password,
            "database": config.database,
        }

    @property
    def connection_pool(self) -> MySQLConnectionPool:
        if not self._pool:
            self._pool = MySQLConnectionPool(**self._pool_config)
        return self._pool

    def _get_connection(self):
        try:
            return self.connection_pool.get_connection()
        except Error as exc:
            raise ConnectionError(f"Failed to get database connection: {exc}")

    def test_connection(self) -> bool:
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result[0] == 1
        except Exception:
            return False

    def execute_query(
        self, query: str, params: Optional[tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        conn = None
        cursor = None
        try:
            conn = self._get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchall()
        except Error as exc:
            raise RuntimeError(f"Query execution failed: {exc}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_table_list(self) -> List[Dict[str, Any]]:
        query = """
        SELECT TABLE_NAME, TABLE_COMMENT
        FROM INFORMATION_SCHEMA.TABLES
        WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        return self.execute_query(query, (self.database_name,))

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        query = """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            CHARACTER_MAXIMUM_LENGTH,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            COLUMN_COMMENT,
            COLUMN_KEY
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """
        return self.execute_query(query, (self.database_name, table_name))

    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        query = """
        SELECT DISTINCT
            INDEX_NAME,
            COLUMN_NAME,
            NON_UNIQUE,
            INDEX_TYPE
        FROM INFORMATION_SCHEMA.STATISTICS
        WHERE TABLE_SCHEMA = %s AND TABLE_NAME = %s
        ORDER BY INDEX_NAME, SEQ_IN_INDEX
        """
        return self.execute_query(query, (self.database_name, table_name))

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        query = """
        SELECT
            CONSTRAINT_NAME,
            TABLE_NAME,
            COLUMN_NAME,
            REFERENCED_TABLE_NAME,
            REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = %s
            AND REFERENCED_TABLE_NAME IS NOT NULL
        ORDER BY TABLE_NAME, CONSTRAINT_NAME
        """
        return self.execute_query(query, (self.database_name,))

    def get_create_table_ddl(self, table_name: str) -> str:
        results = self.execute_query(f"SHOW CREATE TABLE `{table_name}`")
        if results:
            return results[0].get("Create Table", "")
        return ""

    def connect_vanna(self, vn: Any) -> None:
        vn.connect_to_mysql(
            host=self.config.host,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            port=self.config.port or 3306,
        )
