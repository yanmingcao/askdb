"""Database configuration module for AskDB."""

import os
from typing import Dict, Optional, List, Any, Tuple
from dotenv import load_dotenv
import mysql.connector
from mysql.connector import Error
from mysql.connector.pooling import MySQLConnectionPool

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database configuration handler."""

    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.host = os.getenv("MYSQL_HOST")
        self.port = int(os.getenv("MYSQL_PORT", "3306"))
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DATABASE")

        # Validate required configuration
        self._validate_config()

        # Connection pool configuration
        self.pool_config: Dict[str, Any] = {
            "pool_name": "askdb_pool",
            "pool_size": 5,
            "pool_reset_session": True,
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }

        self._connection_pool: Optional[MySQLConnectionPool] = None

    def _validate_config(self):
        """Validate that all required configuration is present."""
        required_vars = {
            "MYSQL_HOST": self.host,
            "MYSQL_USER": self.user,
            "MYSQL_PASSWORD": self.password,
            "MYSQL_DATABASE": self.database,
        }

        missing = [var for var, value in required_vars.items() if not value]
        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}"
            )

    @property
    def connection_pool(self) -> MySQLConnectionPool:
        """Get or create connection pool."""
        if not self._connection_pool:
            self._connection_pool = MySQLConnectionPool(**self.pool_config)
        return self._connection_pool

    def get_connection(self):
        """Get a connection from the pool."""
        try:
            return self.connection_pool.get_connection()
        except Error as e:
            raise ConnectionError(f"Failed to get database connection: {e}")

    def test_connection(self) -> bool:
        """Test database connection."""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            return result[0] == 1
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False

    def get_connection_string(self) -> Dict[str, Any]:
        """Get connection parameters for Vanna."""
        return {
            "host": self.host,
            "dbname": self.database,
            "user": self.user,
            "password": self.password,
            "port": self.port,
        }


class DatabaseUtils:
    """Database utility functions."""

    def __init__(self, db_config: DatabaseConfig):
        """Initialize with database configuration."""
        self.db_config = db_config

    def execute_query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        """Execute a SELECT query and return results."""
        conn = None
        cursor = None
        try:
            conn = self.db_config.get_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute(query, params or ())
            return cursor.fetchall()
        except Error as e:
            raise RuntimeError(f"Query execution failed: {e}")
        finally:
            if cursor:
                cursor.close()
            if conn:
                conn.close()

    def get_table_list(self) -> List[Dict[str, Any]]:
        """Get list of all tables in the database."""
        query = """
        SELECT TABLE_NAME, TABLE_COMMENT 
        FROM INFORMATION_SCHEMA.TABLES 
        WHERE TABLE_SCHEMA = %s AND TABLE_TYPE = 'BASE TABLE'
        ORDER BY TABLE_NAME
        """
        return self.execute_query(query, (self.db_config.database,))

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        """Get schema information for a specific table."""
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
        return self.execute_query(query, (self.db_config.database, table_name))

    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        """Get index information for a table."""
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
        return self.execute_query(query, (self.db_config.database, table_name))

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        """Get all foreign key relationships in the database."""
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
        return self.execute_query(query, (self.db_config.database,))

    def get_create_table_ddl(self, table_name: str) -> str:
        """Get the actual CREATE TABLE DDL using SHOW CREATE TABLE."""
        results = self.execute_query(f"SHOW CREATE TABLE `{table_name}`")
        if results:
            return results[0].get("Create Table", "")
        return ""


# Singleton instances
_db_config = None
_db_utils = None


def get_db_config() -> DatabaseConfig:
    """Get singleton database configuration."""
    global _db_config
    if _db_config is None:
        _db_config = DatabaseConfig()
    return _db_config


def get_db_utils() -> DatabaseUtils:
    """Get singleton database utilities."""
    global _db_utils
    if _db_utils is None:
        _db_utils = DatabaseUtils(get_db_config())
    return _db_utils
