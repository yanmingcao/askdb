"""Database configuration module for AskDB."""

import os
from typing import Dict, Optional, List, Any, Tuple
from dotenv import load_dotenv

from src.db.factory import create_adapter
from src.db.types import DBConfig

# Load environment variables
load_dotenv()


class DatabaseConfig:
    """Database configuration handler."""

    def __init__(self):
        """Initialize database configuration from environment variables."""
        self.db_type = os.getenv("DB_TYPE", "mysql").strip().lower()
        self.host = os.getenv("MYSQL_HOST")
        self.port = (
            int(os.getenv("MYSQL_PORT", "3306")) if os.getenv("MYSQL_PORT") else None
        )
        self.user = os.getenv("MYSQL_USER")
        self.password = os.getenv("MYSQL_PASSWORD")
        self.database = os.getenv("MYSQL_DATABASE")
        self.sqlite_path = os.getenv("SQLITE_PATH")

        if self.db_type == "postgres":
            self.host = os.getenv("POSTGRES_HOST", self.host)
            self.port = (
                int(os.getenv("POSTGRES_PORT", "5432"))
                if os.getenv("POSTGRES_PORT")
                else self.port
            )
            self.user = os.getenv("POSTGRES_USER", self.user)
            self.password = os.getenv("POSTGRES_PASSWORD", self.password)
            self.database = os.getenv("POSTGRES_DATABASE", self.database)

        self._validate_config()

    def _validate_config(self):
        """Validate that all required configuration is present."""
        if self.db_type == "sqlite":
            if not self.sqlite_path:
                raise ValueError("SQLITE_PATH is required for sqlite")
            return

        required = {
            "host": self.host,
            "user": self.user,
            "password": self.password,
            "database": self.database,
        }
        missing = [k for k, v in required.items() if not v]
        if missing:
            raise ValueError(f"Missing required DB configuration: {', '.join(missing)}")

    def to_db_config(self) -> DBConfig:
        return DBConfig(
            db_type=self.db_type,
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database,
            sqlite_path=self.sqlite_path,
        )


class DatabaseUtils:
    """Database utility functions (adapter-backed)."""

    def __init__(self, db_config: DatabaseConfig):
        self.db_config = db_config
        self.adapter = create_adapter(db_config.to_db_config())

    def test_connection(self) -> bool:
        return self.adapter.test_connection()

    def execute_query(
        self, query: str, params: Optional[Tuple[Any, ...]] = None
    ) -> List[Dict[str, Any]]:
        return self.adapter.execute_query(query, params)

    def get_table_list(self) -> List[Dict[str, Any]]:
        return self.adapter.get_table_list()

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        return self.adapter.get_table_schema(table_name)

    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        return self.adapter.get_table_indexes(table_name)

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        return self.adapter.get_foreign_keys()

    def get_create_table_ddl(self, table_name: str) -> str:
        return self.adapter.get_create_table_ddl(table_name)

    def get_view_list(self) -> List[Dict[str, Any]]:
        return self.adapter.get_view_list()

    def get_create_view_ddl(self, view_name: str) -> str:
        return self.adapter.get_create_view_ddl(view_name)

    def connect_vanna(self, vn: Any) -> None:
        return self.adapter.connect_vanna(vn)


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
