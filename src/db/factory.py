"""Database adapter factory."""

from src.db.types import DBConfig
from src.db.mysql_adapter import MySQLAdapter
from src.db.postgres_adapter import PostgresAdapter
from src.db.sqlite_adapter import SQLiteAdapter


def create_adapter(config: DBConfig):
    if config.db_type == "mysql":
        return MySQLAdapter(config)
    if config.db_type == "postgres":
        return PostgresAdapter(config)
    if config.db_type == "sqlite":
        return SQLiteAdapter(config)
    raise ValueError(f"Unsupported DB_TYPE: {config.db_type}")
