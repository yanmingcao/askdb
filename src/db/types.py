"""Shared types for DB adapters."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class DBConfig:
    db_type: str
    host: Optional[str] = None
    port: Optional[int] = None
    user: Optional[str] = None
    password: Optional[str] = None
    database: Optional[str] = None
    sqlite_path: Optional[str] = None
