"""PostgreSQL adapter implementation."""

from typing import Any, Dict, List, Optional

try:
    import psycopg2
    import psycopg2.extras
except ImportError:  # pragma: no cover
    psycopg2 = None

from src.db.types import DBConfig


class PostgresAdapter:
    db_type = "postgres"

    def __init__(self, config: DBConfig):
        self.config = config
        if psycopg2 is None:
            raise ImportError("psycopg2 is required for PostgreSQL support")
        if not all([config.host, config.user, config.password, config.database]):
            raise ValueError("Missing PostgreSQL connection configuration")
        self.database_name = config.database or ""

    def _get_connection(self):
        return psycopg2.connect(
            host=self.config.host,
            port=self.config.port or 5432,
            user=self.config.user,
            password=self.config.password,
            dbname=self.config.database,
        )

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
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
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
        SELECT table_name AS "TABLE_NAME", COALESCE(obj_description((table_schema||'.'||table_name)::regclass), '') AS "TABLE_COMMENT"
        FROM information_schema.tables
        WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
        ORDER BY table_name
        """
        return self.execute_query(query)

    def get_table_schema(self, table_name: str) -> List[Dict[str, Any]]:
        query = """
        SELECT
            column_name AS "COLUMN_NAME",
            data_type AS "DATA_TYPE",
            character_maximum_length AS "CHARACTER_MAXIMUM_LENGTH",
            is_nullable AS "IS_NULLABLE",
            column_default AS "COLUMN_DEFAULT",
            COALESCE(col_description((table_schema||'.'||table_name)::regclass, ordinal_position), '') AS "COLUMN_COMMENT",
            '' AS "COLUMN_KEY"
        FROM information_schema.columns
        WHERE table_schema = 'public' AND table_name = %s
        ORDER BY ordinal_position
        """
        return self.execute_query(query, (table_name,))

    def get_table_indexes(self, table_name: str) -> List[Dict[str, Any]]:
        query = """
        SELECT
            indexname AS "INDEX_NAME",
            indexdef AS "INDEX_TYPE",
            '' AS "COLUMN_NAME",
            0 AS "NON_UNIQUE"
        FROM pg_indexes
        WHERE schemaname = 'public' AND tablename = %s
        ORDER BY indexname
        """
        return self.execute_query(query, (table_name,))

    def get_foreign_keys(self) -> List[Dict[str, Any]]:
        query = """
        SELECT
            tc.constraint_name AS "CONSTRAINT_NAME",
            tc.table_name AS "TABLE_NAME",
            kcu.column_name AS "COLUMN_NAME",
            ccu.table_name AS "REFERENCED_TABLE_NAME",
            ccu.column_name AS "REFERENCED_COLUMN_NAME"
        FROM information_schema.table_constraints AS tc
        JOIN information_schema.key_column_usage AS kcu
          ON tc.constraint_name = kcu.constraint_name
        JOIN information_schema.constraint_column_usage AS ccu
          ON ccu.constraint_name = tc.constraint_name
        WHERE tc.constraint_type = 'FOREIGN KEY' AND tc.table_schema = 'public'
        ORDER BY tc.table_name, tc.constraint_name
        """
        return self.execute_query(query)

    def get_create_table_ddl(self, table_name: str) -> str:
        columns = self.get_table_schema(table_name)
        col_defs = []
        for col in columns:
            parts = [f"  {col['COLUMN_NAME']} {col['DATA_TYPE']}"]
            max_len = col.get("CHARACTER_MAXIMUM_LENGTH")
            if max_len is not None:
                parts.append(f"({max_len})")
            if col.get("IS_NULLABLE") == "NO":
                parts.append(" NOT NULL")
            default = col.get("COLUMN_DEFAULT")
            if default is not None:
                parts.append(f" DEFAULT {default}")
            col_defs.append("".join(parts))

        ddl = f"CREATE TABLE {table_name} (\n"
        ddl += ",\n".join(col_defs)
        ddl += "\n);"
        return ddl

    def connect_vanna(self, vn: Any) -> None:
        if not hasattr(vn, "connect_to_postgres"):
            raise RuntimeError("Vanna does not support PostgreSQL in this build")
        vn.connect_to_postgres(
            host=self.config.host,
            dbname=self.config.database,
            user=self.config.user,
            password=self.config.password,
            port=self.config.port or 5432,
        )
