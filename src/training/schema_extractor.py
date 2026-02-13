"""Extract database schema and train Vanna on DDL statements."""

import os
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.semantic.store import (
    load_semantic_store,
    save_semantic_store,
    semantic_store_path,
)

from src.config.database import get_db_utils
from src.config.vanna_config import get_vanna, get_table_allowlist, AskDBVanna


def extract_ddl_for_table(table_name: str) -> str:
    """Generate CREATE TABLE DDL from INFORMATION_SCHEMA metadata."""
    db_utils = get_db_utils()
    columns = db_utils.get_table_schema(table_name)

    col_defs: list[str] = []
    for col in columns:
        parts = [f"  `{col['COLUMN_NAME']}` {col['DATA_TYPE']}"]

        max_len = col.get("CHARACTER_MAXIMUM_LENGTH")
        if max_len is not None:
            parts.append(f"({max_len})")

        if col["IS_NULLABLE"] == "NO":
            parts.append(" NOT NULL")

        default = col.get("COLUMN_DEFAULT")
        if default is not None:
            parts.append(f" DEFAULT {default}")

        comment = col.get("COLUMN_COMMENT")
        if comment:
            parts.append(f" COMMENT '{comment}'")

        if col.get("COLUMN_KEY") == "PRI":
            parts.append(" PRIMARY KEY")

        col_defs.append("".join(parts))

    ddl = f"CREATE TABLE `{table_name}` (\n"
    ddl += ",\n".join(col_defs)
    ddl += "\n);"
    return ddl


def extract_all_ddl(allowlist: Optional[List[str]] = None) -> List[str]:
    """Extract DDL for tables. Filters by allowlist if provided."""
    db_utils = get_db_utils()
    tables = db_utils.get_table_list()

    if allowlist:
        tables = [t for t in tables if t["TABLE_NAME"] in allowlist]

    ddl_statements: list[str] = []
    for table in tables:
        ddl = extract_ddl_for_table(table["TABLE_NAME"])
        ddl_statements.append(ddl)

    return ddl_statements


def train_vanna_on_schema(vn: Optional[AskDBVanna] = None) -> int:
    """Extract DDL and train Vanna. Returns count of trained tables."""
    if vn is None:
        vn = get_vanna()

    allowlist = get_table_allowlist()
    ddl_statements = extract_all_ddl(allowlist=allowlist)

    for ddl in ddl_statements:
        vn.train(ddl=ddl)

    return len(ddl_statements)


def train_vanna_on_relationships(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on foreign key relationships as documentation."""
    if vn is None:
        vn = get_vanna()

    db_utils = get_db_utils()
    fks = db_utils.get_foreign_keys()

    allowlist = get_table_allowlist()
    if allowlist:
        fks = [
            fk
            for fk in fks
            if fk["TABLE_NAME"] in allowlist or fk["REFERENCED_TABLE_NAME"] in allowlist
        ]

    if fks:
        doc = "Foreign key relationships:\n"
        for fk in fks:
            doc += (
                f"- {fk['TABLE_NAME']}.{fk['COLUMN_NAME']} -> "
                f"{fk['REFERENCED_TABLE_NAME']}.{fk['REFERENCED_COLUMN_NAME']}\n"
            )
        vn.train(documentation=doc)

    return len(fks)


def append_semantic_note(title: str, text: str) -> None:
    """Append a semantic note entry to the semantic store JSON file."""
    payload: Dict[str, Any] = {"notes": []}
    if os.path.exists(semantic_store_path()):
        payload = load_semantic_store()

    notes = payload.get("notes", [])
    notes.append({"title": title.strip(), "text": text.strip()})
    payload["notes"] = notes

    save_semantic_store(payload)


def _table_in_allowlist(table_full_name: str, allowlist: Optional[List[str]]) -> bool:
    if not allowlist:
        return True
    raw_table = table_full_name.split(".")[-1]
    return raw_table in allowlist


def _build_semantic_docs(
    store: Dict[str, Any], allowlist: Optional[List[str]]
) -> List[str]:
    docs: list[str] = []
    tables = store.get("tables", {})

    for table_full_name, table_info in tables.items():
        if not _table_in_allowlist(table_full_name, allowlist):
            continue

        description = table_info.get("description", "")
        header = f"Table {table_full_name}"
        if description:
            header += f": {description}"

        lines = [header, "Columns:"]
        columns = table_info.get("columns", {})
        for col_name, col_info in columns.items():
            if col_info.get("excluded") is True:
                continue
            col_desc = col_info.get("description", "")
            col_type = col_info.get("data_type", "")
            col_line = f"- {col_name}"
            if col_type:
                col_line += f" ({col_type})"
            if col_desc:
                col_line += f": {col_desc}"
            lines.append(col_line)

        docs.append("\n".join(lines))

    return docs


def train_vanna_on_semantic_schema(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on semantic schema documentation from JSON."""
    if vn is None:
        vn = get_vanna()

    allowlist = get_table_allowlist()
    store = load_semantic_store()
    docs = _build_semantic_docs(store, allowlist)

    for doc in docs:
        vn.train(documentation=doc)

    return len(docs)


def train_vanna_on_semantic_notes(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on general semantic notes."""
    if vn is None:
        vn = get_vanna()

    store = load_semantic_store()
    entries = store.get("notes", [])
    docs: list[str] = []
    for entry in entries:
        title = entry.get("title", "").strip()
        text = entry.get("text", "").strip()
        if title and text:
            docs.append(f"{title}: {text}")
        elif text:
            docs.append(text)

    for doc in docs:
        vn.train(documentation=doc)

    return len(docs)


def train_vanna_on_semantic_examples(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on explicit question-to-SQL examples."""
    if vn is None:
        vn = get_vanna()

    store = load_semantic_store()
    entries = store.get("examples", [])
    count = 0
    for entry in entries:
        question = entry.get("question", "").strip()
        sql = entry.get("sql", "").strip()
        if not question or not sql:
            continue
        vn.train(question=question, sql=sql)
        count += 1

    return count


def dump_ddl_to_file(path: str) -> int:
    """Dump DDL using SHOW CREATE TABLE into a JSON file."""
    db_utils = get_db_utils()
    allowlist = get_table_allowlist()
    tables = db_utils.get_table_list()
    if allowlist:
        tables = [t for t in tables if t["TABLE_NAME"] in allowlist]

    ddl_map: Dict[str, str] = {}
    for table in tables:
        table_name = table["TABLE_NAME"]
        ddl = db_utils.get_create_table_ddl(table_name)
        if ddl:
            ddl_map[table_name] = ddl

    payload = {
        "schema": db_utils.db_config.database,
        "dumped_at": datetime.utcnow().isoformat() + "Z",
        "tables": ddl_map,
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return len(ddl_map)


if __name__ == "__main__":
    print("Extracting schema...")
    count = train_vanna_on_schema()
    print(f"Trained Vanna on {count} table(s).")

    print("Training on relationships...")
    fk_count = train_vanna_on_relationships()
    print(f"Trained on {fk_count} foreign key relationship(s).")

    print("Training on semantic schema JSON...")
    doc_count = train_vanna_on_semantic_schema()
    print(f"Trained on {doc_count} semantic docs.")

    print("Training on semantic notes...")
    note_count = train_vanna_on_semantic_notes()
    print(f"Trained on {note_count} semantic notes.")

    print("Training on semantic examples...")
    ex_count = train_vanna_on_semantic_examples()
    print(f"Trained on {ex_count} semantic example(s).")
