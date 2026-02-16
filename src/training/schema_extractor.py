"""Extract database schema and train Vanna on DDL statements."""

import os
import json
import re
from pathlib import Path
from datetime import datetime
from typing import List, Optional, Dict, Any

from src.semantic.store import (
    load_semantic_store,
    save_semantic_store,
    semantic_store_path,
)
from src.training.training_state import (
    load_training_state,
    save_training_state,
    reset_chromadb_data,
    hash_text,
)

from src.config.database import get_db_utils
from src.config.vanna_config import (
    get_vanna,
    get_table_allowlist,
    get_view_allowlist,
    AskDBVanna,
)


def _get_excluded_columns(table_name: str) -> set[str]:
    """Get set of excluded column names for a table from semantic_store.

    Returns empty set if table not in semantic_store or has no excluded columns.
    """
    try:
        store = load_semantic_store()
        tables = store.get("tables", {})

        # Try exact match first, then try with schema prefix
        table_info = None
        for full_name, info in tables.items():
            if full_name.endswith(f".{table_name}") or full_name == table_name:
                table_info = info
                break

        if not table_info:
            return set()

        excluded = set()
        columns = table_info.get("columns", {})
        for col_name, col_info in columns.items():
            if col_info.get("excluded") is True:
                excluded.add(col_name)

        return excluded
    except Exception:
        # If semantic store doesn't exist or has issues, return empty set
        return set()


def extract_ddl_for_table(table_name: str) -> str:
    """Generate CREATE TABLE DDL from INFORMATION_SCHEMA metadata.

    Excludes columns marked as excluded in semantic_store.
    """
    db_utils = get_db_utils()
    columns = db_utils.get_table_schema(table_name)
    excluded_cols = _get_excluded_columns(table_name)

    col_defs: list[str] = []
    for col in columns:
        col_name = col["COLUMN_NAME"]

        # Skip excluded columns
        if col_name in excluded_cols:
            continue

        parts = [f"  `{col_name}` {col['DATA_TYPE']}"]

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


def extract_ddl_for_view(view_name: str) -> str:
    """Generate a view DDL stub using exposed column names/types.

    This avoids training on underlying view SQL definitions.
    """
    db_utils = get_db_utils()
    columns = db_utils.get_table_schema(view_name)

    col_defs: list[str] = []
    for col in columns:
        col_name = col["COLUMN_NAME"]
        parts = [f"  `{col_name}` {col['DATA_TYPE']}"]

        max_len = col.get("CHARACTER_MAXIMUM_LENGTH")
        if max_len is not None:
            parts.append(f"({max_len})")

        col_defs.append("".join(parts))

    ddl = f"CREATE VIEW `{view_name}` (\n"
    ddl += ",\n".join(col_defs)
    ddl += "\n);"
    return ddl


def extract_all_ddl(allowlist: Optional[List[str]] = None) -> List[str]:
    """Extract DDL for tables. Filters by allowlist if provided."""
    db_utils = get_db_utils()
    tables = db_utils.get_table_list()

    if allowlist is not None:
        tables = [t for t in tables if t["TABLE_NAME"] in allowlist]

    ddl_statements: list[str] = []
    for table in tables:
        ddl = extract_ddl_for_table(table["TABLE_NAME"])
        ddl_statements.append(ddl)

    return ddl_statements


def _load_ddl_dump() -> Dict[str, Any]:
    ddl_path = Path(__file__).resolve().parents[1] / "semantic" / "ddl_dump.json"
    if not ddl_path.exists():
        raise FileNotFoundError("ddl_dump.json not found. Run 'dump-ddl' first.")
    with ddl_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _filter_ddl_map(
    ddl_map: Dict[str, Any], allowlist: Optional[List[str]]
) -> Dict[str, Any]:
    if allowlist is None:
        return dict(ddl_map)
    if not allowlist:
        return {}
    return {k: v for k, v in ddl_map.items() if k in allowlist}


def train_vanna_on_schema(vn: Optional[AskDBVanna] = None) -> tuple[int, int]:
    """Train Vanna on DDL from ddl_dump.json. Returns (table_count, view_count)."""
    if vn is None:
        vn = get_vanna()

    ddl_dump = _load_ddl_dump()
    table_allowlist = get_table_allowlist()
    view_allowlist = get_view_allowlist()
    tables_ddl = _filter_ddl_map(ddl_dump.get("tables", {}), table_allowlist)
    views_ddl = _filter_ddl_map(ddl_dump.get("views", {}), view_allowlist)

    ddl_statements: list[str] = []
    for name in sorted(tables_ddl.keys()):
        ddl_statements.append(tables_ddl[name])
    for name in sorted(views_ddl.keys()):
        ddl_statements.append(views_ddl[name])

    for ddl in ddl_statements:
        vn.train(ddl=ddl)

    return len(tables_ddl), len(views_ddl)


def train_vanna_on_relationships(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on foreign key relationships as documentation."""
    if vn is None:
        vn = get_vanna()

    ddl_dump = _load_ddl_dump()
    relationships = ddl_dump.get("relationships", [])
    if not isinstance(relationships, list):
        relationships = []

    allowlist = get_table_allowlist()
    if allowlist is not None:
        relationships = [
            fk
            for fk in relationships
            if fk.get("TABLE_NAME") in allowlist
            or fk.get("REFERENCED_TABLE_NAME") in allowlist
        ]

    if relationships:
        doc = "Foreign key relationships:\n"
        for fk in relationships:
            table_name = fk.get("TABLE_NAME", "")
            column_name = fk.get("COLUMN_NAME", "")
            ref_table = fk.get("REFERENCED_TABLE_NAME", "")
            ref_column = fk.get("REFERENCED_COLUMN_NAME", "")
            if not table_name or not column_name or not ref_table or not ref_column:
                continue
            doc += f"- {table_name}.{column_name} -> {ref_table}.{ref_column}\n"
        if doc.strip() != "Foreign key relationships:":
            vn.train(documentation=doc)

    return len(relationships)


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


def _view_in_allowlist(view_full_name: str, allowlist: Optional[List[str]]) -> bool:
    if not allowlist:
        return True
    raw_view = view_full_name.split(".")[-1]
    return raw_view in allowlist


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

    view_allowlist = get_view_allowlist()
    views = store.get("views", {})

    for view_full_name, view_info in views.items():
        if not _view_in_allowlist(view_full_name, view_allowlist):
            continue

        description = view_info.get("description", "")
        header = f"View {view_full_name}"
        if description:
            header += f": {description}"

        lines = [header, "Columns:"]
        columns = view_info.get("columns", {})
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


def _build_semantic_docs_map(
    store: Dict[str, Any], allowlist: Optional[List[str]]
) -> Dict[str, str]:
    docs: Dict[str, str] = {}
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

        docs[table_full_name] = "\n".join(lines)

    view_allowlist = get_view_allowlist()
    views = store.get("views", {})

    for view_full_name, view_info in views.items():
        if not _view_in_allowlist(view_full_name, view_allowlist):
            continue

        description = view_info.get("description", "")
        header = f"View {view_full_name}"
        if description:
            header += f": {description}"

        lines = [header, "Columns:"]
        columns = view_info.get("columns", {})
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

        docs[view_full_name] = "\n".join(lines)

    return docs


def train_vanna_on_semantic_schema(vn: Optional[AskDBVanna] = None) -> int:
    """Train Vanna on semantic schema documentation from JSON."""
    if vn is None:
        vn = get_vanna()

    allowlist = get_table_allowlist()
    store = load_semantic_store()
    docs_map = _build_semantic_docs_map(store, allowlist)

    for doc in docs_map.values():
        vn.train(documentation=doc)

    return len(docs_map)


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


def _compute_semantic_items() -> Dict[str, Dict[str, Any]]:
    allowlist = get_table_allowlist() or []
    view_allowlist = get_view_allowlist() or []
    store = load_semantic_store()
    items: Dict[str, Dict[str, Any]] = {}

    allowlist_hash = hash_text(
        json.dumps(
            {
                "tables": sorted(allowlist),
                "views": sorted(view_allowlist),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )
    items["allowlist"] = {
        "type": "allowlist",
        "hash": allowlist_hash,
    }

    docs_map = _build_semantic_docs_map(store, allowlist)
    for table_full_name, doc in docs_map.items():
        item_id = f"schema_doc:{table_full_name}"
        items[item_id] = {
            "type": "schema_doc",
            "hash": hash_text(doc),
            "payload": doc,
        }

    notes = store.get("notes", [])
    for idx, entry in enumerate(notes):
        title = str(entry.get("title", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not title and not text:
            continue
        note_id = f"note:{title}" if title else f"note:{idx}"
        content = f"{title}\n{text}".strip()
        items[note_id] = {
            "type": "note",
            "hash": hash_text(content),
            "payload": content,
        }

    examples = store.get("examples", [])
    for idx, entry in enumerate(examples):
        question = str(entry.get("question", "")).strip()
        sql = str(entry.get("sql", "")).strip()
        notes_text = str(entry.get("notes", "")).strip()
        if not question or not sql:
            continue
        example_id = f"example:{question}" if question else f"example:{idx}"
        content = f"{question}\n{sql}\n{notes_text}".strip()
        items[example_id] = {
            "type": "example",
            "hash": hash_text(content),
            "payload": {"question": question, "sql": sql},
        }

    return items


def _diff_training_state(
    state_items: Dict[str, Dict[str, Any]],
    current_items: Dict[str, Dict[str, Any]],
) -> tuple[Dict[str, Dict[str, Any]], list[str]]:
    changed: Dict[str, Dict[str, Any]] = {}
    for item_id, item in current_items.items():
        prev = state_items.get(item_id, {})
        if prev.get("hash") != item.get("hash"):
            changed[item_id] = item

    removed = [item_id for item_id in state_items if item_id not in current_items]
    return changed, removed


def train_vanna_incremental(
    sync: bool = True, allow_reset_failure: bool = False
) -> Dict[str, Any]:
    """Incremental train on semantic store changes with deletion sync.

    Returns a dict with mode and counts.
    """
    state = load_training_state()
    state_items = state.get("items", {}) if isinstance(state, dict) else {}
    current_items = _compute_semantic_items()

    changed, removed = _diff_training_state(state_items, current_items)
    allowlist_changed = "allowlist" in changed or (
        "allowlist" in state_items and "allowlist" not in current_items
    )

    if sync and (removed or allowlist_changed):
        try:
            reset_chromadb_data()
        except PermissionError:
            if not allow_reset_failure:
                raise
            return {
                "mode": "deferred",
                "reason": "chromadb_data_in_use",
                "schema_tables": 0,
                "schema_views": 0,
                "relationships": 0,
                "semantic_docs": 0,
                "notes": 0,
                "examples": 0,
            }
        table_count, view_count = train_vanna_on_schema()
        fk_count = train_vanna_on_relationships()
        doc_count = train_vanna_on_semantic_schema()
        note_count = train_vanna_on_semantic_notes()
        ex_count = train_vanna_on_semantic_examples()
        save_training_state(current_items)
        return {
            "mode": "full",
            "schema_tables": table_count,
            "schema_views": view_count,
            "relationships": fk_count,
            "semantic_docs": doc_count,
            "notes": note_count,
            "examples": ex_count,
        }

    doc_count = 0
    note_count = 0
    ex_count = 0
    vn = get_vanna()
    for item in changed.values():
        item_type = item.get("type")
        if item_type == "schema_doc":
            vn.train(documentation=item.get("payload", ""))
            doc_count += 1
        elif item_type == "note":
            vn.train(documentation=item.get("payload", ""))
            note_count += 1
        elif item_type == "example":
            payload = item.get("payload", {})
            question = str(payload.get("question", "")).strip()
            sql = str(payload.get("sql", "")).strip()
            if question and sql:
                vn.train(question=question, sql=sql)
                ex_count += 1

    save_training_state(current_items)
    return {
        "mode": "incremental",
        "semantic_docs": doc_count,
        "notes": note_count,
        "examples": ex_count,
    }


def init_semantic_store_from_ddl(
    ddl_path: str, store_path: str, allowlist: Optional[set[str]] = None
) -> int:
    """Initialize semantic_store.json from ddl_dump.json.

    Args:
        ddl_path: Path to ddl_dump.json
        store_path: Path to semantic_store.json (will be created/overwritten)
        allowlist: Optional set of table names to include

    Returns:
        Number of tables initialized
    """
    with open(ddl_path, "r", encoding="utf-8") as handle:
        ddl_dump = json.load(handle)

    schema_name = ddl_dump.get("schema", "")
    tables_ddl = ddl_dump.get("tables", {})
    views_ddl = ddl_dump.get("views", {})

    if allowlist:
        tables_ddl = {k: v for k, v in tables_ddl.items() if k in allowlist}
        views_ddl = {k: v for k, v in views_ddl.items() if k in allowlist}

    table_names = sorted(tables_ddl.keys())
    view_names = sorted(views_ddl.keys())

    store: Dict[str, Any] = {
        "schema_prefix": schema_name,
        "allowlist": {
            "tables": table_names,
            "views": view_names,
        },
        "tables": {},
        "views": {},
        "notes": [],
        "examples": [],
    }

    column_id = 1
    for table_name in sorted(tables_ddl.keys()):
        full_name = f"{schema_name}.{table_name}" if schema_name else table_name
        columns: Dict[str, Any] = {}
        ddl = tables_ddl.get(table_name, "")
        for col_name, col_type in _extract_columns_from_ddl(ddl):
            col_full_name = f"{full_name}.{col_name}"
            columns[col_name] = {
                "id": column_id,
                "name": col_name,
                "full_name": col_full_name,
                "data_type": col_type,
                "description": "",
            }
            column_id += 1
        store["tables"][full_name] = {
            "id": len(store["tables"]) + 1,
            "name": table_name,
            "full_name": full_name,
            "table_type": "table",
            "description": "",
            "columns": columns,
        }

    for view_name in sorted(views_ddl.keys()):
        full_name = f"{schema_name}.{view_name}" if schema_name else view_name
        columns: Dict[str, Any] = {}
        ddl = views_ddl.get(view_name, "")
        for col_name, col_type in _extract_columns_from_ddl(ddl):
            col_full_name = f"{full_name}.{col_name}"
            columns[col_name] = {
                "id": column_id,
                "name": col_name,
                "full_name": col_full_name,
                "data_type": col_type,
                "description": "",
            }
            column_id += 1
        store["views"][full_name] = {
            "id": len(store["views"]) + 1,
            "name": view_name,
            "full_name": full_name,
            "table_type": "view",
            "description": "",
            "columns": columns,
        }

    with open(store_path, "w", encoding="utf-8") as handle:
        json.dump(store, handle, ensure_ascii=False, indent=2)

    return len(store["tables"]) + len(store["views"])


def _split_ddl_columns(body: str) -> list[str]:
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    for ch in body:
        if ch == "(":
            depth += 1
        elif ch == ")":
            if depth > 0:
                depth -= 1
        if ch == "," and depth == 0:
            chunk = "".join(current).strip()
            if chunk:
                parts.append(chunk)
            current = []
            continue
        current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _extract_data_type(rest: str) -> str:
    keywords = {
        "not",
        "null",
        "default",
        "primary",
        "unique",
        "references",
        "constraint",
        "check",
        "collate",
        "comment",
        "key",
        "generated",
        "identity",
        "auto_increment",
        "autoincrement",
        "on",
        "using",
    }
    tokens = rest.split()
    data_tokens: list[str] = []
    for token in tokens:
        if token.lower() in keywords:
            break
        data_tokens.append(token)
    return " ".join(data_tokens).strip()


def _extract_columns_from_ddl(ddl: str) -> list[tuple[str, str]]:
    if not ddl:
        return []
    start = ddl.find("(")
    end = ddl.rfind(")")
    if start == -1 or end == -1 or end <= start:
        return []
    body = ddl[start + 1 : end]
    chunks = _split_ddl_columns(body)
    results: list[tuple[str, str]] = []
    for chunk in chunks:
        line = chunk.strip().rstrip(",")
        if not line:
            continue
        lower = line.lower()
        if lower.startswith(
            (
                "primary key",
                "unique key",
                "unique index",
                "key ",
                "index ",
                "constraint ",
                "foreign key",
                "check ",
            )
        ):
            continue

        match = re.match(r"`([^`]+)`\s+(.*)", line)
        if not match:
            match = re.match(r"\"([^\"]+)\"\s+(.*)", line)
        if not match:
            match = re.match(r"([A-Za-z0-9_]+)\s+(.*)", line)
        if not match:
            continue

        col_name = match.group(1).strip()
        rest = match.group(2).strip()
        data_type = _extract_data_type(rest)
        if not col_name or not data_type:
            continue
        results.append((col_name, data_type))

    return results


def dump_ddl_to_file(path: str, force: bool = False) -> tuple[int, int]:
    """Dump DDL using SHOW CREATE TABLE/VIEW into a JSON file.

    Args:
        path: Output file path for ddl_dump.json
        force: If True, dump all tables and views ignoring allowlist
    """
    from src.semantic.store import load_semantic_store, semantic_store_path

    db_utils = get_db_utils()

    # Get allowlist for tables and views
    # None = no allowlist (dump all), [] = empty allowlist (dump nothing), [...] = filter
    table_allowlist = None
    view_allowlist = None
    if not force and os.path.exists(semantic_store_path()):
        store = load_semantic_store()
        allowlist_obj = store.get("allowlist")
        if allowlist_obj is not None and isinstance(allowlist_obj, dict):
            # Check if "tables" key exists
            if "tables" in allowlist_obj:
                tables_raw = allowlist_obj.get("tables", [])
                if not tables_raw:
                    # Empty list -> dump nothing
                    table_allowlist = []
                else:
                    # Non-empty list -> filter
                    table_allowlist = [
                        t.strip()
                        for t in tables_raw
                        if isinstance(t, str) and t.strip()
                    ]

            # Check if "views" key exists
            if "views" in allowlist_obj:
                views_raw = allowlist_obj.get("views", [])
                if not views_raw:
                    # Empty list -> dump nothing
                    view_allowlist = []
                else:
                    # Non-empty list -> filter
                    view_allowlist = [
                        v.strip() for v in views_raw if isinstance(v, str) and v.strip()
                    ]

    # Dump tables
    tables = db_utils.get_table_list()
    if table_allowlist is not None:
        # If allowlist is [], this will result in empty tables list
        tables = [t for t in tables if t["TABLE_NAME"] in table_allowlist]

    table_ddl_map: Dict[str, str] = {}
    for table in tables:
        table_name = table["TABLE_NAME"]
        ddl = db_utils.get_create_table_ddl(table_name)
        if ddl:
            table_ddl_map[table_name] = ddl

    # Dump views (exposed columns only)
    views = db_utils.get_view_list()
    if view_allowlist is not None:
        # If allowlist is [], this will result in empty views list
        views = [v for v in views if v["TABLE_NAME"] in view_allowlist]

    view_ddl_map: Dict[str, str] = {}
    for view in views:
        view_name = view["TABLE_NAME"]
        ddl = extract_ddl_for_view(view_name)
        if ddl:
            view_ddl_map[view_name] = ddl

    # Dump relationships (foreign keys)
    relationships = db_utils.get_foreign_keys()
    if table_allowlist is not None:
        relationships = [
            fk
            for fk in relationships
            if fk.get("TABLE_NAME") in table_allowlist
            or fk.get("REFERENCED_TABLE_NAME") in table_allowlist
        ]

    schema_name = (
        db_utils.db_config.database
        or db_utils.db_config.sqlite_path
        or db_utils.db_config.db_type
    )
    payload = {
        "schema": schema_name,
        "dumped_at": datetime.utcnow().isoformat() + "Z",
        "tables": table_ddl_map,
        "views": view_ddl_map,
        "relationships": relationships,
    }

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

    return len(table_ddl_map), len(view_ddl_map)


if __name__ == "__main__":
    print("Extracting schema...")
    table_count, view_count = train_vanna_on_schema()
    print(f"Trained Vanna on {table_count} table(s) and {view_count} view(s).")

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
