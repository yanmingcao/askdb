"""Terminal UI for semantic store editing."""

from typing import Dict, Any, List

from prompt_toolkit.application import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.output.win32 import NoConsoleScreenBufferError
from prompt_toolkit.shortcuts import (
    input_dialog,
    message_dialog,
    radiolist_dialog,
    yes_no_dialog,
)
from prompt_toolkit.widgets import Button, Dialog, Label, TextArea

from src.config.vanna_config import get_table_allowlist
from src.semantic.store import load_semantic_store, save_semantic_store


def _select_table(
    tables: Dict[str, Any], allowlist: List[str] | None, show_all: bool
) -> str | None:
    items = []
    for full_name, info in tables.items():
        short_name = info.get("name", full_name.split(".")[-1])
        if allowlist and not show_all:
            if short_name not in allowlist and full_name not in allowlist:
                continue
        label = f"{short_name} - {info.get('description', '')}"
        items.append((full_name, label))

    items.sort(key=lambda x: x[0])
    if not items:
        message_dialog(title="No tables", text="No tables available.").run()
        return None
    return _action_dialog(title="Tables", text="Select a table", values=items)


def _multiline_dialog(title: str, text: str, default: str = "") -> str | None:
    kb = KeyBindings()
    textarea = TextArea(
        text=default,
        multiline=True,
        wrap_lines=True,
        scrollbar=True,
        height=Dimension.exact(10),
    )

    def ok_handler() -> None:
        app.exit(result=textarea.text)

    def cancel_handler() -> None:
        app.exit(result=None)

    @kb.add("c-s")
    def _(event) -> None:
        ok_handler()

    dialog = Dialog(
        title=title,
        body=HSplit(
            [
                Label(text=text),
                textarea,
                Label(text="Ctrl+S to save. Enter inserts newline."),
            ]
        ),
        buttons=[
            Button(text="OK", handler=ok_handler),
            Button(text="Cancel", handler=cancel_handler),
        ],
    )

    app = Application(layout=Layout(dialog), key_bindings=kb)
    return app.run()


def _action_dialog(title: str, text: str, values: list[tuple[Any, str]]) -> Any | None:
    return radiolist_dialog(title=title, text=text, values=values).run()


def _edit_table_description(store: Dict[str, Any], show_all: bool) -> None:
    tables = store.get("tables", {})
    allowlist = get_table_allowlist()
    while True:
        selected = _select_table(tables, allowlist, show_all)
        if not selected:
            return
        table = tables[selected]
        current = table.get("description", "")
        new_desc = _multiline_dialog(
            title="Edit Table Description",
            text=f"{table.get('name', selected)} description:",
            default=current,
        )
        if new_desc is None:
            continue
        table["description"] = new_desc
        save_semantic_store(store)
        message_dialog(title="Saved", text="Table description updated.").run()


def _edit_column_description(store: Dict[str, Any], show_all: bool) -> None:
    tables = store.get("tables", {})
    allowlist = get_table_allowlist()
    while True:
        selected = _select_table(tables, allowlist, show_all)
        if not selected:
            return
        table = tables[selected]
        while True:
            columns = table.get("columns", {})
            if not columns:
                message_dialog(title="No columns", text="No columns found.").run()
                break

            items = [
                (
                    col_name,
                    f"{'[x]' if col_info.get('excluded') is True else '[ ]'} "
                    f"{col_name} - {col_info.get('description', '')}",
                )
                for col_name, col_info in columns.items()
            ]
            items.sort(key=lambda x: x[0])

            column = _action_dialog(
                title="Columns",
                text=f"Select a column in {table.get('name', selected)}",
                values=items,
            )
            if not column:
                break

            col_info = columns[column]
            while True:
                excluded = col_info.get("excluded") is True
                status_label = (
                    "Exclude from NL2SQL" if not excluded else "Include in NL2SQL"
                )
                action = _action_dialog(
                    title="Column Actions",
                    text=f"{column}",
                    values=[
                        ("edit", "Edit description"),
                        ("toggle", status_label),
                        ("back", "Back"),
                    ],
                )
                if action in (None, "back"):
                    break
                if action == "edit":
                    current = col_info.get("description", "")
                    new_desc = _multiline_dialog(
                        title="Edit Column Description",
                        text=f"{column} description:",
                        default=current,
                    )
                    if new_desc is None:
                        continue
                    col_info["description"] = new_desc
                    save_semantic_store(store)
                    message_dialog(
                        title="Saved", text="Column description updated."
                    ).run()
                elif action == "toggle":
                    col_info["excluded"] = not excluded
                    save_semantic_store(store)
                    message_dialog(
                        title="Saved",
                        text="Column exclusion updated.",
                    ).run()


def _manage_notes(store: Dict[str, Any]) -> None:
    notes = store.get("notes", [])
    while True:
        choice = _action_dialog(
            title="Notes",
            text="Select action",
            values=[
                ("add", "Add note"),
                ("edit", "Edit note"),
                ("delete", "Delete note"),
                ("back", "Back"),
            ],
        )
        if choice in (None, "back"):
            return

        if choice == "add":
            title = input_dialog(title="Note Title", text="Title:").run()
            if title is None:
                continue
            text = _multiline_dialog(title="Note Text", text="Text:", default="")
            if text is None:
                continue
            notes.append({"title": title, "text": text})
            store["notes"] = notes
            save_semantic_store(store)
            message_dialog(title="Saved", text="Note added.").run()

        elif choice == "edit":
            if not notes:
                message_dialog(title="No notes", text="No notes to edit.").run()
                continue
            items = [
                (str(idx), f"{note.get('title', '')}: {note.get('text', '')}")
                for idx, note in enumerate(notes)
            ]
            idx_value = _action_dialog(
                title="Edit Note", text="Select note", values=items
            )
            if idx_value is None:
                continue
            note = notes[int(idx_value)]
            title = input_dialog(
                title="Note Title", text="Title:", default=note.get("title", "")
            ).run()
            if title is None:
                continue
            text = _multiline_dialog(
                title="Note Text", text="Text:", default=note.get("text", "")
            )
            if text is None:
                continue
            note["title"] = title
            note["text"] = text
            save_semantic_store(store)
            message_dialog(title="Saved", text="Note updated.").run()

        elif choice == "delete":
            if not notes:
                message_dialog(title="No notes", text="No notes to delete.").run()
                continue
            items = [
                (str(idx), f"{note.get('title', '')}: {note.get('text', '')}")
                for idx, note in enumerate(notes)
            ]
            idx_value = _action_dialog(
                title="Delete Note", text="Select note", values=items
            )
            if idx_value is None:
                continue
            if not yes_no_dialog(title="Confirm", text="Delete selected note?").run():
                continue
            notes.pop(int(idx_value))
            save_semantic_store(store)
            message_dialog(title="Deleted", text="Note deleted.").run()


def _manage_examples(store: Dict[str, Any]) -> None:
    examples = store.get("examples", [])
    while True:
        choice = _action_dialog(
            title="Examples",
            text="Select action",
            values=[
                ("add", "Add example"),
                ("edit", "Edit example"),
                ("delete", "Delete example"),
                ("back", "Back"),
            ],
        )
        if choice in (None, "back"):
            return

        if choice == "add":
            question = input_dialog(title="Question", text="Question:").run()
            if question is None:
                continue
            sql = _multiline_dialog(title="SQL", text="SQL:", default="")
            if sql is None:
                continue
            notes = _multiline_dialog(
                title="Notes", text="Notes (optional):", default=""
            )
            if notes is None:
                notes = ""
            examples.append({"question": question, "sql": sql, "notes": notes})
            store["examples"] = examples
            save_semantic_store(store)
            message_dialog(title="Saved", text="Example added.").run()

        elif choice == "edit":
            if not examples:
                message_dialog(title="No examples", text="No examples to edit.").run()
                continue
            items = [
                (str(idx), f"{ex.get('question', '')}")
                for idx, ex in enumerate(examples)
            ]
            idx_value = _action_dialog(
                title="Edit Example", text="Select example", values=items
            )
            if idx_value is None:
                continue
            ex = examples[int(idx_value)]
            question = input_dialog(
                title="Question", text="Question:", default=ex.get("question", "")
            ).run()
            if question is None:
                continue
            sql = _multiline_dialog(title="SQL", text="SQL:", default=ex.get("sql", ""))
            if sql is None:
                continue
            notes = _multiline_dialog(
                title="Notes", text="Notes (optional):", default=ex.get("notes", "")
            )
            if notes is None:
                notes = ""
            ex["question"] = question
            ex["sql"] = sql
            ex["notes"] = notes
            save_semantic_store(store)
            message_dialog(title="Saved", text="Example updated.").run()

        elif choice == "delete":
            if not examples:
                message_dialog(title="No examples", text="No examples to delete.").run()
                continue
            items = [
                (str(idx), f"{ex.get('question', '')}")
                for idx, ex in enumerate(examples)
            ]
            idx_value = _action_dialog(
                title="Delete Example", text="Select example", values=items
            )
            if idx_value is None:
                continue
            if not yes_no_dialog(
                title="Confirm", text="Delete selected example?"
            ).run():
                continue
            examples.pop(int(idx_value))
            save_semantic_store(store)
            message_dialog(title="Deleted", text="Example deleted.").run()


def run_tui() -> None:
    try:
        store = load_semantic_store()
        show_all_tables = False
        while True:
            allowlist = get_table_allowlist()
            allowlist_label = "ON" if allowlist and not show_all_tables else "OFF"
            choice = _action_dialog(
                title="Semantic Editor",
                text="Select action",
                values=[
                    ("table", "Edit table description"),
                    ("column", "Edit column description"),
                    ("notes", "Manage notes"),
                    ("examples", "Manage examples"),
                    ("allowlist", f"Allowlist filter: {allowlist_label}"),
                    ("exit", "Exit"),
                ],
            )
            if choice in (None, "exit"):
                return
            if choice == "table":
                _edit_table_description(store, show_all_tables)
            elif choice == "column":
                _edit_column_description(store, show_all_tables)
            elif choice == "notes":
                _manage_notes(store)
            elif choice == "examples":
                _manage_examples(store)
            elif choice == "allowlist":
                if allowlist:
                    show_all_tables = not show_all_tables
                else:
                    message_dialog(
                        title="No allowlist",
                        text="Allowlist is not set in semantic_store.json.",
                    ).run()
    except NoConsoleScreenBufferError:
        print(
            "Semantic TUI requires a Windows console. Try running from cmd.exe or use WinPTY:\n"
            "  winpty python -m src.cli.main semantic-tui"
        )
