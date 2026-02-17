"""AskDB CLI - Natural language to SQL query tool."""

import re
import shlex
import shutil
import threading
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table
from rich.cells import cell_len
import pandas as pd

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="askdb")
def cli() -> None:
    """AskDB - Ask your database questions in natural language."""


@cli.command("test-connection")
def test_connection() -> None:
    """Test the configured database connection."""
    from src.config.database import get_db_utils

    db = get_db_utils()
    if db.test_connection():
        console.print("[green]Database connection successful![/green]")
    else:
        console.print("[red]Database connection failed.[/red]", style="bold")
        raise SystemExit(1)


@cli.command()
@click.option(
    "--full",
    is_flag=True,
    help="Force full retrain (clears chromadb_data and retrains all).",
)
def train(full: bool) -> None:
    """Train Vanna on ddl_dump.json and semantic metadata (respects allowlist)."""
    from src.training.schema_extractor import (
        train_vanna_incremental,
        reset_chromadb_data,
        train_vanna_on_schema,
        train_vanna_on_relationships,
        train_vanna_on_semantic_schema,
        train_vanna_on_semantic_notes,
        train_vanna_on_semantic_examples,
    )
    from src.config.vanna_config import (
        get_table_allowlist,
        get_view_allowlist,
        vanna_lock,
    )

    table_allowlist = get_table_allowlist()
    view_allowlist = get_view_allowlist()
    if table_allowlist is None and view_allowlist is None:
        console.print("[dim]No allowlist set — training on all tables and views.[/dim]")
    else:
        table_label = "<all>" if table_allowlist is None else ", ".join(table_allowlist)
        view_label = "<all>" if view_allowlist is None else ", ".join(view_allowlist)
        if table_label == "":
            table_label = "<none>"
        if view_label == "":
            view_label = "<none>"
        console.print(
            f"[dim]Allowlist tables: {table_label}[/dim]\n"
            f"[dim]Allowlist views: {view_label}[/dim]"
        )

    if full:
        with vanna_lock():
            with console.status("Clearing local training data..."):
                reset_chromadb_data()
            with console.status("Extracting and training schema..."):
                table_count, view_count = train_vanna_on_schema()
            console.print(
                f"[green]Trained on {table_count} table(s) and {view_count} view(s).[/green]"
            )

            with console.status("Training on foreign key relationships..."):
                fk_count = train_vanna_on_relationships()
            console.print(f"[green]Trained on {fk_count} relationship(s).[/green]")

            with console.status("Training on semantic schema documentation..."):
                doc_count = train_vanna_on_semantic_schema()
            console.print(
                f"[green]Trained on {doc_count} semantic table/view metadata entries.[/green]"
            )

            with console.status("Training on semantic notes..."):
                note_count = train_vanna_on_semantic_notes()
            console.print(f"[green]Trained on {note_count} semantic note(s).[/green]")

            with console.status("Training on semantic examples..."):
                ex_count = train_vanna_on_semantic_examples()
            console.print(f"[green]Trained on {ex_count} semantic example(s).[/green]")
        return

    with console.status("Training on changes..."):
        result = train_vanna_incremental(sync=True)

    mode = result.get("mode")
    if mode == "full":
        console.print(
            "[yellow]Detected removals or allowlist changes. Full retrain performed.[/yellow]"
        )
        console.print(
            f"[green]Trained on {result.get('schema_tables', 0)} table(s) and "
            f"{result.get('schema_views', 0)} view(s).[/green]"
        )
        console.print(
            f"[green]Trained on {result.get('relationships', 0)} relationship(s).[/green]"
        )
        console.print(
            "[green]Trained on "
            f"{result.get('semantic_docs', 0)} semantic table/view metadata entries.[/green]"
        )
        console.print(
            f"[green]Trained on {result.get('notes', 0)} semantic note(s).[/green]"
        )
        console.print(
            f"[green]Trained on {result.get('examples', 0)} semantic example(s).[/green]"
        )
    else:
        console.print(
            f"[green]Trained on {result.get('semantic_docs', 0)} semantic doc(s).[/green]"
        )
        console.print(
            f"[green]Trained on {result.get('notes', 0)} semantic note(s).[/green]"
        )
        console.print(
            f"[green]Trained on {result.get('examples', 0)} semantic example(s).[/green]"
        )


@cli.command("reset-training")
@click.option(
    "--yes",
    is_flag=True,
    help="Confirm deletion of local vector store data.",
)
def reset_training(yes: bool) -> None:
    """Delete local ChromaDB training data (chromadb_data/)."""
    data_dir = Path(__file__).resolve().parents[2] / "chromadb_data"
    if not data_dir.exists():
        console.print("[yellow]No chromadb_data directory found.[/yellow]")
        return

    if not yes:
        console.print(
            "[red]This will delete chromadb_data/. Re-run with --yes to confirm.[/red]"
        )
        raise SystemExit(1)

    shutil.rmtree(data_dir)
    console.print("[green]Deleted chromadb_data/.[/green]")


@cli.command("dump-ddl")
@click.option(
    "--all",
    "all_",
    is_flag=True,
    help="Dump all tables and views, ignoring allowlist in semantic_store.json.",
)
def dump_ddl(all_: bool) -> None:
    """Dump DB DDL to src/semantic/ddl_dump.json (respects allowlist unless --all)."""
    from src.training.schema_extractor import dump_ddl_to_file

    ddl_path = Path(__file__).resolve().parents[1] / "semantic" / "ddl_dump.json"
    table_count, view_count = dump_ddl_to_file(str(ddl_path), force=all_)
    console.print(
        f"[green]Dumped DDL for {table_count} table(s) and {view_count} view(s).[/green]"
    )


@cli.command("init-semantic-store")
@click.option(
    "--allowlist",
    multiple=True,
    help="Table names to include in allowlist (can be specified multiple times).",
)
def init_semantic_store(allowlist: tuple[str, ...]) -> None:
    """Initialize semantic_store.json from ddl_dump.json.

    Creates a scaffold semantic store with table/column structure from DDL dump.
    Optionally specify --allowlist to filter tables (can be used multiple times).
    If no allowlist is provided, all tables from ddl_dump.json are included.
    """
    from src.training.schema_extractor import init_semantic_store_from_ddl

    ddl_path = Path(__file__).resolve().parents[1] / "semantic" / "ddl_dump.json"
    store_path = (
        Path(__file__).resolve().parents[1] / "semantic" / "semantic_store.json"
    )

    if not ddl_path.exists():
        console.print("[red]ddl_dump.json not found. Run 'dump-ddl --all' first.[/red]")
        raise SystemExit(1)

    if store_path.exists():
        console.print("[yellow]semantic_store.json already exists.[/yellow]")
        if not click.confirm("Overwrite?"):
            console.print("[dim]Aborted.[/dim]")
            raise SystemExit(0)

    allowlist_set = set(allowlist) if allowlist else None
    count = init_semantic_store_from_ddl(str(ddl_path), str(store_path), allowlist_set)
    console.print(
        f"[green]Initialized semantic_store.json with {count} table(s).[/green]"
    )


@cli.command("semantic-tui")
def semantic_tui() -> None:
    """Launch semantic editor TUI."""
    from src.cli.semantic_tui import run_tui

    run_tui()


@cli.command()
@click.argument("question", required=False)
@click.option(
    "--max-retries",
    default=1,
    type=int,
    show_default=True,
    help="How many times to retry SQL on error.",
)
@click.option(
    "--verbose",
    is_flag=True,
    help="Show prompt/response details and LLM call diagnostics.",
)
@click.option(
    "--clear-context",
    is_flag=True,
    help="Clear previous conversation context before asking.",
)
@click.option(
    "--insights/--no-insights",
    default=True,
    help="Generate short LLM insights after results.",
)
@click.option(
    "--show-sql/--hide-sql",
    default=True,
    help="Show generated SQL in output.",
)
def ask(
    question: str | None,
    max_retries: int,
    verbose: bool,
    clear_context: bool,
    insights: bool,
    show_sql: bool,
) -> None:
    """Ask a natural language question about your database.

    If QUESTION is provided, runs once and exits (non-interactive mode).
    If QUESTION is omitted, enters interactive loop mode where you can ask multiple questions.

    Interactive mode:
    - Type your questions and press Enter
    - Type 'exit', 'quit', or 'q' to leave the loop
    - Context is maintained across questions automatically
    - Use --clear-context flag to start with fresh context

    Each ask automatically uses context from the previous turn (question, SQL, result columns)
    to enable conversational refinement.
    """
    from src.cli.session_state import clear_session_state

    if clear_context:
        clear_session_state()

    # Interactive mode: loop until exit
    if question is None:
        console.print(
            "[bold]Interactive mode:[/bold] Type your questions (exit/quit/q to leave)\n"
        )
        last_success: dict[str, str] | None = None
        pending_examples: list[dict[str, str]] = []
        while True:
            try:
                user_input = input("❯ ").strip()
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Exiting...[/dim]")
                break

            if not user_input:
                continue

            if user_input.startswith("/save"):
                handled = _handle_save_command(
                    user_input, last_success, pending_examples
                )
                if handled:
                    console.print()
                    continue

            if user_input.startswith("/ok") or user_input.startswith("/reject"):
                handled = _handle_turn_status_command(user_input)
                if handled:
                    console.print()
                    continue

            # Check for exit keywords
            if user_input.lower() in {"exit", "quit", "q"}:
                console.print("[dim]Exiting...[/dim]")
                break

            # Process the question
            result = _process_question(
                user_input, max_retries, verbose, insights, show_sql
            )
            if result:
                last_success = result
            console.print()  # Blank line between turns
        if pending_examples:
            from src.training.schema_extractor import train_vanna_incremental
            from src.config.vanna_config import vanna_lock

            with vanna_lock():
                result = train_vanna_incremental(sync=False)
            console.print(
                f"[dim]Trained on {result.get('examples', 0)} new example(s).[/dim]"
            )
    else:
        # Non-interactive mode: single question
        _process_question(question, max_retries, verbose, insights, show_sql)


def _process_question(
    question: str, max_retries: int, verbose: bool, insights: bool, show_sql: bool
) -> dict[str, str] | None:
    """Process a single question (used by both interactive and non-interactive modes)."""
    from src.config.vanna_config import get_vanna, vanna_lock
    from src.cli.session_state import (
        load_session_state,
        save_session_state,
    )

    vn = get_vanna()
    vn.reset_llm_metrics()
    if vn.config is not None:
        vn.config["verbose"] = verbose

    # Load conversation history (sliding window of max 5 turns)
    turns = load_session_state()
    enhanced_question = question
    if turns:
        # Build context from all available turns
        context_lines = ["Conversation history:"]
        for i, turn in enumerate(turns, 1):
            q = turn.get("question", "")
            sql = turn.get("sql", "")
            cols = turn.get("columns", [])
            status = turn.get("status", "ok")
            context_lines.append(f"Turn {i}:")
            context_lines.append(f"  Question: {q}")
            if status == "ok":
                context_lines.append(f"  SQL: {sql}")
                context_lines.append(f"  Columns: {', '.join(cols)}")

        refinement_prompt = (
            "\n".join(context_lines)
            + f"\n\nNew question (may be a refinement or follow-up): {question}"
        )
        enhanced_question = refinement_prompt

        if verbose:
            console.print(
                f"[dim]Using conversation history ({len(turns)} turn(s)) for context.[/dim]"
            )

    with vanna_lock():
        with console.status("Generating SQL..."):
            sql = vn.generate_sql(enhanced_question)

        sql, df, error = _execute_with_retry(
            vn, enhanced_question, sql, max_retries=max_retries
        )

    if show_sql:
        console.print("\n[bold]Generated SQL:[/bold]")
        console.print(sql)

    if error is not None:
        console.print("\n[red]SQL execution failed.[/red]")
        console.print(str(error))
        # In interactive mode, don't exit on error - just continue
        return None

    _render_llm_metrics(vn, verbose=verbose)
    if verbose:
        _render_llm_debug(vn)

    _render_dataframe(df, question=question)
    if insights and df is not None and len(df) >= 4:
        _render_insights(vn, df, question, sql)

    # Save current turn context for next ask
    columns = list(df.columns) if df is not None and hasattr(df, "columns") else []
    save_session_state(question, sql, columns)
    return {"question": question, "sql": sql}


def _handle_save_command(
    command: str,
    last_success: dict[str, str] | None,
    pending_examples: list[dict[str, str]],
) -> bool:
    if not command.startswith("/save"):
        return False

    if (
        not last_success
        or not last_success.get("question")
        or not last_success.get("sql")
    ):
        console.print("[yellow]No successful query to save yet.[/yellow]")
        return True

    try:
        args = shlex.split(command)
    except ValueError:
        args = command.split()

    notes_text = ""
    if "--notes" in args:
        idx = args.index("--notes")
        if idx + 1 < len(args):
            notes_text = args[idx + 1].strip()
        else:
            notes_text = input("Notes (optional): ").strip()

    from src.semantic.store import load_semantic_store, save_semantic_store

    store = load_semantic_store()
    examples = store.get("examples", [])
    question = last_success["question"].strip()
    sql = last_success["sql"].strip()

    for entry in examples:
        if entry.get("question") == question and entry.get("sql") == sql:
            console.print("[yellow]Example already exists.[/yellow]")
            return True

    examples.append({"question": question, "sql": sql, "notes": notes_text})
    store["examples"] = examples
    save_semantic_store(store)
    console.print("[green]Saved example to semantic_store.json.[/green]")
    pending_examples.append({"question": question, "sql": sql})
    console.print("[dim]Queued example for training at exit.[/dim]")
    return True


def _handle_turn_status_command(command: str) -> bool:
    if command not in {"/ok", "/reject"}:
        return False

    from src.cli.session_state import update_last_turn_status

    status = "ok" if command == "/ok" else "rejected"
    updated = update_last_turn_status(status)
    if not updated:
        console.print("[yellow]No previous turn to update.[/yellow]")
        return True

    if status == "ok":
        console.print("[green]Marked last turn as ok.[/green]")
    else:
        console.print("[yellow]Marked last turn as rejected.[/yellow]")
    return True


@cli.command()
def schema() -> None:
    """Show database tables (respects allowlist in semantic_store)."""
    from src.config.database import get_db_utils
    from src.config.vanna_config import get_table_allowlist

    db_utils = get_db_utils()
    tables = db_utils.get_table_list()
    allowlist = get_table_allowlist()

    if allowlist:
        tables = [t for t in tables if t["TABLE_NAME"] in allowlist]

    console.print(f"\n[bold]Tables ({len(tables)}):[/bold]")
    for table in tables:
        name = table["TABLE_NAME"]
        comment = table.get("TABLE_COMMENT", "")
        line = f"  - {name}"
        if comment:
            line += f"  [dim]({comment})[/dim]"
        console.print(line)


@cli.command("add-note")
@click.option("--title", "title", required=True, help="Short note title.")
@click.option("--text", "text", required=True, help="Note content.")
def add_note(title: str, text: str) -> None:
    """Append a semantic note to the notes store."""
    from src.training.schema_extractor import append_semantic_note

    append_semantic_note(title=title, text=text)
    console.print("[green]Added semantic note.[/green]")


def _execute_with_retry(vn, question: str, sql: str, max_retries: int):
    attempts = 0
    last_error = None
    while True:
        try:
            _validate_allowlist(sql)
            _preflight_explain(vn, sql)
            df = vn.run_sql(sql)
            return sql, df, None
        except Exception as exc:
            last_error = exc
            if attempts >= max_retries:
                return sql, None, last_error
            attempts += 1
            if vn.config and vn.config.get("verbose"):
                console.print("\n[bold]EXPLAIN error:[/bold]")
                console.print(str(exc))
            repair_prompt = (
                f"{question}\n\n"
                f"The previous SQL was:\n{sql}\n\n"
                f"It failed with error:\n{exc}\n\n"
                "Please provide a corrected SQL for MySQL."
            )
            with console.status("Repairing SQL..."):
                sql = vn.generate_sql(repair_prompt)


def _render_dataframe(df, question: str | None = None) -> None:
    if df is None:
        console.print("\n[dim]No results returned.[/dim]")
        return

    if hasattr(df, "empty") and df.empty:
        console.print("\n[dim]Query returned 0 rows.[/dim]")
        return

    display_df = _format_currency_columns(df)

    if len(display_df) == 1:
        row = display_df.iloc[0].tolist()
        values = ["" if v is None else str(v) for v in row]
        answer = None
        if question:
            for value in values:
                if value and value not in question:
                    answer = value
                    break
        if answer is None and values:
            answer = values[0]
        if answer:
            console.print(f"\n{answer}")
            return

    table = Table(show_header=True, header_style="bold")
    columns = list(display_df.columns)
    for col in columns:
        table.add_column(str(col))

    max_rows = 200
    rows = display_df.itertuples(index=False)
    for i, row in enumerate(rows):
        if i >= max_rows:
            break
        table.add_row(*["" if v is None else str(v) for v in row])

    console.print("\n[bold]Results:[/bold]")
    console.print(table)
    if len(display_df) > max_rows:
        console.print(f"\n[dim]Showing first {max_rows} rows.[/dim]")

    # Auto-render bar chart for 2-column results
    _render_bar_chart(df)


def _render_bar_chart(df) -> None:
    """Render horizontal bar chart for 2-column results (category + numeric)."""
    if df is None or len(df.columns) != 2:
        return

    col1, col2 = df.columns[0], df.columns[1]

    # Try to identify which column is numeric and which is categorical
    numeric_col = None
    category_col = None

    # Check if col2 is numeric
    try:
        numeric_vals = pd.to_numeric(df[col2], errors="coerce")
        if isinstance(numeric_vals, pd.Series) and numeric_vals.notna().sum() > 0:
            numeric_col = col2
            category_col = col1
    except Exception:
        pass

    # If col2 is not numeric, try col1
    if numeric_col is None:
        try:
            numeric_vals = pd.to_numeric(df[col1], errors="coerce")
            if isinstance(numeric_vals, pd.Series) and numeric_vals.notna().sum() > 0:
                numeric_col = col1
                category_col = col2
        except Exception:
            pass

    # If we couldn't identify numeric column, skip
    if numeric_col is None or category_col is None:
        return

    # Filter out rows with None/NaN values
    try:
        chart_df = df[[category_col, numeric_col]].copy()
        chart_df[numeric_col] = pd.to_numeric(chart_df[numeric_col], errors="coerce")
        chart_df = chart_df.dropna()

        if len(chart_df) == 0:
            return

        # Limit to 20 rows for readability
        chart_df = chart_df.head(20)

        # Find max value for scaling
        max_val = chart_df[numeric_col].max()
        if max_val <= 0:
            return

        console.print("[bold]Chart:[/bold]")

        # Render horizontal bar chart (ASCII) with aligned columns
        bar_width = 40
        labels = [str(row[category_col]) for _, row in chart_df.iterrows()]
        label_width = max((cell_len(label) for label in labels), default=0)

        rows: list[tuple[str, str, int]] = []
        for _, row in chart_df.iterrows():
            label = str(row[category_col])
            value = row[numeric_col]

            # Skip None/NaN values
            if pd.isna(value):
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                continue

            # Calculate bar length
            bar_len = int((value / max_val) * bar_width) if max_val > 0 else 0
            bar_len = max(1, bar_len) if value > 0 else 0

            # Format value for display
            try:
                if isinstance(value, float) and value == int(value):
                    value_str = f"{int(value):,}"
                else:
                    value_str = f"{value:,.2f}".rstrip("0").rstrip(".")
            except (ValueError, TypeError):
                value_str = str(value)

            rows.append((label, value_str, bar_len))

        value_width = max((cell_len(value_str) for _, value_str, _ in rows), default=0)

        for label, value_str, bar_len in rows:
            # Build bar (ASCII)
            bar = "#" * bar_len
            pad = label_width - cell_len(label) + 2
            label_padded = f"{label}{' ' * max(pad, 2)}"
            value_pad = " " * max(value_width - cell_len(value_str), 0)
            console.print(f"  {label_padded}{bar:<{bar_width}} {value_pad}{value_str}")

    except Exception:
        # Silently fail if chart rendering has issues
        pass


def _render_insights(vn, df, question: str, sql: str) -> None:
    if df is None or (hasattr(df, "empty") and df.empty):
        return

    try:
        display_df = _format_currency_columns(df)
        sample_df = display_df.head(20)
        try:
            table_text = sample_df.to_markdown(index=False)
        except Exception:
            table_text = sample_df.to_string(index=False)

        system_prompt = (
            "You are a data analyst. Provide 3-5 concise insights in Chinese. "
            "Base only on the data shown. Avoid speculation and do not invent facts. "
            "If the data is insufficient for insights, say so briefly."
        )
        user_prompt = (
            f"Question: {question}\nSQL:\n{sql}\n\nData (top rows):\n{table_text}\n"
        )

        messages = [vn.system_message(system_prompt), vn.user_message(user_prompt)]
        with console.status("Generating insights..."):
            response = vn.submit_prompt(messages)
        if response:
            console.print("\n[bold]Insights:[/bold]")
            cleaned = "\n".join(line for line in response.splitlines() if line.strip())
            console.print(cleaned)
    except Exception:
        return


def _format_currency_columns(df):
    keywords = [
        "amount",
        "sales",
        "revenue",
        "price",
        "total",
        "tax",
        "金额",
        "销售额",
        "含税",
        "税额",
    ]

    def should_format(col_name: str) -> bool:
        lower = col_name.lower()
        return any(k in lower for k in keywords if k.isascii()) or any(
            k in col_name for k in keywords if not k.isascii()
        )

    def format_value(value):
        if value is None:
            return ""
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return ""
            stripped = stripped.replace(",", "")
            value = stripped
        try:
            dec = Decimal(str(value))
        except (InvalidOperation, ValueError, TypeError):
            return value
        dec = dec.quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        return f"{int(dec):,}"

    try:
        display_df = df.copy()
    except Exception:
        return df

    for col in display_df.columns:
        if should_format(str(col)):
            try:
                display_df[col] = display_df[col].apply(format_value)
            except Exception:
                continue

    return display_df


def _preflight_explain(vn, sql: str) -> None:
    trimmed = sql.strip()
    if not trimmed:
        raise ValueError("SQL is empty")
    if trimmed.lower().startswith("explain"):
        return
    vn.run_sql(f"EXPLAIN {trimmed}")


def _extract_table_names(sql: str) -> set[str]:
    table_names: set[str] = set()
    pattern = re.compile(r"\b(from|join)\s+([`\w.]+)", re.IGNORECASE)
    for match in pattern.finditer(sql):
        raw = match.group(2).strip("`")
        if raw.startswith("("):
            continue
        if raw.lower() in {"select"}:
            continue
        table = raw.split(".")[-1]
        if table:
            table_names.add(table)
    return table_names


def _validate_allowlist(sql: str) -> None:
    from src.config.vanna_config import get_table_allowlist

    allowlist = get_table_allowlist()
    if not allowlist:
        return
    tables = _extract_table_names(sql)
    if not tables:
        return
    disallowed = sorted(t for t in tables if t not in allowlist)
    if disallowed:
        raise ValueError(
            "SQL references tables outside allowlist: " + ", ".join(disallowed)
        )


def _render_llm_metrics(vn, verbose: bool = False) -> None:
    metrics = vn.get_llm_metrics_summary()
    if not metrics:
        return
    approx = " (approx)" if metrics.get("approx") else ""
    console.print(
        f"\n[dim]LLM calls: {metrics['calls']}, time: {metrics['duration_ms']}ms, "
        f"prompt tokens: {metrics['prompt_tokens']}, completion tokens: {metrics['completion_tokens']}{approx}[/dim]"
    )


def _render_llm_debug(vn) -> None:
    if vn.last_prompt is not None:
        console.print("\n[bold]LLM Prompt:[/bold]")
        console.print(vn.last_prompt)
    if vn.last_response is not None:
        console.print("\n[bold]LLM Response:[/bold]")
        console.print(vn.last_response)


if __name__ == "__main__":
    cli()
