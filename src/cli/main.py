"""AskDB CLI - Natural language to SQL query tool."""

import re
import shutil
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path
import click
from rich.console import Console
from rich.table import Table

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="askdb")
def cli() -> None:
    """AskDB - Ask your database questions in natural language."""


@cli.command("test-connection")
def test_connection() -> None:
    """Test the MySQL database connection."""
    from src.config.database import get_db_config

    db = get_db_config()
    if db.test_connection():
        console.print("[green]Database connection successful![/green]")
    else:
        console.print("[red]Database connection failed.[/red]", style="bold")
        raise SystemExit(1)


@cli.command()
def train() -> None:
    """Train Vanna on the database schema (respects allowlist in semantic_store)."""
    from src.training.schema_extractor import (
        train_vanna_on_schema,
        train_vanna_on_relationships,
        train_vanna_on_semantic_schema,
        train_vanna_on_semantic_notes,
        train_vanna_on_semantic_examples,
    )
    from src.config.vanna_config import get_table_allowlist

    allowlist = get_table_allowlist()
    if allowlist:
        console.print(f"[dim]Allowlist: {', '.join(allowlist)}[/dim]")
    else:
        console.print("[dim]No allowlist set — training on all tables.[/dim]")

    with console.status("Extracting and training schema..."):
        count = train_vanna_on_schema()
    console.print(f"[green]Trained on {count} table(s).[/green]")

    with console.status("Training on foreign key relationships..."):
        fk_count = train_vanna_on_relationships()
    console.print(f"[green]Trained on {fk_count} relationship(s).[/green]")

    with console.status("Training on semantic schema documentation..."):
        doc_count = train_vanna_on_semantic_schema()
    console.print(f"[green]Trained on {doc_count} semantic doc(s).[/green]")

    with console.status("Training on semantic notes..."):
        note_count = train_vanna_on_semantic_notes()
    console.print(f"[green]Trained on {note_count} semantic note(s).[/green]")

    with console.status("Training on semantic examples..."):
        ex_count = train_vanna_on_semantic_examples()
    console.print(f"[green]Trained on {ex_count} semantic example(s).[/green]")


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
def dump_ddl() -> None:
    """Dump DB DDL to src/semantic/ddl_dump.json."""
    from src.training.schema_extractor import dump_ddl_to_file

    ddl_path = Path(__file__).resolve().parents[1] / "semantic" / "ddl_dump.json"
    count = dump_ddl_to_file(str(ddl_path))
    console.print(f"[green]Dumped DDL for {count} table(s).[/green]")


@cli.command("semantic-tui")
def semantic_tui() -> None:
    """Launch semantic editor TUI."""
    from src.cli.semantic_tui import run_tui

    run_tui()


@cli.command()
@click.argument("question")
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
def ask(question: str, max_retries: int, verbose: bool) -> None:
    """Ask a natural language question about your database."""
    from src.config.vanna_config import get_vanna

    vn = get_vanna()
    vn.reset_llm_metrics()
    if vn.config is not None:
        vn.config["verbose"] = verbose

    with console.status("Generating SQL..."):
        sql = vn.generate_sql(question)

    sql, df, error = _execute_with_retry(vn, question, sql, max_retries=max_retries)

    console.print("\n[bold]Generated SQL:[/bold]")
    console.print(sql)

    if error is not None:
        console.print("\n[red]SQL execution failed.[/red]")
        console.print(str(error))
        raise SystemExit(1)

    _render_llm_metrics(vn, verbose=verbose)
    if verbose:
        _render_llm_debug(vn)

    _render_dataframe(df)


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


def _render_dataframe(df) -> None:
    if df is None:
        console.print("\n[dim]No results returned.[/dim]")
        return

    if hasattr(df, "empty") and df.empty:
        console.print("\n[dim]Query returned 0 rows.[/dim]")
        return

    display_df = _format_currency_columns(df)
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
