import argparse
import sys

import pandas as pd
from rich.console import Console

from src.config.vanna_config import get_vanna, vanna_lock
from src.semantic.store import load_semantic_store


def _normalize_sql(sql: str) -> str:
    return " ".join(sql.strip().split()).lower()


def _truncate(value: str, limit: int = 40) -> str:
    text = (value or "").strip().replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit].rstrip() + "..."


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    try:
        normalized = df.copy()
    except Exception:
        normalized = pd.DataFrame(df)
    normalized = normalized.fillna("")
    try:
        normalized.columns = [str(c) for c in normalized.columns]
    except Exception:
        pass
    return normalized


def _compare_results(expected: pd.DataFrame, actual: pd.DataFrame) -> tuple[bool, str]:
    expected = _normalize_df(expected)
    actual = _normalize_df(actual)

    if expected.shape != actual.shape:
        return False, "shape"

    if list(expected.columns) != list(actual.columns):
        return False, "columns"

    if expected.equals(actual):
        return True, "exact"

    try:
        sort_cols = list(expected.columns)
        expected_sorted = expected.sort_values(sort_cols).reset_index(drop=True)
        actual_sorted = actual.sort_values(sort_cols).reset_index(drop=True)
        if expected_sorted.equals(actual_sorted):
            return True, "row-order"
    except Exception:
        pass

    return False, "values"


def main() -> int:
    console = Console()
    parser = argparse.ArgumentParser(
        description="Verify semantic_store examples via LLM and SQL execution."
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit examples")
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        help="Specific example index (1-based). Can be used multiple times.",
    )
    args = parser.parse_args()

    store = load_semantic_store()
    examples = store.get("examples", [])
    if not examples:
        print("No examples found in semantic_store.json")
        return 1

    indices = None
    if args.index:
        indices = sorted(set(i for i in args.index if i > 0))

    selected = []
    for idx, ex in enumerate(examples, 1):
        if indices and idx not in indices:
            continue
        selected.append((idx, ex))
        if args.limit and len(selected) >= args.limit:
            break

    if not selected:
        print("No examples selected.")
        return 1

    vn = get_vanna()
    vn.reset_llm_metrics()
    if vn.config is not None:
        vn.config["verbose"] = False

    total = 0
    sql_match = 0
    result_match = 0
    both_match = 0
    sql_mismatch_indices = []
    result_mismatch_indices = []

    for idx, ex in selected:
        question = str(ex.get("question", "")).strip()
        example_sql = str(ex.get("sql", "")).strip()
        if not question or not example_sql:
            continue
        total += 1

        with vanna_lock():
            generated_sql = vn.generate_sql(question)

        sql_same = _normalize_sql(example_sql) == _normalize_sql(generated_sql)
        console.print(f"#{idx}: Question: {_truncate(question)}")
        console.print(f"         SQL: {_truncate(example_sql)}")
        console.print(f"     Gen-SQL: {_truncate(generated_sql)}")
        if sql_same:
            sql_match += 1
            result_match += 1
            both_match += 1
            console.print("     Compare: [green]SQL matches[/green]")
            console.print("      Result: [green]Skipped (SQL match)[/green]")
            continue
        sql_mismatch_indices.append(idx)
        console.print("     Compare: [yellow]SQL mismatch[/yellow]")

        try:
            expected_df = vn.run_sql(example_sql)
        except Exception as exc:
            result_mismatch_indices.append(idx)
            console.print(f"      Result: [yellow]Example SQL error: {exc}[/yellow]")
            continue

        try:
            actual_df = vn.run_sql(generated_sql)
        except Exception as exc:
            result_mismatch_indices.append(idx)
            console.print(f"      Result: [yellow]Generated SQL error: {exc}[/yellow]")
            continue

        results_same, reason = _compare_results(expected_df, actual_df)
        expected_rows = len(expected_df)
        actual_rows = len(actual_df)

        if results_same:
            result_match += 1
            console.print(
                f"      Result: [green]{expected_rows} rows returned, result matches[/green]"
            )
        else:
            result_mismatch_indices.append(idx)
            console.print(
                f"      Result: [yellow]{expected_rows} vs {actual_rows} rows, result mismatch[/yellow]"
            )

    print()
    console.print(f"Example checked: {total}")
    console.print(f"    SQL matched: {sql_match}")
    console.print(
        f"   SQL mismatch: {', '.join(f'#{i}' for i in sql_mismatch_indices) or 'None'}"
    )
    console.print(f" Result matched: {result_match}")
    console.print(
        f"Result mismatch: {', '.join(f'#{i}' for i in result_mismatch_indices) or 'None'}"
    )

    return 0 if total > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
