# AskDB

Ask questions in natural language and query a MySQL database with Vanna AI.

## Quick Start

### Bootstrap (First Time Setup)

1. Create `.env` from `.env.example` and fill in credentials.
2. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
3. (Optional) Install CLI locally:
   ```bash
   pip install -e .
   ```
4. Dump full DDL (ignores allowlist):
   ```bash
   python -m src.cli.main dump-ddl --force
   ```
5. Initialize semantic store from DDL dump:
   ```bash
   # Option A: Include all tables
   python -m src.cli.main init-semantic-store
   
   # Option B: Specify allowlist
   python -m src.cli.main init-semantic-store --allowlist table1 --allowlist table2
   ```
6. Edit semantic store (describe tables/fields in detail and add plenty of examples):
   - Directly edit `src/semantic/semantic_store.json`, or
   - Use the TUI:
     ```bash
     python -m src.cli.main semantic-tui
     ```
7. Train:
   ```bash
   python -m src.cli.main train
   ```
8. Ask (conversational by default):
   ```bash
   # Interactive mode (no question argument - enter a loop)
   python -m src.cli.main ask
   # Type questions interactively, type 'exit', 'quit', or 'q' to leave
   
   # Non-interactive mode (single question)
   python -m src.cli.main ask "查询2025年全产品线的销售达成总金额"
   # Follow-up questions automatically use previous context
   python -m src.cli.main ask "只看前10名"
   # Clear context to start fresh
   python -m src.cli.main ask "新问题" --clear-context
   ```

### Maintenance (Schema Updates)

When database schema changes:

1. Re-dump DDL (respects current allowlist):
   ```bash
   python -m src.cli.main dump-ddl
   ```
2. Manually merge new tables/columns into `semantic_store.json` or re-initialize:
   ```bash
   python -m src.cli.main init-semantic-store --allowlist table1 --allowlist table2
   ```
3. Reset training and re-train:
   ```bash
   python -m src.cli.main reset-training --yes
   python -m src.cli.main train
   ```

If installed, you can use:
```bash
# Interactive mode
askdb ask

# Non-interactive mode
askdb ask "查询2025年全产品线的销售达成总金额"
```

## Commands

- `ask [QUESTION]` — generate SQL, validate, execute, and display results (conversational: uses previous turn context automatically)
  - If QUESTION is omitted, enters **interactive loop mode** where you can ask multiple questions
  - Type `exit`, `quit`, or `q` to leave interactive mode
  - If QUESTION is provided, runs once and exits (non-interactive mode)
  - `--clear-context` — start fresh without previous conversation context
  - `--insights/--no-insights` — generate short LLM insights after results (default: on)
  - `--max-retries` — retry SQL generation on errors (default: 1)
  - `--verbose` — show LLM prompt/response details
- `train` — train on schema + semantic store + examples
- `dump-ddl` — dump raw DDL into `src/semantic/ddl_dump.json` (respects allowlist unless --force)
- `init-semantic-store` — scaffold semantic_store.json from ddl_dump.json (optionally with --allowlist)
- `semantic-tui` — edit semantic store (tables/columns/notes/examples)
- `schema` — list tables (respects allowlist)
- `reset-training` — delete local ChromaDB vectors
- `test-connection` — verify DB connection

## Semantic Store

All semantic data lives in:
`src/semantic/semantic_store.json`

This includes:
- `tables` and `columns` (descriptions, exclusions)
- `notes`
- `examples` (question → SQL)
- `allowlist` (tables to include in training and SQL generation)

Excluded columns are skipped when building semantic docs to reduce noise.

### Allowlist Behavior

The `allowlist` in `semantic_store.json` controls which tables are:
- Included in training (`train` command)
- Validated in generated SQL (`ask` command)
- Dumped by default (`dump-ddl` without --force)

Use `dump-ddl --force` to dump all tables regardless of allowlist (useful for bootstrap).
Use `init-semantic-store` to scaffold the semantic store from a full DDL dump.

## Conversational Behavior

The `ask` command automatically maintains conversation context between turns:
- Maintains a sliding window of the last 5 turns (questions, SQL, result columns)
- Each new question receives full context from all previous turns in the window
- Follow-up questions like "只看前10名" or "按日期排序" work naturally
- Context is stored in `chromadb_data/session_state.json` (git-ignored)
- Use `--clear-context` flag to start a fresh conversation
- Backward compatible: automatically converts old single-turn format to new multi-turn format

### Interactive Mode

Run `askdb ask` without a question to enter interactive loop mode:
```bash
askdb ask
# Interactive mode: Type your questions (exit/quit/q to leave)
❯ 查询2025年销售总额
# ... results displayed ...
❯ 只看前10名
# ... refined results ...
❯ 按金额降序排列
# ... further refinement ...
❯ exit
```

### Non-Interactive Mode

Provide a question argument for single-shot execution:
```bash
# First question
askdb ask "查询2025年销售总额"
# Follow-up (uses previous SQL and columns)
askdb ask "只看前10名"
# Another refinement
askdb ask "按金额降序排列"
# Start fresh
askdb ask "新问题" --clear-context
```

## LLM Providers

Supported via `.env`:

- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_MODEL`
- **OpenRouter**: `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`, `OPENROUTER_BASE_URL`

Model selection priority:
1. `OPENROUTER_MODEL` (when OpenRouter is configured)
2. `OPENAI_MODEL`
3. `VANNA_MODEL`

## Database Types

AskDB supports MySQL, PostgreSQL, and SQLite via `DB_TYPE`.

Examples:

```env
DB_TYPE=mysql
MYSQL_HOST=...
MYSQL_PORT=3306
MYSQL_USER=...
MYSQL_PASSWORD=...
MYSQL_DATABASE=...
```

```env
DB_TYPE=postgres
POSTGRES_HOST=...
POSTGRES_PORT=5432
POSTGRES_USER=...
POSTGRES_PASSWORD=...
POSTGRES_DATABASE=...
```

```env
DB_TYPE=sqlite
SQLITE_PATH=/absolute/path/to/database.sqlite
```

## Output Formatting

Currency columns are formatted in CLI output with thousands separators and no decimals.
Heuristics include column names like: `amount`, `sales`, `revenue`, `price`, `total`, `tax`, `金额`, `销售额`, `含税`, `税额`.

When results have exactly two columns (one category and one numeric), AskDB also prints a simple horizontal bar chart below the table.
Insights are on by default; use `--no-insights` to disable them.

## Notes

- Use `reset-training --yes` when you want to fully re-index semantic data.
