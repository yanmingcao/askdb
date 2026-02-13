# AskDB

Ask questions in natural language and query a MySQL database with Vanna AI.

## Quick Start

1. Create `.env` from `.env.example` and fill in credentials.
2. Create `src/semantic/semantic_store.json` by copying:
   ```bash
   cp src/semantic/semantic_store.example.json src/semantic/semantic_store.json
   ```
3. Install deps:
   ```bash
   pip install -r requirements.txt
   ```
4. (Optional) Install CLI locally:
   ```bash
   pip install -e .
   ```
5. Dump DDL:
   ```bash
   python -m src.cli.main dump-ddl
   ```
6. Edit semantic store (describe tables/fields in detail and add plenty of examples):
   - Directly edit `src/semantic/semantic_store.json`, or
   - Use the TUI:
     ```bash
     python -m src.cli.main semantic-tui
     ```
7. (Optional) Reset training data:
   ```bash
   python -m src.cli.main reset-training --yes
   ```
8. Train:
   ```bash
   python -m src.cli.main train
   ```
9. Ask:
   ```bash
   python -m src.cli.main ask "查询2025年全产品线的销售达成总金额"
   ```

If installed, you can use:
```bash
askdb ask "查询2025年全产品线的销售达成总金额"
```

## Commands

- `ask` — generate SQL, validate, execute, and display results
- `train` — train on schema + semantic store + examples
- `dump-ddl` — dump raw DDL into `src/semantic/ddl_dump.json`
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

## Notes

- `.env` and `src/semantic/*.json` are ignored by git.
- Use `reset-training --yes` when you want to fully re-index semantic data.
