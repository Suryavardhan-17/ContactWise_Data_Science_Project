"""Execute checked-in, inspectable SQL queries against SQLite in memory."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from .config import PROJECT_ROOT
from .data import add_split_column


def run_queries(raw: pd.DataFrame, output_dir: Path) -> None:
    tagged = add_split_column(raw)
    with sqlite3.connect(":memory:") as conn:
        tagged.to_sql("campaigns", conn, index=False, if_exists="replace")
        for script in sorted((PROJECT_ROOT / "sql").glob("*.sql")):
            result = pd.read_sql_query(script.read_text(encoding="utf-8"), conn)
            result.to_csv(output_dir / f"{script.stem}.csv", index=False)
            print(f"SQL -> {script.stem}.csv ({len(result)} rows)")
