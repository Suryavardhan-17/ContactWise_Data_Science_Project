import sqlite3
import pandas as pd

from contactwise.data import add_split_column
from contactwise.config import PROJECT_ROOT


def test_checked_in_sql_is_executable(tiny_campaigns):
    with sqlite3.connect(":memory:") as con:
        add_split_column(tiny_campaigns).to_sql("campaigns", con, index=False)
        for query in sorted((PROJECT_ROOT / "sql").glob("*.sql")):
            result = pd.read_sql_query(query.read_text(), con)
            assert len(result.columns) > 0, query.name
