import logging
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from config import DATABASE_PATH, MAX_RESULT_ROWS, QUERY_TIMEOUT_SECONDS
from src.database import connect_readonly, describe_schema
from src.sql_validator import validate_sql

logger = logging.getLogger(__name__)


@dataclass
class QueryResult:
    sql: str
    dataframe: pd.DataFrame
    row_count: int
    duration_seconds: float
    status: str = "success"
    error: str | None = None


def execute_query(
    sql: str,
    table_name: str,
    path: Path = DATABASE_PATH,
    max_rows: int = MAX_RESULT_ROWS,
) -> QueryResult:
    schema = describe_schema(table_name, path)
    validated = validate_sql(sql, table_name, max_rows=max_rows, schema=schema)
    started = time.monotonic()
    deadline = started + QUERY_TIMEOUT_SECONDS

    def time_budget_exceeded() -> int:
        return int(time.monotonic() > deadline)

    try:
        with connect_readonly(path, table_name, time_budget_exceeded) as conn:
            cursor = conn.execute(validated.sql)
            rows = cursor.fetchmany(max_rows)
            columns = [item[0] for item in cursor.description or []]
            frame = pd.DataFrame.from_records(rows, columns=columns)
        duration = time.monotonic() - started
        return QueryResult(validated.sql, frame, len(frame), duration)
    except sqlite3.OperationalError as exc:
        logger.warning("Read-only query execution failed: %s", exc)
        duration = time.monotonic() - started
        if "interrupted" in str(exc).lower():
            message = "The query exceeded the safe execution time limit."
        else:
            message = "The query could not be executed against the selected dataset."
        return QueryResult(validated.sql, pd.DataFrame(), 0, duration, "error", message)
    except sqlite3.DatabaseError:
        logger.exception("The business database could not execute an approved query.")
        duration = time.monotonic() - started
        return QueryResult(
            validated.sql,
            pd.DataFrame(),
            0,
            duration,
            "error",
            "The database returned an error while running the query.",
        )
    except OSError:
        logger.exception("The business database file was not available.")
        duration = time.monotonic() - started
        return QueryResult(
            validated.sql,
            pd.DataFrame(),
            0,
            duration,
            "error",
            "The selected database is not available.",
        )
