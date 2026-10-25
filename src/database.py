import json
import logging
import re
import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Callable

import pandas as pd

from config import DATABASE_PATH, QUERY_TIMEOUT_SECONDS
from src.data_loader import ORDERS_COLUMNS, dataset_table_name, generate_sample_data

logger = logging.getLogger(__name__)

ORDERS_SCHEMA = {
    "order_id": "INTEGER",
    "order_date": "TEXT",
    "product": "TEXT",
    "category": "TEXT",
    "region": "TEXT",
    "customer_segment": "TEXT",
    "sales": "REAL",
    "quantity": "INTEGER",
    "discount": "REAL",
    "profit": "REAL",
}
ALLOWED_TABLES = ("orders", "uploaded_orders")
DATASET_REGISTRY = "__bi_datasets"
_DYNAMIC_TABLE = re.compile(r"^dataset_[0-9a-f]{16}$")
SAFE_FUNCTIONS = {
    "abs", "avg", "cast", "coalesce", "count", "date", "datetime", "ifnull",
    "julianday", "length", "lower", "max", "min", "nullif", "round", "strftime",
    "substr", "sum", "total", "trim", "upper",
}


def _quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def _valid_table_name(table_name: str, path: Path = DATABASE_PATH) -> bool:
    if table_name in ALLOWED_TABLES:
        return True
    return bool(_DYNAMIC_TABLE.fullmatch(table_name)) and _registered_table(table_name, path)


def _registered_table(table_name: str, path: Path) -> bool:
    if not path.exists():
        return False
    try:
        with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
            has_registry = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                (DATASET_REGISTRY,),
            ).fetchone()
            if not has_registry:
                return False
            return conn.execute(
                f"SELECT 1 FROM {_quote_identifier(DATASET_REGISTRY)} WHERE table_name=?",
                (table_name,),
            ).fetchone() is not None
    except sqlite3.Error:
        return False


def _create_table(conn: sqlite3.Connection, table_name: str) -> None:
    if table_name != "orders":
        raise ValueError("Unsupported built-in dataset table.")
    definitions = ", ".join(
        f"{_quote_identifier(name)} {kind} NOT NULL" for name, kind in ORDERS_SCHEMA.items()
    )
    conn.execute(f'CREATE TABLE IF NOT EXISTS "orders" ({definitions})')
    actual_columns = {
        row[1]: row[2].upper()
        for row in conn.execute('PRAGMA table_info("orders")').fetchall()
    }
    if actual_columns != ORDERS_SCHEMA:
        raise ValueError(
            "The existing 'orders' table does not match the approved sample schema. "
            "Its data was left unchanged."
        )


def _insert_sample(conn: sqlite3.Connection, frame: pd.DataFrame) -> None:
    column_sql = ", ".join(_quote_identifier(column) for column in ORDERS_COLUMNS)
    placeholders = ", ".join("?" for _ in ORDERS_COLUMNS)
    conn.executemany(
        f'INSERT INTO "orders" ({column_sql}) VALUES ({placeholders})',
        frame.itertuples(index=False, name=None),
    )


def initialize_database(path: Path = DATABASE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        _create_table(conn, "orders")
        if conn.execute('SELECT COUNT(*) FROM "orders"').fetchone()[0] == 0:
            _insert_sample(conn, generate_sample_data())
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {_quote_identifier(DATASET_REGISTRY)} '
            "(table_name TEXT PRIMARY KEY, display_name TEXT NOT NULL)"
        )


def import_dataset(
    frame: pd.DataFrame,
    filename: str,
    file_bytes: bytes,
    path: Path = DATABASE_PATH,
) -> str:
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The uploaded dataset must contain at least one row and one column.")
    if len(frame.columns) != len(set(column.casefold() for column in frame.columns)):
        raise ValueError("Dataset column names must be unique after normalization.")
    table_name = dataset_table_name(file_bytes, filename)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = {column: _sqlite_type(frame[column]) for column in frame.columns}
    definitions = ", ".join(
        f"{_quote_identifier(column)} {kind}" for column, kind in schema.items()
    )
    columns = ", ".join(_quote_identifier(column) for column in frame.columns)
    placeholders = ", ".join("?" for _ in frame.columns)
    with sqlite3.connect(path) as conn:
        conn.execute(
            f'CREATE TABLE IF NOT EXISTS {_quote_identifier(DATASET_REGISTRY)} '
            "(table_name TEXT PRIMARY KEY, display_name TEXT NOT NULL)"
        )
        conn.execute(f"DROP TABLE IF EXISTS {_quote_identifier(table_name)}")
        conn.execute(f"CREATE TABLE {_quote_identifier(table_name)} ({definitions})")
        conn.executemany(
            f"INSERT INTO {_quote_identifier(table_name)} ({columns}) VALUES ({placeholders})",
            (_sqlite_row(row) for row in frame.itertuples(index=False, name=None)),
        )
        conn.execute(
            f"INSERT OR REPLACE INTO {_quote_identifier(DATASET_REGISTRY)} "
            "(table_name, display_name) VALUES (?, ?)",
            (table_name, filename),
        )
    return table_name


def replace_uploaded_data(frame: pd.DataFrame, path: Path = DATABASE_PATH) -> int:
    """Replace the legacy uploaded_orders table for existing callers."""
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The uploaded dataset must contain at least one row and one column.")
    if len(frame.columns) != len(set(column.casefold() for column in frame.columns)):
        raise ValueError("Dataset column names must be unique after normalization.")
    schema = {column: _sqlite_type(frame[column]) for column in frame.columns}
    definitions = ", ".join(
        f"{_quote_identifier(column)} {kind}" for column, kind in schema.items()
    )
    columns = ", ".join(_quote_identifier(column) for column in frame.columns)
    placeholders = ", ".join("?" for _ in frame.columns)
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.execute('DROP TABLE IF EXISTS "uploaded_orders"')
        conn.execute(f'CREATE TABLE "uploaded_orders" ({definitions})')
        conn.executemany(
            f'INSERT INTO "uploaded_orders" ({columns}) VALUES ({placeholders})',
            (_sqlite_row(row) for row in frame.itertuples(index=False, name=None)),
        )
    return len(frame)


def list_datasets(path: Path = DATABASE_PATH) -> list[str]:
    if not path.exists():
        return []
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
        }
        result = ["orders"] if "orders" in tables and _has_rows(conn, "orders") else []
        if "uploaded_orders" in tables and _has_rows(conn, "uploaded_orders"):
            result.append("uploaded_orders")
        if DATASET_REGISTRY in tables:
            rows = conn.execute(
                f"SELECT table_name FROM {_quote_identifier(DATASET_REGISTRY)} ORDER BY display_name"
            ).fetchall()
            result.extend(
                table_name
                for (table_name,) in rows
                if table_name in tables and _has_rows(conn, table_name)
            )
        return result


def dataset_display_name(table_name: str, path: Path = DATABASE_PATH) -> str:
    if table_name == "orders":
        return "Sample orders (synthetic)"
    if table_name == "uploaded_orders":
        return "Imported orders"
    if not _DYNAMIC_TABLE.fullmatch(table_name) or not path.exists():
        return table_name
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        row = conn.execute(
            f"SELECT display_name FROM {_quote_identifier(DATASET_REGISTRY)} WHERE table_name=?",
            (table_name,),
        ).fetchone()
    return row[0] if row else table_name


def _has_rows(conn: sqlite3.Connection, table_name: str) -> bool:
    if table_name not in ALLOWED_TABLES and not _DYNAMIC_TABLE.fullmatch(table_name):
        return False
    return conn.execute(f"SELECT 1 FROM {_quote_identifier(table_name)} LIMIT 1").fetchone() is not None


def load_dataset(table_name: str, path: Path = DATABASE_PATH) -> pd.DataFrame:
    if not _valid_table_name(table_name, path):
        raise ValueError("Unsupported dataset table.")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        return pd.read_sql_query(
            f"SELECT * FROM {_quote_identifier(table_name)}",
            conn,
        )


def connect_readonly(
    path: Path,
    table_name: str,
    deadline_check: Callable[[], int] | None = None,
) -> sqlite3.Connection:
    if not _valid_table_name(table_name, path):
        raise ValueError("Unsupported dataset table.")
    uri = f"{path.resolve().as_uri()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True, timeout=QUERY_TIMEOUT_SECONDS)
    allowed_columns = {name.casefold() for name in describe_schema(table_name, path)}

    def authorize(
        action: int,
        arg1: str | None,
        arg2: str | None,
        database_name: str | None,
        trigger_name: str | None,
    ) -> int:
        _ = (database_name, trigger_name)
        if action == sqlite3.SQLITE_SELECT:
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            if arg1 == table_name and (not arg2 or arg2.casefold() in allowed_columns):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION and (arg2 or arg1 or "").lower() in SAFE_FUNCTIONS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    conn.set_authorizer(authorize)
    if deadline_check is not None:
        conn.set_progress_handler(deadline_check, 1_000)
    return conn


def describe_schema(table_name: str = "orders", path: Path = DATABASE_PATH) -> dict[str, str]:
    if not _valid_table_name(table_name, path):
        raise ValueError("Unsupported dataset table.")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            f"PRAGMA table_info({_quote_identifier(table_name)})"
        ).fetchall()
    if not rows:
        raise ValueError("The selected dataset does not have a readable schema.")
    return {row[1]: row[2].upper() or "TEXT" for row in rows}


def _sqlite_type(series: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(series.dtype) or pd.api.types.is_integer_dtype(series.dtype):
        return "INTEGER"
    if pd.api.types.is_numeric_dtype(series.dtype):
        return "REAL"
    return "TEXT"


def _sqlite_row(row: tuple) -> tuple:
    values = []
    for value in row:
        if value is None or value is pd.NA or value is pd.NaT:
            values.append(None)
        elif isinstance(value, (pd.Timestamp,)):
            values.append(value.isoformat())
        elif isinstance(value, (str, int, float, bytes)):
            values.append(value)
        elif isinstance(value, bool):
            values.append(int(value))
        else:
            try:
                missing = pd.isna(value)
                if isinstance(missing, bool) and missing:
                    values.append(None)
                    continue
            except (TypeError, ValueError):
                pass
            if isinstance(value, (date, datetime)):
                values.append(value.isoformat())
            elif isinstance(value, (list, tuple, dict)):
                values.append(json.dumps(value, ensure_ascii=True, default=str))
            else:
                values.append(str(value))
    return tuple(values)
