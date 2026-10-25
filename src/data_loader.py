import csv
import hashlib
import io
import re
import zipfile
from datetime import date

import pandas as pd

from config import MAX_CSV_BYTES, MAX_CSV_ROWS
from src.dataset_profiler import infer_dataset_types

try:
    from xlrd.biffh import XLRDError
except ImportError:
    EXCEL_READER_ERRORS: tuple[type[BaseException], ...] = ()
else:
    EXCEL_READER_ERRORS = (XLRDError,)

FILE_READ_ERRORS = (
    ImportError,
    ValueError,
    OSError,
    pd.errors.ParserError,
    zipfile.BadZipFile,
) + EXCEL_READER_ERRORS

ORDERS_COLUMNS = (
    "order_id",
    "order_date",
    "product",
    "category",
    "region",
    "customer_segment",
    "sales",
    "quantity",
    "discount",
    "profit",
)


def generate_sample_data(row_count: int = 600, seed: int = 42) -> pd.DataFrame:
    import random
    from datetime import timedelta

    if row_count < 1:
        raise ValueError("row_count must be greater than zero")

    rng = random.Random(seed)
    products = {
        "Technology": ["Laptop", "Monitor", "Keyboard", "Printer", "Tablet"],
        "Furniture": ["Desk", "Chair", "Bookcase", "Table"],
        "Office Supplies": ["Paper", "Binder", "Pen", "Storage Box"],
    }
    regions = ["East", "West", "Central", "South"]
    segments = ["Consumer", "Corporate", "Small Business"]
    start_date = date(2024, 1, 1)
    rows = []

    for order_id in range(1, row_count + 1):
        category = rng.choice(list(products))
        product = rng.choice(products[category])
        sales = round(rng.uniform(15, 1_500), 2)
        quantity = rng.randint(1, 8)
        discount = rng.choice([0, 0, 0.05, 0.1, 0.15, 0.2, 0.3])
        margin = rng.uniform(-0.25, 0.35) - discount * 0.35
        profit = round(sales * margin, 2)
        rows.append(
            {
                "order_id": order_id,
                "order_date": (start_date + timedelta(days=rng.randrange(730))).isoformat(),
                "product": product,
                "category": category,
                "region": rng.choice(regions),
                "customer_segment": rng.choice(segments),
                "sales": sales,
                "quantity": quantity,
                "discount": discount,
                "profit": profit,
            }
        )

    return pd.DataFrame(rows, columns=ORDERS_COLUMNS)


def excel_sheet_names(file_bytes: bytes) -> list[str]:
    if len(file_bytes) > MAX_CSV_BYTES:
        raise ValueError(f"Files must be {MAX_CSV_BYTES // (1024 * 1024)} MB or smaller.")
    try:
        return pd.ExcelFile(io.BytesIO(file_bytes)).sheet_names
    except FILE_READ_ERRORS as exc:
        raise ValueError("The Excel file could not be read. Check its format and installed Excel support.") from exc


def load_tabular_data(
    file_bytes: bytes,
    filename: str,
    sheet_name: str | int | None = None,
) -> pd.DataFrame:
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")
    if len(file_bytes) > MAX_CSV_BYTES:
        raise ValueError(f"Files must be {MAX_CSV_BYTES // (1024 * 1024)} MB or smaller.")

    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    try:
        if suffix == "csv":
            frame = _read_csv_with_encoding_fallback(file_bytes)
        elif suffix in {"xlsx", "xls"}:
            sheets = excel_sheet_names(file_bytes)
            if not sheets:
                raise ValueError("The Excel file contains no worksheets.")
            selected_sheet = sheet_name if sheet_name is not None else sheets[0]
            if isinstance(selected_sheet, str) and selected_sheet not in sheets:
                raise ValueError("Select a worksheet that exists in the uploaded Excel file.")
            frame = pd.read_excel(io.BytesIO(file_bytes), sheet_name=selected_sheet)
        else:
            raise ValueError("Unsupported file type. Upload a CSV, XLSX, or XLS file.")
    except ImportError as exc:
        raise ValueError("Excel support is unavailable. Install the compatible Excel reader dependency.") from exc
    except (csv.Error,) + FILE_READ_ERRORS as exc:
        if isinstance(exc, ValueError) and str(exc).startswith(("Unsupported file", "Select a worksheet", "The Excel file")):
            raise
        raise ValueError("The file could not be read. Check its format, encoding, and worksheet.") from exc

    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Select one worksheet to import.")
    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f"Files are limited to {MAX_CSV_ROWS:,} rows.")
    if frame.empty or len(frame.columns) == 0:
        raise ValueError("The file must contain at least one header row and one data row.")

    frame = frame.dropna(axis=0, how="all").copy()
    frame.columns = _unique_column_names(frame.columns)
    empty_columns = [column for column in frame if frame[column].isna().all()]
    if empty_columns:
        frame = frame.drop(columns=empty_columns)
    if frame.empty or frame.shape[1] == 0:
        raise ValueError("The file does not contain usable tabular data.")
    if len(frame) > MAX_CSV_ROWS:
        raise ValueError(f"Files are limited to {MAX_CSV_ROWS:,} rows.")

    return infer_dataset_types(frame).reset_index(drop=True)


def _read_csv_with_encoding_fallback(file_bytes: bytes) -> pd.DataFrame:
    decode_errors = []
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                encoding_errors="strict",
                low_memory=False,
            )
        except UnicodeDecodeError as exc:
            decode_errors.append(exc)
    raise ValueError("The CSV text encoding could not be detected or decoded.") from decode_errors[-1]


def validate_csv(file_bytes: bytes) -> pd.DataFrame:
    return load_tabular_data(file_bytes, "upload.csv")


def dataset_table_name(file_bytes: bytes, filename: str) -> str:
    digest = hashlib.sha256(filename.encode("utf-8") + b"\0" + file_bytes).hexdigest()[:16]
    return f"dataset_{digest}"


def _unique_column_names(columns) -> list[str]:
    used: set[str] = set()
    names = []
    for index, value in enumerate(columns, start=1):
        base = str(value).replace("\x00", "_").strip()
        if not base or base.lower().startswith("unnamed:"):
            base = f"column_{index}"
        candidate = base
        suffix = 2
        while candidate.casefold() in used:
            candidate = f"{base}_{suffix}"
            suffix += 1
        used.add(candidate.casefold())
        names.append(candidate)
    return names
