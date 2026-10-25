from __future__ import annotations

import re
from typing import Any

import pandas as pd

MAX_PROFILE_CHARS = 8_000
MAX_COLUMNS = 80
MAX_CATEGORY_COLUMNS = 12
MAX_CATEGORY_VALUES = 5
MAX_NUMERIC_COLUMNS = 20
MAX_SENSITIVE_NAME_TOKENS = {
    "account",
    "address",
    "birth",
    "dob",
    "employee",
    "email",
    "first",
    "id",
    "last",
    "member",
    "name",
    "patient",
    "phone",
    "postal",
    "ssn",
    "student",
    "token",
    "user",
    "uuid",
    "zip",
}


def _normalized_tokens(column: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", column.casefold()))


def is_sensitive_column(column: str) -> bool:
    tokens = _normalized_tokens(column)
    return bool(tokens & MAX_SENSITIVE_NAME_TOKENS)


def infer_dataset_types(frame: pd.DataFrame) -> pd.DataFrame:
    prepared = frame.copy()
    for column in prepared.columns:
        series = prepared[column]
        if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
            continue
        if not (pd.api.types.is_object_dtype(series) or pd.api.types.is_string_dtype(series)):
            continue

        text = series.astype("string").str.strip().replace("", pd.NA)
        prepared[column] = text
        non_missing = text.dropna()
        if non_missing.empty:
            continue

        boolean_values = {"true", "false", "yes", "no", "y", "n"}
        normalized = non_missing.str.casefold()
        if normalized.isin(boolean_values).all():
            bool_map = {
                "true": True,
                "yes": True,
                "y": True,
                "false": False,
                "no": False,
                "n": False,
            }
            prepared[column] = text.str.casefold().map(bool_map).astype("boolean")
            continue

        numeric = pd.to_numeric(non_missing.str.replace(",", "", regex=False), errors="coerce")
        if numeric.notna().sum() >= 2 and numeric.notna().mean() >= 0.5:
            prepared[column] = pd.to_numeric(
                text.str.replace(",", "", regex=False),
                errors="coerce",
            )
            continue

        date_named = bool(re.search(r"date|time|timestamp|created|updated", column, re.I))
        date_like = non_missing.str.match(
            r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}|^\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
            na=False,
        ).mean()
        if date_named or date_like >= 0.8:
            dates = pd.to_datetime(text, errors="coerce", format="mixed", utc=True)
            if dates.notna().sum() / len(non_missing) >= 0.8:
                prepared[column] = dates
    return prepared


def _type_name(series: pd.Series) -> str:
    if pd.api.types.is_datetime64_any_dtype(series):
        return "datetime"
    if pd.api.types.is_bool_dtype(series):
        return "boolean"
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if series.nunique(dropna=True) <= min(50, max(10, len(series) // 5)):
        return "categorical"
    return "text"


def build_dataset_profile(frame: pd.DataFrame) -> dict[str, Any]:
    data = infer_dataset_types(frame)
    columns = []
    numeric_summary: dict[str, dict[str, int | float | None]] = {}
    category_frequencies: dict[str, list[dict[str, object]]] = {}
    date_ranges: dict[str, dict[str, str | None]] = {}

    for column in list(data.columns)[:MAX_COLUMNS]:
        series = data[column]
        missing = int(series.isna().sum())
        info: dict[str, object] = {
            "name": str(column)[:120],
            "type": _type_name(series),
            "missing_count": missing,
            "missing_percent": round(missing / len(data) * 100, 2) if len(data) else 0.0,
            "distinct_count": int(series.nunique(dropna=True)),
        }
        columns.append(info)

        if (
            pd.api.types.is_numeric_dtype(series)
            and not pd.api.types.is_bool_dtype(series)
            and len(numeric_summary) < MAX_NUMERIC_COLUMNS
            and not is_sensitive_column(str(column))
        ):
            values = pd.to_numeric(series, errors="coerce")
            numeric_summary[str(column)] = {
                key: float(value) if pd.notna(value) else None
                for key, value in values.agg(["count", "mean", "median", "min", "max"]).items()
            }
        elif (
            _type_name(series) == "categorical"
            and len(category_frequencies) < MAX_CATEGORY_COLUMNS
            and not is_sensitive_column(str(column))
        ):
            counts = series.dropna().astype(str).value_counts().head(MAX_CATEGORY_VALUES)
            category_frequencies[str(column)] = [
                {"value": value[:80], "count": int(count)}
                for value, count in counts.items()
            ]
        elif pd.api.types.is_datetime64_any_dtype(series):
            valid_dates = series.dropna()
            date_ranges[str(column)] = {
                "minimum": valid_dates.min().isoformat() if not valid_dates.empty else None,
                "maximum": valid_dates.max().isoformat() if not valid_dates.empty else None,
            }

    profile: dict[str, Any] = {
        "dimensions": {"rows": int(len(data)), "columns": int(len(data.columns))},
        "columns": columns,
        "duplicate_rows": int(data.duplicated().sum()) if not data.empty else 0,
        "numeric_summary": numeric_summary,
        "category_frequencies": category_frequencies,
        "date_ranges": date_ranges,
        "sensitive_values_included": False,
    }
    return profile


def compact_profile_json(profile: dict[str, Any], max_chars: int = MAX_PROFILE_CHARS) -> str:
    import json

    serialized = json.dumps(profile, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(serialized) <= max_chars:
        return serialized

    reduced = dict(profile)
    reduced["columns"] = [
        {**column, "name": str(column.get("name", ""))[:60]}
        for column in list(profile.get("columns", []))[:20]
    ]
    reduced["numeric_summary"] = dict(list(profile.get("numeric_summary", {}).items())[:6])
    reduced["category_frequencies"] = {
        column: values[:3]
        for column, values in list(profile.get("category_frequencies", {}).items())[:3]
    }
    reduced["date_ranges"] = dict(list(profile.get("date_ranges", {}).items())[:3])
    serialized = json.dumps(reduced, ensure_ascii=True, separators=(",", ":"), default=str)
    if len(serialized) > max_chars:
        raise ValueError("Dataset profile is too large to send safely; reduce the number of columns.")
    return serialized
