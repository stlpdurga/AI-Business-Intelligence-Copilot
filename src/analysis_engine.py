from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from src.dataset_profiler import build_dataset_profile, infer_dataset_types, is_sensitive_column

MAX_ANALYSIS_ROWS = 30
MAX_ANALYSIS_COLUMNS = 12


@dataclass(frozen=True)
class DatasetAnalysis:
    operation: str
    title: str
    dataframe: pd.DataFrame
    verified_summary: str
    clarification: str | None = None
    visualization: str | None = None


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def _column_types(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    numeric = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
        and not is_sensitive_column(str(column))
    ]
    dates = [
        str(column)
        for column in frame.columns
        if pd.api.types.is_datetime64_any_dtype(frame[column])
    ]
    categories = [
        str(column)
        for column in frame.columns
        if str(column) not in numeric
        and str(column) not in dates
        and not is_sensitive_column(str(column))
        and 1 < frame[column].nunique(dropna=True) <= 50
    ]
    return numeric, dates, categories


def _mentioned_columns(question: str, columns: list[str]) -> list[str]:
    normalized_question = f" {_normalize(question)} "
    matches = [
        column for column in columns
        if f" {_normalize(column)} " in normalized_question
    ]
    return sorted(matches, key=lambda column: len(_normalize(column)), reverse=True)


def _clarification(
    question: str,
    numeric: list[str],
    dates: list[str],
    categories: list[str],
) -> str:
    candidates = ", ".join((numeric + dates + categories)[:10]) or "no analyzable columns"
    return (
        "I could not safely infer the requested operation or columns. Please name the column(s) "
        f"you want to analyze. Available numeric/date/category columns: {candidates}."
    )


def _references_unknown_column(question: str, columns: list[str]) -> str | None:
    normalized_columns = {_normalize(column) for column in columns}
    generic_terms = {
        "category",
        "categories",
        "count",
        "records",
        "rows",
        "total value",
        "value",
    }
    quoted = re.findall(r"[\"'`](.+?)[\"'`]", question)
    for candidate in quoted:
        if _normalize(candidate) not in normalized_columns:
            return candidate.strip()
    match = re.search(
        r"\b(?:average|mean|median|sum|total|minimum|maximum|count|trend|by|per|across)\s+"
        r"([A-Za-z][A-Za-z0-9 _()$.-]{1,80}?)(?=\s+\b(?:by|per|across|over|for|and|with)\b|[?.!,]|$)",
        question,
        flags=re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        normalized = _normalize(candidate)
        if normalized.startswith("total ") and normalized[6:] in normalized_columns:
            return None
        if (
            normalized
            and normalized not in normalized_columns
            and normalized not in generic_terms
        ):
            return candidate
    return None


def _bounded_frame(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.iloc[:MAX_ANALYSIS_ROWS, :MAX_ANALYSIS_COLUMNS].copy()


def _summary_analysis(frame: pd.DataFrame) -> DatasetAnalysis:
    profile = build_dataset_profile(frame)
    numeric = profile["numeric_summary"]
    records = [
        {"Column": name, **values}
        for name, values in numeric.items()
    ]
    table = pd.DataFrame(records)
    dimensions = profile["dimensions"]
    summary = (
        f"Dataset contains {dimensions['rows']:,} rows and {dimensions['columns']:,} columns; "
        f"{profile['duplicate_rows']:,} duplicate rows. "
        f"Numeric summaries computed for {len(records)} columns. "
        f"{sum(item['missing_count'] for item in profile['columns']):,} missing cells across "
        f"{sum(item['missing_count'] > 0 for item in profile['columns'])} columns with missing data."
    )
    return DatasetAnalysis("summary", "Dataset overview", table, summary, visualization=None)


def analyze_dataset_question(question: str, frame: pd.DataFrame) -> DatasetAnalysis:
    if not question.strip():
        return DatasetAnalysis(
            "clarification",
            "Clarification needed",
            pd.DataFrame(),
            "",
            "Enter a question about this dataset.",
        )
    if frame.empty or frame.shape[1] == 0:
        return DatasetAnalysis(
            "clarification",
            "Dataset unavailable",
            pd.DataFrame(),
            "",
            "The selected dataset has no rows or columns to analyze.",
        )

    data = infer_dataset_types(frame)
    numeric, dates, categories = _column_types(data)
    all_columns = [str(column) for column in data.columns]
    mentioned = _mentioned_columns(question, all_columns)
    mentioned_numeric = [column for column in mentioned if column in numeric]
    mentioned_dates = [column for column in mentioned if column in dates]
    mentioned_categories = [column for column in mentioned if column in categories]
    text = question.casefold()
    unknown_column = _references_unknown_column(question, all_columns)
    if unknown_column:
        return DatasetAnalysis(
            "clarification",
            "Unknown column",
            pd.DataFrame(),
            "",
            f"I could not find a column named “{unknown_column}”. Available columns: "
            f"{', '.join(all_columns[:20])}.",
        )

    if any(word in text for word in ("missing", "null", "empty", "incomplete")):
        report = pd.DataFrame(
            {
                "column": all_columns,
                "missing_count": [int(data[column].isna().sum()) for column in all_columns],
                "missing_percent": [
                    round(float(data[column].isna().mean() * 100), 2) if len(data) else 0
                    for column in all_columns
                ],
            }
        )
        report = report.loc[report["missing_count"] > 0].sort_values(
            "missing_count", ascending=False
        )
        return DatasetAnalysis(
            "missing",
            "Missing values",
            _bounded_frame(report.reset_index(drop=True)),
            f"Computed missing counts and percentages locally for {len(all_columns)} columns; "
            f"{len(report)} columns contain missing values.",
            visualization="bar" if not report.empty else None,
        )

    if any(word in text for word in ("outlier", "anomal", "unusual value")):
        targets = mentioned_numeric or (numeric if len(numeric) == 1 else [])
        if not targets:
            clarification = (
                "Which numeric column should I check for outliers? "
                f"Available numeric columns: {', '.join(numeric[:10]) or 'none'}."
            )
            return DatasetAnalysis("clarification", "Clarification needed", pd.DataFrame(), "", clarification)
        report = []
        for column in targets[:MAX_ANALYSIS_COLUMNS]:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if values.empty:
                continue
            q1, q3 = values.quantile([0.25, 0.75])
            iqr = q3 - q1
            low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            report.append(
                {
                    "column": column,
                    "outlier_count": int(((values < low) | (values > high)).sum()),
                    "lower_iqr_bound": float(low),
                    "upper_iqr_bound": float(high),
                    "observations": int(values.count()),
                }
            )
        result = pd.DataFrame(report)
        return DatasetAnalysis(
            "outliers",
            "Potential numeric outliers",
            result,
            "Outliers were counted locally using the 1.5×IQR rule; bounds and counts are based on observed non-missing values.",
            visualization="bar" if not result.empty else None,
        )

    if any(word in text for word in ("trend", "monthly", "over time", "by month", "timeline")):
        target_dates = mentioned_dates or (dates if len(dates) == 1 else [])
        target_numeric = mentioned_numeric or (numeric if len(numeric) == 1 else [])
        if len(target_dates) != 1 or len(target_numeric) != 1:
            return DatasetAnalysis(
                "clarification",
                "Clarification needed",
                pd.DataFrame(),
                "",
                "Please specify one date/time column and one numeric column for the trend. "
                f"Date columns: {', '.join(dates[:8]) or 'none'}; numeric columns: "
                f"{', '.join(numeric[:8]) or 'none'}.",
            )
        date_column, value_column = target_dates[0], target_numeric[0]
        dated = data[[date_column, value_column]].dropna().copy()
        dated[date_column] = pd.to_datetime(dated[date_column], errors="coerce", format="mixed", utc=True)
        dated[value_column] = pd.to_numeric(dated[value_column], errors="coerce")
        dated = dated.dropna()
        if dated.empty:
            return DatasetAnalysis(
                "clarification",
                "Trend unavailable",
                pd.DataFrame(),
                "",
                "The selected date and numeric columns have no overlapping usable values.",
            )
        dated["period"] = dated[date_column].dt.to_period("M").astype(str)
        trend = (
            dated.groupby("period", as_index=False)[value_column]
            .sum()
            .rename(columns={"period": date_column})
            .sort_values(date_column)
            .reset_index(drop=True)
        )
        return DatasetAnalysis(
            "trend",
            f"Monthly {value_column} trend",
            trend,
            f"Calculated monthly sums for {value_column} across {len(trend)} observed periods using {date_column}.",
            visualization="line",
        )

    if any(word in text for word in ("average", "mean", "median", "minimum", "maximum", "describe", "statistics", "numeric")) and not re.search(r"\b(by|per|across|each)\b", text):
        all_numeric_requested = any(
            phrase in text
            for phrase in ("all numeric", "numeric columns", "descriptive statistics", "statistics")
        )
        if not mentioned_numeric and len(numeric) > 1 and not all_numeric_requested:
            return DatasetAnalysis(
                "clarification",
                "Clarification needed",
                pd.DataFrame(),
                "",
                "Which numeric column should I calculate statistics for? "
                f"Available: {', '.join(numeric[:10])}.",
            )
        targets = mentioned_numeric or numeric
        if not targets:
            return DatasetAnalysis(
                "clarification",
                "No numeric columns",
                pd.DataFrame(),
                "",
                "This dataset has no numeric columns available for the requested statistics.",
            )
        summaries = []
        for column in targets[:MAX_ANALYSIS_COLUMNS]:
            values = pd.to_numeric(data[column], errors="coerce").dropna()
            if not values.empty:
                summaries.append(
                    {
                        "column": column,
                        "count": int(values.count()),
                        "mean": float(values.mean()),
                        "median": float(values.median()),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                    }
                )
        result = pd.DataFrame(summaries)
        return DatasetAnalysis(
            "describe",
            "Numeric summary statistics",
            result,
            f"Computed count, mean, median, minimum, and maximum locally for {len(result)} numeric columns.",
        )

    if any(word in text for word in ("top", "highest", "most", "compare", "comparison", "group", "category", "categories", "each")):
        dimensions = mentioned_categories or (categories if len(categories) == 1 else [])
        measures = mentioned_numeric or (numeric if len(numeric) == 1 else [])
        if len(dimensions) != 1:
            return DatasetAnalysis(
                "clarification",
                "Clarification needed",
                pd.DataFrame(),
                "",
                "Please specify one category column to compare. Available suitable category columns: "
                f"{', '.join(categories[:10]) or 'none'}.",
            )
        dimension = dimensions[0]
        if len(measures) > 1:
            return DatasetAnalysis(
                "clarification",
                "Clarification needed",
                pd.DataFrame(),
                "",
                f"Which numeric measure should I compare across {dimension}? "
                f"Available: {', '.join(measures[:10])}.",
            )
        top_match = re.search(r"\btop\s+(\d{1,2})\b", text)
        limit = min(max(int(top_match.group(1)), 1), MAX_ANALYSIS_ROWS) if top_match else 10
        if measures:
            measure = measures[0]
            operation = "mean" if any(word in text for word in ("average", "mean")) else "sum"
            grouped = (
                data.groupby(dimension, dropna=True)[measure]
                .agg(operation)
                .sort_values(ascending=False)
                .head(limit)
                .rename(f"{operation}_{measure}")
                .reset_index()
            )
            summary = f"Calculated {operation} of {measure} per {dimension} from the dataset, then selected the top {len(grouped)} groups."
        else:
            grouped = (
                data[dimension]
                .dropna()
                .astype(str)
                .value_counts()
                .head(limit)
                .rename_axis(dimension)
                .reset_index(name="record_count")
            )
            summary = f"Counted non-missing records per {dimension} and selected the top {len(grouped)} categories."
        return DatasetAnalysis(
            "group_compare",
            f"Results by {dimension}",
            _bounded_frame(grouped),
            summary,
            visualization="bar",
        )

    if any(word in text for word in ("insight", "summar", "kpi", "visual", "overview", "important finding")):
        return _summary_analysis(data)

    # Generic counts and dataset-level questions can be answered without guessing columns.
    if any(word in text for word in ("how many rows", "row count", "number of records", "dataset size")):
        result = pd.DataFrame([{"rows": len(data), "columns": len(data.columns)}])
        return DatasetAnalysis(
            "dimensions",
            "Dataset dimensions",
            result,
            f"The dataset contains {len(data):,} rows and {len(data.columns):,} columns.",
        )

    if any(word in text for word in ("correlation", "relationship")):
        selected = mentioned_numeric or numeric
        if len(selected) < 2:
            return DatasetAnalysis(
                "clarification",
                "Correlation unavailable",
                pd.DataFrame(),
                "",
                "Correlation requires at least two numeric columns. "
                f"Available numeric columns: {', '.join(numeric[:10]) or 'none'}.",
            )
        matrix = data[selected[:MAX_ANALYSIS_COLUMNS]].corr(numeric_only=True)
        result = matrix.rename_axis("column").reset_index()
        return DatasetAnalysis(
            "correlation",
            "Numeric correlations",
            result,
            f"Computed Pearson correlations locally for {matrix.shape[0]} numeric columns.",
            visualization="heatmap",
        )

    return DatasetAnalysis(
        "clarification",
        "Clarification needed",
        pd.DataFrame(),
        "",
        _clarification(question, numeric, dates, categories),
    )
