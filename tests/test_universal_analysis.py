import io

import pandas as pd
import pytest

from src.analysis_engine import analyze_dataset_question
from src.data_loader import load_tabular_data
from src.dataset_profiler import build_dataset_profile, infer_dataset_types


def test_profile_reports_types_missing_values_numeric_and_category_summary():
    frame = pd.DataFrame(
        {
            "Sales Amount": ["10", "20", None],
            "Department": ["HR", "IT", "HR"],
            "Created On": ["2025-01-01", "2025-02-01", None],
            "Email Address": ["a@example.test", "b@example.test", None],
        }
    )

    profile = build_dataset_profile(frame)

    assert profile["dimensions"] == {"rows": 3, "columns": 4}
    by_name = {column["name"]: column for column in profile["columns"]}
    assert by_name["Sales Amount"]["type"] == "numeric"
    assert by_name["Created On"]["type"] == "datetime"
    assert by_name["Department"]["missing_count"] == 0
    assert "Sales Amount" in profile["numeric_summary"]
    assert "Department" in profile["category_frequencies"]
    assert "Email Address" not in profile["category_frequencies"]
    assert profile["sensitive_values_included"] is False


def test_pandas_inference_normalizes_mixed_numeric_boolean_and_dates():
    frame = pd.DataFrame(
        {
            "score.value": [" 1,000 ", "2000", "bad"],
            "Active?": [" Yes ", "no", None],
            "created_at": ["2025-01-01", "2025-01-02", None],
            "Notes": [" high ", "", " low "],
        }
    )

    inferred = infer_dataset_types(frame)

    assert inferred["score.value"].tolist()[:2] == [1000, 2000]
    assert inferred["Active?"].dtype.name == "boolean"
    assert inferred["Active?"].iloc[0]
    assert pd.api.types.is_datetime64_any_dtype(inferred["created_at"])
    assert pd.isna(inferred.loc[1, "Notes"])
    assert inferred.loc[0, "Notes"] == "high"


def test_csv_uses_fallback_encoding_and_keeps_unusual_headers():
    frame = load_tabular_data(
        "employee,note\nA,caf\u00e9".encode("cp1252"),
        "staff.csv",
    )

    assert frame.loc[0, "note"] == "café"


def test_csv_invalid_empty_unsupported_and_large_payloads_fail_cleanly():
    for payload, filename in (
        (b"", "empty.csv"),
        (b"header\n", "no_rows.csv"),
        (b"not a table", "not-a-table.csv"),
        (b"a,b\n1,2\n", "unsupported.json"),
    ):
        with pytest.raises(ValueError):
            load_tabular_data(payload, filename)


def test_excel_upload_with_duplicate_columns_and_missing_values():
    workbook = io.BytesIO()
    with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
        pd.DataFrame({"Person": ["Ada", None], "Score": [4, 5]}).to_excel(
            writer, index=False, sheet_name="Results"
        )

    result = load_tabular_data(workbook.getvalue(), "scores.xlsx", "Results")

    assert result.shape == (2, 2)
    assert result["Person"].isna().sum() == 1


def test_large_dataset_analysis_uses_exact_local_numeric_statistics():
    frame = pd.DataFrame(
        {
            "unit": range(100_000),
            "amount": range(100_000),
        }
    )

    result = analyze_dataset_question("What is the average amount?", frame)

    assert result.dataframe.loc[0, "count"] == 100_000
    assert result.dataframe.loc[0, "mean"] == pytest.approx(49_999.5)


def test_missing_values_analysis_reports_actual_counts_and_percentages():
    frame = pd.DataFrame({"A": [1, None, 3, None], "B": ["x", "y", None, "x"]})

    result = analyze_dataset_question("Which columns have missing values?", frame)

    values = result.dataframe.set_index("column")
    assert values.loc["A", "missing_count"] == 2
    assert values.loc["A", "missing_percent"] == 50
    assert values.loc["B", "missing_count"] == 1


def test_unknown_columns_and_ambiguous_operations_ask_for_clarification():
    frame = pd.DataFrame(
        {"Region": ["East", "West"], "Department": ["HR", "IT"], "Sales": [20, 30], "Profit": [3, 4]}
    )

    unknown = analyze_dataset_question("What is the average unicorn score?", frame)
    ambiguous = analyze_dataset_question("Show the top categories", frame)

    assert "could not find a column" in unknown.clarification.lower()
    assert ambiguous.clarification
    assert ambiguous.dataframe.empty


def test_unqualified_average_with_multiple_measures_asks_for_clarification():
    result = analyze_dataset_question(
        "What is the average?",
        pd.DataFrame({"Revenue": [10, 20], "Cost": [3, 5]}),
    )

    assert result.clarification
    assert "Revenue" in result.clarification
    assert "Cost" in result.clarification


def test_grouped_comparisons_return_actual_values_and_request_clarification():
    frame = pd.DataFrame(
        {"Region": ["East", "East", "West"], "Sales": [10, 20, 50]}
    )

    result = analyze_dataset_question("What are the top 2 regions by total sales?", frame)

    assert result.dataframe.iloc[0]["Region"] == "West"
    assert result.dataframe.iloc[0]["sum_Sales"] == 50


def test_monthly_trend_and_outlier_counts_are_computed_locally():
    frame = pd.DataFrame(
        {
            "event_date": ["2025-01-01", "2025-01-12", "2025-02-01", "2025-02-22"],
            "revenue": [10, 20, 30, 1_000],
        }
    )

    trend = analyze_dataset_question("Show monthly trends for event_date and revenue", frame)
    outliers = analyze_dataset_question("Identify outliers in revenue", frame)

    assert trend.dataframe["revenue"].tolist() == [30, 1030]
    assert outliers.dataframe.loc[0, "outlier_count"] == 1


def test_no_numeric_columns_returns_clear_clarification():
    result = analyze_dataset_question(
        "What is the average?",
        pd.DataFrame({"Department": ["HR", "IT"], "Status": ["Open", "Closed"]}),
    )

    assert "no numeric columns" in result.clarification.lower()
