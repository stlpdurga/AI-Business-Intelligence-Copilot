import pytest
import pandas as pd

from src.database import import_dataset, initialize_database, load_dataset
from src.query_executor import execute_query
from src.sql_validator import SQLValidationError


def test_executes_readonly_query_and_returns_real_rows(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    result = execute_query("SELECT COUNT(*) AS n FROM orders", "orders", path)

    assert result.status == "success"
    assert result.dataframe.iloc[0]["n"] == 600


def test_empty_query_result_is_successful_empty_frame(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    result = execute_query("SELECT order_id FROM orders WHERE order_id < 0", "orders", path)

    assert result.status == "success"
    assert result.dataframe.empty


def test_enforces_configured_row_limit(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    result = execute_query("SELECT order_id FROM orders ORDER BY order_id", "orders", path, max_rows=7)

    assert result.row_count == 7


def test_invalid_write_is_rejected_and_does_not_change_data(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    before = len(load_dataset("orders", path))

    with pytest.raises(SQLValidationError):
        execute_query("DELETE FROM orders", "orders", path)

    assert len(load_dataset("orders", path)) == before


def test_sqlite_authorizer_rejects_write_even_if_parser_is_bypassed(tmp_path):
    from src.database import connect_readonly
    import sqlite3

    path = tmp_path / "business.db"
    initialize_database(path)
    with connect_readonly(path, "orders") as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM orders")


def test_executes_dynamic_schema_with_quoted_punctuation_columns(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    frame = pd.DataFrame(
        {"Department Name": ["HR", "HR", "IT"], "Annual Salary ($)": [40_000, 60_000, 70_000]}
    )
    table = import_dataset(frame, "employees.csv", b"employee-content", path)

    result = execute_query(
        f'SELECT "Department Name", AVG("Annual Salary ($)") AS average_salary '
        f'FROM "{table}" GROUP BY "Department Name" ORDER BY average_salary DESC',
        table,
        path,
    )

    assert result.status == "success"
    assert result.dataframe.iloc[0]["Department Name"] == "IT"
    assert result.dataframe.iloc[0]["average_salary"] == 70_000


def test_dynamic_dataset_rejects_injection_and_unrelated_tables(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    table = import_dataset(
        pd.DataFrame({"Department": ["HR"], "Salary": [50_000]}),
        "employees.csv",
        b"employees",
        path,
    )

    for query in (
        f'SELECT * FROM "{table}"; DROP TABLE orders',
        f'SELECT * FROM "{table}" UNION SELECT * FROM orders',
        f'SELECT load_extension("anything") FROM "{table}"',
        f'SELECT "Missing Column" FROM "{table}"',
    ):
        with pytest.raises(SQLValidationError):
            execute_query(query, table, path)

    assert len(load_dataset("orders", path)) == 600


def test_churn_rate_query_uses_observed_values_and_excludes_missing_outcomes(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    frame = pd.DataFrame({"Churn": ["Yes", "No", "Yes", None]})
    table = import_dataset(frame, "churn.csv", b"churn-data", path)
    sql = (
        f'SELECT 100.0 * SUM(CASE WHEN LOWER(TRIM("Churn")) = \'yes\' THEN 1 ELSE 0 END) '
        f'/ NULLIF(COUNT("Churn"), 0) AS churn_rate_percent FROM "{table}" '
        'WHERE "Churn" IS NOT NULL'
    )

    result = execute_query(sql, table, path)

    assert result.status == "success"
    assert result.dataframe.iloc[0]["churn_rate_percent"] == pytest.approx(200 / 3)


def test_demo_churn_query_executes_against_uploaded_telco_data(tmp_path):
    from src.llm_service import generate_query_plan

    path = tmp_path / "business.db"
    initialize_database(path)
    frame = pd.DataFrame(
        {
            "Churn": ["Yes", "No", "Yes", None],
            "Contract": ["Month-to-month", "Month-to-month", "One year", "Two year"],
            "MonthlyCharges": [100, 50, 70, 80],
        }
    )
    table = import_dataset(frame, "telco.csv", b"telco-demo", path)
    plan = generate_query_plan(
        "Which contract type has the highest churn?",
        table,
        schema={"Churn": "TEXT", "Contract": "TEXT", "MonthlyCharges": "REAL"},
        categorical_values={"Churn": ("Yes", "No"), "Contract": ("Month-to-month", "One year", "Two year")},
    )

    result = execute_query(plan.sql, table, path)
    churn_plan = generate_query_plan(
        "What is the churn rate?",
        table,
        schema={"Churn": "TEXT", "Contract": "TEXT", "MonthlyCharges": "REAL"},
        categorical_values={"Churn": ("Yes", "No"), "Contract": ("Month-to-month", "One year", "Two year")},
    )
    churn_result = execute_query(churn_plan.sql, table, path)
    average_plan = generate_query_plan(
        "What is the average monthly charge?",
        table,
        schema={"Churn": "TEXT", "Contract": "TEXT", "MonthlyCharges": "REAL"},
    )
    average_result = execute_query(average_plan.sql, table, path)

    assert result.status == "success"
    assert result.dataframe.iloc[0]["contract_type"] == "One year"
    assert result.dataframe.iloc[0]["churn_rate_percent"] == pytest.approx(100.0)
    assert churn_result.status == "success"
    assert churn_result.dataframe.iloc[0]["churn_rate_percent"] == pytest.approx(200 / 3)
    assert average_result.status == "success"
    assert average_result.dataframe.iloc[0]["average_monthly_charges"] == pytest.approx(75.0)
