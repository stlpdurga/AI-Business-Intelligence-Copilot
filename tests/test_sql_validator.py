import pytest

from src.sql_validator import SQLValidationError, validate_sql


def test_accepts_select_and_caps_result_limit():
    validated = validate_sql("SELECT category, SUM(profit) AS total FROM orders GROUP BY category", "orders")

    assert "LIMIT 1000" in validated.sql.upper()


def test_accepts_safe_cte():
    validated = validate_sql(
        "WITH totals AS (SELECT category, SUM(sales) AS amount FROM orders GROUP BY category) "
        "SELECT category, amount FROM totals ORDER BY amount DESC",
        "orders",
    )

    assert "WITH" in validated.sql.upper()


def test_accepts_selecting_all_approved_dataset_columns():
    validated = validate_sql("SELECT * FROM orders", "orders")

    assert "SELECT * FROM orders" in validated.sql


@pytest.mark.parametrize(
    "sql",
    [
        "DELETE FROM orders",
        "UPDATE orders SET profit = 0",
        "DROP TABLE orders",
        "PRAGMA table_info(orders)",
        "SELECT * FROM orders; SELECT * FROM orders",
        "SELECT * FROM private_data",
        "SELECT 1",
        "SELECT 1 FROM orders",
        "SELECT missing_column FROM orders",
        "SELECT load_extension('malicious') FROM orders",
        "SELECT readfile('business.db') FROM orders",
        "SELECT random() FROM orders",
        "SELECT FROM orders",
        "WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM x) SELECT n FROM x",
    ],
)
def test_rejects_unsafe_or_invalid_sql(sql):
    with pytest.raises(SQLValidationError):
        validate_sql(sql, "orders")


def test_rejects_database_qualified_names_and_unsafe_functions():
    for sql in (
        "SELECT * FROM main.orders",
        "SELECT readfile('business.db')",
    ):
        with pytest.raises(SQLValidationError):
            validate_sql(sql, "orders")


def test_smaller_existing_limit_is_preserved():
    validated = validate_sql("SELECT order_id FROM orders LIMIT 3", "orders")

    assert "LIMIT 3" in validated.sql.upper()


def test_negative_limit_is_capped():
    validated = validate_sql("SELECT order_id FROM orders LIMIT -1", "orders")

    assert "LIMIT 1000" in validated.sql.upper()


def test_validates_dynamic_dataset_columns_and_cte():
    schema = {"Employee Name": "TEXT", "Salary ($)": "REAL"}
    validated = validate_sql(
        'WITH payroll AS (SELECT "Employee Name", AVG("Salary ($)") AS average_pay '
        'FROM "dataset_0123456789abcdef" GROUP BY "Employee Name") '
        'SELECT "Employee Name", average_pay FROM payroll ORDER BY average_pay DESC',
        "dataset_0123456789abcdef",
        schema=schema,
    )

    assert "WITH" in validated.sql.upper()
    assert '"Salary ($)"' in validated.sql


def test_rejects_missing_dynamic_column():
    with pytest.raises(SQLValidationError, match="outside the approved dataset"):
        validate_sql(
            'SELECT "Secret" FROM "dataset_0123456789abcdef"',
            "dataset_0123456789abcdef",
            schema={"Department": "TEXT"},
        )
