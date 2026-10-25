import sqlite3

import pandas as pd

from src.database import (
    dataset_display_name,
    import_dataset,
    initialize_database,
    list_datasets,
    load_dataset,
    replace_uploaded_data,
)
from src.data_loader import generate_sample_data


def test_initializes_reproducible_sample_data(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    first = load_dataset("orders", path)
    initialize_database(path)
    second = load_dataset("orders", path)

    assert len(first) == 600
    assert first.equals(second)
    assert first["profit"].lt(0).any()
    assert {"category", "region", "customer_segment"} <= set(first.columns)
    assert "orders" in list_datasets(path)


def test_initialization_preserves_legacy_sales_table(tmp_path):
    path = tmp_path / "business.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE sales (order_id INTEGER, sales INTEGER)")
        connection.execute("INSERT INTO sales VALUES (1, 900)")

    initialize_database(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT order_id, sales FROM sales").fetchall() == [(1, 900)]
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 600


def test_initialization_rejects_incompatible_orders_without_modifying_it(tmp_path):
    import pytest

    path = tmp_path / "business.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE orders (private_note TEXT)")
        connection.execute("INSERT INTO orders VALUES ('keep this row')")

    with pytest.raises(ValueError, match="left unchanged"):
        initialize_database(path)

    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT private_note FROM orders").fetchone()[0] == "keep this row"


def test_imports_and_isolates_multiple_non_sales_datasets(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    employees = pd.DataFrame(
        {"Employee Name": ["Ari", "Bea"], "Department": ["HR", "Finance"], "Annual Salary": [50_000, 60_000]}
    )
    inventory = pd.DataFrame(
        {"Item": ["Widget"], "Stock Level": [12], "Unit Cost ($)": [3.5]}
    )
    employee_table = import_dataset(employees, "employees.csv", b"employee-file", path)
    inventory_table = import_dataset(inventory, "inventory.csv", b"inventory-file", path)

    assert employee_table != inventory_table
    assert employee_table in list_datasets(path)
    assert inventory_table in list_datasets(path)
    assert dataset_display_name(employee_table, path) == "employees.csv"
    assert load_dataset(employee_table, path).equals(employees)
    assert load_dataset(inventory_table, path).equals(inventory)


def test_legacy_uploaded_orders_import_remains_available(tmp_path):
    path = tmp_path / "business.db"
    initialize_database(path)
    frame = generate_sample_data(3)

    assert replace_uploaded_data(frame, path) == 3
    assert "uploaded_orders" in list_datasets(path)
    assert load_dataset("uploaded_orders", path).equals(frame)
