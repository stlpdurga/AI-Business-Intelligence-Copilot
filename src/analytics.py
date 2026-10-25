import math
import re
from collections.abc import Iterable

import pandas as pd


_POSITIVE_CHURN_VALUES = {
    "1", "true", "yes", "y", "churn", "churned", "left", "exited",
    "terminated", "cancelled", "canceled", "departed", "lost",
}
_NEGATIVE_CHURN_VALUES = {
    "0", "false", "no", "n", "stayed", "retained", "active", "current",
    "notchurned",
}


def _normalized_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _find_named_column(columns: Iterable[str], names: set[str]) -> str | None:
    return next((column for column in columns if _normalized_name(column) in names), None)


def find_churn_column(columns: Iterable[str]) -> str | None:
    preferred = ("Churn Label", "Churn", "Customer Churn", "Churn Status", "Is Churned", "Churn Value")
    normalized_to_column = {_normalized_name(column): column for column in columns}
    return next(
        (
            normalized_to_column[_normalized_name(name)]
            for name in preferred
            if _normalized_name(name) in normalized_to_column
        ),
        None,
    )


def find_contract_column(columns: Iterable[str]) -> str | None:
    return _find_named_column(columns, {"contract", "contracttype"})


def find_monthly_charges_column(columns: Iterable[str]) -> str | None:
    return _find_named_column(
        columns,
        {"monthlycharge", "monthlycharges", "monthlyfee", "monthlycost"},
    )


def find_customer_id_column(columns: Iterable[str]) -> str | None:
    return _find_named_column(
        columns,
        {"customerid", "subscriberid", "accountid", "clientid"},
    )


def churn_value_status(value: object) -> bool | None:
    if pd.isna(value):
        return None
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        numeric_value = None
    if numeric_value == 1:
        return True
    if numeric_value == 0:
        return False
    normalized = re.sub(r"[^a-z0-9]", "", str(value).casefold())
    if normalized in _POSITIVE_CHURN_VALUES:
        return True
    if normalized in _NEGATIVE_CHURN_VALUES:
        return False
    return None


def churn_labels(values: Iterable[object], positive: bool) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            str(value).strip()
            for value in values
            if churn_value_status(value) is positive
        )
    )


def calculate_dashboard_metrics(frame: pd.DataFrame) -> dict[str, int | float | None]:
    customer_id = find_customer_id_column(frame.columns)
    churn_column = find_churn_column(frame.columns)
    monthly_charges = find_monthly_charges_column(frame.columns)

    total_customers = (
        int(frame[customer_id].nunique(dropna=True))
        if customer_id
        else len(frame)
    )
    churned_customers = None
    churn_rate = None
    if churn_column:
        statuses = frame[churn_column].map(churn_value_status)
        eligible = statuses.notna()
        if eligible.any():
            churned_customers = int(statuses[eligible].sum())
            churn_rate = float(statuses[eligible].mean())

    average_monthly_charges = None
    if monthly_charges:
        amounts = pd.to_numeric(frame[monthly_charges], errors="coerce")
        if amounts.notna().any():
            average_monthly_charges = float(amounts.mean())

    return {
        "total_customers": total_customers,
        "churned_customers": churned_customers,
        "churn_rate": churn_rate,
        "average_monthly_charges": average_monthly_charges,
    }


def churn_by_contract(frame: pd.DataFrame) -> pd.DataFrame:
    churn_column = find_churn_column(frame.columns)
    contract_column = find_contract_column(frame.columns)
    if not churn_column or not contract_column:
        return pd.DataFrame()

    statuses = frame[churn_column].map(churn_value_status)
    eligible = statuses.notna() & frame[contract_column].notna()
    if not eligible.any():
        return pd.DataFrame()

    grouped = pd.DataFrame(
        {
            "contract": frame.loc[eligible, contract_column],
            "churned": statuses.loc[eligible].astype(bool),
        }
    )
    return (
        grouped.groupby("contract", dropna=True)["churned"]
        .agg(churn_rate="mean", customers="size")
        .reset_index()
        .assign(churn_rate_percent=lambda result: result["churn_rate"] * 100)
        .drop(columns="churn_rate")
        .sort_values("churn_rate_percent", ascending=False, ignore_index=True)
    )


def calculate_kpis(frame: pd.DataFrame) -> dict[str, float | int | None]:
    sales = float(frame["sales"].sum()) if "sales" in frame else 0.0
    profit = float(frame["profit"].sum()) if "profit" in frame else 0.0
    order_count = int(frame["order_id"].nunique()) if "order_id" in frame else len(frame)
    margin = profit / sales if sales else None
    return {
        "total_sales": sales,
        "total_profit": profit,
        "total_orders": order_count,
        "profit_margin": margin if margin is None or math.isfinite(margin) else None,
    }


def summarize_results(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "The executed query returned no rows, so there are no results to summarize."

    facts = []
    for column in frame.columns:
        value = frame.iloc[0][column]
        if isinstance(value, (int, float)) and not pd.isna(value):
            label = column.replace("_", " ")
            if any(token in column.lower() for token in ("sales", "profit", "value")):
                rendered = f"${value:,.2f}"
            elif "margin" in column.lower():
                rendered = f"{value:.1%}"
            elif "percent" in column.lower():
                rendered = f"{value:.2f}%"
            elif "rate" in column.lower():
                rendered = f"{value * 100:.2f}%" if abs(value) <= 1 else f"{value:.2f}%"
            else:
                rendered = f"{value:,.2f}" if isinstance(value, float) else f"{value:,}"
            facts.append(f"{label}: {rendered}")
        elif not pd.isna(value):
            facts.append(f"{column.replace('_', ' ')}: {value}")

    first_row = "; ".join(facts) if facts else "The result contains no displayable values."
    if len(frame) == 1:
        return f"The executed query returned one row. Observed result: {first_row}."
    return f"The executed query returned {len(frame):,} rows. First row: {first_row}."
