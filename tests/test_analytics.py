import pandas as pd

from src.analytics import (
    calculate_dashboard_metrics,
    calculate_kpis,
    churn_by_contract,
    summarize_results,
)
from src.chart_builder import PALETTE, build_dashboard_charts


def test_kpis_and_zero_sales_margin():
    frame = pd.DataFrame({"order_id": [1, 2, 2], "sales": [10, 0, 0], "profit": [4, -1, 0]})

    assert calculate_kpis(frame) == {
        "total_sales": 10.0,
        "total_profit": 3.0,
        "total_orders": 2,
        "profit_margin": 0.3,
    }
    assert calculate_kpis(pd.DataFrame({"order_id": [1], "sales": [0], "profit": [2]}))["profit_margin"] is None


def test_insight_uses_actual_returned_values():
    summary = summarize_results(pd.DataFrame([{"category": "Technology", "total_profit": 125.5}]))

    assert "Technology" in summary
    assert "$125.50" in summary
    assert "highest" not in summary.lower()


def test_empty_result_has_no_claimed_findings():
    assert "no rows" in summarize_results(pd.DataFrame()).lower()


def test_telco_dashboard_metrics_exclude_missing_churn_and_charge_values():
    frame = pd.DataFrame(
        {
            "CustomerID": ["001", "002", "003", "004"],
            "Churn Label": ["Yes", "No", "Yes", None],
            "Monthly Charges": [100, 50, "70", None],
            "Contract": ["Month-to-month", "One year", "Month-to-month", "Two year"],
        }
    )

    assert calculate_dashboard_metrics(frame) == {
        "total_customers": 4,
        "churned_customers": 2,
        "churn_rate": 2 / 3,
        "average_monthly_charges": 220 / 3,
    }


def test_churn_by_contract_and_dashboard_charts_use_observed_data():
    frame = pd.DataFrame(
        {
            "Churn": ["Yes", "No", "Yes", "No", None],
            "Contract": ["Monthly", "Monthly", "Yearly", "Annual", "Two year"],
            "MonthlyCharges": [30, 40, 100, 120, 75],
        }
    )

    result = churn_by_contract(frame)
    charts = build_dashboard_charts(frame)

    assert result.iloc[0]["contract"] == "Yearly"
    assert result.iloc[0]["churn_rate_percent"] == 100.0
    assert [title for title, _ in charts] == [
        "Records by Churn",
        "MonthlyCharges distribution",
    ]
    assert all(figure.layout.paper_bgcolor == "#FFFFFF" for _, figure in charts)
    assert all(tuple(figure.layout.colorway) == PALETTE for _, figure in charts)
    for _, figure in charts:
        for trace in figure.data:
            marker_color = trace.marker.color
            if isinstance(marker_color, str):
                assert marker_color in PALETTE
            elif marker_color is not None:
                assert set(marker_color) <= set(PALETTE)


def test_churn_metrics_do_not_guess_for_unrecognized_labels():
    metrics = calculate_dashboard_metrics(
        pd.DataFrame({"Churn": ["unknown", "pending"], "MonthlyCharges": ["n/a", None]})
    )

    assert metrics["churned_customers"] is None
    assert metrics["churn_rate"] is None
    assert metrics["average_monthly_charges"] is None
