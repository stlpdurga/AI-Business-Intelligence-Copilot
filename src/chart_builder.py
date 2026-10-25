import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.dataset_profiler import infer_dataset_types, is_sensitive_column

PALETTE = ("#91B6F8", "#B0D0F8", "#C5E1FA", "#B59BE4", "#A386D9", "#9270D2")
NAVY = "#172554"
WHITE = "#FFFFFF"


def _apply_chart_theme(figure):
    figure.update_layout(
        template="plotly_white",
        colorway=PALETTE,
        paper_bgcolor=WHITE,
        plot_bgcolor=WHITE,
        font={"color": NAVY},
        title={"font": {"color": NAVY}},
        legend={"bgcolor": WHITE, "font": {"color": NAVY}},
        margin={"l": 24, "r": 18, "t": 54, "b": 28},
    )
    figure.update_xaxes(
        gridcolor=PALETTE[1],
        linecolor=PALETTE[0],
        tickfont={"color": NAVY},
        title_font={"color": NAVY},
        zerolinecolor=PALETTE[0],
    )
    figure.update_yaxes(
        gridcolor=PALETTE[1],
        linecolor=PALETTE[0],
        tickfont={"color": NAVY},
        title_font={"color": NAVY},
        zerolinecolor=PALETTE[0],
    )
    return figure


def build_dashboard_charts(frame: pd.DataFrame) -> list[tuple[str, object]]:
    charts: list[tuple[str, object]] = []
    if frame.empty:
        return charts

    data = infer_dataset_types(frame)
    numeric = [
        column for column in data.select_dtypes(include="number").columns
        if not pd.api.types.is_bool_dtype(data[column])
        and not is_sensitive_column(str(column))
    ]
    dates = list(data.select_dtypes(include=["datetime", "datetimetz"]).columns)
    categories = [
        column for column in data.columns
        if column not in numeric
        and column not in dates
        and not is_sensitive_column(str(column))
        and 1 < data[column].nunique(dropna=True) <= 20
    ]

    if dates and numeric:
        date_column, measure = dates[0], numeric[0]
        trend = data[[date_column, measure]].dropna().copy()
        if not trend.empty:
            trend["_period"] = trend[date_column].dt.to_period("M").astype(str)
            trend = trend.groupby("_period", as_index=False)[measure].sum()
            charts.append(
                (
                    f"{measure} over time",
                    _apply_chart_theme(
                        px.line(
                            trend,
                            x="_period",
                            y=measure,
                            markers=True,
                            title=f"{measure} by month",
                            color_discrete_sequence=[PALETTE[5]],
                        )
                    ),
                )
            )

    if categories:
        dimension = categories[0]
        counts = (
            data[dimension].dropna().astype(str).value_counts().head(12)
            .rename_axis(dimension).reset_index(name="record_count")
        )
        if not counts.empty:
            charts.append(
                (
                    f"Records by {dimension}",
                    _apply_chart_theme(
                        px.bar(
                            counts,
                            x=dimension,
                            y="record_count",
                            title=f"Records by {dimension}",
                            color_discrete_sequence=[PALETTE[0]],
                        )
                    ),
                )
            )

    for column in numeric[:2]:
        values = pd.to_numeric(data[column], errors="coerce").dropna()
        if not values.empty:
            charts.append(
                (
                    f"{column} distribution",
                    _apply_chart_theme(
                        px.histogram(
                            pd.DataFrame({column: values}),
                            x=column,
                            nbins=24,
                            title=f"{column} distribution",
                            color_discrete_sequence=[PALETTE[3]],
                        )
                    ),
                )
            )

    if len(numeric) >= 2:
        pair = data[numeric[:2]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(pair) >= 2:
            charts.append(
                (
                    f"{numeric[1]} vs {numeric[0]}",
                    _apply_chart_theme(
                        px.scatter(
                            pair,
                            x=numeric[0],
                            y=numeric[1],
                            title=f"{numeric[1]} vs {numeric[0]}",
                            color_discrete_sequence=[PALETTE[4]],
                        )
                    ),
                )
            )
    if len(numeric) >= 3:
        correlation = data[numeric[:10]].corr(numeric_only=True)
        heatmap = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale=[[0, PALETTE[2]], [0.5, "#FFFFFF"], [1, PALETTE[5]]],
                zmin=-1,
                zmax=1,
            )
        )
        heatmap.update_layout(title="Numeric correlation")
        charts.append(("Numeric correlation", _apply_chart_theme(heatmap)))
    return charts[:5]


def build_chart(
    frame: pd.DataFrame,
    question: str,
    suggested_type: str | None = None,
):
    if frame.empty:
        return None

    dimensions = [
        column for column in frame.columns
        if not pd.api.types.is_numeric_dtype(frame[column])
    ]
    if not dimensions:
        return None
    numeric = list(frame.select_dtypes(include="number").columns)
    measure = next(
        (
            column
            for column in numeric
            if column.lower() in question.lower()
            or column.lower().replace("_", " ") in question.lower()
        ),
        numeric[0] if numeric else None,
    )
    if measure is None:
        return None

    dimension = dimensions[0]
    if frame[dimension].nunique(dropna=True) < 2 or len(frame) > 100:
        return None
    title = f"{measure.replace('_', ' ').title()} by {dimension.replace('_', ' ').title()}"
    if suggested_type == "heatmap" and len(numeric) >= 2:
        correlation = frame[numeric].corr(numeric_only=True)
        figure = go.Figure(
            data=go.Heatmap(
                z=correlation.values,
                x=correlation.columns,
                y=correlation.index,
                colorscale=[[0, PALETTE[2]], [0.5, WHITE], [1, PALETTE[5]]],
                zmin=-1,
                zmax=1,
            )
        )
        figure.update_layout(title="Numeric correlation")
    elif suggested_type == "scatter" and len(numeric) >= 2:
        figure = px.scatter(
            frame,
            x=numeric[0],
            y=numeric[1],
            title=f"{numeric[1]} vs {numeric[0]}",
            color_discrete_sequence=[PALETTE[4]],
        )
    elif pd.api.types.is_datetime64_any_dtype(frame[dimension]) or any(
        token in dimension.lower() for token in ("date", "time")
    ) or suggested_type == "line":
        figure = px.line(
            frame,
            x=dimension,
            y=measure,
            markers=True,
            title=title,
            color_discrete_sequence=[PALETTE[5]],
        )
    else:
        figure = px.bar(
            frame,
            x=dimension,
            y=measure,
            title=title,
            labels={
                dimension: dimension.replace("_", " ").title(),
                measure: measure.replace("_", " ").title(),
            },
            color_discrete_sequence=[PALETTE[0]],
        )
    return _apply_chart_theme(figure)
