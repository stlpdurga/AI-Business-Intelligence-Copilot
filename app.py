import logging
import re
import sqlite3
import time

import pandas as pd
import streamlit as st

from config import (
    DATABASE_PATH,
    GEMINI_MODEL,
    get_google_api_key,
    get_google_api_key_source,
)
from src.analytics import summarize_results
from src.audit_logger import create_audit_record
from src.chart_builder import build_chart, build_dashboard_charts
from src.data_loader import excel_sheet_names, load_tabular_data
from src.analysis_engine import analyze_dataset_question
from src.database import (
    dataset_display_name,
    import_dataset,
    initialize_database,
    list_datasets,
    load_dataset,
)
from src.llm_service import (
    LLMServiceError,
    QueryPlan,
    explain_verified_analysis,
    generate_query_plan,
    list_gemini_text_models,
    suggest_questions,
    test_gemini_connection,
)
from src.dataset_profiler import build_dataset_profile, infer_dataset_types
from src.query_executor import QueryResult, execute_query
from src.sql_validator import SQLValidationError, validate_sql

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def show_dataframe(frame: pd.DataFrame, **kwargs) -> None:
    try:
        st.dataframe(frame, **kwargs)
    except ImportError:
        logger.exception("Streamlit's interactive table dependency could not be loaded.")
        st.warning(
            "The interactive table is unavailable because a local display dependency could not load. "
            "The data below is the actual dataset/query output."
        )
        st.code(frame.to_json(orient="records", date_format="iso", indent=2), language="json")


def _categorical_values(
    frame: pd.DataFrame,
    columns: list[str],
) -> dict[str, tuple[str, ...]]:
    values_by_column = {}
    sensitive_tokens = {
        "id", "name", "email", "phone", "address", "postal", "zip",
        "account", "subscriber", "customer", "client", "employee", "student", "user",
    }
    for column in columns:
        normalized_name = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", column)
        name_tokens = set(re.split(r"[_\s-]+", normalized_name.lower()))
        if sensitive_tokens & name_tokens:
            continue
        distinct = frame[column].dropna().astype(str).drop_duplicates()
        if not 1 <= len(distinct) <= 12:
            continue
        values = tuple(distinct.tolist())
        if all(len(value) <= 80 for value in values):
            values_by_column[column] = values
    return values_by_column


st.set_page_config(
    page_title="AI Business Intelligence Copilot",
    page_icon="📊",
    layout="wide",
)
st.markdown(
    """
    <style>
    :root {
        --bi-light-blue: #91B6F8; --bi-soft-blue: #B0D0F8;
        --bi-sky-blue: #C5E1FA; --bi-light-lavender: #B59BE4;
        --bi-purple: #A386D9; --bi-deep-purple: #9270D2;
        --bi-white: #FFFFFF; --bi-navy: #172554;
    }
    [data-testid="stAppViewContainer"] {
        color: var(--bi-navy);
        background: linear-gradient(145deg, var(--bi-sky-blue), var(--bi-white) 42%, var(--bi-soft-blue));
    }
    [data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stAppViewContainer"] [data-testid="stCaptionContainer"],
    [data-testid="stAppViewContainer"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stWidgetLabel"] {color: var(--bi-navy);}
    header[data-testid="stHeader"] {background: var(--bi-sky-blue);}
    [data-testid="stToolbar"] {background: transparent;}
    .block-container {max-width: 1320px; padding-top: 2rem; padding-bottom: 3rem;}
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, var(--bi-sky-blue), var(--bi-white) 46%, var(--bi-soft-blue));
        border-right: 1px solid var(--bi-light-blue);
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {color: var(--bi-navy);}
    [data-testid="stMetric"] {
        background: var(--bi-sky-blue); border: 1px solid var(--bi-light-blue);
        border-top: 3px solid var(--bi-light-blue); padding: 1rem 1.1rem;
        border-radius: .9rem; box-shadow: 0 5px 18px rgba(23, 37, 84, .09); min-height: 7rem;
        transition: transform .18s ease, box-shadow .18s ease;
    }
    [data-testid="stMetric"]:nth-child(even) {
        background: #FFFFFF; border-color: var(--bi-light-lavender); border-top-color: var(--bi-deep-purple);
    }
    [data-testid="stMetric"]:hover {transform: translateY(-2px); box-shadow: 0 9px 22px rgba(23, 37, 84, .13);}
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {color: var(--bi-navy);}
    [data-testid="stForm"] {
        background: var(--bi-white); border: 1px solid var(--bi-light-blue);
        border-radius: 1rem; padding: 1.25rem; box-shadow: 0 8px 26px rgba(23, 37, 84, .09);
    }
    [data-testid="stExpander"] {background: rgba(255,255,255,.94); border: 1px solid var(--bi-soft-blue); border-radius: .8rem;}
    div.stButton > button[kind="primary"], div.stFormSubmitButton > button {
        background: linear-gradient(105deg, var(--bi-light-blue), var(--bi-purple), var(--bi-deep-purple));
        color: var(--bi-navy); border: 1px solid var(--bi-deep-purple);
        border-radius: .7rem; font-weight: 700; transition: transform .18s ease, box-shadow .18s ease;
    }
    div.stButton > button[kind="primary"]:hover, div.stFormSubmitButton > button:hover {
        border-color: var(--bi-deep-purple); color: var(--bi-navy);
        box-shadow: 0 6px 16px rgba(23, 37, 84, .16); transform: translateY(-1px);
    }
    div.stButton > button:not([kind="primary"]), div[data-testid="stFileUploader"] button {
        color: var(--bi-navy); border-color: var(--bi-light-blue); background: var(--bi-white);
        border-radius: .65rem;
    }
    div.stButton > button:not([kind="primary"]):hover, div[data-testid="stFileUploader"] button:hover {
        color: var(--bi-navy); border-color: var(--bi-deep-purple); background: var(--bi-sky-blue);
    }
    h1, h2, h3 {color: var(--bi-navy);}
    h1 {letter-spacing: -.035em; font-weight: 750; background: linear-gradient(95deg, var(--bi-light-blue), var(--bi-deep-purple)); background-clip: text; color: transparent;}
    .bi-subtitle {color: var(--bi-navy); font-size: 1.08rem; margin-top: -.6rem; margin-bottom: 1.4rem;}
    .muted {color: var(--bi-navy); font-size: 0.95rem;}
    [data-testid="stAlert"] {
        background: var(--bi-sky-blue); color: var(--bi-navy);
        border: 1px solid var(--bi-light-blue); border-radius: .75rem;
    }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"] {color: var(--bi-navy);}
    [data-testid="stAlert"] svg {color: var(--bi-deep-purple); fill: var(--bi-deep-purple);}
    [data-testid="stTextInput"] input, [data-testid="stSelectbox"] [data-baseweb="select"] > div {
        border-color: var(--bi-light-blue); color: var(--bi-navy); background: var(--bi-white);
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stSelectbox"] [data-baseweb="select"] > div:focus-within {
        border-color: var(--bi-deep-purple); box-shadow: 0 0 0 .15rem var(--bi-soft-blue);
    }
    [data-testid="stRadio"] [role="radio"][aria-checked="true"] > div:first-child {
        border-color: var(--bi-deep-purple); background: var(--bi-deep-purple);
    }
    [data-testid="stRadio"] [role="radio"] > div:first-child {border-color: var(--bi-light-blue);}
    [data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {
        background: var(--bi-white); border: 1px solid var(--bi-soft-blue); border-radius: .85rem;
        padding: .35rem; box-shadow: 0 5px 18px rgba(23, 37, 84, .07);
    }
    @media (max-width: 768px) {
        .block-container {padding: 1.1rem 1rem 2rem;}
        h1 {font-size: 1.8rem;}
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("AI Business Intelligence Copilot")
st.markdown(
    '<div class="bi-subtitle">Turn business data into actionable insights with AI.</div>',
    unsafe_allow_html=True,
)

try:
    initialize_database()
    datasets = list_datasets()
except (OSError, sqlite3.Error, ValueError):
    logger.exception("Could not initialize the business database.")
    st.error("The business database could not be opened. Check the configured database path and permissions.")
    st.stop()

api_key = get_google_api_key()
api_key_source = get_google_api_key_source()
with st.sidebar:
    st.header("📊 Copilot")
    st.caption("Ask a question, review its SQL, and verify the results.")
    st.subheader("Data source")
    if not datasets:
        st.error("No usable datasets are available.")
        st.stop()

    pending_dataset = st.session_state.pop("pending_dataset", None)
    if pending_dataset in datasets:
        st.session_state["dataset_selector"] = pending_dataset
    default_dataset = "orders" if "orders" in datasets else datasets[0]
    dataset_name = st.selectbox(
        "Active dataset",
        datasets,
        index=datasets.index(default_dataset),
        key="dataset_selector",
        format_func=dataset_display_name,
    )
    previous_dataset = st.session_state.get("active_dataset")
    if previous_dataset != dataset_name:
        st.session_state.pop("last_query", None)
        st.session_state.pop("query_error", None)
        st.session_state["active_dataset"] = dataset_name
    st.caption(f"Database: `{DATABASE_PATH.name}`")

    uploaded_file = st.file_uploader(
        "Upload a tabular dataset",
        type=["csv", "xlsx", "xls"],
        help="CSV and Excel files up to 20 MB and 100,000 rows are supported.",
    )
    selected_sheet = None
    if uploaded_file is not None and uploaded_file.name.lower().endswith((".xlsx", ".xls")):
        try:
            sheets = excel_sheet_names(uploaded_file.getvalue())
            if sheets:
                selected_sheet = st.selectbox("Excel worksheet", sheets)
        except ValueError as exc:
            st.error(str(exc))
    if st.button("Validate and import dataset", disabled=uploaded_file is None, use_container_width=True):
        try:
            file_bytes = uploaded_file.getvalue()
            imported = load_tabular_data(file_bytes, uploaded_file.name, selected_sheet)
            table_name = import_dataset(imported, uploaded_file.name, file_bytes)
            st.session_state["pending_dataset"] = table_name
            st.success(f"Imported {len(imported):,} rows and {len(imported.columns):,} columns.")
            st.rerun()
        except ValueError as exc:
            st.error(str(exc))
        except (OSError, sqlite3.Error):
            logger.exception("Could not import the uploaded dataset.")
            st.error("The dataset could not be saved to the local database.")

    st.subheader("AI provider status")
    selected_model = GEMINI_MODEL
    if api_key:
        st.info(
            f"Google Gemini API key loaded from {api_key_source}. "
            "Provider connection has not been verified in this session."
        )
    else:
        st.warning(
            "Google Gemini API key missing. Set `GOOGLE_API_KEY` in the project-root `.env` or process "
            "environment, then restart Streamlit. Never paste the key into chat or source code."
        )

    if st.button(
        "Discover available Gemini text models",
        use_container_width=True,
        disabled=not api_key,
    ):
        try:
            with st.spinner("Listing models available to this API key..."):
                discovered_models = list_gemini_text_models(api_key)
                st.session_state["gemini_text_models"] = discovered_models
            if st.session_state.get("gemini_model_selector") not in discovered_models:
                st.session_state.pop("gemini_model_selector", None)
            st.session_state.pop("gemini_model_discovery_error", None)
        except LLMServiceError as exc:
            st.session_state["gemini_text_models"] = ()
            st.session_state["gemini_model_discovery_error"] = str(exc)

    discovery_error = st.session_state.get("gemini_model_discovery_error")
    if discovery_error:
        st.error(discovery_error)
    available_models = st.session_state.get("gemini_text_models", ()) if api_key else ()
    if available_models:
        configured_model = GEMINI_MODEL.removeprefix("models/")
        configured_match = next(
            (
                index for index, name in enumerate(available_models)
                if name.removeprefix("models/") == configured_model
            ),
            0,
        )
        selected_model = st.selectbox(
            "Available text-generation model",
            available_models,
            index=configured_match,
            help="These exact model identifiers were returned by Google for this API key.",
        )
        previous_model = st.session_state.get("selected_gemini_model")
        if previous_model != selected_model:
            st.session_state.pop("provider_test", None)
        st.session_state["selected_gemini_model"] = selected_model
    else:
        st.caption(
            f"Configured model: `{GEMINI_MODEL}`. Discover available models to choose one "
            "returned by Google."
        )
        if "gemini_text_models" in st.session_state:
            st.warning("Google returned no eligible text-generation models for this key.")

    if st.button("Test Google Gemini connection", use_container_width=True):
        if not api_key:
            st.session_state["provider_test"] = (
                "error",
                "API key missing. Add GOOGLE_API_KEY to the local .env file or process environment.",
            )
        else:
            try:
                with st.spinner("Checking Google Gemini connectivity..."):
                    message = test_gemini_connection(api_key, selected_model)
                st.session_state["provider_test"] = ("success", message)
            except LLMServiceError as exc:
                st.session_state["provider_test"] = ("error", str(exc))
    provider_test = st.session_state.get("provider_test")
    if provider_test:
        if provider_test[0] == "success":
            st.success(provider_test[1])
        else:
            st.error(provider_test[1])

    default_mode = "AI" if api_key else "Demo"
    mode = st.radio(
        "Select query mode",
        ("AI", "Demo"),
        index=("AI", "Demo").index(default_mode),
        horizontal=True,
        help="Demo Mode uses local, rule-based query patterns. AI Mode calculates supported results with Pandas, then sends a compact profile and verified aggregates to Google Gemini for explanation.",
    )
    if mode == "Demo":
        st.info("Demo Mode is rule-based; answers are limited to supported patterns.")

try:
    data = infer_dataset_types(load_dataset(dataset_name))
    schema = {
        str(column): (
            "INTEGER"
            if pd.api.types.is_bool_dtype(data[column])
            or pd.api.types.is_integer_dtype(data[column])
            else "REAL"
            if pd.api.types.is_numeric_dtype(data[column])
            else "TEXT"
        )
        for column in data.columns
    }
except (OSError, sqlite3.Error, ValueError):
    logger.exception("Could not read the selected dataset.")
    st.error("The selected dataset could not be read.")
    st.stop()

st.subheader(dataset_display_name(dataset_name))
st.caption(f"{len(data):,} rows · {len(data.columns):,} columns")
profile = build_dataset_profile(data)
numeric_columns = list(data.select_dtypes(include="number").columns)
date_columns = list(data.select_dtypes(include=["datetime", "datetimetz"]).columns)
categorical_columns = [
    str(column)
    for column in data.columns
    if column not in numeric_columns and column not in date_columns
]
profile_columns = st.columns(4)
profile_columns[0].metric("Rows", f"{len(data):,}")
profile_columns[1].metric("Columns", f"{len(data.columns):,}")
profile_columns[2].metric("Duplicate rows", f"{profile['duplicate_rows']:,}")
profile_columns[3].metric(
    "Missing values",
    f"{sum(item['missing_count'] for item in profile['columns']):,}",
)

try:
    dashboard_charts = build_dashboard_charts(data)
except (KeyError, TypeError, ValueError):
    logger.exception("Could not build the dashboard charts for the selected dataset.")
    dashboard_charts = []
if dashboard_charts:
    st.subheader("Dataset insights")
    chart_columns = st.columns(len(dashboard_charts))
    for column, (chart_title, chart) in zip(chart_columns, dashboard_charts):
        with column:
            st.markdown(f"**{chart_title}**")
            try:
                st.plotly_chart(chart, width="stretch", key=f"dashboard_{chart_title}")
            except (ImportError, TypeError, ValueError):
                logger.exception("Streamlit could not render a dashboard chart.")
                st.info(f"{chart_title} is available in the dataset but could not be rendered.")

with st.expander("Dataset profile and preview", expanded=True):
    left, right = st.columns([1, 2])
    with left:
        st.markdown("**Detected column types**")
        show_dataframe(
            pd.DataFrame(
                {
                    "Column": data.columns,
                    "Type": [str(data[column].dtype) for column in data.columns],
                    "Missing": [int(data[column].isna().sum()) for column in data.columns],
                    "Missing %": [
                        round(float(data[column].isna().mean() * 100), 2)
                        if len(data) else 0
                        for column in data.columns
                    ],
                    "Distinct": [int(data[column].nunique(dropna=True)) for column in data.columns],
                }
            ),
            hide_index=True,
            width="stretch",
        )
        st.caption(f"Numeric: {', '.join(numeric_columns) or 'none'}")
        st.caption(f"Categorical: {', '.join(categorical_columns) or 'none'}")
        st.caption(f"Date/time: {', '.join(date_columns) or 'none'}")
        if profile["date_ranges"]:
            st.markdown("**Date ranges**")
            show_dataframe(
                pd.DataFrame(
                    [
                        {"Column": column, **limits}
                        for column, limits in profile["date_ranges"].items()
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
        if profile["category_frequencies"]:
            st.markdown("**Frequent category values**")
            show_dataframe(
                pd.DataFrame(
                    [
                        {"Column": column, **item}
                        for column, values in profile["category_frequencies"].items()
                        for item in values
                    ]
                ),
                hide_index=True,
                width="stretch",
            )
    with right:
        st.markdown("**Data preview**")
        preview_page_size = 20
        preview_page_count = max(1, (len(data) + preview_page_size - 1) // preview_page_size)
        preview_page = st.number_input(
            "Preview page",
            min_value=1,
            max_value=preview_page_count,
            value=1,
            step=1,
            key=f"preview_page_{dataset_name}",
        )
        preview_start = (int(preview_page) - 1) * preview_page_size
        show_dataframe(
            data.iloc[preview_start : preview_start + preview_page_size],
            width="stretch",
            hide_index=True,
        )
        if numeric_columns:
            st.markdown("**Numeric descriptive statistics**")
            statistics = (
                data[numeric_columns]
                .describe()
                .T
                .rename_axis("Metric")
                .reset_index()
                .round(2)
            )
            show_dataframe(statistics, width="stretch", hide_index=True)

questions = list(suggest_questions(schema))
for column in categorical_columns:
    sample_values = data[column].dropna().astype(str).drop_duplicates().head(2).tolist()
    if sample_values:
        sample = ", ".join(f"“{value[:32]}”" for value in sample_values)
        questions.append(f"How many records belong to each {column}? Example values: {sample}.")
        break

with st.sidebar:
    st.subheader("Suggested questions")
    for index, example in enumerate(questions[:4]):
        if st.button(example, key=f"suggested_{dataset_name}_{index}", use_container_width=True):
            st.session_state["question_input"] = example
            st.rerun()

st.subheader("Ask your data")
with st.form("question_form"):
    question = st.text_input(
        "Business question",
        key="question_input",
        placeholder="For example: Which department has the highest average salary?",
        max_chars=1_000,
    )
    submitted = st.form_submit_button("Ask Copilot", type="primary")

if submitted:
    st.session_state.pop("last_query", None)
    st.session_state.pop("query_error", None)
    if not question.strip():
        st.session_state["query_error"] = "Enter a business question first."
    elif mode == "AI":
        started = time.monotonic()
        try:
            analysis = analyze_dataset_question(question, data)
            if analysis.clarification:
                plan = QueryPlan(
                    sql="",
                    explanation="",
                    assumptions=(),
                    mode="ai",
                    clarification=analysis.clarification,
                )
                result = None
                validation_status = "needs clarification"
                warning = None
            else:
                warning = None
                explanation = analysis.verified_summary
                if api_key:
                    try:
                        explanation = explain_verified_analysis(
                            question=question,
                            profile=profile,
                            verified_summary=analysis.verified_summary,
                            result_frame=analysis.dataframe,
                            api_key=api_key,
                            model=selected_model,
                        )
                    except LLMServiceError as exc:
                        warning = str(exc)
                else:
                    warning = (
                        "Gemini is not configured; showing the verified local calculation without "
                        "an AI explanation."
                    )
                plan = QueryPlan(
                    sql="",
                    explanation=explanation,
                    assumptions=(analysis.verified_summary,),
                    visualization=analysis.visualization,
                    mode="ai",
                )
                elapsed = time.monotonic() - started
                result = QueryResult(
                    sql="",
                    dataframe=analysis.dataframe,
                    row_count=len(analysis.dataframe),
                    duration_seconds=elapsed,
                )
                validation_status = "calculated locally"
            st.session_state["last_query"] = {
                "dataset": dataset_name,
                "question": question,
                "plan": plan,
                "sql": "",
                "validation_status": validation_status,
                "validation_error": None,
                "result": result,
                "analysis_operation": analysis.operation,
                "analysis_title": analysis.title,
                "analysis_warning": warning,
            }
            history = st.session_state.get("query_history")
            if not isinstance(history, dict):
                history = {}
                st.session_state["query_history"] = history
            dataset_history = history.setdefault(dataset_name, [])
            dataset_history.insert(
                0,
                create_audit_record(
                    question,
                    None,
                    validation_status,
                    "success" if result is not None else "clarification",
                    time.monotonic() - started,
                    result.row_count if result is not None else 0,
                ),
            )
            history[dataset_name] = dataset_history[:30]
        except (LLMServiceError, ValueError) as exc:
            st.session_state["query_error"] = str(exc)
    else:
        try:
            plan = None
            validated = None
            validation_error = None
            attempts = 1
            with st.spinner("Generating and validating a read-only SQL query..."):
                for attempt in range(attempts):
                    plan = generate_query_plan(
                        question=question,
                        table_name=dataset_name,
                        mode=mode.lower(),
                        schema=schema,
                        feedback=str(validation_error) if validation_error else None,
                        categorical_values=_categorical_values(data, categorical_columns),
                    )
                    if not plan.sql:
                        st.session_state["last_query"] = {
                            "dataset": dataset_name,
                            "question": question,
                            "plan": plan,
                            "sql": "",
                            "validation_status": "not generated",
                            "validation_error": None,
                            "result": None,
                        }
                        break
                    try:
                        validated = validate_sql(plan.sql, dataset_name, schema=schema)
                        validation_error = None
                        break
                    except SQLValidationError as exc:
                        validation_error = exc
                        if attempt + 1 == attempts:
                            st.session_state["last_query"] = {
                                "dataset": dataset_name,
                                "question": question,
                                "plan": plan,
                                "sql": plan.sql,
                                "validation_status": "rejected",
                                "validation_error": str(exc),
                                "result": None,
                            }
                if validated is not None and plan is not None:
                    result = execute_query(validated.sql, dataset_name)
                    st.session_state["last_query"] = {
                        "dataset": dataset_name,
                        "question": question,
                        "plan": plan,
                        "sql": result.sql,
                        "validation_status": "passed",
                        "validation_error": None,
                        "result": result,
                    }
                    history = st.session_state.get("query_history")
                    if not isinstance(history, dict):
                        history = {}
                        st.session_state["query_history"] = history
                    dataset_history = history.setdefault(dataset_name, [])
                    dataset_history.insert(
                        0,
                        create_audit_record(
                            question,
                            result.sql,
                            "passed",
                            result.status,
                            result.duration_seconds,
                            result.row_count,
                        ),
                    )
                    history[dataset_name] = dataset_history[:30]
        except LLMServiceError as exc:
            st.session_state["query_error"] = str(exc)

if st.session_state.get("query_error"):
    st.error(st.session_state["query_error"])

last_query = st.session_state.get("last_query")
if last_query and last_query.get("dataset") == dataset_name:
    plan = last_query["plan"]
    result = last_query["result"]
    st.divider()
    st.subheader("Answer")
    st.caption(f"Generation mode: {plan.mode.upper()} · Dataset: {dataset_display_name(dataset_name)}")
    if plan.clarification:
        st.info(plan.clarification)
    if plan.explanation:
        st.write(plan.explanation)
    if last_query.get("analysis_warning"):
        st.warning(
            f"Local analysis completed, but Gemini could not explain it: "
            f"{last_query['analysis_warning']}"
        )
    if last_query["validation_status"] == "calculated locally":
        st.success(
            f"Verified local Pandas calculation: {last_query['analysis_title']} "
            f"({last_query['analysis_operation']})."
        )
    elif last_query["validation_status"] == "passed":
        st.success("SQL validation passed: one read-only query restricted to the active dataset.")
    elif last_query["validation_status"] == "rejected":
        st.error(f"SQL validation rejected the query: {last_query['validation_error']}")
    if last_query["sql"]:
        with st.expander("Generated SQL"):
            st.code(last_query["sql"], language="sql")
    if result is not None:
        if result.status == "success":
            st.success(f"Query executed successfully in {result.duration_seconds:.3f}s.")
            st.markdown("**Observed result**")
            st.write(summarize_results(result.dataframe))
            if result.dataframe.empty:
                st.info("The approved query returned no rows.")
            else:
                show_dataframe(result.dataframe, width="stretch", hide_index=True)
                try:
                    chart = build_chart(result.dataframe, last_query["question"], plan.visualization)
                except (KeyError, TypeError, ValueError):
                    logger.exception("Could not build a chart for the query results.")
                    chart = None
                if chart is not None:
                    st.subheader("Visualization")
                    try:
                        st.plotly_chart(chart, width="stretch")
                    except (ImportError, TypeError, ValueError):
                        logger.exception("Streamlit could not render the result visualization.")
                        st.info("The query results are available, but the chart could not be rendered.")
        else:
            st.error(result.error or "Query execution failed.")

with st.expander("Query history for this dataset"):
    history = st.session_state.get("query_history", {}).get(dataset_name, [])
    if not history:
        st.caption("Queries run against this dataset during the current browser session appear here.")
    else:
        for item in history:
            st.markdown(
                f"**{item['timestamp']}** — {item['question']}  \n"
                f"Validation: {item['validation_status']} · Execution: {item['execution_status']} · "
                f"Rows: {item['row_count']} · Duration: {item['duration_seconds']:.3f}s"
            )
            if item["sql"]:
                st.code(item["sql"], language="sql")
