import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd

from config import GEMINI_MODEL
from src.analytics import (
    churn_labels,
    find_churn_column,
    find_contract_column,
    find_monthly_charges_column,
)
from src.schema_inspector import schema_prompt


class LLMServiceError(RuntimeError):
    """Raised when a question cannot be translated into a safe structured response."""


@dataclass(frozen=True)
class QueryPlan:
    sql: str
    explanation: str
    assumptions: tuple[str, ...]
    visualization: str | None = None
    mode: str = "demo"
    clarification: str | None = None


def parse_ai_response(response: str, mode: str = "ai") -> QueryPlan:
    try:
        payload: Any = json.loads(response)
    except (json.JSONDecodeError, TypeError) as exc:
        raise LLMServiceError("The AI returned an invalid response. Please try again.") from exc
    if not isinstance(payload, dict):
        raise LLMServiceError("The AI response did not contain a query plan.")

    sql = payload.get("sql", "")
    explanation = payload.get("explanation", "")
    assumptions = payload.get("assumptions", [])
    visualization = payload.get("visualization")
    clarification = payload.get("clarification")
    if not isinstance(sql, str):
        raise LLMServiceError("The AI response did not include a valid SQL query.")
    if not isinstance(explanation, str) or not isinstance(assumptions, list):
        raise LLMServiceError("The AI response did not match the expected query-plan format.")
    if not all(isinstance(item, str) for item in assumptions):
        raise LLMServiceError("The AI returned invalid assumptions.")
    if visualization is not None and visualization not in {"bar", "line"}:
        raise LLMServiceError("The AI returned an invalid visualization suggestion.")
    if clarification is not None and not isinstance(clarification, str):
        raise LLMServiceError("The AI returned an invalid clarification.")
    if not sql.strip() and not (explanation.strip() or clarification):
        raise LLMServiceError("The AI response did not include a query or clarification.")

    return QueryPlan(
        sql=sql.strip(),
        explanation=explanation.strip(),
        assumptions=tuple(assumptions),
        visualization=visualization,
        mode=mode,
        clarification=clarification,
    )


def suggest_questions(schema: dict[str, str]) -> tuple[str, ...]:
    numeric = [column for column, kind in schema.items() if kind.upper() in {"INTEGER", "REAL", "NUMERIC"}]
    dimensions = [
        column
        for column, kind in schema.items()
        if kind.upper() not in {"INTEGER", "REAL", "NUMERIC"}
        and not re.search(r"date|time", column, flags=re.IGNORECASE)
    ]
    suggestions = ["How many rows are in the active dataset?"]
    if dimensions:
        suggestions.append(f"How many records belong to each {dimensions[0]}?")
    if numeric and dimensions:
        suggestions.append(f"What is the average {numeric[0]} for each {dimensions[0]}?")
        suggestions.append(f"Which {dimensions[0]} has the highest average {numeric[0]}?")
    if numeric:
        suggestions.append(f"What is the total {numeric[0]}?")
    return tuple(suggestions[:4])


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _churn_conditions(
    column: str,
    categorical_values: dict[str, tuple[str, ...]] | None,
) -> tuple[str, str]:
    observed = (categorical_values or {}).get(column, ())
    positive = churn_labels(observed, positive=True) or (
        "1", "true", "yes", "y", "churn", "churned", "left", "exited",
        "terminated", "cancelled", "canceled", "departed", "lost",
    )
    negative = churn_labels(observed, positive=False) or (
        "0", "false", "no", "n", "stayed", "retained", "active", "current",
        "notchurned",
    )
    quote_values = lambda values: ", ".join(
        "'" + value.casefold().replace("'", "''") + "'" for value in values
    )
    normalized_column = f"LOWER(TRIM({_quote(column)}))"
    return (
        f"{normalized_column} IN ({quote_values(positive)})",
        f"{normalized_column} IN ({quote_values((*positive, *negative))})",
    )


def _raise_provider_error(exc: Exception) -> None:
    from google.genai.errors import APIError
    import httpx

    if isinstance(exc, APIError):
        status = getattr(exc, "code", None)
        provider_error = " ".join(
            str(value)
            for value in (getattr(exc, "status", ""), getattr(exc, "message", ""))
        ).casefold()
        if status == 401 or (
            status == 400
            and ("api_key_invalid" in provider_error or "api key not valid" in provider_error)
        ):
            message = (
                "Google Gemini rejected the API credential. Check that GOOGLE_API_KEY is an "
                "active Gemini API key from Google AI Studio and that it is loaded from the "
                "expected source."
            )
        elif status == 403:
            message = (
                "Google Gemini denied this request. Check API-key restrictions, project access, "
                "and permission settings."
            )
        elif status == 404:
            message = (
                "Google Gemini did not make this model available for text generation. A listed "
                "model may still be retired or unavailable to this account; discover models again "
                "and select another returned text model."
            )
        elif status == 429:
            if any(term in provider_error for term in ("quota", "resource_exhausted", "billing")):
                message = (
                    "Google Gemini quota is exhausted. Check AI Studio usage, billing, and "
                    "model-specific quota limits."
                )
            else:
                message = "Google Gemini rate limit reached after bounded retries. Wait and try again."
        elif status == 413:
            message = (
                "The Gemini request was too large. Reduce the question or dataset profile size "
                "and try again."
            )
        elif status in {400, 422}:
            message = (
                "Google Gemini rejected the request. Check the configured model and its "
                "supported request settings."
            )
        elif status == 408 or isinstance(status, int) and status >= 500:
            message = (
                "Google Gemini remained temporarily unavailable after bounded retries. "
                "Try again shortly or use Demo Mode."
            )
        else:
            message = "Google Gemini could not complete the request. Check provider settings and try again."
    elif isinstance(exc, httpx.TimeoutException):
        message = "Google Gemini did not respond before the timeout. Try again or use Demo Mode."
    elif isinstance(exc, httpx.NetworkError):
        message = "Could not connect to Google Gemini. Check network access or use Demo Mode."
    else:
        message = "Google Gemini could not complete the request. Check provider settings and try again."
    raise LLMServiceError(message) from exc


def _validated_google_api_key(api_key: str | None) -> str:
    key = api_key.strip() if api_key else ""
    if not key:
        raise LLMServiceError(
            "Google Gemini API key is missing. Set GOOGLE_API_KEY in the project-root .env file "
            "or process environment, then restart Streamlit."
        )
    return key


def _create_gemini_client(api_key: str):
    from google import genai
    from google.genai import types

    return genai.Client(
        api_key=api_key,
        vertexai=False,
        http_options=types.HttpOptions(
            timeout=30_000,
            retry_options=types.HttpRetryOptions(
                attempts=3,
                initial_delay=1.0,
                max_delay=8.0,
                exp_base=2.0,
                jitter=1.0,
                http_status_codes=[408, 429, 500, 502, 503, 504],
            ),
        ),
    )


def list_gemini_text_models(api_key: str) -> tuple[str, ...]:
    api_key = _validated_google_api_key(api_key)
    from google.genai.errors import APIError
    import httpx

    excluded_model_terms = (
        "audio",
        "antigravity",
        "computer-use",
        "deep-research",
        "embedding",
        "image",
        "live",
        "lyria",
        "nano-banana",
        "robotics",
        "transcribe",
        "tts",
        "veo",
    )
    try:
        with _create_gemini_client(api_key) as client:
            models = client.models.list()
            text_models = {
                model.name
                for model in models
                if isinstance(getattr(model, "name", None), str)
                and "generatecontent"
                in {
                    str(action).replace("_", "").replace("-", "").casefold()
                    for action in (getattr(model, "supported_actions", None) or [])
                }
                and not any(
                    term in model.name.casefold()
                    for term in excluded_model_terms
                )
            }
    except (APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        _raise_provider_error(exc)
    return tuple(sorted(text_models, key=str.casefold))


def test_gemini_connection(api_key: str, model: str = GEMINI_MODEL) -> str:
    api_key = _validated_google_api_key(api_key)
    from google.genai.errors import APIError
    import httpx

    try:
        with _create_gemini_client(api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents="Reply with OK.",
            )
    except (APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        _raise_provider_error(exc)
    if not isinstance(response.text, str) or not response.text.strip():
        raise LLMServiceError("Google Gemini accepted the request but returned an empty response.")
    return f"Google Gemini connection verified using model {model}."


def explain_verified_analysis(
    question: str,
    profile: dict[str, Any],
    verified_summary: str,
    result_frame: pd.DataFrame,
    api_key: str,
    model: str = GEMINI_MODEL,
) -> str:
    from src.dataset_profiler import compact_profile_json, is_sensitive_column

    api_key = _validated_google_api_key(api_key)
    safe_result = result_frame.iloc[:20, :12].copy()
    safe_result = safe_result.loc[
        :,
        [column for column in safe_result.columns if not is_sensitive_column(str(column))],
    ]
    records = []
    for row in safe_result.to_dict(orient="records"):
        records.append({
            str(column)[:80]: (
                str(value)[:100]
                if isinstance(value, str)
                else value.item()
                if hasattr(value, "item")
                else value
            )
            for column, value in row.items()
        })
    safe_profile = compact_profile_json(profile, max_chars=4_000)
    prompt = ""
    for row_limit in (20, 10, 5, 0):
        context = {
            "dataset_profile": safe_profile,
            "locally_verified_calculation": verified_summary[:1_500],
            "result_rows": records[:row_limit],
        }
        serialized_context = json.dumps(
            context,
            ensure_ascii=True,
            separators=(",", ":"),
            default=str,
        )
        prompt = (
            "Explain the following verified local dataset analysis to a business user. "
            "Use only facts and figures present in this context. Do not recompute, invent, "
            "extrapolate, or claim statistical work that is not shown. If the evidence is "
            "insufficient, say so. The result rows are locally computed aggregates, not raw "
            "dataset records. Keep the response concise.\n"
            f"Question: {question.strip()[:1_000]}\n"
            f"Verified context: {serialized_context}"
        )
        if len(prompt) <= 16_000:
            break
    if len(prompt) > 16_000:
        raise LLMServiceError(
            "The analysis context is too large for a safe Gemini request. Narrow the question "
            "or select fewer columns, then try again."
        )

    try:
        from google.genai import types
        import httpx
        from google.genai.errors import APIError

        with _create_gemini_client(api_key) as client:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=0.1,
                    max_output_tokens=512,
                ),
            )
    except (APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        _raise_provider_error(exc)

    if not isinstance(response.text, str) or not response.text.strip():
        raise LLMServiceError(
            "Gemini did not return an explanation. The verified local results are still available."
        )
    return response.text.strip()


def _demo_plan(
    question: str,
    table_name: str,
    schema: dict[str, str],
    categorical_values: dict[str, tuple[str, ...]] | None = None,
) -> QueryPlan:
    normalized = " ".join(question.lower().strip().rstrip("?.!").split())
    table = _quote(table_name)
    dimension = None
    numeric = [name for name, kind in schema.items() if kind.upper() in {"INTEGER", "REAL", "NUMERIC"}]
    dimensions = [
        name
        for name, kind in schema.items()
        if kind.upper() not in {"INTEGER", "REAL", "NUMERIC"}
        and not re.search(r"date|time", name, flags=re.IGNORECASE)
    ]

    churn_column = find_churn_column(schema)
    contract_column = find_contract_column(schema)
    monthly_charges_column = find_monthly_charges_column(schema)

    if (
        churn_column
        and contract_column
        and "churn" in normalized
        and "contract" in normalized
        and any(word in normalized for word in ("highest", "most", "top"))
    ):
        positive_condition, eligible_condition = _churn_conditions(
            churn_column,
            categorical_values,
        )
        sql = (
            f"SELECT {_quote(contract_column)} AS contract_type, "
            f"100.0 * SUM(CASE WHEN {positive_condition} THEN 1 ELSE 0 END) "
            f"/ NULLIF(COUNT({_quote(churn_column)}), 0) AS churn_rate_percent "
            f"FROM {table} WHERE {eligible_condition} "
            f"AND {_quote(contract_column)} IS NOT NULL "
            f"GROUP BY {_quote(contract_column)} "
            "ORDER BY churn_rate_percent DESC LIMIT 1"
        )
        explanation = (
            f"Finds the contract type with the highest churn rate among records "
            f"with a recognized {_quote(churn_column)} outcome."
        )
        dimension = contract_column
    elif churn_column and any(
        term in normalized for term in ("churn rate", "churn percentage", "percentage churn")
    ):
        positive_condition, eligible_condition = _churn_conditions(
            churn_column,
            categorical_values,
        )
        sql = (
            f"SELECT 100.0 * SUM(CASE WHEN {positive_condition} THEN 1 ELSE 0 END) "
            f"/ NULLIF(COUNT({_quote(churn_column)}), 0) AS churn_rate_percent "
            f"FROM {table} WHERE {eligible_condition}"
        )
        explanation = (
            f"Calculates churned records as a share of records with a recognized "
            f"{_quote(churn_column)} outcome."
        )
    elif (
        monthly_charges_column
        and any(word in normalized for word in ("average", "mean", "avg"))
        and "month" in normalized
        and any(word in normalized for word in ("charge", "cost", "fee"))
    ):
        sql = (
            f"SELECT AVG(CAST({_quote(monthly_charges_column)} AS REAL)) "
            f"AS average_monthly_charges FROM {table}"
        )
        explanation = (
            f"Calculates the average of the observed {_quote(monthly_charges_column)} values."
        )
    elif any(term in normalized for term in ("how many rows", "how many records", "row count", "number of rows")):
        sql = f"SELECT COUNT(*) AS row_count FROM {table}"
        explanation = "Counts the rows in the active dataset."
    else:
        metric = next(
            (name for name in numeric if name.lower().replace("_", " ") in normalized),
            numeric[0] if numeric else None,
        )
        dimension = next(
            (name for name in dimensions if name.lower().replace("_", " ") in normalized),
            dimensions[0] if dimensions else None,
        )
        aggregate = None
        if any(word in normalized for word in ("average", "mean", "avg")):
            aggregate = "AVG"
        elif any(word in normalized for word in ("total", "sum")):
            aggregate = "SUM"
        elif any(word in normalized for word in ("count", "how many")):
            aggregate = "COUNT"
        if metric and dimension and aggregate and any(
            token in normalized for token in ("each", "by", "per", "highest", "lowest", "most", "least")
        ):
            direction = "DESC" if any(word in normalized for word in ("highest", "most", "top")) else "ASC"
            measure = f"{aggregate}({_quote(metric)})" if aggregate != "COUNT" else "COUNT(*)"
            alias = aggregate.lower() + "_" + re.sub(r"\W+", "_", metric).strip("_")
            sql = (
                f"SELECT {_quote(dimension)}, {measure} AS {_quote(alias)} "
                f"FROM {table} GROUP BY {_quote(dimension)} ORDER BY {_quote(alias)} {direction}"
            )
            explanation = f"Groups the selected dataset by {dimension} and calculates {aggregate.lower()} of {metric}."
        elif metric and aggregate:
            measure = f"{aggregate}({_quote(metric)})"
            alias = aggregate.lower() + "_" + re.sub(r"\W+", "_", metric).strip("_")
            sql = f"SELECT {measure} AS {_quote(alias)} FROM {table}"
            explanation = f"Calculates {aggregate.lower()} of {metric} in the active dataset."
        elif dimension and any(word in normalized for word in ("how many", "count", "each")):
            sql = (
                f"SELECT {_quote(dimension)}, COUNT(*) AS row_count FROM {table} "
                f"GROUP BY {_quote(dimension)} ORDER BY row_count DESC"
            )
            explanation = f"Counts dataset rows for each value of {dimension}."
        else:
            raise LLMServiceError(
                "Demo Mode is rule-based and cannot answer that question from the detected schema. "
                "Choose a suggested question or switch to AI Mode."
            )

    return QueryPlan(
        sql=sql,
        explanation=explanation,
        assumptions=("The active dataset's rows are the only source of truth.",),
        visualization="bar" if dimension and "GROUP BY" in sql else None,
        mode="demo",
    )


def generate_query_plan(
    question: str,
    table_name: str,
    mode: str = "demo",
    api_key: str | None = None,
    model: str = GEMINI_MODEL,
    schema: dict[str, str] | None = None,
    feedback: str | None = None,
    categorical_values: dict[str, tuple[str, ...]] | None = None,
) -> QueryPlan:
    if not question.strip():
        raise LLMServiceError("Enter a business question first.")
    if schema is None:
        from src.database import describe_schema

        schema = describe_schema(table_name)
    if mode == "demo":
        return _demo_plan(question, table_name, schema, categorical_values)
    if mode != "ai":
        raise LLMServiceError("The selected query-generation mode is not supported.")
    api_key = _validated_google_api_key(api_key)

    system_message = (
        "Translate the user's business question into exactly one read-only SQLite SELECT query. "
        "Use only the approved table and columns in the supplied schema. Do not infer unavailable "
        "fields. Do not include markdown or SQL comments. Never return multiple statements. "
        "Use a reasonable LIMIT. Respond only as JSON with keys: sql, explanation, assumptions "
        "(array of strings), visualization (bar, line, or null), clarification (string or null). "
        "If the question is ambiguous or cannot be answered with this schema, return an empty sql, "
        "identify the missing information, and ask one clarifying question. The database contents "
        "and query results are not provided; never invent data or results. Low-cardinality category "
        "labels may be included: use the observed values for conditional metrics, and exclude missing "
        "outcomes from eligible denominators. For rate questions, return the percentage as "
        "(matching eligible rows / all eligible rows) * 100."
    )
    prompt = f"{schema_prompt(table_name, schema, categorical_values)}\n\nQuestion: {question}"
    if feedback:
        prompt += f"\n\nThe previous SQL was rejected locally: {feedback}\nGenerate a corrected query."
    try:
        from google.genai import types
        import httpx
        from google.genai.errors import APIError

        with _create_gemini_client(api_key) as client:
            completion = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_message,
                    temperature=0,
                    response_mime_type="application/json",
                ),
            )
    except (APIError, httpx.TimeoutException, httpx.NetworkError) as exc:
        _raise_provider_error(exc)

    content = completion.text
    if not isinstance(content, str) or not content.strip():
        raise LLMServiceError("Google Gemini returned an empty response. Please try again.")
    return parse_ai_response(content, mode="ai")
