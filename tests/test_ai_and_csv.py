import io
import json

import pandas as pd
import pytest

from config import GEMINI_MODEL, BASE_DIR, get_google_api_key, get_google_api_key_source
from src.data_loader import (
    ORDERS_COLUMNS,
    generate_sample_data,
    load_tabular_data,
    validate_csv,
)
from src.llm_service import (
    LLMServiceError,
    _create_gemini_client,
    generate_query_plan,
    list_gemini_text_models,
    parse_ai_response,
    suggest_questions,
    test_gemini_connection as check_gemini_connection,
)
from src.schema_inspector import schema_prompt


ORDERS_SCHEMA = {
    "order_id": "INTEGER",
    "order_date": "TEXT",
    "product": "TEXT",
    "category": "TEXT",
    "customer_segment": "TEXT",
    "sales": "REAL",
    "profit": "REAL",
}
TEST_API_KEY = "google-test-key"


def _assert_gemini_client_settings(kwargs):
    assert kwargs["api_key"] == TEST_API_KEY
    assert kwargs["vertexai"] is False
    options = kwargs["http_options"]
    assert options.timeout == 30_000
    retries = options.retry_options
    assert retries.attempts == 3
    assert retries.http_status_codes == [408, 429, 500, 502, 503, 504]


def test_google_key_is_trimmed_and_unquoted_from_environment(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", '  "google-test-key"  ')

    assert get_google_api_key() == "google-test-key"
    assert get_google_api_key_source() == "process environment"


def test_google_key_placeholder_is_not_treated_as_configured(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "your-google-api-key")

    assert get_google_api_key() == ""


def test_project_root_dotenv_path_is_absolute():
    assert BASE_DIR.is_absolute()
    assert (BASE_DIR / ".env").parent == BASE_DIR


def test_gemini_client_uses_explicit_developer_api_key(monkeypatch):
    from google import genai

    arguments = {}

    class FakeClient:
        def __init__(self, **kwargs):
            arguments.update(kwargs)

    monkeypatch.setattr(genai, "Client", FakeClient)
    _create_gemini_client(TEST_API_KEY)

    _assert_gemini_client_settings(arguments)


def test_parses_structured_ai_response():
    plan = parse_ai_response(
        json.dumps(
            {
                "sql": 'SELECT SUM("sales") AS total_sales FROM "orders"',
                "explanation": "Totals sales.",
                "assumptions": ["Use the selected dataset."],
                "visualization": None,
            }
        )
    )

    assert plan.sql.startswith("SELECT")
    assert plan.assumptions == ("Use the selected dataset.",)


def test_rejects_malformed_ai_response():
    with pytest.raises(LLMServiceError):
        parse_ai_response("not json")
    with pytest.raises(LLMServiceError):
        parse_ai_response('{"sql": 4, "explanation": "x", "assumptions": []}')


def test_parses_ai_clarification_without_sql():
    plan = parse_ai_response(
        json.dumps(
            {
                "sql": "",
                "explanation": "The dataset has no department field.",
                "assumptions": [],
                "clarification": "Which available field should represent a department?",
            }
        )
    )
    assert not plan.sql
    assert "Which available field" in plan.clarification


def test_demo_answers_supported_patterns_without_api_key():
    plan = generate_query_plan(
        "What is the total sales?",
        "orders",
        schema=ORDERS_SCHEMA,
    )

    assert "SUM" in plan.sql
    assert plan.mode == "demo"
    assert "sales" in plan.sql


def test_demo_works_for_non_sales_schema():
    schema = {"Employee Name": "TEXT", "Department": "TEXT", "Annual Salary": "REAL"}
    plan = generate_query_plan(
        "What is the average Annual Salary by Department?",
        "dataset_aabbccddeeff0011",
        schema=schema,
    )

    assert '"Annual Salary"' in plan.sql
    assert '"Department"' in plan.sql
    assert "AVG" in plan.sql


def test_demo_handles_telco_questions_from_the_active_schema():
    schema = {
        "Churn Label": "TEXT",
        "Contract Type": "TEXT",
        "Monthly Charges": "REAL",
    }
    categories = {
        "Churn Label": ("Yes", "No"),
        "Contract Type": ("Month-to-month", "One year", "Two year"),
    }

    churn_rate = generate_query_plan(
        "What is the churn rate?",
        "customers",
        schema=schema,
        categorical_values=categories,
    )
    highest_contract = generate_query_plan(
        "Which contract type has the highest churn?",
        "customers",
        schema=schema,
        categorical_values=categories,
    )
    average_charges = generate_query_plan(
        "What is the average monthly charge?",
        "customers",
        schema=schema,
        categorical_values=categories,
    )

    assert "churn_rate_percent" in churn_rate.sql
    assert "'yes'" in churn_rate.sql
    assert "'no'" in churn_rate.sql
    assert "ORDER BY churn_rate_percent DESC LIMIT 1" in highest_contract.sql
    assert '"Contract Type"' in highest_contract.sql
    assert 'AVG(CAST("Monthly Charges" AS REAL))' in average_charges.sql


def test_suggestions_use_uploaded_schema():
    suggestions = suggest_questions(
        {"department": "TEXT", "salary": "REAL", "hire_date": "TEXT"},
    )

    assert any("department" in question.lower() for question in suggestions)
    assert any("salary" in question.lower() for question in suggestions)


def test_schema_prompt_includes_only_distinct_category_labels():
    prompt = schema_prompt(
        "customers",
        {"Churn": "TEXT", "Monthly Charges": "REAL"},
        {"Churn": ("Yes", "No")},
    )

    assert '"Churn" observed category values' in prompt
    assert "distinct values only; no rows" in prompt
    assert "'Yes', 'No'" in prompt


def test_gemini_connection_check_makes_a_real_provider_call(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs == {
                "model": GEMINI_MODEL,
                "contents": "Reply with OK.",
            }
            return type("Response", (), {"text": "OK"})()

    class FakeClient:
        def __init__(self, **kwargs):
            _assert_gemini_client_settings(kwargs)
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    assert "Google Gemini connection verified" in check_gemini_connection(TEST_API_KEY)


def test_lists_only_text_generation_models_and_keeps_exact_ids(monkeypatch):
    from google import genai

    class FakeModels:
        def list(self):
            return [
                type(
                    "Model",
                    (),
                    {
                        "name": "models/gemini-flash-latest",
                        "supported_actions": ["generateContent", "countTokens"],
                    },
                )(),
                type(
                    "Model",
                    (),
                    {
                        "name": "models/gemini-flash-image",
                        "supported_actions": ["generateContent"],
                    },
                )(),
                type(
                    "Model",
                    (),
                    {
                        "name": "models/gemini-flash-tts",
                        "supported_actions": ["generateContent"],
                    },
                )(),
                type(
                    "Model",
                    (),
                    {
                        "name": "models/text-embedding",
                        "supported_actions": ["embedContent"],
                    },
                )(),
            ]

    class FakeClient:
        def __init__(self, **kwargs):
            _assert_gemini_client_settings(kwargs)
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)

    assert list_gemini_text_models(TEST_API_KEY) == ("models/gemini-flash-latest",)


def test_connection_check_passes_exact_discovered_model_identifier(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "models/gemini-flash-latest"
            return type("Response", (), {"text": "OK"})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)

    result = check_gemini_connection(TEST_API_KEY, "models/gemini-flash-latest")

    assert "models/gemini-flash-latest" in result


def test_gemini_connection_check_sanitizes_authentication_error(monkeypatch):
    from google import genai
    from google.genai.errors import APIError

    class FakeClient:
        def __init__(self, **kwargs):
            raise APIError(401, {"message": "private credential detail"})

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(LLMServiceError, match="Google Gemini rejected") as error:
        check_gemini_connection(TEST_API_KEY)
    assert "private credential detail" not in str(error.value)


def test_demo_explains_unsupported_question():
    with pytest.raises(LLMServiceError, match="rule-based"):
        generate_query_plan("Which customer should we call?", "orders", schema=ORDERS_SCHEMA)


def test_ai_mode_requires_a_key():
    with pytest.raises(LLMServiceError, match="API key is missing"):
        generate_query_plan("What are total sales?", "orders", mode="ai", schema=ORDERS_SCHEMA)


def test_connection_handles_unavailable_model_and_quota_errors(monkeypatch):
    from google import genai
    from google.genai.errors import APIError

    class FakeModels:
        def generate_content(self, **kwargs):
            raise APIError(FakeClient.error_code, FakeClient.error_payload)

    class FakeClient:
        error_code = 404
        error_payload = {"message": "private provider detail"}

        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    failures = (
        (
            404,
            {"message": "private provider detail"},
            "did not make this model available",
        ),
        (429, {"message": "private provider detail"}, "rate limit"),
        (
            400,
            {"message": "API key not valid. Please pass a valid API key.", "status": "INVALID_ARGUMENT"},
            "rejected GOOGLE_API_KEY",
        ),
    )
    for code, payload, message in failures:
        FakeClient.error_code = code
        FakeClient.error_payload = payload
        with pytest.raises(LLMServiceError, match=message) as error:
            check_gemini_connection(TEST_API_KEY)
        assert "private" not in str(error.value)


def test_nonstandard_key_format_is_left_for_google_to_validate(monkeypatch):
    from google import genai

    credential = "provider-issued-credential"

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("Response", (), {"text": "OK"})()

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == credential
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)

    assert "connection verified" in check_gemini_connection(credential)


def test_ai_provider_failure_is_reported_without_network(monkeypatch):
    from google import genai
    from google.genai.errors import APIError

    class FakeModels:
        def generate_content(self, **kwargs):
            raise APIError(500, {"message": "simulated provider failure"})

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == TEST_API_KEY
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(LLMServiceError, match="temporarily unavailable after bounded retries"):
        generate_query_plan(
            "What are total sales?",
            "orders",
            mode="ai",
            api_key=TEST_API_KEY,
            schema=ORDERS_SCHEMA,
        )


def test_ai_uses_active_schema_and_returns_query_plan(monkeypatch):
    from google import genai

    result_json = json.dumps(
        {
            "sql": 'SELECT AVG("Annual Salary") AS average_salary '
            'FROM "dataset_aabbccddeeff0011"',
            "explanation": "Calculates the average salary.",
            "assumptions": [],
            "visualization": None,
        }
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["model"] == "models/gemini-3.8-flash"
            assert '"Annual Salary": REAL' in kwargs["contents"]
            assert "Question: What is the average salary?" in kwargs["contents"]
            assert kwargs["config"].response_mime_type == "application/json"
            assert kwargs["config"].temperature == 0
            return type("Response", (), {"text": result_json})()

    class FakeClient:
        def __init__(self, **kwargs):
            _assert_gemini_client_settings(kwargs)
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    plan = generate_query_plan(
        "What is the average salary?",
        "dataset_aabbccddeeff0011",
        mode="ai",
        api_key=TEST_API_KEY,
        model="models/gemini-3.8-flash",
        schema={"Employee Name": "TEXT", "Annual Salary": "REAL"},
    )

    assert plan.mode == "ai"
    assert "AVG" in plan.sql


def test_churn_question_receives_observed_values_for_sql_generation(monkeypatch):
    from google import genai

    result_json = json.dumps(
        {
            "sql": 'SELECT 100.0 * SUM(CASE WHEN LOWER(TRIM("Churn")) = \'yes\' '
            'THEN 1 ELSE 0 END) / NULLIF(COUNT("Churn"), 0) AS churn_rate_percent '
            'FROM "customers" WHERE "Churn" IS NOT NULL',
            "explanation": "Calculates churn among non-missing outcomes.",
            "assumptions": [],
            "visualization": None,
        }
    )

    class FakeModels:
        def generate_content(self, **kwargs):
            prompt = kwargs["contents"]
            assert '"Churn" observed category values:' in prompt
            assert "distinct values only; no rows" in prompt
            assert "'Yes', 'No'" in prompt
            return type("Response", (), {"text": result_json})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    plan = generate_query_plan(
        "What is the churn rate?",
        "customers",
        mode="ai",
        api_key=TEST_API_KEY,
        schema={"Churn": "TEXT", "Monthly Charges": "REAL"},
        categorical_values={"Churn": ("Yes", "No")},
    )

    assert "churn_rate_percent" in plan.sql


def test_ai_authentication_error_is_safe(monkeypatch):
    from google import genai
    from google.genai.errors import APIError

    class FakeModels:
        def generate_content(self, **kwargs):
            raise APIError(401, {"message": "private response detail"})

    class FakeClient:
        def __init__(self, **kwargs):
            assert kwargs["api_key"] == TEST_API_KEY
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(LLMServiceError, match="Google Gemini rejected") as error:
        generate_query_plan("Question", "orders", mode="ai", api_key=TEST_API_KEY, schema=ORDERS_SCHEMA)
    assert "private response detail" not in str(error.value)


def test_timeout_and_rate_limit_errors_are_safe(monkeypatch):
    import httpx
    from google import genai
    from google.genai.errors import APIError

    class FakeModels:
        def __init__(self, error):
            self.error = error

        def generate_content(self, **kwargs):
            raise self.error

    class FakeClient:
        failure = None

        def __init__(self, **kwargs):
            self.models = FakeModels(self.failure)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    failures = (
        (httpx.TimeoutException("private timeout"), "timeout"),
        (APIError(429, {"message": "private rate limit detail"}), "rate limit"),
    )
    for failure, expected_message in failures:
        FakeClient.failure = failure
        with pytest.raises(LLMServiceError, match=expected_message) as error:
            generate_query_plan("Question", "orders", mode="ai", api_key=TEST_API_KEY, schema=ORDERS_SCHEMA)
        assert "private" not in str(error.value)


def test_malformed_model_response_is_reported(monkeypatch):
    from google import genai

    class FakeModels:
        def generate_content(self, **kwargs):
            assert kwargs["config"].response_mime_type == "application/json"
            return type("Response", (), {"text": "not json"})()

    class FakeClient:
        def __init__(self, **kwargs):
            self.models = FakeModels()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(genai, "Client", FakeClient)
    with pytest.raises(LLMServiceError, match="invalid response"):
        generate_query_plan("Question", "orders", mode="ai", api_key=TEST_API_KEY, schema=ORDERS_SCHEMA)


def test_loads_sales_and_non_sales_csv_and_keeps_missing_values():
    sales = generate_sample_data(2)
    sales_result = validate_csv(sales.to_csv(index=False).encode("utf-8"))
    people = pd.DataFrame(
        {"Employee Name": ["Ari", "Bea"], "Department": ["HR", None], "Annual Salary": [50_000, 60_000]}
    )
    people_result = load_tabular_data(people.to_csv(index=False).encode("utf-8"), "people.csv")

    assert tuple(sales_result.columns) == ORDERS_COLUMNS
    assert list(people_result.columns) == ["Employee Name", "Department", "Annual Salary"]
    assert people_result["Department"].isna().sum() == 1


def test_duplicate_and_blank_headers_are_normalized():
    result = load_tabular_data(b"name,name,,amount\nA,B,C,10\n", "duplicates.csv")

    assert list(result.columns) == ["name", "name.1", "column_3", "amount"]


def test_invalid_empty_and_unsupported_files_are_rejected():
    for payload, name in (
        (b"", "empty.csv"),
        (b"not a dataset", "empty.csv"),
        (b"1,2\n", "unsupported.txt"),
    ):
        with pytest.raises(ValueError):
            load_tabular_data(payload, name)


def test_loads_selected_excel_worksheet():
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame({"student": ["A"], "score": [80]}).to_excel(
            writer, index=False, sheet_name="Scores"
        )
        pd.DataFrame({"student": ["B"], "score": [90]}).to_excel(
            writer, index=False, sheet_name="Other"
        )

    frame = load_tabular_data(buffer.getvalue(), "students.xlsx", "Other")

    assert frame.iloc[0]["student"] == "B"
    assert frame.iloc[0]["score"] == 90
