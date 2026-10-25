import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv

BASE_DIR = Path(__file__).resolve().parent
_GOOGLE_API_KEY_WAS_SET_BEFORE_DOTENV = bool(
    os.getenv("GOOGLE_API_KEY", "").strip()
)
load_dotenv(BASE_DIR / ".env", override=False)
DATABASE_PATH = Path(os.getenv("BI_DATABASE_PATH", BASE_DIR / "business.db")).resolve()
MAX_CSV_BYTES = 20 * 1024 * 1024
MAX_CSV_ROWS = 100_000
MAX_RESULT_ROWS = int(os.getenv("BI_MAX_RESULT_ROWS", "1000"))
if not 1 <= MAX_RESULT_ROWS <= 10_000:
    raise ValueError("BI_MAX_RESULT_ROWS must be between 1 and 10000.")
QUERY_TIMEOUT_SECONDS = 2.0
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash").strip() or "gemini-3.6-flash"
_GOOGLE_API_KEY_PLACEHOLDERS = {
    "your-google-api-key",
    "your-google-ai-studio-api-key",
    "replace-with-your-api-key",
}


def _normalize_api_key(value: str) -> str:
    key = value.strip()
    if len(key) >= 2 and key[0] in {"'", '"'} and key[-1] == key[0]:
        return key[1:-1].strip()
    return key


def get_google_api_key() -> str:
    key = _normalize_api_key(os.getenv("GOOGLE_API_KEY", ""))
    if key and key.lower() not in _GOOGLE_API_KEY_PLACEHOLDERS:
        return key

    try:
        import streamlit as st
    except ImportError:
        return ""
    try:
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return ""
    try:
        return _normalize_api_key(str(st.secrets.get("GOOGLE_API_KEY", "")))
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return ""


def get_google_api_key_source() -> str:
    key = _normalize_api_key(os.getenv("GOOGLE_API_KEY", ""))
    if key and key.lower() not in _GOOGLE_API_KEY_PLACEHOLDERS:
        dotenv_key = _normalize_api_key(
            str(dotenv_values(BASE_DIR / ".env").get("GOOGLE_API_KEY") or "")
        )
        if _GOOGLE_API_KEY_WAS_SET_BEFORE_DOTENV or key != dotenv_key:
            return "process environment"
        return "project-root .env"

    try:
        import streamlit as st
    except ImportError:
        return "not configured"
    try:
        from streamlit.errors import StreamlitSecretNotFoundError
    except ImportError:
        return "not configured"
    try:
        secret = _normalize_api_key(str(st.secrets.get("GOOGLE_API_KEY", "")))
    except (FileNotFoundError, StreamlitSecretNotFoundError):
        return "not configured"
    if secret and secret.lower() not in _GOOGLE_API_KEY_PLACEHOLDERS:
        return "Streamlit secrets"
    return "not configured"
