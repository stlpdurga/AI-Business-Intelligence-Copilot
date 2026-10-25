import importlib
import sys


REQUIRED_MODULES = ("streamlit", "pandas", "plotly", "sqlglot", "google.genai", "pytest")


def main() -> int:
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    failures = []
    for module_name in REQUIRED_MODULES:
        try:
            imported = importlib.import_module(module_name)
            version = getattr(imported, "__version__", "available")
            print(f"{module_name}: {version}")
        except ImportError as exc:
            failures.append(module_name)
            print(f"{module_name}: missing ({exc})")
    if failures:
        print("Install the project dependencies with: python -m pip install -r requirements.txt")
        return 1
    print("Environment validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
