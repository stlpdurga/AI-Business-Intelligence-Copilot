from datetime import datetime, timezone


def create_audit_record(
    question: str,
    sql: str | None,
    validation_status: str,
    execution_status: str,
    duration_seconds: float | None,
    row_count: int,
) -> dict[str, str | float | int | None]:
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "question": question,
        "sql": sql,
        "validation_status": validation_status,
        "execution_status": execution_status,
        "duration_seconds": duration_seconds,
        "row_count": row_count,
    }
