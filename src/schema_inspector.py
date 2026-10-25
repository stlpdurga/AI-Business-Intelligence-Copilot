from src.database import describe_schema


def schema_prompt(
    table_name: str = "orders",
    schema: dict[str, str] | None = None,
    categorical_values: dict[str, tuple[str, ...]] | None = None,
) -> str:
    columns = schema if schema is not None else describe_schema(table_name)
    rendered = "\n".join(f'- "{name}": {kind}' for name, kind in columns.items())
    value_context = ""
    if categorical_values:
        rendered_values = "\n".join(
            f'- "{column}" observed category values: {", ".join(repr(value) for value in values)}'
            for column, values in categorical_values.items()
            if values
        )
        if rendered_values:
            value_context = (
                "\nObserved low-cardinality category labels (distinct values only; no rows):\n"
                f"{rendered_values}\n"
            )
    return (
        f'Approved SQLite table: "{table_name}"\n'
        "Only these columns are available:\n"
        f"{rendered}\n"
        f"{value_context}"
        "Use only this table and these columns. Column names must be quoted when needed."
    )
