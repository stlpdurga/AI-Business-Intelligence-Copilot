from dataclasses import dataclass

import sqlglot
from sqlglot import exp
from sqlglot.errors import ParseError

from config import MAX_RESULT_ROWS
from src.database import ORDERS_SCHEMA, describe_schema

SAFE_FUNCTIONS = {
    "abs", "avg", "cast", "case", "coalesce", "count", "date", "datetime", "if", "ifnull",
    "julianday", "length", "lower", "max", "min", "nullif", "round", "strftime",
    "substr", "sum", "total", "trim", "upper",
}
MAX_AST_NODES = 200
MAX_SQL_CHARS = 10_000


class SQLValidationError(ValueError):
    """Raised when a generated query fails the SQL safety policy."""


@dataclass(frozen=True)
class ValidatedQuery:
    sql: str
    table_name: str


def validate_sql(
    sql: str,
    table_name: str,
    max_rows: int = MAX_RESULT_ROWS,
    schema: dict[str, str] | None = None,
) -> ValidatedQuery:
    try:
        allowed_columns = {
            column.casefold()
            for column in (schema if schema is not None else describe_schema(table_name))
        }
    except (OSError, ValueError):
        if table_name == "orders":
            allowed_columns = {column.casefold() for column in ORDERS_SCHEMA}
        else:
            raise SQLValidationError("The selected dataset schema is not available.")
    if not sql or not sql.strip():
        raise SQLValidationError("The generated query was empty.")
    if len(sql) > MAX_SQL_CHARS:
        raise SQLValidationError("The generated query is too long to run safely.")
    if max_rows < 1 or max_rows > MAX_RESULT_ROWS:
        raise ValueError(f"max_rows must be between 1 and {MAX_RESULT_ROWS}")

    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except ParseError as exc:
        raise SQLValidationError("The generated SQL could not be parsed safely.") from exc
    if len(statements) != 1 or statements[0] is None:
        raise SQLValidationError("Only one SQL statement is allowed.")

    query = statements[0]
    if not isinstance(query, (exp.Select, exp.Union)):
        raise SQLValidationError("Only read-only SELECT queries are allowed.")
    if any(isinstance(node, exp.Into) for node in query.walk()):
        raise SQLValidationError("SELECT INTO is not supported.")
    if len(list(query.walk())) > MAX_AST_NODES:
        raise SQLValidationError("The query is too complex to run safely.")
    if any(not select.expressions for select in query.find_all(exp.Select)):
        raise SQLValidationError("Every SELECT must return at least one expression.")

    if any(
        with_clause.args.get("recursive")
        for with_clause in query.find_all(exp.With)
    ):
        raise SQLValidationError("Recursive CTEs are not supported.")

    cte_names = {
        cte.alias_or_name.lower()
        for cte in query.find_all(exp.CTE)
        if cte.alias_or_name
    }
    table_aliases = {}
    dataset_referenced = False
    for table in query.find_all(exp.Table):
        name = table.name.lower()
        if table.db or table.catalog or (name != table_name.lower() and name not in cte_names):
            raise SQLValidationError("The query references a table outside the approved dataset.")
        if name == table_name.lower():
            dataset_referenced = True
            table_aliases[(table.alias_or_name or name).lower()] = name
            if table.alias:
                table_aliases[table.alias_or_name.lower()] = name
    if not dataset_referenced:
        raise SQLValidationError("The query must read from the selected dataset.")

    for select in query.find_all(exp.Select):
        for projection in select.expressions:
            references_column = any(projection.find_all(exp.Column))
            selects_dataset_columns = any(
                isinstance(node, exp.Star) for node in projection.walk()
            )
            counts_rows = any(
                isinstance(node, exp.Count)
                and any(isinstance(child, exp.Star) for child in node.walk())
                for node in projection.walk()
            )
            if not references_column and not selects_dataset_columns and not counts_rows:
                raise SQLValidationError("Every selected value must be derived from the dataset.")

    select_aliases = {
        alias.alias.lower()
        for alias in query.find_all(exp.Alias)
        if alias.alias
    }
    for column in query.find_all(exp.Column):
        name = column.name.lower()
        if name == "*":
            continue
        qualifier = column.table.lower()
        if qualifier and qualifier not in table_aliases and qualifier not in cte_names:
            raise SQLValidationError("The query references a column outside the approved dataset.")
        if name.casefold() not in allowed_columns and name not in select_aliases:
            raise SQLValidationError("The query references a column outside the approved dataset.")

    structural_expressions = (exp.And, exp.Case, exp.If, exp.In, exp.Is, exp.Not, exp.Or)
    for function in query.find_all(exp.Func):
        if isinstance(function, structural_expressions):
            continue
        function_name = (
            function.name
            if isinstance(function, exp.Anonymous)
            else function.sql_name()
        ).lower()
        if function_name not in SAFE_FUNCTIONS:
            raise SQLValidationError("The query uses a SQL function that is not approved.")

    limit = query.args.get("limit")
    if limit is None:
        query.set("limit", exp.Limit(expression=exp.Literal.number(max_rows)))
    else:
        expression = limit.args.get("expression")
        try:
            if isinstance(expression, exp.Neg) and isinstance(expression.this, exp.Literal):
                requested = -int(expression.this.this)
            elif isinstance(expression, exp.Literal):
                requested = int(expression.this)
            else:
                requested = max_rows + 1
        except (TypeError, ValueError):
            requested = max_rows + 1
        if requested < 0 or requested > max_rows:
            limit.set("expression", exp.Literal.number(max_rows))

    return ValidatedQuery(sql=query.sql(dialect="sqlite"), table_name=table_name)
