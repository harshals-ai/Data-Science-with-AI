import os
import re
import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from huggingface_hub import InferenceClient


DB_PATH = os.environ.get("NL2SQL_DB_PATH", "meta_ads.db")
HF_TOKEN = os.environ.get("HF_TOKEN")  # required
HF_MODEL_ID = os.environ.get("HF_MODEL_ID") or "google/gemma-3-27b-it"


FEW_SHOT = """
Example 1
Question: What is total revenue?
SQL: SELECT SUM(t.revenue_usd) AS total_revenue_usd FROM ad_transactions t;

Example 2
Question: Revenue by placement
SQL: SELECT p.placement, SUM(t.revenue_usd) AS revenue_usd
     FROM ad_transactions t
     JOIN ad_products p ON p.product_id = t.product_id
     GROUP BY p.placement
     ORDER BY revenue_usd DESC;
""".strip()

PROMPT_TEMPLATE = """
You are an expert data analyst.
Given a SQLite database schema, write ONE SQLite SQL query that answers the question.

Rules:
- Output ONLY the SQL query.
- Must be a single SELECT statement.
- Do NOT use ``` fences.
- Use only table/column names that exist in the schema.

Schema:
{schema}

{few_shot}

Question: {question}
SQL:
""".strip()


class NL2SQLGuardError(ValueError):
    pass


def _get_schema_paragraph(conn: sqlite3.Connection) -> str:
    def get_tables() -> list[str]:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        return [r[0] for r in rows]

    def get_table_info(table: str) -> list[tuple]:
        # PRAGMA table_info returns: cid, name, type, notnull, dflt_value, pk
        return conn.execute(f"PRAGMA table_info({table})").fetchall()

    parts: list[str] = []
    for t in get_tables():
        parts.append(f"TABLE {t}:")
        cols = get_table_info(t)
        for c in cols:
            col_name, col_type = c[1], c[2]
            parts.append(f"  - {col_name} {col_type}")
        parts.append("")  # blank line between tables
    return "\n".join(parts).strip()


def _extract_first_select(text: str) -> str:
    # best-effort: find the first SELECT ... until end of string
    m = re.search(r"\bselect\b[\s\S]*", text, flags=re.IGNORECASE)
    if not m:
        return text.strip()
    sql = m.group(0).strip()
    # remove fenced code blocks if the model used them
    sql = re.sub(r"```[\s\S]*?```", "", sql).strip()
    return sql


def is_readonly_select(sql: str) -> bool:
    s = sql.strip()
    if not s.lower().startswith("select"):
        return False
    # block multi-statement/chaining
    if ";" in s[:-1]:  # allow trailing semicolon
        return False
    lowered = s.lower()
    blocked = [
        "insert",
        "update",
        "delete",
        "drop",
        "alter",
        "create",
        "attach",
        "detach",
        "pragma",
        "reindex",
        "vacuum",
        "replace",
        "truncate",
    ]
    return not any(k in lowered for k in blocked)


def build_prompt(schema: str, question: str) -> str:
    return PROMPT_TEMPLATE.format(schema=schema, few_shot=FEW_SHOT, question=question)


@dataclass
class NL2SQLConfig:
    db_path: str = DB_PATH
    model_id: str = HF_MODEL_ID


def main() -> None:
    if not HF_TOKEN:
        raise RuntimeError("Missing HF_TOKEN env var.")

    cfg = NL2SQLConfig()
    if not os.path.exists(cfg.db_path):
        raise FileNotFoundError(f"DB file not found: {cfg.db_path}")

    # Connect to SQLite
    conn = sqlite3.connect(cfg.db_path)
    schema = _get_schema_paragraph(conn)

    # Initialize HF client once
    client = InferenceClient(api_key=HF_TOKEN)

    print("NL→SQL CLI ready.")
    print("Type a natural-language question about the ad revenue dataset.")
    print("Type 'exit' to quit.")

    while True:
        question = input(
            " what would you like to know about our Ad products revenue data ? "
        ).strip()
        if question.lower() in {"exit", "quit", "q"}:
            break
        if not question:
            continue

        prompt = build_prompt(schema=schema, question=question)
        messages = [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model=cfg.model_id,
            messages=messages,
            max_tokens=256,
            temperature=0.0,
        )
        raw = response.choices[0].message.content

        sql = _extract_first_select(raw)
        if not is_readonly_select(sql):
            conn.close()
            raise NL2SQLGuardError("Generated SQL failed read-only SELECT guard.")

        df = pd.read_sql_query(sql, conn)
        if len(df) > 20:
            print("\nResult (first 20 rows):")
            print(df.head(20).to_string(index=False))
        else:
            print("\nResult:")
            print(df.to_string(index=False))

        print("\nSQL used:\n" + sql)

        print("\n" + "-" * 80)

    conn.close()


if __name__ == "__main__":
    main()

