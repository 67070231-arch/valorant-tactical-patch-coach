import os
import re
import sqlite3
from pathlib import Path
from typing import Any, Dict

DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "valorant.db",
)

FORBIDDEN_KEYWORDS = {
    "insert",
    "update",
    "delete",
    "drop",
    "alter",
    "create",
    "replace",
    "truncate",
    "attach",
    "detach",
    "pragma",
    "vacuum",
    "reindex",
    "grant",
    "revoke",
    "commit",
    "rollback",
    "savepoint",
    "begin",
}


def is_safe_select(sql: str) -> bool:
    if not isinstance(sql, str):
        return False
    cleaned = sql.strip()
    if not cleaned:
        return False
    if "--" in cleaned or "/*" in cleaned or "*/" in cleaned:
        return False
    if cleaned.endswith(";"):
        cleaned = cleaned[:-1].strip()
    if ";" in cleaned:
        return False
    tokens = re.findall(r"[a-zA-Z_]+", cleaned.lower())
    if not tokens or tokens[0] != "select":
        return False
    for tok in tokens:
        if tok in FORBIDDEN_KEYWORDS:
            return False
    return True


def run_readonly_sql(sql: str, db_path: str = DB_PATH) -> Dict[str, Any]:
    if not is_safe_select(sql):
        return {
            "ok": False,
            "sql": sql,
            "error": "Unsafe SQL blocked by allow-list policy: only single read-only SELECT statements without comments or stacked queries are permitted.",
            "rows": [],
        }
    if not os.path.exists(db_path):
        from scripts.init_db import init_database
        init_database()

    cleaned_sql = sql.strip()
    if cleaned_sql.endswith(";"):
        cleaned_sql = cleaned_sql[:-1].strip()

    db_uri = f"{Path(db_path).resolve().as_uri()}?mode=ro"
    try:
        conn = sqlite3.connect(db_uri, uri=True, timeout=5.0)
        conn.row_factory = sqlite3.Row
        cur = conn.cursor()
        cur.execute(cleaned_sql)
        fetched = cur.fetchmany(50)
        rows = [dict(r) for r in fetched]
        conn.close()
        return {
            "ok": True,
            "sql": cleaned_sql,
            "row_count": len(rows),
            "rows": rows,
        }
    except Exception as exc:
        return {
            "ok": False,
            "sql": cleaned_sql,
            "error": str(exc),
            "rows": [],
        }
