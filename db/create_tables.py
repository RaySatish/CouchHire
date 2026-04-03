#!/usr/bin/env python3
"""One-off script to create the CouchHire applications table in Supabase.

Reads db/schema.sql and executes it against the Supabase database.

Supports two methods (tries in order):
  1. Direct Postgres connection via SUPABASE_DB_URL env var
     (e.g. postgresql://postgres.<ref>:<password>@aws-0-<region>.pooler.supabase.com:6543/postgres)
  2. Supabase Management API via SUPABASE_ACCESS_TOKEN env var
     (POST https://api.supabase.com/v1/projects/{ref}/database/query)

Set at least one of these in your .env file. SUPABASE_DB_URL is preferred
because it's simpler and doesn't require a separate access token.

Usage:
    python db/create_tables.py
"""

import os
import sys
from pathlib import Path
from urllib.parse import urlparse, quote_plus, unquote, urlunparse

# Ensure project root is on sys.path so we can import config
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from config import SUPABASE_URL, SUPABASE_ACCESS_TOKEN


def _read_schema() -> str:
    """Read the SQL schema file, prepended with DROP TABLE for a clean slate."""
    schema_path = _PROJECT_ROOT / "db" / "schema.sql"
    if not schema_path.exists():
        print(f"ERROR: Schema file not found at {schema_path}")
        sys.exit(1)
    drop_stmt = "DROP TABLE IF EXISTS public.applications CASCADE;\n\n"
    return drop_stmt + schema_path.read_text(encoding="utf-8")


def _extract_project_ref(supabase_url: str) -> str:
    """Extract the project ref from a Supabase URL like https://<ref>.supabase.co."""
    parsed = urlparse(supabase_url)
    hostname = parsed.hostname or ""
    # hostname is like: abcdefghij.supabase.co
    parts = hostname.split(".")
    if len(parts) >= 3 and "supabase" in parts[-2]:
        return parts[0]
    raise ValueError(
        f"Could not extract project ref from SUPABASE_URL: {supabase_url}"
    )


def _sanitize_db_url(db_url: str) -> str:
    """URL-encode the password in a Postgres connection string.

    Handles passwords containing special characters like @, #, %, etc.
    that would otherwise break URL parsing.
    urlparse returns percent-encoded values, so we must decode first
    to get the real password, then re-encode for safe URL construction.
    """
    parsed = urlparse(db_url)
    if parsed.password:
        # Decode first (urlparse keeps %40 as-is), then re-encode
        real_password = unquote(parsed.password)
        encoded_password = quote_plus(real_password)
        # Rebuild the netloc: user:encoded_password@host:port
        if parsed.port:
            netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}:{parsed.port}"
        else:
            netloc = f"{parsed.username}:{encoded_password}@{parsed.hostname}"
        return urlunparse((
            parsed.scheme,
            netloc,
            parsed.path,
            parsed.params,
            parsed.query,
            parsed.fragment,
        ))
    return db_url


def _try_direct_postgres(sql: str) -> bool:
    """Attempt to execute SQL via direct Postgres connection (psycopg2)."""
    db_url = os.environ.get("SUPABASE_DB_URL", "").strip()
    if not db_url:
        return False

    try:
        import psycopg2  # noqa: F811
    except ImportError:
        print("INFO: psycopg2 not installed, skipping direct Postgres method.")
        print("      Install with: pip install psycopg2-binary")
        return False

    # Sanitize the URL to handle special characters in password
    safe_url = _sanitize_db_url(db_url)

    print("Connecting via direct Postgres connection...")
    try:
        conn = psycopg2.connect(safe_url)
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(sql)
        conn.close()
        return True
    except Exception as exc:
        print(f"ERROR (direct Postgres): {exc}")
        return False


def _try_management_api(sql: str) -> bool:
    """Attempt to execute SQL via Supabase Management API."""
    if not SUPABASE_ACCESS_TOKEN:
        return False

    try:
        import httpx
    except ImportError:
        print("INFO: httpx not installed, skipping Management API method.")
        print("      Install with: pip install httpx")
        return False

    try:
        project_ref = _extract_project_ref(SUPABASE_URL)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return False

    url = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"
    print(f"Executing SQL via Supabase Management API (project: {project_ref})...")

    try:
        response = httpx.post(
            url,
            json={"query": sql},
            headers={
                "Authorization": f"Bearer {SUPABASE_ACCESS_TOKEN}",
                "Content-Type": "application/json",
            },
            timeout=30.0,
        )
        if response.status_code in (200, 201):
            return True
        else:
            print(f"ERROR (Management API): HTTP {response.status_code}")
            print(f"  Response: {response.text[:500]}")
            return False
    except Exception as exc:
        print(f"ERROR (Management API): {exc}")
        return False


def main() -> None:
    """Execute schema.sql against Supabase."""
    sql = _read_schema()
    print(f"Read schema.sql ({len(sql)} chars)")

    # Method 1: Direct Postgres connection (preferred)
    if _try_direct_postgres(sql):
        print("Table created successfully")
        return

    # Method 2: Supabase Management API
    if _try_management_api(sql):
        print("Table created successfully")
        return

    # Neither method worked — guide the user
    print()
    print("=" * 70)
    print("Could not execute SQL automatically. Set one of these in .env:")
    print()
    print("  Option A (recommended): SUPABASE_DB_URL")
    print("    Find it in Supabase Dashboard → Settings → Database → Connection string")
    print("    Format: postgresql://postgres.<ref>:<password>@<host>:6543/postgres")
    print("    Then: pip install psycopg2-binary")
    print()
    print("  Option B: SUPABASE_ACCESS_TOKEN")
    print("    Generate at: https://supabase.com/dashboard/account/tokens")
    print("    This uses the Supabase Management API.")
    print()
    print("  Option C: Copy-paste db/schema.sql into the Supabase SQL Editor manually:")
    print("    https://supabase.com/dashboard/project/_/sql/new")
    print("=" * 70)
    sys.exit(1)


if __name__ == "__main__":
    main()
