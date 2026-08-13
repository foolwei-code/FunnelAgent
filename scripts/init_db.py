"""PostgreSQL document table initialization script — create tables, indexes, and support schema migration."""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.config.settings import PostgresSettings
from python.utils.logging_config import get_logger

logger = get_logger(__name__)

# ============================================================================
# SQL Definitions
# ============================================================================

DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    id            SERIAL       PRIMARY KEY,
    doc_id        VARCHAR(255) NOT NULL UNIQUE,
    title         VARCHAR(512) DEFAULT '',
    content       TEXT         NOT NULL,
    source        VARCHAR(512) DEFAULT '',
    category      VARCHAR(128) DEFAULT '',
    language      VARCHAR(16)  DEFAULT 'zh',
    metadata      JSONB        DEFAULT '{}',
    content_hash  VARCHAR(64)  DEFAULT '',
    created_at    TIMESTAMP    DEFAULT NOW(),
    updated_at    TIMESTAMP    DEFAULT NOW()
);
"""

# Indexes for common query patterns
INDEXES_SQL = [
    # Fast lookup by doc_id
    """
    CREATE INDEX IF NOT EXISTS idx_documents_doc_id
    ON documents(doc_id);
    """,
    # Full-text search index on content (using GIN + tsvector)
    """
    CREATE INDEX IF NOT EXISTS idx_documents_content_fts
    ON documents USING gin(to_tsvector('simple', content));
    """,
    # Index on source for filtering
    """
    CREATE INDEX IF NOT EXISTS idx_documents_source
    ON documents(source);
    """,
    # Index on category for filtering
    """
    CREATE INDEX IF NOT EXISTS idx_documents_category
    ON documents(category);
    """,
    # Index on created_at for time-based queries
    """
    CREATE INDEX IF NOT EXISTS idx_documents_created_at
    ON documents(created_at);
    """,
    # GIN index on metadata for JSON queries
    """
    CREATE INDEX IF NOT EXISTS idx_documents_metadata_gin
    ON documents USING gin(metadata);
    """,
]

# Trigger to auto-update updated_at on row modification
UPDATED_AT_TRIGGER_SQL = """
CREATE OR REPLACE FUNCTION update_documents_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_documents_updated_at ON documents;
CREATE TRIGGER trg_documents_updated_at
    BEFORE UPDATE ON documents
    FOR EACH ROW
    EXECUTE FUNCTION update_documents_updated_at();
"""

# Expected columns for migration check
EXPECTED_COLUMNS = {
    "id":           "integer",
    "doc_id":       "character varying",
    "title":        "character varying",
    "content":      "text",
    "source":       "character varying",
    "category":     "character varying",
    "language":     "character varying",
    "metadata":     "jsonb",
    "content_hash": "character varying",
    "created_at":   "timestamp without time zone",
    "updated_at":   "timestamp without time zone",
}

# Columns to add if missing (migration support)
MIGRATION_COLUMNS = {
    "category":     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS category VARCHAR(128) DEFAULT '';",
    "language":     "ALTER TABLE documents ADD COLUMN IF NOT EXISTS language VARCHAR(16) DEFAULT 'zh';",
    "content_hash": "ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64) DEFAULT '';",
    "updated_at":   "ALTER TABLE documents ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP DEFAULT NOW();",
}


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL database for FunnelRAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="PostgreSQL host (overrides settings/env)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="PostgreSQL port (overrides settings/env)",
    )
    parser.add_argument(
        "--user", type=str, default=None,
        help="PostgreSQL user (overrides settings/env)",
    )
    parser.add_argument(
        "--database", type=str, default=None,
        help="PostgreSQL database (overrides settings/env)",
    )
    parser.add_argument(
        "--migrate", action="store_true", default=False,
        help="Run schema migration (add missing columns) instead of full init",
    )
    parser.add_argument(
        "--verify-only", action="store_true", default=False,
        help="Only verify the schema, do not create or migrate",
    )
    return parser.parse_args()


async def get_existing_columns(conn: Any, table_name: str = "documents") -> set[str]:
    """Get the set of existing column names for a table."""
    rows = await conn.fetch(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = $1
        """,
        table_name,
    )
    return {r["column_name"] for r in rows}


async def check_table_exists(conn: Any, table_name: str = "documents") -> bool:
    """Check if a table exists in the database."""
    row = await conn.fetchval(
        """
        SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = $1
        )
        """,
        table_name,
    )
    return row


async def init_db(settings: PostgresSettings, migrate: bool = False) -> None:
    """Initialize the PostgreSQL database with tables, indexes, and triggers."""
    import asyncpg

    logger.info("Connecting to PostgreSQL at %s:%d/%s", settings.host, settings.port, settings.database)

    try:
        pool = await asyncpg.create_pool(settings.dsn, min_size=1, max_size=5)
    except Exception as e:
        logger.error("Failed to connect to PostgreSQL: %s", e)
        raise

    async with pool.acquire() as conn:
        # --- Check existing schema ---
        table_exists = await check_table_exists(conn)
        existing_columns = set()
        if table_exists:
            existing_columns = await get_existing_columns(conn)
            logger.info("Table 'documents' exists with %d columns: %s",
                        len(existing_columns), sorted(existing_columns))
        else:
            logger.info("Table 'documents' does not exist — will create")

        # --- Migration mode ---
        if migrate and table_exists:
            missing = set(MIGRATION_COLUMNS.keys()) - existing_columns
            if not missing:
                logger.info("Migration: no missing columns — schema is up to date")
            else:
                logger.info("Migration: adding %d missing columns: %s", len(missing), sorted(missing))
                for col_name in sorted(missing):
                    sql = MIGRATION_COLUMNS[col_name]
                    logger.info("  Adding column '%s'...", col_name)
                    await conn.execute(sql)
                logger.info("Migration complete")
            await pool.close()
            return

        # --- Create table ---
        if not table_exists:
            logger.info("Creating 'documents' table...")
            await conn.execute(DOCUMENTS_TABLE_SQL)
            logger.info("Table 'documents' created successfully")
        else:
            logger.info("Table 'documents' already exists, skipping creation")

        # --- Create indexes ---
        logger.info("Creating indexes...")
        for i, index_sql in enumerate(INDEXES_SQL):
            try:
                await conn.execute(index_sql)
                logger.info("  Index %d/%d created", i + 1, len(INDEXES_SQL))
            except Exception as e:
                logger.warning("  Index %d/%d creation failed (may already exist): %s", i + 1, len(INDEXES_SQL), e)
        logger.info("Indexes created")

        # --- Create trigger ---
        logger.info("Creating updated_at trigger...")
        try:
            await conn.execute(UPDATED_AT_TRIGGER_SQL)
            logger.info("Trigger created successfully")
        except Exception as e:
            logger.warning("Trigger creation failed: %s", e)

        # --- Run migration for any missing columns ---
        if table_exists:
            missing = set(MIGRATION_COLUMNS.keys()) - existing_columns
            if missing:
                logger.info("Adding %d missing columns: %s", len(missing), sorted(missing))
                for col_name in sorted(missing):
                    sql = MIGRATION_COLUMNS[col_name]
                    await conn.execute(sql)
                    logger.info("  Added column '%s'", col_name)

    await pool.close()
    logger.info("PostgreSQL database initialization complete")


async def verify_db(settings: PostgresSettings) -> bool:
    """Verify the database schema is correct."""
    import asyncpg

    logger.info("Verifying PostgreSQL schema...")

    try:
        pool = await asyncpg.create_pool(settings.dsn, min_size=1, max_size=2)
    except Exception as e:
        logger.error("Cannot connect to PostgreSQL for verification: %s", e)
        return False

    async with pool.acquire() as conn:
        # Check table exists
        if not await check_table_exists(conn):
            logger.error("Verification failed: table 'documents' does not exist")
            await pool.close()
            return False

        # Check columns
        existing = await get_existing_columns(conn)
        required = set(EXPECTED_COLUMNS.keys())
        missing = required - existing

        if missing:
            logger.error("Verification failed: missing columns: %s", sorted(missing))
            await pool.close()
            return False

        extra = existing - required - {"id"}  # id is auto-managed
        if extra:
            logger.info("Note: extra columns present: %s", sorted(extra))

        # Check row count
        count = await conn.fetchval("SELECT COUNT(*) FROM documents")
        logger.info("Verification passed: %d columns, %d rows in 'documents'",
                    len(existing), count)

    await pool.close()
    return True


def main() -> None:
    """Main entry point."""
    args = parse_args()

    settings = PostgresSettings()
    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.user is not None:
        settings.user = args.user
    if args.database is not None:
        settings.database = args.database

    logger.info("PostgreSQL init_db for FunnelRAG")
    logger.info("  Host:     %s:%d", settings.host, settings.port)
    logger.info("  Database: %s", settings.database)

    if args.verify_only:
        ok = asyncio.run(verify_db(settings))
        if not ok:
            sys.exit(1)
        print("Verification passed.")
        return

    try:
        asyncio.run(init_db(settings, migrate=args.migrate))
    except Exception as e:
        logger.error("Database initialization failed: %s", e)
        sys.exit(1)

    # Post-init verification
    if not args.migrate:
        ok = asyncio.run(verify_db(settings))
        if not ok:
            logger.warning("Post-init verification failed — please check the schema")
        else:
            logger.info("Post-init verification passed")

    print("Done.")


if __name__ == "__main__":
    main()
