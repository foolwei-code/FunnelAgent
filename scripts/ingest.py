"""Document ingestion script — load, validate, deduplicate, embed, and write documents to Milvus + PostgreSQL."""

import argparse
import asyncio
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from python.config.settings import AppSettings, MilvusSettings, PostgresSettings
from python.tools.doc_store import DocStore
from python.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)

# ============================================================================
# Constants
# ============================================================================

MAX_CONTENT_LENGTH = 100_000   # Max content length in characters
MAX_DOC_ID_LENGTH  = 255       # Max doc_id length
REQUIRED_FIELDS    = {"doc_id", "content"}
BATCH_LOG_INTERVAL = 100       # Log progress every N documents


# ============================================================================
# Argument parsing
# ============================================================================

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest documents into FunnelRAG (Milvus + PostgreSQL)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "data", type=str,
        help="Path to JSONL data file",
    )
    parser.add_argument(
        "--batch-size", type=int, default=500,
        help="Number of documents per batch for PostgreSQL insertion",
    )
    parser.add_argument(
        "--dry-run", action="store_true", default=False,
        help="Validate and process documents without writing to database",
    )
    parser.add_argument(
        "--validate-only", action="store_true", default=False,
        help="Only validate the data file, do not process or ingest",
    )
    parser.add_argument(
        "--skip-embedding", action="store_true", default=False,
        help="Skip embedding generation (only write text to PostgreSQL)",
    )
    parser.add_argument(
        "--check-duplicates", action="store_true", default=False,
        help="Check for duplicate doc_ids against existing database before ingestion",
    )
    parser.add_argument(
        "--max-content-length", type=int, default=MAX_CONTENT_LENGTH,
        help="Maximum content length in characters (longer content will be truncated)",
    )
    return parser.parse_args()


# ============================================================================
# Document loading
# ============================================================================

def load_documents(data_path: str) -> list[dict]:
    """Load documents from a JSONL file.

    Each line should be a JSON object with at least 'doc_id' and 'content' fields.
    Empty lines are skipped. Parse errors are logged and skipped.
    """
    docs = []
    path = Path(data_path)

    if not path.exists():
        logger.error("Data file not found: %s", data_path)
        raise FileNotFoundError(f"Data file not found: {data_path}")

    logger.info("Loading documents from: %s", data_path)
    start_time = time.time()

    with open(data_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                doc = json.loads(line)
                docs.append(doc)
            except json.JSONDecodeError as e:
                logger.warning("Line %d: JSON parse error — %s (skipping)", line_no, e)

    elapsed = time.time() - start_time
    logger.info("Loaded %d documents in %.2f seconds (%.0f docs/sec)",
                len(docs), elapsed, len(docs) / max(elapsed, 0.001))

    return docs


# ============================================================================
# Document validation
# ============================================================================

def validate_document(doc: dict, doc_index: int, max_content_length: int) -> list[str]:
    """Validate a single document. Returns a list of error messages (empty if valid)."""
    errors = []

    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in doc:
            errors.append(f"missing required field '{field}'")
        elif not doc[field]:
            errors.append(f"field '{field}' is empty")

    # Check doc_id length
    if "doc_id" in doc and len(str(doc["doc_id"])) > MAX_DOC_ID_LENGTH:
        errors.append(f"doc_id exceeds {MAX_DOC_ID_LENGTH} characters")

    # Check content length
    if "content" in doc and len(doc["content"]) > max_content_length:
        # Truncate content rather than reject
        doc["content"] = doc["content"][:max_content_length]
        doc["_truncated"] = True
        # This is a warning, not an error
        logger.debug("Doc #%d: content truncated to %d characters", doc_index, max_content_length)

    # Validate metadata if present
    if "metadata" in doc and not isinstance(doc["metadata"], dict):
        errors.append(f"'metadata' must be a dict, got {type(doc['metadata']).__name__}")

    return errors


def validate_documents(docs: list[dict], max_content_length: int = MAX_CONTENT_LENGTH) -> tuple[list[dict], list[dict]]:
    """Validate all documents. Returns (valid_docs, invalid_docs_with_errors)."""
    valid = []
    invalid = []

    for i, doc in enumerate(docs):
        errors = validate_document(doc, i, max_content_length)
        if errors:
            doc_id = doc.get("doc_id", f"<index {i}>")
            invalid.append({"doc_id": doc_id, "errors": errors})
            logger.warning("Doc '%s': validation failed — %s", doc_id, "; ".join(errors))
        else:
            valid.append(doc)

    logger.info("Validation: %d valid, %d invalid out of %d total",
                len(valid), len(invalid), len(docs))

    return valid, invalid


# ============================================================================
# Duplicate detection
# ============================================================================

async def check_existing_doc_ids(doc_ids: list[str], settings: PostgresSettings) -> set[str]:
    """Check which doc_ids already exist in the database. Returns set of existing IDs."""
    if not doc_ids:
        return set()

    import asyncpg

    logger.info("Checking %d doc_ids against existing database...", len(doc_ids))
    pool = await asyncpg.create_pool(settings.dsn, min_size=1, max_size=5)

    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM documents WHERE doc_id = ANY($1)",
                doc_ids,
            )
            existing = {r["doc_id"] for r in rows}
    except Exception as e:
        logger.warning("Could not check existing doc_ids: %s", e)
        existing = set()
    finally:
        await pool.close()

    logger.info("Found %d existing doc_ids in database", len(existing))
    return existing


# ============================================================================
# Content hashing
# ============================================================================

def compute_content_hash(content: str) -> str:
    """Compute SHA-256 hash of document content for change detection."""
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


# ============================================================================
# Embedding generation (stub)
# ============================================================================

async def generate_embeddings(docs: list[dict]) -> list[dict]:
    """Generate embeddings for documents.

    This is a stub that would be replaced with an actual embedding model
    (e.g., bge-large-zh-v1.5 via HuggingFace, OpenAI, or local ONNX).

    In production, this would:
    1. Batch documents for efficient inference
    2. Call the embedding model
    3. Attach the resulting vectors to each document
    """
    logger.info("Embedding generation: %d documents (stub — returning empty vectors)", len(docs))
    for doc in docs:
        doc["embedding"] = []  # Placeholder; real implementation would produce float vectors
    return docs


# ============================================================================
# Milvus vector insertion
# ============================================================================

async def insert_to_milvus(docs: list[dict], settings: MilvusSettings) -> int:
    """Insert document vectors into Milvus.

    Returns the number of documents inserted.
    Currently a placeholder that logs the intent.
    """
    from pymilvus import MilvusClient

    docs_with_embeddings = [d for d in docs if d.get("embedding")]
    if not docs_with_embeddings:
        logger.info("No documents with embeddings — skipping Milvus insertion")
        return 0

    logger.info("Inserting %d vectors into Milvus collection '%s'...",
                len(docs_with_embeddings), settings.collection)

    try:
        client = MilvusClient(uri=f"http://{settings.host}:{settings.port}")

        data = []
        for doc in docs_with_embeddings:
            data.append({
                "doc_id":   doc["doc_id"],
                "text":     doc["content"][:65535],  # Milvus VARCHAR max
                "source":   doc.get("source", ""),
                "embedding": doc["embedding"],
            })

        client.insert(
            collection_name=settings.collection,
            data=data,
        )
        logger.info("Inserted %d vectors into Milvus", len(data))
        return len(data)

    except Exception as e:
        logger.error("Milvus insertion failed: %s", e)
        return 0


# ============================================================================
# Main ingestion flow
# ============================================================================

async def ingest(
    data_path: str,
    batch_size: int = 500,
    dry_run: bool = False,
    validate_only: bool = False,
    skip_embedding: bool = False,
    check_duplicates: bool = False,
    max_content_length: int = MAX_CONTENT_LENGTH,
) -> dict:
    """Execute the full ingestion pipeline.

    Returns a dict with ingestion statistics.
    """
    stats = {
        "total_loaded": 0,
        "valid": 0,
        "invalid": 0,
        "duplicates_skipped": 0,
        "pg_inserted": 0,
        "milvus_inserted": 0,
        "errors": 0,
    }

    settings = AppSettings()
    start_time = time.time()

    # 1. Load documents
    docs = load_documents(data_path)
    stats["total_loaded"] = len(docs)

    if not docs:
        logger.error("No documents loaded from %s", data_path)
        return stats

    # 2. Validate documents
    valid_docs, invalid_docs = validate_documents(docs, max_content_length)
    stats["valid"] = len(valid_docs)
    stats["invalid"] = len(invalid_docs)

    if validate_only:
        logger.info("Validate-only mode: %d valid, %d invalid", len(valid_docs), len(invalid_docs))
        return stats

    if not valid_docs:
        logger.error("No valid documents to ingest")
        return stats

    # 3. Duplicate detection (optional)
    if check_duplicates and not dry_run:
        doc_ids = [d["doc_id"] for d in valid_docs]
        existing_ids = await check_existing_doc_ids(doc_ids, settings.postgres)
        before = len(valid_docs)
        valid_docs = [d for d in valid_docs if d["doc_id"] not in existing_ids]
        stats["duplicates_skipped"] = before - len(valid_docs)
        if stats["duplicates_skipped"] > 0:
            logger.info("Skipped %d duplicate documents", stats["duplicates_skipped"])

    # 4. Compute content hashes for change detection
    for doc in valid_docs:
        doc["content_hash"] = compute_content_hash(doc["content"])

    # 5. Embedding generation (optional)
    if not skip_embedding and not dry_run:
        valid_docs = await generate_embeddings(valid_docs)

    # 6. Write to PostgreSQL (batch insertion)
    if not dry_run:
        logger.info("Writing %d documents to PostgreSQL (batch_size=%d)...",
                    len(valid_docs), batch_size)
        store = DocStore(settings.postgres)
        await store.init()

        total_inserted = 0
        total_errors = 0

        for batch_start in range(0, len(valid_docs), batch_size):
            batch = valid_docs[batch_start:batch_start + batch_size]
            batch_num = batch_start // batch_size + 1
            total_batches = (len(valid_docs) + batch_size - 1) // batch_size

            try:
                await store.batch_insert(batch)
                total_inserted += len(batch)
            except Exception as e:
                total_errors += 1
                logger.error("Batch %d/%d insertion failed: %s", batch_num, total_batches, e)
                # Attempt individual inserts for error recovery
                for doc in batch:
                    try:
                        await store.batch_insert([doc])
                        total_inserted += 1
                    except Exception as e2:
                        logger.error("  Individual insert failed for doc_id='%s': %s",
                                     doc.get("doc_id", "?"), e2)

            if batch_num % max(1, total_batches // 10) == 0 or batch_num == total_batches:
                logger.info("  Batch %d/%d: %d/%d documents inserted",
                            batch_num, total_batches, total_inserted, len(valid_docs))

        stats["pg_inserted"] = total_inserted
        stats["errors"] = total_errors
        await store.close()
        logger.info("PostgreSQL: %d documents inserted, %d batch errors",
                    total_inserted, total_errors)

        # 7. Write to Milvus (if embeddings available)
        if not skip_embedding:
            milvus_count = await insert_to_milvus(valid_docs, settings.milvus)
            stats["milvus_inserted"] = milvus_count
        else:
            logger.info("Embedding generation skipped — Milvus insertion skipped")
    else:
        logger.info("Dry-run mode: would process %d valid documents", len(valid_docs))
        stats["pg_inserted"] = len(valid_docs)

    elapsed = time.time() - start_time
    stats["elapsed_seconds"] = elapsed

    return stats


# ============================================================================
# Entry point
# ============================================================================

def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging("INFO")

    try:
        stats = asyncio.run(ingest(
            data_path=args.data,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
            validate_only=args.validate_only,
            skip_embedding=args.skip_embedding,
            check_duplicates=args.check_duplicates,
            max_content_length=args.max_content_length,
        ))
    except Exception as e:
        logger.error("Ingestion failed: %s", e)
        sys.exit(1)

    # Print summary
    print(f"\n{'='*60}")
    print("Ingestion Summary")
    print(f"{'='*60}")
    print(f"  Total loaded:        {stats['total_loaded']}")
    print(f"  Valid:               {stats['valid']}")
    print(f"  Invalid:             {stats['invalid']}")
    print(f"  Duplicates skipped:  {stats['duplicates_skipped']}")
    print(f"  PG inserted:         {stats['pg_inserted']}")
    print(f"  Milvus inserted:     {stats['milvus_inserted']}")
    print(f"  Batch errors:        {stats['errors']}")
    if "elapsed_seconds" in stats:
        print(f"  Elapsed:             {stats['elapsed_seconds']:.2f}s")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
