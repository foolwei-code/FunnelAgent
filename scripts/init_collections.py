"""Milvus collection initialization script — create funnelrag_docs collection with full schema and indexes."""

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusClient
from python.config.settings import MilvusSettings
from python.utils.logging_config import get_logger

logger = get_logger(__name__)

# Default embedding dimension (must match the embedding model used at ingestion;
# default 1024 corresponds to bge-large-zh-v1.5)
DEFAULT_EMBEDDING_DIM = 1024

# HNSW index defaults
DEFAULT_HNSW_M = 16
DEFAULT_HNSW_EF_CONSTRUCTION = 256


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize Milvus collections for FunnelRAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--host", type=str, default=None,
        help="Milvus host (overrides settings/env)",
    )
    parser.add_argument(
        "--port", type=int, default=None,
        help="Milvus port (overrides settings/env)",
    )
    parser.add_argument(
        "--collection", type=str, default=None,
        help="Collection name (overrides settings/env)",
    )
    parser.add_argument(
        "--dimension", type=int, default=DEFAULT_EMBEDDING_DIM,
        help="Embedding vector dimension",
    )
    parser.add_argument(
        "--hnsw-m", type=int, default=DEFAULT_HNSW_M,
        help="HNSW index parameter M (connections per layer)",
    )
    parser.add_argument(
        "--hnsw-ef-construction", type=int, default=DEFAULT_HNSW_EF_CONSTRUCTION,
        help="HNSW index parameter efConstruction (build-time search width)",
    )
    parser.add_argument(
        "--drop-existing", action="store_true", default=False,
        help="Drop existing collection before creating (destructive!)",
    )
    parser.add_argument(
        "--verify", action="store_true", default=True,
        help="Verify collection after creation by inserting and querying a dummy vector",
    )
    parser.add_argument(
        "--no-verify", dest="verify", action="store_false",
        help="Skip verification step",
    )
    return parser.parse_args()


def build_collection_schema(dimension: int) -> CollectionSchema:
    """Build a detailed collection schema with typed fields.

    Fields:
        - id:        auto-increment primary key
        - doc_id:    VARCHAR document identifier (for cross-referencing with PostgreSQL)
        - text:      VARCHAR document text snippet (for quick display / reranker input)
        - source:    VARCHAR data source identifier
        - embedding: FLOAT_VECTOR embedding vector
    """
    fields = [
        FieldSchema(
            name="id",
            dtype=DataType.INT64,
            is_primary=True,
            auto_id=True,
            description="Auto-increment primary key",
        ),
        FieldSchema(
            name="doc_id",
            dtype=DataType.VARCHAR,
            max_length=255,
            description="Document identifier (matches PostgreSQL doc_id)",
        ),
        FieldSchema(
            name="text",
            dtype=DataType.VARCHAR,
            max_length=65535,
            description="Document text content snippet",
        ),
        FieldSchema(
            name="source",
            dtype=DataType.VARCHAR,
            max_length=512,
            description="Data source identifier",
        ),
        FieldSchema(
            name="embedding",
            dtype=DataType.FLOAT_VECTOR,
            dim=dimension,
            description="Document embedding vector",
        ),
    ]
    return CollectionSchema(
        fields=fields,
        description="FunnelRAG document collection with vector embeddings",
    )


def create_collection(
    settings: MilvusSettings | None = None,
    dimension: int = DEFAULT_EMBEDDING_DIM,
    hnsw_m: int = DEFAULT_HNSW_M,
    hnsw_ef_construction: int = DEFAULT_HNSW_EF_CONSTRUCTION,
    drop_existing: bool = False,
) -> dict:
    """Create Milvus collection with schema and indexes.

    Returns:
        dict with collection info (name, dimension, index params, row count, etc.)
    """
    if settings is None:
        settings = MilvusSettings()

    uri = f"http://{settings.host}:{settings.port}"
    logger.info("Connecting to Milvus at %s", uri)

    try:
        client = MilvusClient(uri=uri)
    except Exception as e:
        logger.error("Failed to connect to Milvus at %s: %s", uri, e)
        raise

    collection_name = settings.collection
    existing_collections = client.list_collections()

    # Drop existing if requested
    if drop_existing and collection_name in existing_collections:
        logger.warning("Dropping existing collection: %s", collection_name)
        client.drop_collection(collection_name)
        existing_collections = client.list_collections()
        logger.info("Collection %s dropped", collection_name)

    # Skip if already exists
    if collection_name in existing_collections:
        logger.info("Collection %s already exists, skipping creation", collection_name)
        info = client.get_collection_stats(collection_name)
        logger.info("Collection info: %s", info)
        return {"name": collection_name, "status": "existing", "stats": info}

    # --- Create collection with schema ---
    logger.info("Creating collection '%s' with dimension=%d", collection_name, dimension)

    schema = build_collection_schema(dimension)
    collection = Collection(
        name=collection_name,
        schema=schema,
        using="default",
    )
    logger.info("Collection '%s' created with %d fields", collection_name, len(schema.fields))

    # --- Create indexes ---
    # HNSW index on the embedding field for fast approximate nearest neighbor search
    index_params = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {
            "M": hnsw_m,
            "efConstruction": hnsw_ef_construction,
        },
    }
    logger.info(
        "Creating HNSW index on 'embedding' (M=%d, efConstruction=%d, metric=COSINE)",
        hnsw_m, hnsw_ef_construction,
    )
    collection.create_index(
        field_name="embedding",
        index_params=index_params,
        index_name="embedding_hnsw",
    )
    logger.info("HNSW index created successfully")

    # Load collection into memory for search
    collection.load()
    logger.info("Collection loaded into memory")

    return {
        "name": collection_name,
        "status": "created",
        "dimension": dimension,
        "hnsw_m": hnsw_m,
        "hnsw_ef_construction": hnsw_ef_construction,
    }


def verify_collection(settings: MilvusSettings | None = None, dimension: int = DEFAULT_EMBEDDING_DIM) -> bool:
    """Verify the collection is functional by checking its schema and stats."""
    if settings is None:
        settings = MilvusSettings()

    uri = f"http://{settings.host}:{settings.port}"
    collection_name = settings.collection

    try:
        client = MilvusClient(uri=uri)
        collections = client.list_collections()

        if collection_name not in collections:
            logger.error("Verification failed: collection '%s' not found", collection_name)
            return False

        # Check collection stats
        info = client.get_collection_stats(collection_name)
        logger.info("Verification: collection '%s' exists, stats=%s", collection_name, info)

        # Query with a dummy vector to verify the search pipeline works
        import random
        dummy_vector = [random.gauss(0, 1) for _ in range(dimension)]
        # Normalize to unit vector for cosine similarity
        norm = sum(x * x for x in dummy_vector) ** 0.5
        dummy_vector = [x / norm for x in dummy_vector]

        results = client.search(
            collection_name=collection_name,
            data=[dummy_vector],
            limit=1,
            output_fields=["doc_id"],
        )
        logger.info("Verification search returned %d results (collection is queryable)", len(results[0]) if results else 0)
        return True

    except Exception as e:
        logger.error("Verification failed: %s", e)
        return False


def display_collection_info(settings: MilvusSettings | None = None) -> None:
    """Display detailed information about the collection."""
    if settings is None:
        settings = MilvusSettings()

    uri = f"http://{settings.host}:{settings.port}"
    collection_name = settings.collection

    try:
        client = MilvusClient(uri=uri)
        info = client.get_collection_stats(collection_name)
        print(f"\n{'='*60}")
        print(f"Collection: {collection_name}")
        print(f"{'='*60}")
        print(f"  Stats: {info}")
        print(f"  URI:   {uri}")
        print(f"{'='*60}\n")
    except Exception as e:
        logger.warning("Could not retrieve collection info: %s", e)


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Build settings from args (override defaults)
    settings = MilvusSettings()
    if args.host is not None:
        settings.host = args.host
    if args.port is not None:
        settings.port = args.port
    if args.collection is not None:
        settings.collection = args.collection

    logger.info("Initializing Milvus collection for FunnelRAG")
    logger.info("  Host:       %s:%d", settings.host, settings.port)
    logger.info("  Collection: %s", settings.collection)
    logger.info("  Dimension:  %d", args.dimension)
    logger.info("  HNSW M:     %d", args.hnsw_m)
    logger.info("  HNSW efC:   %d", args.hnsw_ef_construction)

    try:
        result = create_collection(
            settings=settings,
            dimension=args.dimension,
            hnsw_m=args.hnsw_m,
            hnsw_ef_construction=args.hnsw_ef_construction,
            drop_existing=args.drop_existing,
        )
        logger.info("Collection operation result: %s", result)
    except Exception as e:
        logger.error("Collection creation failed: %s", e)
        sys.exit(1)

    # Verification step
    if args.verify:
        logger.info("Running verification...")
        if verify_collection(settings, args.dimension):
            logger.info("Verification passed")
        else:
            logger.error("Verification failed — collection may not be fully functional")
            sys.exit(1)

    # Display collection info
    display_collection_info(settings)

    print("Done.")


if __name__ == "__main__":
    main()
