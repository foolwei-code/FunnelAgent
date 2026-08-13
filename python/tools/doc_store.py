"""PostgreSQL 文档存储工具 —— 提供原文与标量元数据的全生命周期管理。"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Dict, List, Optional, Sequence

from python.config.settings import PostgresSettings
from python.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------
from prometheus_client import Counter as _Counter, Histogram as _Histogram

DOCSTORE_QUERY_TOTAL = _Counter(
    "funnelrag_docstore_query_total", "Total DocStore queries"
)
DOCSTORE_ERROR_TOTAL = _Counter(
    "funnelrag_docstore_error_total", "Total DocStore errors"
)
DOCSTORE_QUERY_LATENCY = _Histogram(
    "funnelrag_docstore_query_latency_seconds", "DocStore query latency"
)

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """\
CREATE TABLE IF NOT EXISTS documents (
    id           SERIAL PRIMARY KEY,
    doc_id       VARCHAR(255) NOT NULL UNIQUE,
    title        VARCHAR(512) DEFAULT '',
    content      TEXT NOT NULL DEFAULT '',
    source       VARCHAR(512) DEFAULT '',
    metadata     JSONB DEFAULT '{}',
    created_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at   TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Primary lookup index
CREATE INDEX IF NOT EXISTS idx_documents_doc_id ON documents(doc_id);

-- Full-text search support (GIN index on content)
CREATE INDEX IF NOT EXISTS idx_documents_content_fts ON documents
    USING GIN (to_tsvector('simple', content));

-- JSONB metadata index
CREATE INDEX IF NOT EXISTS idx_documents_metadata ON documents USING GIN (metadata);

-- Source index for filtering
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

-- Auto-update trigger for updated_at
CREATE OR REPLACE FUNCTION _funnelrag_doc_updated_at()
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
    EXECUTE FUNCTION _funnelrag_doc_updated_at();
"""

# Transient-error retry config
_MAX_RETRIES = 3
_RETRY_BASE_DELAY = 0.2  # seconds


class DocStore:
    """PostgreSQL 文档存储，提供原文的快速聚合查询与全生命周期管理。

    支持 CRUD、全文搜索、JSONB 元数据查询、事务上下文管理器、
    连接池监控、瞬态错误自动重试等能力。

    Attributes:
        settings: PostgreSQL 连接配置。
    """

    def __init__(self, settings: PostgresSettings) -> None:
        """初始化 DocStore。

        Args:
            settings: PostgreSQL 连接配置实例。
        """
        self.settings = settings
        self._pool = None

    # ------------------------------------------------------------------
    # Connection pool lifecycle
    # ------------------------------------------------------------------

    async def init(self, *, min_size: int = 2, max_size: int = 10) -> None:
        """初始化连接池并建表。

        Args:
            min_size: 连接池最小连接数。
            max_size: 连接池最大连接数。

        Raises:
            ConnectionError: 连接池创建失败时抛出。
        """
        import asyncpg

        try:
            self._pool = await asyncpg.create_pool(
                self.settings.dsn,
                min_size=min_size,
                max_size=max_size,
            )
            logger.info(
                "PostgreSQL 连接池创建成功 (min=%d, max=%d)", min_size, max_size
            )
        except Exception as exc:
            DOCSTORE_ERROR_TOTAL.inc()
            raise ConnectionError(f"PostgreSQL 连接池创建失败: {exc}") from exc

        async with self._pool.acquire() as conn:
            await conn.execute(_CREATE_TABLE_SQL)
        logger.info("PostgreSQL 文档表初始化完成")

    async def close(self) -> None:
        """关闭连接池并释放所有连接。"""
        if self._pool is not None:
            try:
                await self._pool.close()
                logger.info("PostgreSQL 连接池已关闭")
            except Exception as exc:
                DOCSTORE_ERROR_TOTAL.inc()
                logger.warning("关闭连接池异常: %s", exc)
            finally:
                self._pool = None

    # ------------------------------------------------------------------
    # Connection pool monitoring
    # ------------------------------------------------------------------

    def pool_stats(self) -> Dict[str, Any]:
        """获取连接池状态信息。

        Returns:
            包含 ``minsize``、``maxsize``、``size``、``idle``、``used`` 的字典。
            若连接池未初始化则返回空字典。
        """
        if self._pool is None:
            return {}

        return {
            "minsize": self._pool.get_min_size(),
            "maxsize": self._pool.get_max_size(),
            "size": self._pool.get_size(),
            "idle": self._pool.get_idle_size(),
            "used": self._pool.get_size() - self._pool.get_idle_size(),
        }

    # ------------------------------------------------------------------
    # Retry helper for transient errors
    # ------------------------------------------------------------------

    async def _execute_with_retry(
        self,
        coro_factory,
        *args,
        max_retries: int = _MAX_RETRIES,
        **kwargs,
    ):
        """执行数据库操作，遇到瞬态错误自动重试。

        Args:
            coro_factory: 返回协程的可调用对象。
            *args: 传递给 coro_factory 的位置参数。
            max_retries: 最大重试次数。
            **kwargs: 传递给 coro_factory 的关键字参数。

        Returns:
            协程执行结果。

        Raises:
            Exception: 所有重试均失败时抛出最后一次异常。
        """
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                return await coro_factory(*args, **kwargs)
            except Exception as exc:
                # Classify transient errors: connection issues, serialization failures
                exc_msg = str(exc).lower()
                is_transient = any(
                    kw in exc_msg
                    for kw in (
                        "connection",
                        "timeout",
                        "deadlock",
                        "serialization",
                        "could not",
                    )
                )
                if not is_transient or attempt == max_retries:
                    raise
                last_exc = exc
                delay = _RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "数据库瞬态错误 (attempt=%d/%d): %s, %.2fs 后重试",
                    attempt,
                    max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)
        raise last_exc  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Transaction context manager
    # ------------------------------------------------------------------

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator:
        """提供事务上下文管理器。

        在 ``async with`` 块内执行的操作要么全部提交，要么全部回滚。

        Yields:
            asyncpg.Connection 绑定到当前事务。

        Example::

            async with store.transaction() as conn:
                await conn.execute("INSERT INTO ...")
                await conn.execute("UPDATE ...")
        """
        if self._pool is None:
            raise RuntimeError("DocStore 未初始化，请先调用 init()")

        async with self._pool.acquire() as conn:
            tr = conn.transaction()
            await tr.start()
            try:
                yield conn
                await tr.commit()
            except Exception:
                await tr.rollback()
                raise

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    async def get_by_doc_ids(self, doc_ids: List[str]) -> List[Dict[str, Any]]:
        """根据 doc_id 列表快速聚合文档原文。

        Args:
            doc_ids: 文档 ID 列表。

        Returns:
            文档字典列表，每项含 ``doc_id``、``title``、``content``、``source``、``metadata``。
        """
        if not doc_ids:
            return []

        DOCSTORE_QUERY_TOTAL.inc()
        t0 = time.monotonic()

        async def _query():
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT doc_id, title, content, source, metadata "
                    "FROM documents WHERE doc_id = ANY($1)",
                    doc_ids,
                )
            return rows

        rows = await self._execute_with_retry(_query)
        DOCSTORE_QUERY_LATENCY.observe(time.monotonic() - t0)
        logger.info("按 doc_id 查询返回 %d 条文档", len(rows))

        return [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "content": r["content"],
                "source": r["source"],
                "metadata": r["metadata"],
            }
            for r in rows
        ]

    async def get_by_id(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """根据单个 doc_id 查询文档。

        Args:
            doc_id: 文档 ID。

        Returns:
            文档字典，不存在时返回 None。
        """
        DOCSTORE_QUERY_TOTAL.inc()
        t0 = time.monotonic()

        async def _query():
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT doc_id, title, content, source, metadata "
                    "FROM documents WHERE doc_id = $1",
                    doc_id,
                )
            return row

        row = await self._execute_with_retry(_query)
        DOCSTORE_QUERY_LATENCY.observe(time.monotonic() - t0)

        if row is None:
            return None
        return {
            "doc_id": row["doc_id"],
            "title": row["title"],
            "content": row["content"],
            "source": row["source"],
            "metadata": row["metadata"],
        }

    async def search_by_content(
        self,
        query: str,
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """全文搜索文档内容。

        使用 PostgreSQL ``to_tsvector`` + ``to_tsquery`` 实现全文检索。

        Args:
            query: 搜索关键词，空格分隔多个词。
            limit: 返回结果数量上限。
            offset: 分页偏移量。

        Returns:
            匹配的文档列表，按相关性降序排列。
        """
        DOCSTORE_QUERY_TOTAL.inc()
        t0 = time.monotonic()

        # Build tsquery: split words and join with &
        tokens = query.strip().split()
        if not tokens:
            return []
        ts_query = " & ".join(tokens)

        async def _query():
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT doc_id, title, content, source, metadata, "
                    "ts_rank(to_tsvector('simple', content), to_tsquery('simple', $1)) AS rank "
                    "FROM documents "
                    "WHERE to_tsvector('simple', content) @@ to_tsquery('simple', $1) "
                    "ORDER BY rank DESC LIMIT $2 OFFSET $3",
                    ts_query,
                    limit,
                    offset,
                )
            return rows

        rows = await self._execute_with_retry(_query)
        DOCSTORE_QUERY_LATENCY.observe(time.monotonic() - t0)
        logger.info("全文搜索 '%s' 返回 %d 条结果", query, len(rows))

        return [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "content": r["content"],
                "source": r["source"],
                "metadata": r["metadata"],
                "rank": float(r["rank"]),
            }
            for r in rows
        ]

    async def search_by_metadata(
        self,
        metadata_query: Dict[str, Any],
        *,
        limit: int = 20,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """基于 JSONB 元数据查询文档。

        Args:
            metadata_query: 元数据过滤条件，键值对形式。
                例如 ``{"category": "tech"}`` 查询 metadata->>category = 'tech'。
            limit: 返回结果数量上限。
            offset: 分页偏移量。

        Returns:
            匹配的文档列表。
        """
        if not metadata_query:
            return []

        DOCSTORE_QUERY_TOTAL.inc()
        t0 = time.monotonic()

        # Build WHERE clauses dynamically
        conditions: List[str] = []
        params: List[Any] = []
        param_idx = 1
        for key, value in metadata_query.items():
            conditions.append(f"metadata->>${param_idx} = ${param_idx + 1}::text")
            params.extend([key, str(value)])
            param_idx += 2

        where_clause = " AND ".join(conditions)
        params.extend([limit, offset])

        async def _query():
            sql = (
                "SELECT doc_id, title, content, source, metadata "
                f"FROM documents WHERE {where_clause} "
                f"ORDER BY created_at DESC LIMIT ${param_idx} OFFSET ${param_idx + 1}"
            )
            async with self._pool.acquire() as conn:
                rows = await conn.fetch(sql, *params)
            return rows

        rows = await self._execute_with_retry(_query)
        DOCSTORE_QUERY_LATENCY.observe(time.monotonic() - t0)
        logger.info("元数据查询返回 %d 条结果", len(rows))

        return [
            {
                "doc_id": r["doc_id"],
                "title": r["title"],
                "content": r["content"],
                "source": r["source"],
                "metadata": r["metadata"],
            }
            for r in rows
        ]

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    async def batch_insert(self, docs: List[dict]) -> None:
        """批量插入文档（ON CONFLICT 更新）。

        Args:
            docs: 文档列表，每项须含 ``doc_id`` 和 ``content``。
        """
        if not docs:
            return

        async def _insert():
            async with self._pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO documents (doc_id, title, content, source, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata
                    """,
                    [
                        (
                            d["doc_id"],
                            d.get("title", ""),
                            d["content"],
                            d.get("source", ""),
                            json.dumps(d.get("metadata", {}), ensure_ascii=False),
                        )
                        for d in docs
                    ],
                )

        await self._execute_with_retry(_insert)
        DOCSTORE_QUERY_TOTAL.inc()
        logger.info("写入 %d 条文档到 PostgreSQL", len(docs))

    async def upsert(self, doc: dict) -> bool:
        """插入或更新单篇文档。

        Args:
            doc: 文档字典，须含 ``doc_id`` 和 ``content``。

        Returns:
            ``True`` 表示新建，``False`` 表示更新。
        """
        DOCSTORE_QUERY_TOTAL.inc()

        async def _upsert():
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO documents (doc_id, title, content, source, metadata)
                    VALUES ($1, $2, $3, $4, $5::jsonb)
                    ON CONFLICT (doc_id) DO UPDATE SET
                        title = EXCLUDED.title,
                        content = EXCLUDED.content,
                        source = EXCLUDED.source,
                        metadata = EXCLUDED.metadata
                    RETURNING (xmax = 0) AS inserted
                    """,
                    doc["doc_id"],
                    doc.get("title", ""),
                    doc["content"],
                    doc.get("source", ""),
                    json.dumps(doc.get("metadata", {}), ensure_ascii=False),
                )
            return row

        row = await self._execute_with_retry(_upsert)
        is_insert = row["inserted"] if row else True
        logger.info("文档 %s: %s", doc["doc_id"], "新建" if is_insert else "更新")
        return is_insert

    async def update_document(
        self,
        doc_id: str,
        *,
        title: Optional[str] = None,
        content: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """更新指定文档的字段。

        仅更新提供的字段，未提供的字段保持不变。metadata 字段会与已有值深度合并。

        Args:
            doc_id: 文档 ID。
            title: 新标题，None 表示不更新。
            content: 新内容，None 表示不更新。
            source: 新来源，None 表示不更新。
            metadata: 新元数据（与已有值深度合并），None 表示不更新。

        Returns:
            ``True`` 表示更新成功，``False`` 表示文档不存在。
        """
        set_clauses: List[str] = []
        params: List[Any] = []
        param_idx = 1

        if title is not None:
            set_clauses.append(f"title = ${param_idx}")
            params.append(title)
            param_idx += 1

        if content is not None:
            set_clauses.append(f"content = ${param_idx}")
            params.append(content)
            param_idx += 1

        if source is not None:
            set_clauses.append(f"source = ${param_idx}")
            params.append(source)
            param_idx += 1

        if metadata is not None:
            set_clauses.append(f"metadata = metadata || ${param_idx}::jsonb")
            params.append(json.dumps(metadata, ensure_ascii=False))
            param_idx += 1

        if not set_clauses:
            return False

        params.append(doc_id)

        async def _update():
            sql = (
                f"UPDATE documents SET {', '.join(set_clauses)} "
                f"WHERE doc_id = ${param_idx} "
                f"RETURNING doc_id"
            )
            async with self._pool.acquire() as conn:
                row = await conn.fetchrow(sql, *params)
            return row

        row = await self._execute_with_retry(_update)
        DOCSTORE_QUERY_TOTAL.inc()
        success = row is not None
        if success:
            logger.info("文档 %s 更新成功", doc_id)
        else:
            logger.warning("文档 %s 不存在，更新失败", doc_id)
        return success

    # ------------------------------------------------------------------
    # Delete operations
    # ------------------------------------------------------------------

    async def delete_by_doc_ids(self, doc_ids: List[str]) -> int:
        """按 doc_id 列表删除文档。

        Args:
            doc_ids: 待删除的文档 ID 列表。

        Returns:
            实际删除的行数。
        """
        if not doc_ids:
            return 0

        DOCSTORE_QUERY_TOTAL.inc()

        async def _delete():
            async with self._pool.acquire() as conn:
                result = await conn.execute(
                    "DELETE FROM documents WHERE doc_id = ANY($1)",
                    doc_ids,
                )
            return result

        result = await self._execute_with_retry(_delete)
        # asyncpg returns "DELETE N"
        deleted = int(result.split()[-1]) if result else 0
        logger.info("删除 %d 条文档 (请求 %d 条)", deleted, len(doc_ids))
        return deleted

    # ------------------------------------------------------------------
    # Utility queries
    # ------------------------------------------------------------------

    async def count_documents(self) -> int:
        """获取文档总数。

        Returns:
            文档数量。
        """
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow("SELECT COUNT(*) AS cnt FROM documents")
        return int(row["cnt"]) if row else 0

    async def get_all_doc_ids(self, *, limit: int = 10000) -> List[str]:
        """获取所有文档 ID（用于缓存预热等场景）。

        Args:
            limit: 最大返回数量，防止内存溢出。

        Returns:
            doc_id 列表。
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT doc_id FROM documents ORDER BY created_at DESC LIMIT $1",
                limit,
            )
        logger.info("获取全部 doc_id: %d 条", len(rows))
        return [r["doc_id"] for r in rows]
