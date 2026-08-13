"""Milvus 向量粗筛工具 —— 提供向量检索、文档入库、集合管理等全生命周期能力。"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Sequence

from pymilvus import MilvusClient, DataType, CollectionSchema, FieldSchema

from python.config.settings import MilvusSettings
from python.utils.logging_config import get_logger
from python.utils.metrics import MILVUS_SEARCH_LATENCY

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters for instrumentation
# ---------------------------------------------------------------------------
from prometheus_client import Counter as _Counter

MILVUS_INSERT_TOTAL = _Counter(
    "funnelrag_milvus_insert_total", "Total documents inserted into Milvus"
)
MILVUS_DELETE_TOTAL = _Counter(
    "funnelrag_milvus_delete_total", "Total documents deleted from Milvus"
)
MILVUS_ERROR_TOTAL = _Counter(
    "funnelrag_milvus_error_total", "Total Milvus operation errors"
)


class MilvusSearchTool:
    """Milvus 向量检索工具，用于粗筛阶段。

    提供向量近邻搜索、混合搜索（向量 + 标量过滤）、文档向量入库、
    集合统计信息查询、连接健康检查等能力，内置重试逻辑和指标埋点。

    Attributes:
        name: 工具名称标识。
        description: 工具功能描述，供 Agent 识别调用时机。
    """

    name: str = "milvus_search"
    description: str = (
        "在知识库中进行语义检索，返回与查询最相关的文档。"
        "当需要查找企业内部文档、技术规范、产品手册等内容时使用此工具。"
    )

    # ------------------------------------------------------------------
    # Construction & connection lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        milvus_settings: Optional[MilvusSettings] = None,
        *,
        max_retries: int = 3,
        retry_delay: float = 0.5,
        vector_dim: int = 1024,
    ) -> None:
        """初始化 MilvusSearchTool。

        Args:
            milvus_settings: Milvus 连接配置，为 None 时使用默认值。
            max_retries: 连接/操作最大重试次数。
            retry_delay: 重试间隔基础秒数（指数退避）。
            vector_dim: 向量维度，用于建表和输入校验。
        """
        self.settings = milvus_settings or MilvusSettings()
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.vector_dim = vector_dim
        self._client: Optional[MilvusClient] = None
        self._async_client = None

    def _get_client(self) -> MilvusClient:
        """获取同步 Milvus 客户端，带重试逻辑。

        Returns:
            已连接的 MilvusClient 实例。

        Raises:
            ConnectionError: 所有重试均失败时抛出。
        """
        if self._client is not None:
            return self._client

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                uri = f"http://{self.settings.host}:{self.settings.port}"
                self._client = MilvusClient(uri=uri)
                logger.info("Milvus 客户端连接成功 (attempt=%d, uri=%s)", attempt, uri)
                return self._client
            except Exception as exc:
                last_exc = exc
                delay = self.retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "Milvus 连接失败 (attempt=%d/%d): %s, %.1fs 后重试",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                time.sleep(delay)

        MILVUS_ERROR_TOTAL.inc()
        raise ConnectionError(
            f"无法连接 Milvus ({self.settings.host}:{self.settings.port}): {last_exc}"
        )

    async def _get_async_client(self):
        """获取异步 Milvus 客户端（基于 pymilvus grpc aio）。

        Returns:
            异步 gRPC 客户端实例。

        Raises:
            ConnectionError: 连接失败时抛出。
        """
        if self._async_client is not None:
            return self._async_client

        try:
            from pymilvus import AsyncMilvusClient

            uri = f"http://{self.settings.host}:{self.settings.port}"
            self._async_client = AsyncMilvusClient(uri=uri)
            logger.info("Milvus 异步客户端连接成功 (uri=%s)", uri)
            return self._async_client
        except ImportError:
            logger.warning("pymilvus AsyncMilvusClient 不可用，回退同步客户端")
            return None
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("Milvus 异步客户端连接失败: %s", exc)
            raise ConnectionError(f"Milvus 异步连接失败: {exc}") from exc

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    def health_check(self) -> Dict[str, Any]:
        """检查 Milvus 连接健康状态。

        Returns:
            字典包含 ``status`` (``"healthy"`` / ``"unhealthy"``)、
            ``latency_ms`` 和 ``detail`` 字段。
        """
        start = time.monotonic()
        try:
            client = self._get_client()
            client.list_collections()
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("Milvus 健康检查通过 (%.1fms)", latency_ms)
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "detail": None,
            }
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("Milvus 健康检查失败: %s", exc)
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "detail": str(exc),
            }

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def create_collection_if_not_exists(
        self,
        collection_name: Optional[str] = None,
        vector_dim: Optional[int] = None,
    ) -> bool:
        """若集合不存在则创建，使用默认 schema（向量 + doc_id + text）。

        Args:
            collection_name: 集合名称，默认使用 settings.collection。
            vector_dim: 向量维度，默认使用 self.vector_dim。

        Returns:
            ``True`` 表示新建集合，``False`` 表示集合已存在。
        """
        col = collection_name or self.settings.collection
        dim = vector_dim or self.vector_dim
        client = self._get_client()

        existing = client.list_collections()
        if col in existing:
            logger.info("集合 '%s' 已存在，跳过创建", col)
            return False

        schema = CollectionSchema(
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=255),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dim),
            ],
            description="FunnelRAG document vectors",
        )
        client.create_collection(
            collection_name=col,
            schema=schema,
        )
        # Create index on the vector field
        client.create_index(
            collection_name=col,
            field_name="vector",
            index_params={
                "index_type": "IVF_FLAT",
                "metric_type": "COSINE",
                "params": {"nlist": 1024},
            },
        )
        logger.info("集合 '%s' 创建成功 (dim=%d)", col, dim)
        return True

    # ------------------------------------------------------------------
    # Vector search (sync)
    # ------------------------------------------------------------------

    @MILVUS_SEARCH_LATENCY.time()
    def search(
        self,
        query_vector: List[float],
        top_k: int = 0,
        filter_expr: str = "",
    ) -> List[Dict[str, Any]]:
        """执行向量近邻检索。

        Args:
            query_vector: 查询向量，维度须与集合一致。
            top_k: 返回结果数量，0 表示使用默认值。
            filter_expr: Milvus 标量过滤表达式。

        Returns:
            检索结果列表，每项包含 ``doc_id``、``score``、``text``。

        Raises:
            ValueError: 向量维度不匹配时抛出。
        """
        self._validate_vector(query_vector)

        k = top_k or self.settings.top_k
        client = self._get_client()

        try:
            results = client.search(
                collection_name=self.settings.collection,
                data=[query_vector],
                limit=k,
                output_fields=["doc_id", "text"],
                filter=filter_expr or None,
            )
            if results and len(results) > 0:
                hits = results[0]
                logger.info("Milvus 检索返回 %d 条结果", len(hits))
                return [
                    {
                        "doc_id": hit["entity"].get("doc_id", ""),
                        "score": hit["distance"],
                        "text": hit["entity"].get("text", ""),
                    }
                    for hit in hits
                ]
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("Milvus 检索失败: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Async search
    # ------------------------------------------------------------------

    async def async_search(
        self,
        query_vector: List[float],
        top_k: int = 0,
        filter_expr: str = "",
    ) -> List[Dict[str, Any]]:
        """异步执行向量近邻检索。

        Args:
            query_vector: 查询向量。
            top_k: 返回结果数量。
            filter_expr: 标量过滤表达式。

        Returns:
            检索结果列表，格式同 :meth:`search`。
        """
        self._validate_vector(query_vector)

        k = top_k or self.settings.top_k
        client = await self._get_async_client()

        if client is None:
            logger.warning("异步客户端不可用，回退同步检索")
            return self.search(query_vector, top_k=k, filter_expr=filter_expr)

        t0 = time.monotonic()
        try:
            results = await client.search(
                collection_name=self.settings.collection,
                data=[query_vector],
                limit=k,
                output_fields=["doc_id", "text"],
                filter=filter_expr or None,
            )
            elapsed = time.monotonic() - t0
            logger.info("Milvus 异步检索耗时 %.3fs", elapsed)

            if results and len(results) > 0:
                return [
                    {
                        "doc_id": hit["entity"].get("doc_id", ""),
                        "score": hit["distance"],
                        "text": hit["entity"].get("text", ""),
                    }
                    for hit in results[0]
                ]
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("Milvus 异步检索失败: %s", exc)
        return []

    # ------------------------------------------------------------------
    # Hybrid search (vector + scalar filter)
    # ------------------------------------------------------------------

    def hybrid_search(
        self,
        query_vector: List[float],
        filter_expr: str,
        top_k: int = 0,
    ) -> List[Dict[str, Any]]:
        """向量 + 标量过滤的混合检索。

        在执行向量近邻搜索的同时，通过 Milvus filter 表达式限定
        搜索空间（如按 ``source`` 或 ``metadata`` 字段过滤）。

        Args:
            query_vector: 查询向量。
            filter_expr: Milvus 标量过滤表达式，例如 ``'source == "wiki"'``。
            top_k: 返回结果数量。

        Returns:
            检索结果列表，每项包含 ``doc_id``、``score``、``text``。
        """
        logger.info("执行混合检索 (filter=%s)", filter_expr)
        return self.search(query_vector, top_k=top_k, filter_expr=filter_expr)

    # ------------------------------------------------------------------
    # Document ingestion
    # ------------------------------------------------------------------

    def insert_vectors(
        self,
        vectors: Sequence[List[float]],
        doc_ids: Sequence[str],
        texts: Sequence[str],
    ) -> int:
        """将向量与对应文档信息写入集合。

        Args:
            vectors: 向量列表，每个向量维度须与集合一致。
            doc_ids: 文档 ID 列表，长度须与 vectors 一致。
            texts: 文档文本列表，长度须与 vectors 一致。

        Returns:
            成功写入的行数。

        Raises:
            ValueError: 输入长度不一致或向量维度不匹配时抛出。
        """
        if not (len(vectors) == len(doc_ids) == len(texts)):
            raise ValueError("vectors / doc_ids / texts 长度不一致")

        for vec in vectors:
            self._validate_vector(vec)

        client = self._get_client()
        data = [
            {"vector": vec, "doc_id": did, "text": txt}
            for vec, did, txt in zip(vectors, doc_ids, texts)
        ]

        try:
            client.insert(collection_name=self.settings.collection, data=data)
            MILVUS_INSERT_TOTAL.inc(len(vectors))
            logger.info("Milvus 写入 %d 条向量", len(vectors))
            return len(vectors)
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("Milvus 向量写入失败: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Document deletion
    # ------------------------------------------------------------------

    def delete_by_doc_ids(self, doc_ids: List[str]) -> int:
        """按 doc_id 删除向量记录。

        Args:
            doc_ids: 待删除的文档 ID 列表。

        Returns:
            请求删除的文档数量（Milvus 不一定返回实际删除行数）。
        """
        if not doc_ids:
            return 0

        client = self._get_client()
        try:
            client.delete(
                collection_name=self.settings.collection,
                filter=f"doc_id in {doc_ids}",
            )
            MILVUS_DELETE_TOTAL.inc(len(doc_ids))
            logger.info(
                "Milvus 删除 %d 条记录 (doc_ids=%s…)", len(doc_ids), doc_ids[:3]
            )
            return len(doc_ids)
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("Milvus 删除失败: %s", exc)
            raise

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------

    def get_collection_stats(
        self, collection_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """获取集合统计信息。

        Args:
            collection_name: 集合名称，默认使用 settings.collection。

        Returns:
            包含 ``row_count``、``index_type`` 等统计字段的字典。
        """
        col = collection_name or self.settings.collection
        client = self._get_client()

        try:
            info = client.describe_collection(col)
            stats = client.get_collection_stats(col)
            row_count = int(stats.get("row_count", 0))

            index_info: List[Dict[str, Any]] = []
            for idx in info.get("index_descriptions", []):
                index_info.append(
                    {
                        "field_name": idx.get("field_name"),
                        "index_type": idx.get("index_type"),
                    }
                )

            result = {
                "collection_name": col,
                "row_count": row_count,
                "indexes": index_info,
            }
            logger.info("集合 '%s' 统计: row_count=%d", col, row_count)
            return result
        except Exception as exc:
            MILVUS_ERROR_TOTAL.inc()
            logger.error("获取集合统计信息失败: %s", exc)
            return {
                "collection_name": col,
                "row_count": -1,
                "indexes": [],
                "error": str(exc),
            }

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def _validate_vector(self, vector: List[float]) -> None:
        """校验向量维度与配置一致。

        Args:
            vector: 待校验的向量。

        Raises:
            ValueError: 维度不匹配时抛出。
        """
        if len(vector) != self.vector_dim:
            raise ValueError(
                f"向量维度不匹配: 期望 {self.vector_dim}, 实际 {len(vector)}"
            )

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭同步客户端并释放连接资源。"""
        if self._client is not None:
            try:
                self._client.close()
                logger.info("Milvus 同步客户端已关闭")
            except Exception as exc:
                logger.warning("关闭 Milvus 同步客户端异常: %s", exc)
            finally:
                self._client = None

        if self._async_client is not None:
            try:
                asyncio.get_event_loop().run_until_complete(self._async_client.close())
                logger.info("Milvus 异步客户端已关闭")
            except Exception as exc:
                logger.warning("关闭 Milvus 异步客户端异常: %s", exc)
            finally:
                self._async_client = None

    def __del__(self) -> None:
        """析构时自动释放连接。"""
        self.close()
