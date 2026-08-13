"""C++ 精排服务调用工具 —— 通过 gRPC 异步/同步调用精排服务，带回退与重试。"""

from __future__ import annotations

import asyncio
import math
import re
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

from python.config.settings import RerankerSettings
from python.utils.logging_config import get_logger
from python.utils.metrics import RERANK_LATENCY

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------
from prometheus_client import Counter as _Counter

RERANK_CALL_TOTAL = _Counter(
    "funnelrag_rerank_call_total", "Total rerank service calls"
)
RERANK_FALLBACK_TOTAL = _Counter(
    "funnelrag_rerank_fallback_total", "Total rerank fallback invocations"
)
RERANK_ERROR_TOTAL = _Counter("funnelrag_rerank_error_total", "Total rerank errors")

# Maximum text length (characters) per document sent to the reranker
_MAX_TEXT_LENGTH = 8192


class RerankTool:
    """精排工具，调用 C++ gRPC 精排服务对候选文档重排序。

    当 gRPC 服务不可用时，自动降级到基于 TF-IDF 的本地重排序。
    支持 synchronous / asynchronous / batch 三种调用模式。

    Attributes:
        name: 工具名称标识。
        description: 工具功能描述。
    """

    name: str = "rerank"
    description: str = (
        "对候选文档进行精排重排序，基于 Cross-Encoder 模型计算相关性分数。"
        "输入查询文本和候选文档列表，返回按相关性排序的结果。"
    )

    # ------------------------------------------------------------------
    # Construction & gRPC channel lifecycle
    # ------------------------------------------------------------------

    def __init__(
        self,
        reranker_settings: Optional[RerankerSettings] = None,
        *,
        max_retries: int = 3,
        base_retry_delay: float = 0.1,
        max_text_length: int = _MAX_TEXT_LENGTH,
    ) -> None:
        """初始化 RerankTool。

        Args:
            reranker_settings: 精排服务配置，为 None 时使用默认值。
            max_retries: gRPC 调用最大重试次数。
            base_retry_delay: 重试基础延迟秒数（指数退避）。
            max_text_length: 文档文本最大截断长度。
        """
        self.settings = reranker_settings or RerankerSettings()
        self.max_retries = max_retries
        self.base_retry_delay = base_retry_delay
        self.max_text_length = max_text_length
        self._stub = None
        self._channel = None

    def _create_channel(self):
        """创建 gRPC channel，配置 keepalive 与消息大小限制。

        Returns:
            gRPC channel 实例，或当 grpc 包不可用时返回 None。
        """
        try:
            import grpc

            target = f"{self.settings.host}:{self.settings.port}"
            options = [
                ("grpc.keepalive_time_ms", 30_000),
                ("grpc.keepalive_timeout_ms", 10_000),
                ("grpc.keepalive_permit_without_calls", 1),
                ("grpc.max_receive_message_length", 64 * 1024 * 1024),
                ("grpc.max_send_message_length", 64 * 1024 * 1024),
                ("grpc.http2.max_pings_without_data", 0),
            ]
            channel = grpc.insecure_channel(target, options=options)
            logger.info("gRPC channel 已创建 (target=%s)", target)
            return channel
        except ImportError:
            logger.warning("grpcio 包未安装，gRPC 精排服务不可用")
            return None

    def _get_stub(self):
        """懒加载 gRPC stub，带重试创建逻辑。

        Returns:
            RerankerServiceStub 实例，或当 proto 模块不可用时返回 None。
        """
        if self._stub is not None:
            return self._stub

        try:
            from proto import reranker_pb2_grpc

            channel = self._create_channel()
            if channel is None:
                return None
            self._channel = channel
            self._stub = reranker_pb2_grpc.RerankerServiceStub(channel)
            logger.info("RerankerServiceStub 创建成功")
            return self._stub
        except ImportError:
            logger.warning("gRPC proto 模块未生成，精排服务不可用")
            return None
        except Exception as exc:
            RERANK_ERROR_TOTAL.inc()
            logger.error("创建 gRPC stub 失败: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Health check
    # ------------------------------------------------------------------

    async def health_check(self) -> Dict[str, Any]:
        """检查 gRPC 精排服务健康状态。

        Returns:
            字典包含 ``status``、``latency_ms`` 和 ``detail``。
        """
        stub = self._get_stub()
        if stub is None:
            return {"status": "unavailable", "latency_ms": 0, "detail": "stub 未初始化"}

        start = time.monotonic()
        try:
            from proto import reranker_pb2

            request = reranker_pb2.RerankRequest(query="", items=[], top_k=1)
            await asyncio.wait_for(
                stub.Rerank(request), timeout=self.settings.timeout * 2
            )
            latency_ms = (time.monotonic() - start) * 1000
            logger.info("精排服务健康检查通过 (%.1fms)", latency_ms)
            return {
                "status": "healthy",
                "latency_ms": round(latency_ms, 2),
                "detail": None,
            }
        except Exception as exc:
            latency_ms = (time.monotonic() - start) * 1000
            logger.error("精排服务健康检查失败: %s", exc)
            return {
                "status": "unhealthy",
                "latency_ms": round(latency_ms, 2),
                "detail": str(exc),
            }

    # ------------------------------------------------------------------
    # Text preprocessing
    # ------------------------------------------------------------------

    def _preprocess_text(self, text: str) -> str:
        """文本预处理：归一化空白、截断超长文本。

        Args:
            text: 原始文本。

        Returns:
            处理后的文本。
        """
        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()
        # Truncate
        if len(text) > self.max_text_length:
            text = text[: self.max_text_length]
        return text

    # ------------------------------------------------------------------
    # Async rerank (primary)
    # ------------------------------------------------------------------

    @RERANK_LATENCY.time()
    async def arerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """异步调用精排服务，带重试与回退。

        Args:
            query: 查询文本。
            documents: 候选文档列表，每项须含 ``doc_id`` 和 ``text``。
            top_k: 返回结果数量。

        Returns:
            按相关性降序排列的文档列表，每项含 ``doc_id``、``score``、``text``。
        """
        if not documents:
            return []

        query = self._preprocess_text(query)
        for doc in documents:
            doc["text"] = self._preprocess_text(doc.get("text", ""))

        RERANK_CALL_TOTAL.inc()

        stub = self._get_stub()
        if stub is None:
            logger.warning("精排服务不可用，回退本地排序")
            RERANK_FALLBACK_TOTAL.inc()
            return self.fallback_to_local_rerank(query, documents, top_k)

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                from proto import reranker_pb2

                items = [
                    reranker_pb2.RerankItem(
                        doc_id=str(d.get("doc_id", "")),
                        text=d.get("text", ""),
                    )
                    for d in documents
                ]
                request = reranker_pb2.RerankRequest(
                    query=query, items=items, top_k=top_k
                )

                response = await asyncio.wait_for(
                    stub.Rerank(request),
                    timeout=self.settings.timeout,
                )

                results = [
                    {"doc_id": item.doc_id, "score": item.score, "text": item.text}
                    for item in response.items
                ]
                logger.info(
                    "精排完成 (attempt=%d)，返回 %d 条结果", attempt, len(results)
                )
                return results

            except asyncio.TimeoutError:
                last_exc = TimeoutError("精排服务超时")
                delay = self.base_retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "精排超时 (attempt=%d/%d), %.2fs 后重试",
                    attempt,
                    self.max_retries,
                    delay,
                )
                await asyncio.sleep(delay)

            except Exception as exc:
                last_exc = exc
                delay = self.base_retry_delay * (2 ** (attempt - 1))
                logger.warning(
                    "精排调用失败 (attempt=%d/%d): %s, %.2fs 后重试",
                    attempt,
                    self.max_retries,
                    exc,
                    delay,
                )
                await asyncio.sleep(delay)

        RERANK_ERROR_TOTAL.inc()
        logger.error("精排服务所有重试失败: %s，回退本地排序", last_exc)
        RERANK_FALLBACK_TOTAL.inc()
        return self.fallback_to_local_rerank(query, documents, top_k)

    # ------------------------------------------------------------------
    # Sync rerank
    # ------------------------------------------------------------------

    def sync_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """同步调用精排服务。

        在同步上下文中包装异步调用，适用于非 async 代码路径。

        Args:
            query: 查询文本。
            documents: 候选文档列表。
            top_k: 返回结果数量。

        Returns:
            按相关性降序排列的文档列表。
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(
                    asyncio.run,
                    self.arerank(query, documents, top_k),
                )
                return future.result(
                    timeout=self.settings.timeout * self.max_retries + 5
                )
        else:
            return asyncio.run(self.arerank(query, documents, top_k))

    # ------------------------------------------------------------------
    # Batch rerank
    # ------------------------------------------------------------------

    async def batch_rerank(
        self,
        queries: Sequence[str],
        documents_per_query: Sequence[List[Dict[str, Any]]],
        top_k: int = 5,
    ) -> List[List[Dict[str, Any]]]:
        """批量精排：对多个查询并行执行精排。

        Args:
            queries: 查询文本列表。
            documents_per_query: 每个查询对应的候选文档列表，与 queries 一一对应。
            top_k: 每个查询返回的结果数量。

        Returns:
            列表的列表，外层与 queries 一一对应。

        Raises:
            ValueError: queries 与 documents_per_query 长度不一致时抛出。
        """
        if len(queries) != len(documents_per_query):
            raise ValueError("queries 与 documents_per_query 长度不一致")

        tasks = [
            self.arerank(query, docs, top_k)
            for query, docs in zip(queries, documents_per_query)
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        output: List[List[Dict[str, Any]]] = []
        for idx, result in enumerate(results):
            if isinstance(result, Exception):
                RERANK_ERROR_TOTAL.inc()
                logger.error("批量精排第 %d 个查询失败: %s", idx, result)
                output.append(documents_per_query[idx][:top_k])
            else:
                output.append(result)
        return output

    # ------------------------------------------------------------------
    # Local fallback (TF-IDF based)
    # ------------------------------------------------------------------

    def fallback_to_local_rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """当 gRPC 精排服务不可用时，基于简单 TF-IDF 加权的相关性回退排序。

        使用词频-逆文档频率近似计算查询与文档的相关度，
        仅作为应急回退，精度远低于 Cross-Encoder。

        Args:
            query: 查询文本。
            documents: 候选文档列表。
            top_k: 返回结果数量。

        Returns:
            按近似相关性降序排列的文档列表。
        """
        logger.info(
            "使用本地 TF-IDF 回退精排 (docs=%d, top_k=%d)", len(documents), top_k
        )

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return documents[:top_k]

        # Document frequency
        df = Counter()
        doc_tokens_list: List[List[str]] = []
        for doc in documents:
            tokens = self._tokenize(doc.get("text", ""))
            doc_tokens_list.append(tokens)
            unique = set(tokens)
            for t in unique:
                df[t] += 1

        n_docs = len(documents)
        scored: List[tuple] = []
        for idx, tokens in enumerate(doc_tokens_list):
            tf = Counter(tokens)
            score = 0.0
            for qt in query_tokens:
                if qt in tf:
                    tf_val = 1 + math.log(tf[qt]) if tf[qt] > 0 else 0
                    idf_val = math.log((n_docs + 1) / (df.get(qt, 0) + 1)) + 1
                    score += tf_val * idf_val
            scored.append((score, idx))

        scored.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scored[:top_k]:
            doc = dict(documents[idx])
            doc["score"] = score
            results.append(doc)

        logger.info("本地回退精排完成，返回 %d 条结果", len(results))
        return results

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词：按空白/标点切分并转小写。

        Args:
            text: 原始文本。

        Returns:
            token 列表。
        """
        return re.findall(r"\w+", text.lower())

    # ------------------------------------------------------------------
    # Resource cleanup
    # ------------------------------------------------------------------

    def close(self) -> None:
        """关闭 gRPC channel，释放资源。"""
        if self._channel is not None:
            try:
                self._channel.close()
                logger.info("gRPC channel 已关闭")
            except Exception as exc:
                logger.warning("关闭 gRPC channel 异常: %s", exc)
            finally:
                self._channel = None
                self._stub = None

    def __del__(self) -> None:
        """析构时自动释放 gRPC 资源。"""
        self.close()
