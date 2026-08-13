"""FastAPI 路由定义 —— 提供查询、流式响应、对话、文档摄入、统计等 REST 接口。"""

import asyncio
import json
import time
from typing import Any, AsyncGenerator, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from python.agent.engine import FunnelRAGAgent
from python.config.settings import AppSettings
from python.schemas.models import QueryRequest, QueryResponse
from python.utils.logging_config import get_logger
from python.utils.metrics import QUERY_TOTAL

logger = get_logger(__name__)

router = APIRouter()

# 全局 Agent 实例（由 main.py 初始化时注入）
_agent: FunnelRAGAgent = None


def set_agent(agent: FunnelRAGAgent) -> None:
    """注入 Agent 实例，由应用生命周期调用。"""
    global _agent
    _agent = agent


# ---------------------------------------------------------------------------
# 依赖注入
# ---------------------------------------------------------------------------


def get_request_id(request: Request) -> str:
    """从请求状态中提取请求 ID。

    此 ID 由请求 ID 中间件注入，若不存在则生成临时 ID。

    Args:
        request: FastAPI 请求对象

    Returns:
        请求 ID 字符串
    """
    return getattr(request.state, "request_id", "unknown")


def require_agent() -> FunnelRAGAgent:
    """确保 Agent 已初始化，否则抛出 503。

    Returns:
        可用的 Agent 实例

    Raises:
        HTTPException: Agent 未初始化时
    """
    if _agent is None:
        raise HTTPException(status_code=503, detail="Agent 未初始化，服务暂时不可用")
    return _agent


# ---------------------------------------------------------------------------
# 扩展请求/响应模型
# ---------------------------------------------------------------------------


class ConversationRequest(BaseModel):
    """多轮对话请求。"""

    question: str = Field(..., min_length=1, max_length=2000, description="用户问题")
    conversation_id: Optional[str] = Field(
        default=None, description="对话 ID，为空则新开对话"
    )
    include_sources: bool = Field(default=False, description="是否返回来源文档")
    temperature_override: Optional[float] = Field(
        default=None, ge=0.0, le=2.0, description="覆盖默认温度"
    )


class ConversationResponse(BaseModel):
    """多轮对话响应。"""

    answer: str
    thinking_chain: Optional[List[Dict[str, Any]]] = None
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    conversation_id: str = Field(default="", description="对话 ID")


class IngestRequest(BaseModel):
    """文档摄入请求。"""

    documents: List[Dict[str, Any]] = Field(
        ..., min_length=1, description="待摄入文档列表，每个元素须含 text 字段"
    )
    collection: Optional[str] = Field(default=None, description="目标集合名称")
    batch_size: int = Field(default=100, ge=1, le=1000, description="批量摄入大小")


class IngestResponse(BaseModel):
    """文档摄入响应。"""

    ingested_count: int = Field(description="成功摄入的文档数")
    failed_count: int = Field(default=0, description="失败的文档数")
    errors: List[str] = Field(default_factory=list, description="错误信息列表")


class StatsResponse(BaseModel):
    """系统统计响应。"""

    query_total: int
    agent_info: Dict[str, Any]


# ---------------------------------------------------------------------------
# 速率限制存根
# ---------------------------------------------------------------------------

_rate_limit_store: Dict[str, List[float]] = {}
_RATE_LIMIT_WINDOW: float = 60.0  # 窗口秒数
_RATE_LIMIT_MAX_REQUESTS: int = 60  # 窗口内最大请求数


def _check_rate_limit(client_id: str) -> bool:
    """简易速率限制检查（基于滑动窗口）。

    Args:
        client_id: 客户端标识（如 IP 或 API Key）

    Returns:
        True 表示通过，False 表示超限
    """
    now = time.time()
    window_start = now - _RATE_LIMIT_WINDOW

    timestamps = _rate_limit_store.get(client_id, [])
    # 清理过期时间戳
    timestamps = [ts for ts in timestamps if ts > window_start]

    if len(timestamps) >= _RATE_LIMIT_MAX_REQUESTS:
        return False

    timestamps.append(now)
    _rate_limit_store[client_id] = timestamps
    return True


# ---------------------------------------------------------------------------
# 查询端点
# ---------------------------------------------------------------------------


@router.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
    include_sources: bool = Query(default=False, description="是否返回来源文档"),
    temperature_override: Optional[float] = Query(
        default=None, ge=0.0, le=2.0, description="覆盖默认温度"
    ),
):
    """普通查询接口。

    支持通过查询参数控制是否返回来源文档和覆盖模型温度。

    Args:
        req: 查询请求体
        request: FastAPI 请求对象
        request_id: 请求 ID（依赖注入）
        include_sources: 是否返回来源文档
        temperature_override: 覆盖默认温度

    Returns:
        查询响应
    """
    QUERY_TOTAL.inc()
    agent = require_agent()

    # 速率限制
    client_id = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    logger.info(
        "[%s] 查询请求: %s (sources=%s)",
        request_id[:8],
        req.question[:50],
        include_sources,
    )

    try:
        if include_sources:
            result = await agent.query_with_sources(req.question)
        else:
            result = await agent.aquery(req.question)
    except Exception as exc:
        logger.error("[%s] 查询执行异常: %s", request_id[:8], exc)
        raise HTTPException(status_code=500, detail=f"查询执行失败: {exc}") from exc

    sources = result.get("sources", [])
    if include_sources and "formatted_sources" in result:
        sources = result.get("sources", [])

    return QueryResponse(
        answer=result["answer"],
        sources=[
            s.get("doc_id", "") if isinstance(s, dict) else str(s) for s in sources
        ],
        thinking_chain=result.get("thinking_chain"),
    )


# ---------------------------------------------------------------------------
# 流式查询端点
# ---------------------------------------------------------------------------


@router.post("/query/stream")
async def query_stream(
    req: QueryRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
):
    """SSE 流式响应接口。

    产生的事件类型：
    - start: 查询开始，包含 question
    - thinking: 工具调用，包含 tool 和 input
    - tool_result: 工具返回，包含 tool 和 output
    - token: LLM 输出片段，包含 content
    - sources: 来源文档，包含 sources 列表
    - metadata: 元信息（耗时等）
    - answer: 最终答案
    - error: 错误信息
    - done: 流结束

    Args:
        req: 查询请求体
        request: FastAPI 请求对象
        request_id: 请求 ID

    Returns:
        SSE StreamingResponse
    """
    QUERY_TOTAL.inc()
    agent = require_agent()

    async def event_generator() -> AsyncGenerator[str, None]:
        start_time = time.perf_counter()

        # 发送开始信号
        yield f"data: {json.dumps({'type': 'start', 'question': req.question, 'request_id': request_id}, ensure_ascii=False)}\n\n"

        try:
            async for event in agent.stream_query(req.question):
                event_type = event.get("type", "")

                if event_type == "thinking":
                    yield f"data: {json.dumps({'type': 'thinking', 'tool': event.get('tool'), 'input': event.get('input')}, ensure_ascii=False)}\n\n"
                elif event_type == "tool_result":
                    yield f"data: {json.dumps({'type': 'tool_result', 'tool': event.get('tool'), 'output': event.get('output')}, ensure_ascii=False)}\n\n"
                elif event_type == "token":
                    yield f"data: {json.dumps({'type': 'token', 'content': event.get('content')}, ensure_ascii=False)}\n\n"
                elif event_type == "sources":
                    yield f"data: {json.dumps({'type': 'sources', 'sources': event.get('sources')}, ensure_ascii=False)}\n\n"
                elif event_type == "answer":
                    elapsed = time.perf_counter() - start_time
                    # 发送元信息
                    yield f"data: {json.dumps({'type': 'metadata', 'elapsed_seconds': round(elapsed, 3)}, ensure_ascii=False)}\n\n"
                    yield f"data: {json.dumps({'type': 'answer', 'answer': event.get('answer')}, ensure_ascii=False)}\n\n"
                elif event_type == "error":
                    yield f"data: {json.dumps({'type': 'error', 'message': event.get('message')}, ensure_ascii=False)}\n\n"
                elif event_type == "done":
                    yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

                await asyncio.sleep(0)  # 让出控制权

        except Exception as e:
            logger.error("[%s] 流式查询异常: %s", request_id[:8], e)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"
            yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Request-ID": request_id,
        },
    )


# ---------------------------------------------------------------------------
# 多轮对话端点
# ---------------------------------------------------------------------------


@router.post("/query/conversation", response_model=ConversationResponse)
async def query_conversation(
    req: ConversationRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
):
    """多轮对话接口，支持对话历史和来源返回。

    当 conversation_id 不匹配当前 Agent 会话时，将重置对话历史。
    如果 conversation_id 为空，则开始新的对话。

    Args:
        req: 对话请求体
        request: FastAPI 请求对象
        request_id: 请求 ID

    Returns:
        对话响应
    """
    QUERY_TOTAL.inc()
    agent = require_agent()

    # 速率限制
    client_id = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client_id):
        raise HTTPException(status_code=429, detail="请求过于频繁，请稍后重试")

    # 对话 ID 管理
    current_conv_id = agent.get_agent_info().get("conversation_id", "")
    if req.conversation_id and req.conversation_id != current_conv_id:
        agent.reset_conversation()
        logger.info("[%s] 对话切换: %s -> new", request_id[:8], req.conversation_id[:8])
    elif not req.conversation_id:
        agent.reset_conversation()
        current_conv_id = agent.get_agent_info().get("conversation_id", "")

    logger.info("[%s] 对话请求: %s", request_id[:8], req.question[:50])

    try:
        if req.include_sources:
            result = await agent.query_with_sources(req.question)
        else:
            result = await agent.aquery(req.question)
    except Exception as exc:
        logger.error("[%s] 对话查询异常: %s", request_id[:8], exc)
        raise HTTPException(status_code=500, detail=f"对话查询失败: {exc}") from exc

    return ConversationResponse(
        answer=result["answer"],
        thinking_chain=result.get("thinking_chain"),
        sources=result.get("sources", []),
        conversation_id=agent.get_agent_info().get("conversation_id", ""),
    )


# ---------------------------------------------------------------------------
# 文档摄入端点
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse)
async def ingest(
    req: IngestRequest,
    request: Request,
    request_id: str = Depends(get_request_id),
):
    """文档摄入接口，将文档批量写入知识库。

    每个文档须包含 text 字段，可选 doc_id、metadata 等字段。
    当前为存根实现，实际摄入逻辑依赖 Milvus 和 Embedding 服务。

    Args:
        req: 摄入请求体
        request: FastAPI 请求对象
        request_id: 请求 ID

    Returns:
        摄入结果
    """
    agent = require_agent()

    # 速率限制（摄入操作更严格）
    client_id = request.client.host if request.client else "unknown"
    if not _check_rate_limit(f"ingest:{client_id}"):
        raise HTTPException(status_code=429, detail="摄入请求过于频繁，请稍后重试")

    logger.info(
        "[%s] 文档摄入请求: %d 文档",
        request_id[:8],
        len(req.documents),
    )

    # 校验文档格式
    validated_docs: List[Dict[str, Any]] = []
    errors: List[str] = []

    for idx, doc in enumerate(req.documents):
        if not isinstance(doc, dict):
            errors.append(f"文档 {idx}: 不是有效的字典格式")
            continue
        if "text" not in doc or not doc["text"]:
            errors.append(f"文档 {idx}: 缺少 text 字段或 text 为空")
            continue
        validated_docs.append(doc)

    if not validated_docs:
        raise HTTPException(
            status_code=400,
            detail=f"没有有效的文档可摄入。错误: {errors}",
        )

    # 存根：实际摄入逻辑待实现
    # TODO: 调用 Milvus insert + Embedding 服务
    logger.info(
        "[%s] 文档摄入存根: %d 文档通过校验",
        request_id[:8],
        len(validated_docs),
    )

    return IngestResponse(
        ingested_count=len(validated_docs),
        failed_count=len(req.documents) - len(validated_docs),
        errors=errors,
    )


# ---------------------------------------------------------------------------
# 系统统计端点
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StatsResponse)
async def stats(
    request: Request,
    request_id: str = Depends(get_request_id),
):
    """获取系统统计信息。

    返回查询总数和 Agent 运行时信息。

    Args:
        request: FastAPI 请求对象
        request_id: 请求 ID

    Returns:
        系统统计
    """
    agent = require_agent()
    agent_info = agent.get_agent_info()

    return StatsResponse(
        query_total=agent_info.get("query_count", 0),
        agent_info=agent_info,
    )


# ---------------------------------------------------------------------------
# Agent 信息端点
# ---------------------------------------------------------------------------


@router.get("/agent/info")
async def agent_info(
    request: Request,
    request_id: str = Depends(get_request_id),
):
    """获取 Agent 详细信息，包括模型、工具、对话状态等。

    Args:
        request: FastAPI 请求对象
        request_id: 请求 ID

    Returns:
        Agent 信息字典
    """
    agent = require_agent()
    info = agent.get_agent_info()
    # 附加对话历史摘要
    info["conversation_history_length"] = len(agent.get_conversation_history())
    return info


# ---------------------------------------------------------------------------
# 健康检查端点
# ---------------------------------------------------------------------------


@router.get("/health")
async def health():
    """健康检查接口。

    Returns:
        服务状态信息
    """
    agent_status = "ready" if _agent is not None else "not_initialized"
    return {
        "status": "ok" if agent_status == "ready" else "degraded",
        "service": "FunnelRAG",
        "agent": agent_status,
    }
