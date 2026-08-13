"""FunnelRAG FastAPI 应用入口 —— 应用工厂、中间件配置、异常处理、生命周期管理。"""

import sys
import time
import uuid
from pathlib import Path

# 确保项目根目录在 sys.path 中，使 `from python.xxx` 导入正常工作
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from contextlib import asynccontextmanager
from typing import Callable

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.exceptions import HTTPException as StarletteHTTPException

from python.agent.engine import FunnelRAGAgent
from python.api.routes import router, set_agent
from python.config.settings import AppSettings, load_settings
from python.utils.logging_config import get_logger, setup_logging

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# 中间件配置
# ---------------------------------------------------------------------------


def configure_middleware(app: FastAPI, settings: AppSettings) -> None:
    """配置应用中间件栈。

    按执行顺序（最外层先执行）：
    1. CORS 中间件
    2. 请求 ID 注入中间件
    3. 请求日志中间件

    Args:
        app: FastAPI 应用实例
        settings: 应用配置
    """
    # CORS 配置
    app.add_middleware(
        CORSMiddleware,
        allow_origins=(
            settings.cors_origins if hasattr(settings, "cors_origins") else ["*"]
        ),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    # 请求 ID 注入中间件
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Callable) -> Response:
        """为每个请求生成唯一 ID 并注入请求状态和响应头。"""
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex)
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    # 请求日志中间件
    @app.middleware("http")
    async def request_logging_middleware(
        request: Request, call_next: Callable
    ) -> Response:
        """记录每个请求的方法、路径、状态码和耗时。"""
        start_time = time.perf_counter()
        request_id = getattr(request.state, "request_id", "unknown")

        # 跳过健康检查和指标的日志以减少噪音
        skip_paths = {"/health", "/metrics", "/"}
        if request.url.path in skip_paths:
            return await call_next(request)

        logger.info(
            "[%s] --> %s %s",
            request_id[:8],
            request.method,
            request.url.path,
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            elapsed = time.perf_counter() - start_time
            logger.error(
                "[%s] <-- %s %s | 500 | %.3fs | error: %s",
                request_id[:8],
                request.method,
                request.url.path,
                elapsed,
                exc,
            )
            raise

        elapsed = time.perf_counter() - start_time
        logger.info(
            "[%s] <-- %s %s | %d | %.3fs",
            request_id[:8],
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response


# ---------------------------------------------------------------------------
# 异常处理器配置
# ---------------------------------------------------------------------------


def configure_exception_handlers(app: FastAPI) -> None:
    """注册全局异常处理器。

    处理以下异常类型：
    - StarletteHTTPException: FastAPI/Starlette HTTP 异常
    - RequestValidationError: 请求参数校验失败
    - Exception: 兜底通用异常

    Args:
        app: FastAPI 应用实例
    """

    @app.exception_handler(StarletteHTTPException)
    async def http_exception_handler(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        """处理 HTTP 异常，返回结构化错误响应。"""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.warning(
            "[%s] HTTP %d: %s %s - %s",
            request_id[:8],
            exc.status_code,
            request.method,
            request.url.path,
            exc.detail,
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": True,
                "status_code": exc.status_code,
                "detail": exc.detail,
                "request_id": request_id,
            },
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        """处理请求校验异常，返回详细的字段错误信息。"""
        request_id = getattr(request.state, "request_id", "unknown")
        errors = exc.errors()
        # 提取人类可读的错误描述
        detail_messages = []
        for err in errors:
            loc = " → ".join(str(l) for l in err.get("loc", []))
            msg = err.get("msg", "validation error")
            detail_messages.append(f"{loc}: {msg}")

        logger.warning(
            "[%s] Validation error: %s %s - %s",
            request_id[:8],
            request.method,
            request.url.path,
            detail_messages,
        )
        return JSONResponse(
            status_code=422,
            content={
                "error": True,
                "status_code": 422,
                "detail": detail_messages,
                "errors": errors,
                "request_id": request_id,
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """兜底处理所有未捕获异常，避免泄露内部堆栈。"""
        request_id = getattr(request.state, "request_id", "unknown")
        logger.error(
            "[%s] Unhandled exception: %s %s - %s",
            request_id[:8],
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": True,
                "status_code": 500,
                "detail": "服务器内部错误，请稍后重试",
                "request_id": request_id,
            },
        )


# ---------------------------------------------------------------------------
# 生命周期管理
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理 —— 启动时初始化 Agent，关闭时清理资源。"""
    settings: AppSettings = app.state.settings

    logger.info("FunnelRAG 服务启动中...")

    # 启动连通性检查
    _check_connectivity(settings)

    # 初始化 Agent
    try:
        agent = FunnelRAGAgent(settings=settings)
        set_agent(agent)
        logger.info("FunnelRAG Agent 初始化完成")
    except Exception as e:
        logger.warning("Agent 初始化失败（服务仍可启动，查询时将返回错误）: %s", e)

    logger.info("FunnelRAG 服务启动完成")
    yield

    # 关闭清理
    logger.info("FunnelRAG 服务关闭中...")
    _cleanup_resources()
    logger.info("FunnelRAG 服务已关闭")


def _check_connectivity(settings: AppSettings) -> None:
    """启动时执行轻量级连通性检查，记录但不阻断服务。

    Args:
        settings: 应用配置
    """
    # Milvus 连通性
    try:
        from pymilvus import MilvusClient

        client = MilvusClient(
            uri=f"http://{settings.milvus.host}:{settings.milvus.port}"
        )
        client.close()
        logger.info("Milvus 连通性检查: OK")
    except Exception as exc:
        logger.warning("Milvus 连通性检查: 失败 - %s", exc)

    # Redis 连通性（可选）
    if settings.redis.host:
        try:
            import redis

            r = redis.Redis(
                host=settings.redis.host,
                port=settings.redis.port,
                password=settings.redis.password or None,
                socket_connect_timeout=2,
            )
            r.ping()
            r.close()
            logger.info("Redis 连通性检查: OK")
        except Exception as exc:
            logger.warning("Redis 连通性检查: 失败 - %s", exc)


def _cleanup_resources() -> None:
    """关闭时清理各类资源（连接池、临时文件等）。"""
    # 当前无显式资源需要清理，预留扩展点
    pass


# ---------------------------------------------------------------------------
# 应用工厂
# ---------------------------------------------------------------------------


def create_app(settings: AppSettings | None = None) -> FastAPI:
    """创建 FastAPI 应用实例。

    Args:
        settings: 应用配置，为 None 时从默认路径加载

    Returns:
        配置完成的 FastAPI 应用实例
    """
    if settings is None:
        settings = load_settings()

    setup_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description="FunnelRAG 智能Agent系统 - 企业级私有知识库问答",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.settings = settings

    # 配置中间件
    configure_middleware(app, settings)

    # 配置异常处理器
    configure_exception_handlers(app)

    # 注册路由
    app.include_router(router, prefix="/api/v1")

    # Prometheus 指标端点
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # 根路由
    @app.get("/", include_in_schema=False)
    async def root():
        """API 根路由，返回服务基本信息。"""
        return {
            "name": settings.app_name,
            "version": "1.0.0",
            "description": "FunnelRAG 智能Agent系统 - 企业级私有知识库问答",
            "docs": "/docs",
            "health": "/api/v1/health",
            "metrics": "/metrics",
        }

    return app


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------


def main():
    """CLI 入口函数，启动 Uvicorn 服务器。"""
    settings = load_settings()
    app = create_app(settings)
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
