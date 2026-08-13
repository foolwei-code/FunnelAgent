"""FunnelRAG 全局配置模块 —— 从环境变量和 YAML 文件加载配置。

This module provides a layered configuration system built on top of pydantic-settings.
Configuration values are resolved in the following priority order (highest first):

1. Environment variables (with appropriate prefixes)
2. Explicit keyword arguments passed to the constructor
3. YAML configuration file values
4. Built-in defaults

Example::

    from python.config.settings import load_settings

    settings = load_settings()
    print(settings.llm.model_name)       # "qwen-plus"
    print(settings.embedding.dimension)   # 1024
    print(settings.to_yaml())             # full YAML dump
"""

from __future__ import annotations

import asyncio
import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_CONFIG_PATH = _PROJECT_ROOT / "config.yaml"


# ---------------------------------------------------------------------------
# Sub-section settings
# ---------------------------------------------------------------------------


class LLMSettings(BaseSettings):
    """Language-model provider configuration.

    Attributes:
        provider: LLM provider identifier (e.g. ``"openai"``, ``"anthropic"``).
        model_name: Model identifier understood by the provider.
        api_key: API key for authentication.  **Never** commit a real key.
        api_base: Custom base URL (useful for self-hosted endpoints).
        temperature: Sampling temperature in [0, 2].
        max_tokens: Maximum number of tokens the model may generate.
    """

    provider: str = "openai"
    model_name: str = "qwen-plus"
    api_key: str = ""
    api_base: str = ""
    temperature: float = 0.1
    max_tokens: int = 2048

    @field_validator("temperature")
    @classmethod
    def _validate_temperature(cls, v: float) -> float:
        """Ensure temperature is within the valid range [0, 2]."""
        if not 0.0 <= v <= 2.0:
            raise ValueError(f"temperature must be between 0 and 2, got {v}")
        return v

    @field_validator("max_tokens")
    @classmethod
    def _validate_max_tokens(cls, v: int) -> int:
        """Ensure max_tokens is a positive integer."""
        if v <= 0:
            raise ValueError(f"max_tokens must be positive, got {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"LLMSettings(provider={self.provider!r}, model_name={self.model_name!r}, "
            f"api_base={self.api_base!r}, temperature={self.temperature}, "
            f"max_tokens={self.max_tokens})"
        )

    model_config = {"env_file": ".env", "env_prefix": "LLM_", "extra": "ignore"}


class MilvusSettings(BaseSettings):
    """Milvus vector-database connection and search configuration.

    Attributes:
        host: Milvus server hostname.
        port: Milvus gRPC port.
        collection: Default collection name for document storage.
        top_k: Number of candidates to retrieve during coarse search.
    """

    host: str = "localhost"
    port: int = 19530
    collection: str = "funnelrag_docs"
    top_k: int = 100

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator("top_k")
    @classmethod
    def _validate_top_k(cls, v: int) -> int:
        """Ensure top_k is a positive integer."""
        if v <= 0:
            raise ValueError(f"top_k must be positive, got {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"MilvusSettings(host={self.host!r}, port={self.port}, "
            f"collection={self.collection!r}, top_k={self.top_k})"
        )

    model_config = {"env_file": ".env", "env_prefix": "MILVUS_", "extra": "ignore"}


class RerankerSettings(BaseSettings):
    """Cross-encoder reranker service configuration.

    Attributes:
        host: Reranker gRPC service hostname.
        port: Reranker gRPC service port.
        model_path: Path to the ONNX cross-encoder model file.
        timeout: Timeout in seconds for reranking requests.
    """

    host: str = "localhost"
    port: int = 50051
    model_path: str = "cpp/model/cross_encoder.onnx"
    timeout: float = 0.05

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator("timeout")
    @classmethod
    def _validate_timeout(cls, v: float) -> float:
        """Ensure timeout is non-negative."""
        if v < 0:
            raise ValueError(f"timeout must be non-negative, got {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"RerankerSettings(host={self.host!r}, port={self.port}, "
            f"model_path={self.model_path!r}, timeout={self.timeout})"
        )

    model_config = {"env_file": ".env", "env_prefix": "RERANKER_", "extra": "ignore"}


class PostgresSettings(BaseSettings):
    """PostgreSQL database configuration for structured metadata storage.

    Attributes:
        host: PostgreSQL server hostname.
        port: PostgreSQL server port.
        user: Database user.
        password: Database password.  **Never** commit a real password.
        database: Database name.
    """

    host: str = "localhost"
    port: int = 5432
    user: str = "postgres"
    password: str = "123456abc"
    database: str = "funnelrag"

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @property
    def dsn(self) -> str:
        """Return a PostgreSQL connection DSN string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"

    def __repr__(self) -> str:
        return (
            f"PostgresSettings(host={self.host!r}, port={self.port}, "
            f"user={self.user!r}, database={self.database!r})"
        )

    model_config = {"env_file": ".env", "env_prefix": "POSTGRES_", "extra": "ignore"}


class RedisSettings(BaseSettings):
    """Redis configuration for caching and rate-limiting.

    Attributes:
        host: Redis server hostname.
        port: Redis server port.
        password: Redis authentication password (empty = no auth).
        db: Redis database index.
    """

    host: str = "localhost"
    port: int = 6379
    password: str = ""
    db: int = 0

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator("db")
    @classmethod
    def _validate_db(cls, v: int) -> int:
        """Ensure Redis database index is in the valid range."""
        if not 0 <= v <= 15:
            raise ValueError(f"Redis db must be between 0 and 15, got {v}")
        return v

    @property
    def url(self) -> str:
        """Return a Redis connection URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"

    def __repr__(self) -> str:
        return (
            f"RedisSettings(host={self.host!r}, port={self.port}, db={self.db})"
        )

    model_config = {"env_file": ".env", "env_prefix": "REDIS_", "extra": "ignore"}


class EmbeddingSettings(BaseSettings):
    """Embedding model configuration for vectorisation.

    Attributes:
        model_name: Embedding model identifier.
        model_path: Local path to the embedding model weights.
        dimension: Output embedding dimensionality.
        batch_size: Number of texts to embed per batch.
        device: Compute device (``"cpu"``, ``"cuda"``, ``"mps"``).
    """

    model_name: str = "bge-large-zh-v1.5"
    model_path: str = ""
    dimension: int = 1024
    batch_size: int = 32
    device: str = "cpu"

    @field_validator("dimension")
    @classmethod
    def _validate_dimension(cls, v: int) -> int:
        """Ensure embedding dimension is a positive integer."""
        if v <= 0:
            raise ValueError(f"dimension must be positive, got {v}")
        return v

    @field_validator("batch_size")
    @classmethod
    def _validate_batch_size(cls, v: int) -> int:
        """Ensure batch_size is a positive integer."""
        if v <= 0:
            raise ValueError(f"batch_size must be positive, got {v}")
        return v

    @field_validator("device")
    @classmethod
    def _validate_device(cls, v: str) -> str:
        """Ensure device is a recognised identifier."""
        allowed = {"cpu", "cuda", "mps"}
        if v not in allowed:
            raise ValueError(f"device must be one of {allowed}, got {v!r}")
        return v

    def __repr__(self) -> str:
        return (
            f"EmbeddingSettings(model_name={self.model_name!r}, "
            f"dimension={self.dimension}, batch_size={self.batch_size}, "
            f"device={self.device!r})"
        )

    model_config = {"env_file": ".env", "env_prefix": "EMBEDDING_", "extra": "ignore"}


class CacheSettings(BaseSettings):
    """Caching behaviour configuration.

    Attributes:
        enabled: Whether the caching layer is active.
        ttl_coarse: Time-to-live in seconds for coarse (embedding) cache entries.
        ttl_fine: Time-to-live in seconds for fine (reranked) cache entries.
        max_size: Maximum number of entries per cache namespace.
    """

    enabled: bool = True
    ttl_coarse: int = 3600
    ttl_fine: int = 1800
    max_size: int = 10000

    @field_validator("ttl_coarse", "ttl_fine")
    @classmethod
    def _validate_ttl(cls, v: int) -> int:
        """Ensure TTL values are non-negative."""
        if v < 0:
            raise ValueError(f"TTL must be non-negative, got {v}")
        return v

    @field_validator("max_size")
    @classmethod
    def _validate_max_size(cls, v: int) -> int:
        """Ensure max_size is a positive integer."""
        if v <= 0:
            raise ValueError(f"max_size must be positive, got {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"CacheSettings(enabled={self.enabled}, ttl_coarse={self.ttl_coarse}, "
            f"ttl_fine={self.ttl_fine}, max_size={self.max_size})"
        )

    model_config = {"env_file": ".env", "env_prefix": "CACHE_", "extra": "ignore"}


class ServerSettings(BaseSettings):
    """FastAPI / uvicorn server configuration.

    Attributes:
        host: Bind address.
        port: Bind port.
        workers: Number of uvicorn worker processes.
        cors_origins: List of allowed CORS origin patterns.
    """

    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])

    @field_validator("port")
    @classmethod
    def _validate_port(cls, v: int) -> int:
        """Ensure port is in the valid range."""
        if not 1 <= v <= 65535:
            raise ValueError(f"port must be between 1 and 65535, got {v}")
        return v

    @field_validator("workers")
    @classmethod
    def _validate_workers(cls, v: int) -> int:
        """Ensure workers is a positive integer."""
        if v <= 0:
            raise ValueError(f"workers must be positive, got {v}")
        return v

    def __repr__(self) -> str:
        return (
            f"ServerSettings(host={self.host!r}, port={self.port}, "
            f"workers={self.workers}, cors_origins={self.cors_origins!r})"
        )

    model_config = {"env_file": ".env", "env_prefix": "SERVER_", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Root application settings
# ---------------------------------------------------------------------------


class AppSettings(BaseSettings):
    """Root application settings aggregating all sub-sections.

    Attributes:
        app_name: Human-readable application name.
        version: Application version string.
        debug: Enable debug mode (verbose logging, debug endpoints).
        log_level: Root logging level.
        llm: Language-model settings.
        milvus: Milvus vector-store settings.
        reranker: Cross-encoder reranker settings.
        postgres: PostgreSQL settings.
        redis: Redis settings.
        embedding: Embedding model settings.
        cache: Cache behaviour settings.
        server: HTTP server settings.
    """

    app_name: str = "FunnelRAG"
    version: str = "0.1.0"
    debug: bool = False
    log_level: str = "INFO"

    llm: LLMSettings = Field(default_factory=LLMSettings)
    milvus: MilvusSettings = Field(default_factory=MilvusSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)
    redis: RedisSettings = Field(default_factory=RedisSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)

    @field_validator("log_level")
    @classmethod
    def _validate_log_level(cls, v: str) -> str:
        """Ensure log_level is a valid Python logging level."""
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = v.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got {v!r}")
        return upper

    @model_validator(mode="after")
    def _sync_debug_log_level(self) -> "AppSettings":
        """When debug is True, force log_level to DEBUG."""
        if self.debug and self.log_level != "DEBUG":
            object.__setattr__(self, "log_level", "DEBUG")
        return self

    # -- YAML export --------------------------------------------------------

    def to_yaml(self) -> str:
        """Serialise the full settings tree to a YAML string.

        Sensitive fields (``api_key``, ``password``) are replaced with
        ``"***REDACTED***"`` so the output is safe for logging or config
        sharing.
        """
        data = self.model_dump()
        _redact_sensitive(data, sensitive_keys={"api_key", "password"})
        return yaml.dump(data, default_flow_style=False, sort_keys=False, allow_unicode=True)

    # -- Connectivity check -------------------------------------------------

    async def validate_connections(self) -> Dict[str, bool]:
        """Probe external services and return a mapping of service → reachable.

        Currently checks Milvus, PostgreSQL, and Redis connectivity.
        This method is intended to be called at startup so that the
        application can fail fast when a required service is unavailable.

        Returns:
            A dict like ``{"milvus": True, "postgres": False, "redis": True}``.
        """
        results: Dict[str, bool] = {}

        # Milvus
        try:
            from pymilvus import connections  # type: ignore[import-untyped]

            connections.connect(
                alias="health_check",
                host=self.milvus.host,
                port=self.milvus.port,
                timeout=5,
            )
            connections.disconnect("health_check")
            results["milvus"] = True
        except Exception as exc:
            logger.warning("Milvus connectivity check failed: %s", exc)
            results["milvus"] = False

        # PostgreSQL
        try:
            import asyncpg  # type: ignore[import-untyped]

            conn = await asyncio.wait_for(
                asyncpg.connect(self.postgres.dsn), timeout=5.0
            )
            await conn.execute("SELECT 1")
            await conn.close()
            results["postgres"] = True
        except Exception as exc:
            logger.warning("PostgreSQL connectivity check failed: %s", exc)
            results["postgres"] = False

        # Redis
        try:
            import redis as _redis  # type: ignore[import-untyped]

            client = _redis.from_url(self.redis.url, socket_timeout=5)
            client.ping()
            client.close()
            results["redis"] = True
        except Exception as exc:
            logger.warning("Redis connectivity check failed: %s", exc)
            results["redis"] = False

        return results

    def __repr__(self) -> str:
        return (
            f"AppSettings(app_name={self.app_name!r}, version={self.version!r}, "
            f"debug={self.debug}, log_level={self.log_level!r})"
        )

    model_config = {"env_file": ".env", "env_prefix": "", "extra": "ignore"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_REDACTED = "***REDACTED***"


def _redact_sensitive(obj: Any, sensitive_keys: set[str]) -> None:
    """Recursively redact values whose keys match *sensitive_keys*."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in sensitive_keys:
                obj[key] = _REDACTED
            else:
                _redact_sensitive(value, sensitive_keys)
    elif isinstance(obj, list):
        for item in obj:
            _redact_sensitive(item, sensitive_keys)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict.

    - Dict values are merged recursively.
    - Scalar / list values from *override* replace those in *base*.
    - Neither input dict is mutated.
    """
    merged = deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _env_overrides() -> Dict[str, Any]:
    """Collect environment-variable overrides for the top-level AppSettings.

    Only variables with the ``FUNNELRAG_`` prefix are considered.  Nested
    sections are detected via a double-underscore separator::

        FUNNELRAG_LLM__MODEL_NAME=gpt-4  →  {"llm": {"model_name": "gpt-4"}}
    """
    prefix = "FUNNELRAG_"
    result: Dict[str, Any] = {}
    for env_key, env_value in os.environ.items():
        if not env_key.startswith(prefix):
            continue
        path = env_key[len(prefix):].lower().split("__")
        node: Dict[str, Any] = result
        for part in path[:-1]:
            node = node.setdefault(part, {})
        node[path[-1]] = env_value
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_settings(config_path: Optional[str] = None) -> AppSettings:
    """Load application settings from YAML, environment variables, and defaults.

    Resolution order (highest priority first):

    1. Environment variables with ``FUNNELRAG_`` prefix (nested via ``__``).
    2. Standard pydantic-settings env vars (per-section prefixes like ``LLM_``).
    3. Values from the YAML configuration file at *config_path*.
    4. Built-in field defaults.

    Args:
        config_path: Optional path to a YAML configuration file.  When
            ``None``, defaults to ``<project_root>/config.yaml``.

    Returns:
        A fully-validated :class:`AppSettings` instance.

    Raises:
        pydantic.ValidationError: If any configuration value fails validation.
    """
    path = Path(config_path) if config_path else _CONFIG_PATH
    yaml_data: Dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            yaml_data = yaml.safe_load(fh) or {}

    # Merge FUNNELRAG_ env overrides on top of YAML data
    env_data = _env_overrides()
    merged = _deep_merge(yaml_data, env_data) if env_data else yaml_data

    settings = AppSettings(**merged)
    logger.debug("Loaded settings from %s (env overrides: %s)", path, bool(env_data))
    return settings
