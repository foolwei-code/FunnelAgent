"""直接回答工具 —— 当 Agent 判断无需检索时，直接基于 LLM 自身知识回答。"""

from __future__ import annotations

import re
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from python.utils.logging_config import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prometheus counters
# ---------------------------------------------------------------------------
from prometheus_client import Counter as _Counter

DIRECT_ANSWER_TOTAL = _Counter(
    "funnelrag_direct_answer_total", "Total direct answer invocations"
)
DIRECT_ANSWER_FALLBACK_TOTAL = _Counter(
    "funnelrag_direct_answer_fallback_total",
    "Total direct answer fallbacks (retrieval needed)",
)

# ---------------------------------------------------------------------------
# Sensitive content patterns (basic)
# ---------------------------------------------------------------------------

_SENSITIVE_PATTERNS = [
    re.compile(r"\b\d{16,19}\b"),  # Credit card numbers
    re.compile(r"\b\d{6,8}(?:\d{4})?\b"),  # ID-like numbers
    re.compile(r"(?:password|passwd|pwd)\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"(?:api[_-]?key|secret[_-]?key|token)\s*[:=]\s*\S+", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# Pydantic schemas for tool I/O
# ---------------------------------------------------------------------------


class DirectAnswerInput(BaseModel):
    """DirectAnswerTool 的输入 schema。"""

    query: str = Field(
        ...,
        description="用户提问内容",
    )
    context: Optional[str] = Field(
        default=None,
        description="可选上下文信息，帮助判断是否需要检索",
    )


class DirectAnswerOutput(BaseModel):
    """DirectAnswerTool 的输出 schema。"""

    answer: str = Field(description="直接回答内容")
    category: str = Field(description="问题分类")
    confidence: float = Field(ge=0.0, le=1.0, description="回答置信度")
    needs_retrieval: bool = Field(description="是否需要检索知识库")
    safety_passed: bool = Field(description="是否通过安全检查")


class QuestionCategory(str, Enum):
    """问题分类枚举。"""

    GENERAL = "general"
    CHITCHAT = "chitchat"
    FACTUAL = "factual"
    PROCEDURAL = "procedural"
    REQUIRES_RETRIEVAL = "requires_retrieval"


# ---------------------------------------------------------------------------
# Rate-limit stub (placeholder for integration with Redis / token bucket)
# ---------------------------------------------------------------------------

_rate_limit_store: Dict[str, List[float]] = {}
_RATE_LIMIT_WINDOW = 60  # seconds
_RATE_LIMIT_MAX_CALLS = 100


class DirectAnswerTool(BaseTool):
    """直接回答工具，当 Agent 判断无需检索知识库时使用。

    继承自 LangChain 的 BaseTool，可无缝集成到 Agent 执行循环中。
    提供问题分类、置信度评估、安全检查、速率限制等能力。

    Attributes:
        name: 工具名称标识。
        description: 工具功能描述。
        args_schema: 输入参数的 Pydantic 模型。
    """

    name: str = "direct_answer"
    description: str = (
        "当问题不需要检索知识库时，直接基于 LLM 自身知识回答。"
        "适用于通用问题、闲聊、常识问答等场景。"
    )
    args_schema: Type[BaseModel] = DirectAnswerInput

    # ------------------------------------------------------------------
    # Core execution
    # ------------------------------------------------------------------

    def _run(self, query: str, context: Optional[str] = None) -> str:
        """同步执行直接回答。

        Args:
            query: 用户提问内容。
            context: 可选上下文信息。

        Returns:
            结构化 JSON 字符串，包含 answer / category / confidence 等字段。
        """
        DIRECT_ANSWER_TOTAL.inc()
        logger.info("使用直接回答: %s", query[:80])

        # Rate limit check
        rate_ok = self.rate_limit_check("default")
        if not rate_ok:
            return self.format_response(
                answer="请求过于频繁，请稍后再试。",
                category=QuestionCategory.GENERAL.value,
                confidence=0.0,
                needs_retrieval=False,
                safety_passed=True,
            )

        # Classify the question
        category = self.categorize_question(query, context)

        # If retrieval is needed, signal to the agent
        if category == QuestionCategory.REQUIRES_RETRIEVAL:
            DIRECT_ANSWER_FALLBACK_TOTAL.inc()
            logger.info("问题需要检索: %s", query[:80])
            return self.format_response(
                answer="",
                category=category.value,
                confidence=0.0,
                needs_retrieval=True,
                safety_passed=True,
            )

        # Generate a direct-answer marker for the Agent to identify
        answer = f"DIRECT_ANSWER:{query}"

        # Safety check
        safety_passed = self.safety_check(answer)

        # Confidence score
        confidence = self.confidence_score(query, category)

        return self.format_response(
            answer=answer,
            category=category.value,
            confidence=confidence,
            needs_retrieval=False,
            safety_passed=safety_passed,
        )

    async def _arun(self, query: str, context: Optional[str] = None) -> str:
        """异步执行直接回答。

        Args:
            query: 用户提问内容。
            context: 可选上下文信息。

        Returns:
            结构化 JSON 字符串，格式同 :meth:`_run`。
        """
        return self._run(query, context)

    # ------------------------------------------------------------------
    # Question categorization
    # ------------------------------------------------------------------

    def categorize_question(
        self,
        query: str,
        context: Optional[str] = None,
    ) -> QuestionCategory:
        """判断问题是否需要检索知识库。

        基于关键词和模式匹配进行规则分类：
        - 包含公司/产品特定术语 → 需要检索
        - 闲聊/打招呼 → chitchat
        - 操作步骤类 → procedural
        - 事实类 → factual
        - 其他 → general

        Args:
            query: 用户提问内容。
            context: 可选上下文信息。

        Returns:
            问题分类枚举值。
        """
        query_lower = query.lower().strip()

        # Chitchat detection
        chitchat_patterns = [
            r"^(hi|hello|hey|你好|嗨|您好)[\s!.?]*$",
            r"^(thanks|thank you|谢谢|多谢)[\s!.?]*$",
            r"^(bye|goodbye|再见|拜拜)[\s!.?]*$",
            r"^(how are you|你好吗|最近怎么样)",
        ]
        for pattern in chitchat_patterns:
            if re.match(pattern, query_lower):
                logger.info("问题分类: chitchat")
                return QuestionCategory.CHITCHAT

        # Requires-retrieval detection (enterprise-specific terms)
        retrieval_keywords = [
            "公司",
            "内部",
            "规范",
            "流程",
            "制度",
            "文档编号",
            "产品线",
            "工单",
            "内部系统",
            "sop",
            "内网",
            "knowledge base",
            "公司政策",
            "hr",
            "员工手册",
        ]
        for kw in retrieval_keywords:
            if kw in query_lower:
                logger.info("问题分类: requires_retrieval (keyword=%s)", kw)
                return QuestionCategory.REQUIRES_RETRIEVAL

        # Procedural detection
        procedural_patterns = [
            r"如何|怎么|怎样|步骤|流程|教程|how to|how do i",
        ]
        for pattern in procedural_patterns:
            if re.search(pattern, query_lower):
                logger.info("问题分类: procedural")
                return QuestionCategory.PROCEDURAL

        # Factual detection
        factual_patterns = [
            r"是什么|什么是|定义|含义|是什么意思|what is|define|meaning",
        ]
        for pattern in factual_patterns:
            if re.search(pattern, query_lower):
                logger.info("问题分类: factual")
                return QuestionCategory.FACTUAL

        logger.info("问题分类: general")
        return QuestionCategory.GENERAL

    # ------------------------------------------------------------------
    # Confidence scoring
    # ------------------------------------------------------------------

    def confidence_score(
        self,
        query: str,
        category: QuestionCategory,
    ) -> float:
        """评估直接回答的置信度。

        基于问题分类和查询特征给出 0-1 的置信度评分。
        闲聊类置信度高，通用类置信度中等，需要检索的为 0。

        Args:
            query: 用户提问内容。
            category: 问题分类。

        Returns:
            0.0 ~ 1.0 的置信度分数。
        """
        base_scores: Dict[QuestionCategory, float] = {
            QuestionCategory.CHITCHAT: 0.95,
            QuestionCategory.FACTUAL: 0.6,
            QuestionCategory.PROCEDURAL: 0.4,
            QuestionCategory.GENERAL: 0.5,
            QuestionCategory.REQUIRES_RETRIEVAL: 0.0,
        }

        score = base_scores.get(category, 0.5)

        # Adjust based on query length — very short queries are less reliable
        if len(query.strip()) < 5:
            score *= 0.8

        # Adjust based on specificity markers
        specificity_markers = ["具体", "详细", "精确", "exactly", "specifically"]
        for marker in specificity_markers:
            if marker in query.lower():
                score *= 0.7
                break

        return round(min(max(score, 0.0), 1.0), 2)

    # ------------------------------------------------------------------
    # Safety check
    # ------------------------------------------------------------------

    def safety_check(self, text: str) -> bool:
        """检查回答内容是否包含敏感信息。

        检测信用卡号、身份证号、密码、API Key 等模式。

        Args:
            text: 待检查的文本。

        Returns:
            ``True`` 表示安全，``False`` 表示包含敏感信息。
        """
        for pattern in _SENSITIVE_PATTERNS:
            if pattern.search(text):
                logger.warning("安全检查未通过: 检测到敏感信息模式")
                return False
        return True

    # ------------------------------------------------------------------
    # Rate limit check (stub)
    # ------------------------------------------------------------------

    def rate_limit_check(self, key: str = "default") -> bool:
        """检查速率限制（基于内存的简易实现）。

        使用滑动窗口算法检查每分钟调用次数是否超限。
        生产环境应替换为 Redis 实现。

        Args:
            key: 速率限制的键名，例如用户 ID 或 IP。

        Returns:
            ``True`` 表示允许通过，``False`` 表示超限。
        """
        now = time.monotonic()
        if key not in _rate_limit_store:
            _rate_limit_store[key] = []

        # Remove expired entries
        window = _rate_limit_store[key]
        window[:] = [t for t in window if now - t < _RATE_LIMIT_WINDOW]

        if len(window) >= _RATE_LIMIT_MAX_CALLS:
            logger.warning("速率限制触发 (key=%s, calls=%d)", key, len(window))
            return False

        window.append(now)
        return True

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    def format_response(
        self,
        answer: str,
        category: str,
        confidence: float,
        needs_retrieval: bool,
        safety_passed: bool,
    ) -> str:
        """将回答格式化为结构化 JSON 字符串。

        Args:
            answer: 回答内容。
            category: 问题分类。
            confidence: 置信度分数。
            needs_retrieval: 是否需要检索知识库。
            safety_passed: 是否通过安全检查。

        Returns:
            JSON 格式的结构化字符串。
        """
        import json

        output = DirectAnswerOutput(
            answer=answer,
            category=category,
            confidence=confidence,
            needs_retrieval=needs_retrieval,
            safety_passed=safety_passed,
        )
        return output.model_dump_json()
