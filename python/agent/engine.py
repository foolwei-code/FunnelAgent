"""Agent 编排引擎 —— 基于 LangChain create_agent 实现工具调用循环。

FunnelRAGAgent 是核心编排类，管理 LLM 实例、工具集、Agent 图、
对话历史以及各类查询入口（同步/异步/流式/带来源）。
"""

from __future__ import annotations

import re
import time
import uuid
from collections import deque
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from langchain.agents.factory import create_agent
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langsmith import traceable

from python.agent.prompt_templates import (
    REACT_SYSTEM_PROMPT,
    build_rag_prompt,
    format_documents,
    format_sources,
)
from python.config.settings import AppSettings, LLMSettings
from python.tools.direct_answer import DirectAnswerTool
from python.tools.milvus_search import MilvusSearchTool
from python.tools.rerank_tool import RerankTool
from python.utils.logging_config import get_logger, setup_logging
from python.utils.metrics import LLM_GENERATE_LATENCY, QUERY_LATENCY, QUERY_TOTAL

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# 自定义异常类型
# ---------------------------------------------------------------------------


class AgentNotInitializedError(Exception):
    """Agent 未正确初始化时抛出。"""


class LLMCallError(Exception):
    """LLM 调用失败时抛出，包含可重试标志。"""

    def __init__(self, message: str, retryable: bool = False):
        super().__init__(message)
        self.retryable = retryable


class QueryValidationError(Exception):
    """查询输入校验失败时抛出。"""


class ToolExecutionError(Exception):
    """工具执行失败时抛出。"""


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

_MAX_RETRIES: int = 3
_RETRY_DELAY_BASE: float = 0.5  # 秒，指数退避基数
_CONVERSATION_MAX_TURNS: int = 50
_QUERY_MAX_LENGTH: int = 2000
_TOOL_OUTPUT_MAX_LENGTH: int = 500


# ---------------------------------------------------------------------------
# FunnelRAGAgent
# ---------------------------------------------------------------------------


class FunnelRAGAgent:
    """FunnelRAG Agent 编排引擎，管理工具调用循环与 ReAct 推理。

    职责：
    - 构建 LLM（含 fallback）和工具集（含启动校验）
    - 提供同步 / 异步 / 流式 / 带来源等多种查询入口
    - 管理对话历史和上下文窗口
    - 提取回答、思维链、工具结果
    - 查询校验、检索决策、错误恢复
    - Prometheus 指标记录
    """

    def __init__(self, settings: Optional[AppSettings] = None):
        self.settings = settings or AppSettings()
        self._conversation_id: str = uuid.uuid4().hex
        self._conversation_history: deque[BaseMessage] = deque(maxlen=_CONVERSATION_MAX_TURNS)

        # 构建 LLM（主 + 备）
        self._llm, self._fallback_llm = self._build_llm_with_fallback(self.settings.llm)

        # 构建并校验工具
        self._tools = self._build_tools_with_validation()

        # 构建 Agent 图
        self._agent = self._build_agent()

        # 统计计数器
        self._query_count: int = 0
        self._error_count: int = 0

        logger.info(
            "FunnelRAGAgent 初始化完成: model=%s, tools=%s, conv_id=%s",
            self.settings.llm.model_name,
            [t.name for t in self._tools],
            self._conversation_id[:8],
        )

    # ------------------------------------------------------------------
    # LLM 构建
    # ------------------------------------------------------------------

    def _build_llm_with_fallback(
        self, llm_settings: LLMSettings
    ) -> Tuple[ChatOpenAI, Optional[ChatOpenAI]]:
        """构建主 LLM 和可选的 fallback LLM 实例。

        当主模型不可用时，fallback LLM 将作为降级后端，确保服务可用性。
        fallback 使用相同配置但可指定不同的 model_name。

        Args:
            llm_settings: LLM 配置

        Returns:
            (primary_llm, fallback_llm) 元组，fallback 可能为 None
        """
        primary = self._create_chat_openai(llm_settings)
        logger.info("主 LLM 初始化: %s", llm_settings.model_name)

        # 如果配置了 fallback 模型名且与主模型不同，则创建 fallback
        fallback_model = getattr(llm_settings, "fallback_model_name", None)
        fallback_llm: Optional[ChatOpenAI] = None
        if fallback_model and fallback_model != llm_settings.model_name:
            fallback_settings = llm_settings.model_copy(update={"model_name": fallback_model})
            fallback_llm = self._create_chat_openai(fallback_settings)
            logger.info("Fallback LLM 初始化: %s", fallback_model)

        return primary, fallback_llm

    @staticmethod
    def _create_chat_openai(llm_settings: LLMSettings) -> ChatOpenAI:
        """根据设置创建 ChatOpenAI 实例。"""
        kwargs: Dict[str, Any] = {
            "model": llm_settings.model_name,
            "temperature": llm_settings.temperature,
            "max_tokens": llm_settings.max_tokens,
        }
        if llm_settings.api_key:
            kwargs["api_key"] = llm_settings.api_key
        if llm_settings.api_base:
            kwargs["base_url"] = llm_settings.api_base
        return ChatOpenAI(**kwargs)

    # ------------------------------------------------------------------
    # 工具构建
    # ------------------------------------------------------------------

    def _build_tools_with_validation(self) -> List:
        """构建工具集并在启动时校验工具可用性。

        每个工具被实例化后进行轻量级可用性探测（如连接测试），
        不可用的工具会被标记但不会阻止 Agent 启动。

        Returns:
            可用工具列表

        Raises:
            AgentNotInitializedError: 如果所有工具都不可用
        """
        tool_instances = [
            MilvusSearchTool(milvus_settings=self.settings.milvus),
            RerankTool(reranker_settings=self.settings.reranker),
            DirectAnswerTool(),
        ]

        available_tools: List = []
        for tool_inst in tool_instances:
            try:
                # MilvusSearchTool: 尝试获取 client 以验证连接
                if isinstance(tool_inst, MilvusSearchTool):
                    client = tool_inst._get_client()
                    if client is not None:
                        available_tools.append(tool_inst)
                        logger.info("工具可用: %s", tool_inst.name)
                    else:
                        logger.warning("工具不可用 (连接失败): %s", tool_inst.name)
                else:
                    # 其他工具默认可用
                    available_tools.append(tool_inst)
                    logger.info("工具可用: %s", tool_inst.name)
            except Exception as exc:
                logger.warning("工具初始化校验失败: %s - %s", tool_inst.name, exc)

        if not available_tools:
            logger.error("没有可用的工具，Agent 无法正常工作")
            # DirectAnswerTool 不需要外部连接，至少保留它
            available_tools.append(DirectAnswerTool())

        return available_tools

    # ------------------------------------------------------------------
    # Agent 图构建
    # ------------------------------------------------------------------

    def _build_agent(self):
        """基于 LangChain create_agent 构建 Agent 执行图。"""
        agent = create_agent(
            model=self._llm,
            tools=self._tools,
            system_prompt=REACT_SYSTEM_PROMPT,
            debug=self.settings.debug,
        )
        return agent

    # ------------------------------------------------------------------
    # 查询校验
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_query(question: str) -> str:
        """校验和清洗用户查询输入。

        Args:
            question: 原始用户输入

        Returns:
            清洗后的查询字符串

        Raises:
            QueryValidationError: 输入不合法
        """
        if not question or not question.strip():
            raise QueryValidationError("查询内容不能为空")

        # 去除首尾空白
        cleaned = question.strip()

        # 长度限制
        if len(cleaned) > _QUERY_MAX_LENGTH:
            cleaned = cleaned[:_QUERY_MAX_LENGTH]
            logger.info("查询已截断至 %d 字符", _QUERY_MAX_LENGTH)

        # 基础 XSS 防护：移除 HTML 标签
        cleaned = re.sub(r"<[^>]+>", "", cleaned)

        # 移除控制字符（保留换行和制表符）
        cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)

        if not cleaned:
            raise QueryValidationError("清洗后查询内容为空")

        return cleaned

    # ------------------------------------------------------------------
    # 检索决策
    # ------------------------------------------------------------------

    @staticmethod
    def _should_retrieve(question: str) -> bool:
        """启发式判断当前查询是否需要知识库检索。

        基于关键词和模式匹配来决定是否触发 milvus_search → rerank 流程。
        此方法为轻量级启发式，不调用 LLM。

        Args:
            question: 用户查询

        Returns:
            True 表示建议检索，False 表示可直接回答
        """
        retrieval_keywords = [
            "流程", "规范", "制度", "文档", "手册", "标准", "规定",
            "政策", "指南", "方案", "配置", "部署", "架构", "设计",
            "产品", "系统", "服务", "项目", "平台",
            "查一下", "查下", "有没有", "是什么", "怎么", "如何",
            "帮我查", "找一下", "搜索",
        ]

        direct_keywords = [
            "你好", "hello", "hi", "谢谢", "感谢", "再见",
            "你是谁", "你能做什么", "你叫什么",
        ]

        question_lower = question.lower()

        # 如果匹配直接回答关键词，不检索
        for kw in direct_keywords:
            if kw in question_lower:
                return False

        # 如果匹配检索关键词，需要检索
        for kw in retrieval_keywords:
            if kw in question_lower:
                return True

        # 默认：对较长的问题倾向于检索
        return len(question) > 15

    # ------------------------------------------------------------------
    # 上下文格式化
    # ------------------------------------------------------------------

    @staticmethod
    def _format_retrieved_context(
        documents: List[Dict[str, Any]],
        max_length: int = 4000,
    ) -> str:
        """将检索结果格式化为 LLM 可用的上下文字符串。

        Args:
            documents: 检索结果列表
            max_length: 上下文最大字符长度

        Returns:
            格式化后的上下文字符串
        """
        if not documents:
            return "（未检索到相关文档）"
        return format_documents(documents, include_score=True, max_text_length=max_length // max(len(documents), 1))

    # ------------------------------------------------------------------
    # 结果提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_answer(messages: List[BaseMessage]) -> str:
        """从消息列表中提取最终回答，支持多种提取策略。

        策略优先级：
        1. 最后一条非空 AIMessage（标准提取）
        2. 最后一条 ToolMessage 中包含最终答案的标记
        3. 拼接所有 AIMessage 内容

        Args:
            messages: Agent 执行产生的消息列表

        Returns:
            提取出的回答字符串
        """
        # 策略 1：最后一条非空 AI 消息
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    return content.strip()

        # 策略 2：包含最终答案的 ToolMessage
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage) and msg.content:
                content = str(msg.content)
                if "FINAL_ANSWER:" in content:
                    return content.split("FINAL_ANSWER:", 1)[1].strip()

        # 策略 3：拼接所有 AI 消息
        ai_contents = []
        for msg in messages:
            if isinstance(msg, AIMessage) and msg.content:
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                if content.strip():
                    ai_contents.append(content.strip())
        if ai_contents:
            return "\n".join(ai_contents)

        return "（未能提取到有效回答）"

    @staticmethod
    def _extract_thinking_chain(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """从消息列表中提取思维链（工具调用序列）。

        Args:
            messages: Agent 执行产生的消息列表

        Returns:
            思维链列表，每步包含 tool、input、output
        """
        thinking_chain: List[Dict[str, Any]] = []

        for msg in messages:
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    thinking_chain.append(
                        {
                            "tool": tc.get("name", ""),
                            "input": tc.get("args", {}),
                        }
                    )
            elif hasattr(msg, "name") and hasattr(msg, "content") and msg.name:
                # 工具返回结果 —— 匹配到对应的 thinking_chain 条目
                for tc_entry in reversed(thinking_chain):
                    if tc_entry["tool"] == msg.name and "output" not in tc_entry:
                        output = str(msg.content)
                        tc_entry["output"] = output[:_TOOL_OUTPUT_MAX_LENGTH]
                        break

        return thinking_chain

    @staticmethod
    def _extract_tool_results(messages: List[BaseMessage]) -> List[Dict[str, Any]]:
        """从消息列表中提取所有工具执行结果。

        Args:
            messages: Agent 执行产生的消息列表

        Returns:
            工结果列表，每个元素包含 tool_name 和 content
        """
        results: List[Dict[str, Any]] = []
        for msg in messages:
            if isinstance(msg, ToolMessage):
                results.append(
                    {
                        "tool_name": getattr(msg, "name", "unknown"),
                        "content": str(msg.content)[:_TOOL_OUTPUT_MAX_LENGTH],
                    }
                )
        return results

    # ------------------------------------------------------------------
    # 错误处理
    # ------------------------------------------------------------------

    @staticmethod
    def _handle_tool_error(
        tool_name: str,
        error: Exception,
        default_return: Any = None,
    ) -> Any:
        """优雅处理工具执行失败。

        Args:
            tool_name: 失败的工具名称
            error: 异常实例
            default_return: 降级返回值

        Returns:
            default_return 或错误信息字典
        """
        logger.error("工具执行失败: %s - %s", tool_name, error)
        if default_return is not None:
            return default_return
        return {
            "error": True,
            "tool": tool_name,
            "message": f"工具 {tool_name} 执行失败，请稍后重试",
        }

    # ------------------------------------------------------------------
    # 核心查询方法
    # ------------------------------------------------------------------

    async def _invoke_agent_with_retry(
        self,
        question: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """带重试的 Agent 调用，处理瞬态 LLM 故障。

        使用指数退避重试最多 _MAX_RETRIES 次。

        Args:
            question: 用户查询
            config: 可选的 LangSmith 运行配置

        Returns:
            Agent 执行结果字典

        Raises:
            LLMCallError: 所有重试均失败
        """
        last_error: Optional[Exception] = None
        run_config = config or {"run_name": "funnelrag_agent"}

        for attempt in range(_MAX_RETRIES):
            try:
                result = await self._agent.ainvoke(
                    {"messages": [("user", question)]},
                    config=run_config,
                )
                return result
            except Exception as exc:
                last_error = exc
                error_msg = str(exc).lower()
                # 判断是否可重试（速率限制、超时等）
                retryable = any(
                    kw in error_msg
                    for kw in ["rate_limit", "timeout", "429", "503", "connection"]
                )
                if not retryable or attempt == _MAX_RETRIES - 1:
                    raise LLMCallError(str(exc), retryable=retryable) from exc

                delay = _RETRY_DELAY_BASE * (2 ** attempt)
                logger.warning(
                    "LLM 调用失败 (attempt %d/%d), %ss 后重试: %s",
                    attempt + 1,
                    _MAX_RETRIES,
                    delay,
                    exc,
                )
                time.sleep(delay)

        raise LLMCallError(str(last_error), retryable=False)

    @traceable(name="funnelrag_query", run_type="chain")
    async def aquery(self, question: str, **kwargs) -> Dict[str, Any]:
        """异步查询入口，执行完整的 Agent 推理循环。

        Args:
            question: 用户问题
            **kwargs: 可选参数（temperature_override 等）

        Returns:
            包含 answer、thinking_chain、sources、tool_results 的字典
        """
        start = time.perf_counter()
        self._query_count += 1
        QUERY_TOTAL.inc()

        # 校验输入
        try:
            validated_question = self._validate_query(question)
        except QueryValidationError as exc:
            self._error_count += 1
            return {
                "answer": f"输入校验失败: {exc}",
                "thinking_chain": [],
                "sources": [],
                "tool_results": [],
            }

        # 记录到对话历史
        self._conversation_history.append(HumanMessage(content=validated_question))

        try:
            # 带重试的 Agent 调用
            result = await self._invoke_agent_with_retry(validated_question)

            messages = result.get("messages", [])
            answer = self._extract_answer(messages)
            thinking_chain = self._extract_thinking_chain(messages)
            tool_results = self._extract_tool_results(messages)

            # 提取来源文档
            sources = self._extract_sources_from_results(tool_results)

            # 记录 AI 回答到对话历史
            self._conversation_history.append(AIMessage(content=answer))

            elapsed = time.perf_counter() - start
            QUERY_LATENCY.observe(elapsed)
            LLM_GENERATE_LATENCY.observe(elapsed)
            logger.info("Agent 查询完成，耗时 %.3fs", elapsed)

            return {
                "answer": answer,
                "thinking_chain": thinking_chain,
                "sources": sources,
                "tool_results": tool_results,
            }
        except LLMCallError as exc:
            self._error_count += 1
            logger.error("LLM 调用最终失败: %s", exc)
            return {
                "answer": "抱歉，AI 服务暂时不可用，请稍后重试。" if exc.retryable else f"处理问题时出错: {exc}",
                "thinking_chain": [],
                "sources": [],
                "tool_results": [],
            }
        except Exception as exc:
            self._error_count += 1
            logger.error("Agent 查询异常: %s", exc)
            return {
                "answer": f"抱歉，处理您的问题时出现错误: {exc}",
                "thinking_chain": [],
                "sources": [],
                "tool_results": [],
            }

    @traceable(name="funnelrag_query_sync", run_type="chain")
    def query(self, question: str, **kwargs) -> Dict[str, Any]:
        """同步查询入口。

        自动检测当前是否已在事件循环中运行，避免嵌套循环冲突。

        Args:
            question: 用户问题
            **kwargs: 可选参数

        Returns:
            查询结果字典
        """
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            import concurrent.futures

            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, self.aquery(question, **kwargs))
                return future.result()
        else:
            return asyncio.run(self.aquery(question, **kwargs))

    # ------------------------------------------------------------------
    # 流式查询
    # ------------------------------------------------------------------

    async def stream_query(
        self,
        question: str,
        **kwargs,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """异步流式查询，逐步产生 Agent 推理过程中的中间事件。

        产生的事件类型：
        - {"type": "thinking", "tool": ..., "input": ...} - 工具调用
        - {"type": "tool_result", "tool": ..., "output": ...} - 工具结果
        - {"type": "token", "content": ...} - LLM 输出 token
        - {"type": "answer", "answer": ...} - 最终答案
        - {"type": "sources", "sources": ...} - 来源文档
        - {"type": "done"} - 流结束

        Args:
            question: 用户问题

        Yields:
            事件字典
        """
        validated = self._validate_query(question)
        self._conversation_history.append(HumanMessage(content=validated))

        try:
            result = await self._invoke_agent_with_retry(validated)
            messages = result.get("messages", [])

            # 逐步推送思维链事件
            for msg in messages:
                if hasattr(msg, "tool_calls") and msg.tool_calls:
                    for tc in msg.tool_calls:
                        yield {
                            "type": "thinking",
                            "tool": tc.get("name", ""),
                            "input": tc.get("args", {}),
                        }
                elif isinstance(msg, ToolMessage):
                    yield {
                        "type": "tool_result",
                        "tool": getattr(msg, "name", "unknown"),
                        "output": str(msg.content)[:_TOOL_OUTPUT_MAX_LENGTH],
                    }
                elif isinstance(msg, AIMessage) and msg.content:
                    content = msg.content if isinstance(msg.content, str) else str(msg.content)
                    yield {"type": "token", "content": content}

            # 提取最终答案和来源
            answer = self._extract_answer(messages)
            tool_results = self._extract_tool_results(messages)
            sources = self._extract_sources_from_results(tool_results)

            self._conversation_history.append(AIMessage(content=answer))

            yield {"type": "answer", "answer": answer}
            if sources:
                yield {"type": "sources", "sources": sources}
            yield {"type": "done"}

        except Exception as exc:
            logger.error("流式查询异常: %s", exc)
            yield {"type": "error", "message": str(exc)}
            yield {"type": "done"}

    # ------------------------------------------------------------------
    # 带来源的查询
    # ------------------------------------------------------------------

    async def query_with_sources(
        self,
        question: str,
        **kwargs,
    ) -> Dict[str, Any]:
        """执行查询并返回完整的来源文档信息。

        相比 aquery，此方法额外返回格式化的来源引用字符串。

        Args:
            question: 用户问题

        Returns:
            包含 answer、thinking_chain、sources、formatted_sources 的字典
        """
        result = await self.aquery(question, **kwargs)
        sources = result.get("sources", [])

        result["formatted_sources"] = format_sources(sources) if sources else ""
        return result

    # ------------------------------------------------------------------
    # 来源提取
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_sources_from_results(
        tool_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """从工具结果中提取来源文档信息。

        尝试解析工具结果中的 JSON 格式文档列表，
        提取 doc_id、score、text 等字段。

        Args:
            tool_results: 工结果列表

        Returns:
            来源文档列表
        """
        import json

        sources: List[Dict[str, Any]] = []
        for tr in tool_results:
            content = tr.get("content", "")
            try:
                # 尝试解析为 JSON
                parsed = json.loads(content)
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "doc_id" in item:
                            sources.append(
                                {
                                    "doc_id": item.get("doc_id", ""),
                                    "score": item.get("score", 0.0),
                                    "text": item.get("text", "")[:200],
                                }
                            )
                elif isinstance(parsed, dict) and "doc_id" in parsed:
                    sources.append(
                        {
                            "doc_id": parsed.get("doc_id", ""),
                            "score": parsed.get("score", 0.0),
                            "text": parsed.get("text", "")[:200],
                        }
                    )
            except (json.JSONDecodeError, TypeError):
                # 非 JSON 结果，跳过
                continue
        return sources

    # ------------------------------------------------------------------
    # 对话管理
    # ------------------------------------------------------------------

    def reset_conversation(self) -> None:
        """重置对话历史，开始新的会话。"""
        self._conversation_history.clear()
        self._conversation_id = uuid.uuid4().hex
        logger.info("对话已重置, new conv_id=%s", self._conversation_id[:8])

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """获取当前对话历史的可序列化形式。

        Returns:
            对话历史列表，每个元素包含 role 和 content
        """
        history: List[Dict[str, str]] = []
        for msg in self._conversation_history:
            if isinstance(msg, HumanMessage):
                history.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                content = msg.content if isinstance(msg.content, str) else str(msg.content)
                history.append({"role": "assistant", "content": content})
        return history

    # ------------------------------------------------------------------
    # Agent 信息
    # ------------------------------------------------------------------

    def get_agent_info(self) -> Dict[str, Any]:
        """返回 Agent 的能力、工具和模型等元信息。

        Returns:
            Agent 信息字典
        """
        return {
            "name": "FunnelRAG",
            "version": "1.0.0",
            "model": self.settings.llm.model_name,
            "fallback_model": getattr(self.settings.llm, "fallback_model_name", None),
            "tools": [
                {
                    "name": t.name,
                    "description": t.description,
                }
                for t in self._tools
            ],
            "conversation_id": self._conversation_id,
            "conversation_turns": len(self._conversation_history),
            "query_count": self._query_count,
            "error_count": self._error_count,
            "settings": {
                "temperature": self.settings.llm.temperature,
                "max_tokens": self.settings.llm.max_tokens,
                "debug": self.settings.debug,
            },
        }
