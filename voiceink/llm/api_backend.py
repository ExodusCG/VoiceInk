"""
VoiceInk - 云端 API 后端（OpenAI 兼容格式）

支持所有兼容 OpenAI API 格式的大语言模型服务：
- OpenAI（GPT-4o-mini / GPT-4o 等）
- 阿里通义千问（Qwen）
- DeepSeek
- 月之暗面（Moonshot / Kimi）
- 智谱（GLM）
- 其他兼容 OpenAI 格式的服务

特点：
- 通过 openai 官方 Python 库调用，兼容性强
- 支持流式（streaming）和非流式两种输出模式
- 自动重试机制，应对网络抖动
- 完善的超时和异常处理
"""

import time
import threading
from typing import Optional, List, Iterator

from voiceink.utils.logger import logger
from voiceink.config import LLMConfig
from voiceink.llm.base import (
    LLMBackend,
    PolishResult,
    POLISH_SYSTEM_PROMPT,
    build_polish_prompt,
)


class APIBackend(LLMBackend):
    """
    基于 OpenAI 兼容 API 的云端推理后端

    通过 HTTP API 调用云端大语言模型进行文本润色。
    只需配置 base_url、api_key 和 model 即可对接不同服务商。
    """

    # 各主流 API 服务商的默认 base_url（方便用户参考）
    KNOWN_PROVIDERS = {
        "openai": "https://api.openai.com/v1",
        "deepseek": "https://api.deepseek.com/v1",
        "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "moonshot": "https://api.moonshot.cn/v1",
        "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    }

    def __init__(self, config: LLMConfig):
        """
        初始化 API 后端

        Args:
            config: LLM 配置，其中 config.api 包含 API 相关设置
        """
        self._config = config
        self._api_config = config.api
        self._client = None        # openai.OpenAI 同步客户端
        self._is_loaded = False     # 标记后端是否已初始化就绪
        self._lock = threading.Lock()  # 线程安全锁

        logger.info(
            f"[API] 初始化，base_url={self._api_config.base_url}, "
            f"model={self._api_config.model}"
        )

    def load_model(self) -> None:
        """
        初始化 API 客户端并验证连接

        Raises:
            ValueError: API Key 未配置
            RuntimeError: openai 库未安装或连接验证失败
        """
        # 检查 API Key 是否已配置
        if not self._api_config.api_key:
            raise ValueError(
                "API Key 未配置。\n"
                "请在配置文件中设置 llm.api.api_key，或设置环境变量 OPENAI_API_KEY。"
            )

        try:
            # 延迟导入，避免未安装 openai 时影响其他模块
            from openai import OpenAI

            self._client = OpenAI(
                api_key=self._api_config.api_key,
                base_url=self._api_config.base_url,
                timeout=30.0,    # 请求超时 30 秒
                max_retries=2,   # 自动重试 2 次
            )

            self._is_loaded = True
            logger.info(f"[API] 客户端初始化成功，model={self._api_config.model}")

        except ImportError:
            raise RuntimeError(
                "未安装 openai 库。\n"
                "请运行: pip install openai\n"
                "详情参考: https://github.com/openai/openai-python"
            )
        except Exception as e:
            self._client = None
            self._is_loaded = False
            raise RuntimeError(f"API 客户端初始化失败: {e}")

    def unload(self) -> None:
        """关闭 API 客户端，释放连接资源"""
        with self._lock:
            if self._client is not None:
                try:
                    self._client.close()
                except Exception:
                    pass  # 关闭时的异常无需处理
                self._client = None
            self._is_loaded = False
            logger.info("[API] 客户端已关闭")

    def is_loaded(self) -> bool:
        """检查 API 客户端是否已初始化"""
        return self._is_loaded and self._client is not None

    def polish(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        通过 API 调用润色文本（非流式）

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文（之前说过的话）
            custom_terms: 自定义词汇列表

        Returns:
            PolishResult 润色结果

        Raises:
            RuntimeError: 客户端未初始化
        """
        if not self.is_loaded():
            raise RuntimeError("API 客户端未初始化，请先调用 load_model()")

        raw_text = raw_text.strip()
        if not raw_text:
            return PolishResult(text="", raw_text=raw_text)

        # 构建提示词
        user_prompt = build_polish_prompt(raw_text, context, custom_terms)
        messages = [
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            with self._lock:
                start_time = time.time()

                response = self._client.chat.completions.create(
                    model=self._api_config.model,
                    messages=messages,
                    temperature=self._config.temperature,
                    top_p=self._config.top_p,
                    max_tokens=len(raw_text) * 3 + 100,
                    stream=False,
                )

                elapsed = time.time() - start_time

            # 提取生成的文本
            polished = response.choices[0].message.content.strip()

            # 统计 token 用量
            tokens_used = None
            if response.usage:
                tokens_used = response.usage.total_tokens

            logger.debug(
                f"[API] 润色完成，耗时 {elapsed:.2f}s，"
                f"输入 {len(raw_text)} 字 → 输出 {len(polished)} 字，"
                f"tokens={tokens_used}"
            )

            return PolishResult(
                text=polished,
                raw_text=raw_text,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            error_msg = self._parse_api_error(e)
            logger.error(f"[API] 润色失败: {error_msg}")
            return PolishResult(
                text=raw_text,
                raw_text=raw_text,
                success=False,
                error=error_msg,
            )

    def polish_stream(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> Iterator[str]:
        """
        通过 API 流式调用润色文本

        以迭代器方式逐块返回润色结果，适用于需要实时显示的场景。
        调用方可以逐块拼接获得完整结果。

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文
            custom_terms: 自定义词汇列表

        Yields:
            润色后的文本片段（逐块输出）

        Raises:
            RuntimeError: 客户端未初始化
        """
        if not self.is_loaded():
            raise RuntimeError("API 客户端未初始化，请先调用 load_model()")

        raw_text = raw_text.strip()
        if not raw_text:
            return

        # 构建提示词
        user_prompt = build_polish_prompt(raw_text, context, custom_terms)
        messages = [
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            stream = self._client.chat.completions.create(
                model=self._api_config.model,
                messages=messages,
                temperature=self._config.temperature,
                top_p=self._config.top_p,
                max_tokens=len(raw_text) * 3 + 100,
                stream=True,
            )

            for chunk in stream:
                # 每个 chunk 可能包含一小段文本
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content

        except Exception as e:
            error_msg = self._parse_api_error(e)
            logger.error(f"[API] 流式润色失败: {error_msg}")
            # 流式模式下出错，回退输出原文
            yield raw_text

    def polish_stream_full(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        流式调用的便捷封装：收集所有流式片段并返回完整的 PolishResult

        内部使用流式 API，但对外呈现与 polish() 相同的返回格式。

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文
            custom_terms: 自定义词汇列表

        Returns:
            PolishResult 润色结果
        """
        try:
            parts = []
            for chunk_text in self.polish_stream(raw_text, context, custom_terms):
                parts.append(chunk_text)

            polished = "".join(parts).strip()

            if not polished:
                return PolishResult(
                    text=raw_text,
                    raw_text=raw_text,
                    success=False,
                    error="流式润色结果为空",
                )

            return PolishResult(
                text=polished,
                raw_text=raw_text,
                success=True,
            )

        except Exception as e:
            return PolishResult(
                text=raw_text,
                raw_text=raw_text,
                success=False,
                error=f"流式润色异常: {e}",
            )

    @staticmethod
    def _parse_api_error(error: Exception) -> str:
        """
        解析 API 错误，返回人类可读的错误信息

        针对常见的 API 错误类型提供中文提示。

        Args:
            error: 异常对象

        Returns:
            格式化后的错误信息字符串
        """
        error_type = type(error).__name__
        error_str = str(error)

        # 尝试从 openai 库导入具体的异常类型进行匹配
        try:
            from openai import (
                AuthenticationError,
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
                BadRequestError,
            )

            if isinstance(error, AuthenticationError):
                return "API 认证失败：请检查 API Key 是否正确"
            elif isinstance(error, RateLimitError):
                return "API 请求频率超限：请稍后重试，或升级 API 套餐"
            elif isinstance(error, APIConnectionError):
                return "API 连接失败：请检查网络连接和 base_url 配置"
            elif isinstance(error, APITimeoutError):
                return "API 请求超时：服务响应过慢，请稍后重试"
            elif isinstance(error, BadRequestError):
                return f"API 请求参数错误：{error_str}"
        except ImportError:
            pass

        return f"API 调用失败 ({error_type}): {error_str}"
