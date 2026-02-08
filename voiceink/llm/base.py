"""
VoiceInk - LLM 润色模块抽象基类

定义所有 LLM 后端必须实现的接口，包括模型加载、文本润色、模型卸载等。
无论是本地 llama.cpp 推理还是云端 API 调用，都遵循统一的接口规范。
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, List, AsyncIterator


@dataclass
class PolishResult:
    """
    润色结果数据类

    Attributes:
        text: 润色后的文本
        raw_text: 原始输入文本
        success: 是否润色成功
        error: 如果失败，记录错误信息
        tokens_used: 本次润色消耗的 token 数（可选）
    """
    text: str
    raw_text: str
    success: bool = True
    error: Optional[str] = None
    tokens_used: Optional[int] = None


# ============================================================================
# 通用的润色系统提示词（中英文通用）
# 精心设计用于语音识别后处理：修正错误、添加标点、保持语义
# ============================================================================

POLISH_SYSTEM_PROMPT = """你是一个专业的语音识别文本润色助手。你的任务是对语音识别（ASR）输出的原始文本进行润色修正。

## 润色规则（严格遵守）：

1. **修正语音识别错误**：修正同音字、谐音字、近音词的识别错误。例如："因该" → "应该"，"在坐" → "在座"，"做为" → "作为"。
2. **添加标点符号**：根据语义和语气，添加合适的中文标点（，。！？、；：""）或英文标点（,.!?;:）。
3. **修正语法**：使表述更加流畅自然，但不改变原意。例如去除不必要的口头禅、重复词。
4. **保留原始语义和语气**：不要增删内容的实质含义，不要改变说话者的语气风格（正式/口语/幽默等）。
5. **中英文混合处理**：正确处理中英文混合文本，英文单词保持正确拼写和大小写。
6. **数字与专有名词**：正确处理数字格式，保持专有名词的正确写法。

## 禁止事项：
- 不要添加任何解释、说明或注释
- 不要添加原文中没有的新内容
- 不要改变文本的核心意思
- 不要过度修改，如果原文已经很好就保持不变
- 只输出润色后的文本，不要输出其他任何内容"""


def build_polish_prompt(
    raw_text: str,
    context: Optional[str] = None,
    custom_terms: Optional[List[str]] = None
) -> str:
    """
    构建完整的润色用户提示词

    Args:
        raw_text: 语音识别输出的原始文本
        context: 上下文文本（之前说过的话），帮助模型理解连贯性
        custom_terms: 自定义词汇列表（专业术语、人名等），帮助模型识别特殊词汇

    Returns:
        格式化后的用户提示词字符串
    """
    parts = []

    # 如果有自定义词汇，告知模型优先参考
    if custom_terms:
        terms_str = "、".join(custom_terms[:30])  # 限制数量避免 prompt 过长
        parts.append(f"【参考词汇表】以下是可能出现的专有名词和术语，请优先使用这些词汇的正确写法：\n{terms_str}")

    # 如果有上下文，提供给模型以保持连贯
    if context and context.strip():
        # 只取最近的上下文，避免过长
        ctx = context.strip()[-300:]
        parts.append(f"【上下文】以下是之前的内容（仅供参考，不需要输出）：\n{ctx}")

    # 核心：要润色的原始文本
    parts.append(f"【待润色文本】\n{raw_text}")

    parts.append("【润色结果】/no_think")

    return "\n\n".join(parts)


class LLMBackend(ABC):
    """
    LLM 后端抽象基类

    所有 LLM 后端（本地推理、云端 API 等）都必须实现此接口。
    提供统一的模型加载、文本润色、模型卸载能力。
    """

    @abstractmethod
    def load_model(self) -> None:
        """
        加载模型 / 初始化后端连接

        对于本地推理后端，这意味着将 GGUF 模型加载到内存。
        对于 API 后端，这意味着验证 API 连接可用性。

        Raises:
            FileNotFoundError: 模型文件不存在（本地后端）
            ConnectionError: API 连接失败（API 后端）
            RuntimeError: 其他加载错误
        """
        pass

    @abstractmethod
    def polish(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        对语音识别输出的文本进行润色

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文文本（之前识别的内容），用于保持连贯
            custom_terms: 自定义词汇列表（专业术语、人名等）

        Returns:
            PolishResult 润色结果

        Raises:
            RuntimeError: 模型未加载时调用
        """
        pass

    @abstractmethod
    def unload(self) -> None:
        """
        卸载模型 / 释放资源

        释放模型占用的内存和 GPU 资源。
        对于 API 后端，关闭连接会话。
        调用后 is_loaded 应返回 False。
        """
        pass

    @abstractmethod
    def is_loaded(self) -> bool:
        """
        检查模型是否已加载就绪

        Returns:
            True 表示模型已加载可用，False 表示尚未加载
        """
        pass

    def polish_or_passthrough(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        安全的润色方法：如果润色失败，返回原文

        这是一个便利方法，确保即使 LLM 出错也不会丢失用户的输入。
        调用方通常应该使用此方法而非直接调用 polish()。

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文文本
            custom_terms: 自定义词汇列表

        Returns:
            PolishResult - 成功时返回润色结果，失败时返回原文
        """
        # 空文本直接返回
        if not raw_text or not raw_text.strip():
            return PolishResult(text="", raw_text=raw_text, success=True)

        # 模型未加载时直接返回原文
        if not self.is_loaded():
            return PolishResult(
                text=raw_text,
                raw_text=raw_text,
                success=False,
                error="模型未加载"
            )

        try:
            result = self.polish(raw_text, context, custom_terms)
            # 如果润色结果为空，回退到原文
            if not result.text or not result.text.strip():
                return PolishResult(
                    text=raw_text,
                    raw_text=raw_text,
                    success=False,
                    error="润色结果为空，已回退到原文"
                )
            return result
        except Exception as e:
            return PolishResult(
                text=raw_text,
                raw_text=raw_text,
                success=False,
                error=f"润色异常: {str(e)}"
            )
