"""
VoiceInk - llama.cpp 本地推理后端

使用 llama-cpp-python 库加载 GGUF 格式的本地大语言模型，
在本地 CPU 上进行推理，无需联网。适合对隐私要求高或网络不佳的场景。

特点：
- 纯 CPU 推理（n_gpu_layers=0），兼容无 GPU 环境
- 低温度（temperature=0.1）确保输出稳定一致
- 长文本自动分段处理，避免超出上下文窗口
- 完善的异常处理和超时机制
"""

import re
import time
import threading
from pathlib import Path
from typing import Optional, List

from voiceink.utils.logger import logger
from voiceink.config import LLMConfig
from voiceink.llm.base import (
    LLMBackend,
    PolishResult,
    POLISH_SYSTEM_PROMPT,
    build_polish_prompt,
)


class LlamaCppBackend(LLMBackend):
    """
    基于 llama-cpp-python 的本地推理后端

    使用 GGUF 格式的量化模型在 CPU 上进行推理。
    推荐使用 Qwen3-0.6B 等小型模型以获得较快的推理速度。
    """

    def __init__(self, config: LLMConfig):
        """
        初始化 llama.cpp 后端

        Args:
            config: LLM 配置，包含模型路径、线程数等参数
        """
        self._config = config
        self._model = None  # llama_cpp.Llama 实例
        self._lock = threading.Lock()  # 线程安全锁，防止并发推理冲突

        # 解析模型文件的完整路径
        if config.model_path:
            self._model_file = Path(config.model_path) / config.model_name
        else:
            # 默认在用户目录下查找模型
            self._model_file = Path.home() / ".voiceink" / "models" / "llm" / config.model_name

        logger.info(f"[LlamaCpp] 初始化，模型路径: {self._model_file}")

    def load_model(self) -> None:
        """
        加载 GGUF 模型到内存

        Raises:
            FileNotFoundError: 模型文件不存在
            RuntimeError: 模型加载失败（格式错误、内存不足等）
        """
        # 检查模型文件是否存在
        if not self._model_file.exists():
            raise FileNotFoundError(
                f"模型文件不存在: {self._model_file}\n"
                f"请下载 GGUF 模型并放置到该路径。\n"
                f"推荐模型: Qwen3-0.6B-Q8_0.gguf\n"
                f"下载地址: https://huggingface.co/Qwen/Qwen3-0.6B-GGUF"
            )

        try:
            # 延迟导入，避免未安装 llama-cpp-python 时影响其他模块
            from llama_cpp import Llama

            logger.info(f"[LlamaCpp] 正在加载模型: {self._model_file.name} ...")
            start_time = time.time()

            self._model = Llama(
                model_path=str(self._model_file),
                n_ctx=self._config.n_ctx,        # 上下文窗口大小
                n_threads=self._config.n_threads,  # CPU 线程数
                n_gpu_layers=0,                    # 纯 CPU 推理，不使用 GPU
                verbose=False,                     # 关闭 llama.cpp 的详细日志输出
            )

            elapsed = time.time() - start_time
            logger.info(f"[LlamaCpp] 模型加载完成，耗时 {elapsed:.1f}s")

        except ImportError:
            raise RuntimeError(
                "未安装 llama-cpp-python 库。\n"
                "请运行: pip install llama-cpp-python\n"
                "详情参考: https://github.com/abetlen/llama-cpp-python"
            )
        except Exception as e:
            self._model = None
            raise RuntimeError(f"模型加载失败: {e}")

    def unload(self) -> None:
        """卸载模型，释放内存"""
        with self._lock:
            if self._model is not None:
                # llama-cpp-python 没有显式的 unload 方法，
                # 将引用置空后由 Python GC 回收内存
                del self._model
                self._model = None
                logger.info("[LlamaCpp] 模型已卸载")

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._model is not None

    def polish(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        使用本地模型润色语音识别文本

        对于超长文本（超过 max_polish_length），会自动分段处理，
        每段独立润色后拼接结果。

        Args:
            raw_text: 语音识别输出的原始文本
            context: 上下文（之前说过的话）
            custom_terms: 自定义词汇列表

        Returns:
            PolishResult 润色结果

        Raises:
            RuntimeError: 模型未加载
        """
        if not self.is_loaded():
            raise RuntimeError("模型未加载，请先调用 load_model()")

        raw_text = raw_text.strip()
        if not raw_text:
            return PolishResult(text="", raw_text=raw_text)

        # 检查是否需要分段处理
        max_len = self._config.max_polish_length
        if len(raw_text) > max_len:
            return self._polish_long_text(raw_text, context, custom_terms, max_len)

        # 单段润色
        return self._polish_single(raw_text, context, custom_terms)

    def _polish_single(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None
    ) -> PolishResult:
        """
        对单段文本进行润色（内部方法）

        Args:
            raw_text: 原始文本（已确保长度在限制内）
            context: 上下文
            custom_terms: 自定义词汇

        Returns:
            PolishResult 润色结果
        """
        # 构建提示词
        user_prompt = build_polish_prompt(raw_text, context, custom_terms)

        # 使用 chat completion 格式（适配指令微调模型）
        messages = [
            {"role": "system", "content": POLISH_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            with self._lock:
                start_time = time.time()

                response = self._model.create_chat_completion(
                    messages=messages,
                    temperature=self._config.temperature,  # 低温度 → 稳定输出
                    top_p=self._config.top_p,
                    max_tokens=len(raw_text) * 4 + 300,  # 给足空间（含 think 标签）
                    stop=[
                        "【",             # 防止模型重复生成 prompt 格式
                        "<|im_end|>",     # Qwen 系列模型的结束标记
                        "<|endoftext|>",  # 通用结束标记
                        # 不拦截 "\n\n" 和 "<think>"：Qwen3 的思考内容中
                        # 包含大量换行，会被 "\n\n" 误截导致输出为空。
                        # 由 _strip_think_tags() 在后处理中清理思考标签。
                    ],
                )

                elapsed = time.time() - start_time

            # 提取生成的文本
            raw_content = response["choices"][0]["message"]["content"]
            finish_reason = response["choices"][0].get("finish_reason", "unknown")
            tokens_used = response.get("usage", {}).get("total_tokens")

            logger.info(
                f"[LlamaCpp] 原始输出 ({elapsed:.2f}s, "
                f"finish={finish_reason}, tokens={tokens_used}): "
                f"'{raw_content[:200]}'"
            )

            polished = raw_content.strip()

            # 清理可能残留的 Qwen3 思考标签
            polished = self._strip_think_tags(polished)

            if polished != raw_content.strip():
                logger.info(
                    f"[LlamaCpp] 清理 think 标签后: '{polished[:100]}'"
                )

            logger.info(
                f"[LlamaCpp] 润色: {len(raw_text)} 字 → {len(polished)} 字"
            )

            return PolishResult(
                text=polished,
                raw_text=raw_text,
                success=True,
                tokens_used=tokens_used,
            )

        except Exception as e:
            logger.error(f"[LlamaCpp] 推理失败: {e}")
            return PolishResult(
                text=raw_text,
                raw_text=raw_text,
                success=False,
                error=f"推理失败: {e}",
            )

    # Qwen3 思考标签的正则（匹配 <think>...</think> 及未闭合的 <think>...）
    _THINK_TAG_RE = re.compile(r"<think>.*?</think>|<think>.*", re.DOTALL)

    @classmethod
    def _strip_think_tags(cls, text: str) -> str:
        """
        清除 Qwen3 模型可能输出的 <think>...</think> 思考标签。

        即使通过 /no_think 指令禁用思考模式，某些情况下模型仍可能输出
        思考标签。此方法作为最后一道防线，确保输出不包含思考内容。

        Args:
            text: 模型生成的原始文本

        Returns:
            清除思考标签后的文本
        """
        if "<think>" not in text:
            return text
        cleaned = cls._THINK_TAG_RE.sub("", text).strip()
        return cleaned if cleaned else text  # 如果清理后为空，回退到原文

    def _polish_long_text(
        self,
        raw_text: str,
        context: Optional[str] = None,
        custom_terms: Optional[List[str]] = None,
        max_len: int = 500,
    ) -> PolishResult:
        """
        分段处理超长文本

        将长文本按句子边界拆分为多个段落，每段独立润色后合并。
        后续段落会将前一段的润色结果作为上下文传入，保持连贯性。

        Args:
            raw_text: 超长的原始文本
            context: 初始上下文
            custom_terms: 自定义词汇
            max_len: 每段最大长度

        Returns:
            合并后的 PolishResult
        """
        logger.info(
            f"[LlamaCpp] 文本过长（{len(raw_text)} 字），"
            f"将按 {max_len} 字分段处理"
        )

        # 按句子边界分段
        segments = self._split_text(raw_text, max_len)
        polished_parts = []
        total_tokens = 0
        current_context = context or ""

        for i, segment in enumerate(segments):
            logger.debug(f"[LlamaCpp] 处理第 {i+1}/{len(segments)} 段，长度 {len(segment)} 字")

            result = self._polish_single(segment, current_context, custom_terms)

            if result.success:
                polished_parts.append(result.text)
                # 将本段结果作为下一段的上下文
                current_context = result.text
            else:
                # 单段失败时使用原文，继续处理后续段落
                polished_parts.append(segment)
                current_context = segment
                logger.warning(f"[LlamaCpp] 第 {i+1} 段润色失败，使用原文: {result.error}")

            if result.tokens_used:
                total_tokens += result.tokens_used

        # 合并所有段落
        final_text = "".join(polished_parts)

        return PolishResult(
            text=final_text,
            raw_text=raw_text,
            success=True,
            tokens_used=total_tokens if total_tokens > 0 else None,
        )

    @staticmethod
    def _split_text(text: str, max_len: int) -> List[str]:
        """
        按句子边界将文本拆分为不超过 max_len 的段落

        优先在句号、问号、感叹号处断开；
        其次在逗号、分号处断开；
        最后强制按 max_len 截断。

        Args:
            text: 待拆分文本
            max_len: 每段最大字符数

        Returns:
            文本段落列表
        """
        if len(text) <= max_len:
            return [text]

        segments = []
        remaining = text

        while remaining:
            if len(remaining) <= max_len:
                segments.append(remaining)
                break

            # 在 max_len 范围内寻找最佳断句点
            chunk = remaining[:max_len]

            # 优先级1：句号、问号、感叹号（中英文）
            best_break = -1
            for sep in ["。", "！", "？", ".", "!", "?"]:
                pos = chunk.rfind(sep)
                if pos > best_break:
                    best_break = pos

            # 优先级2：逗号、分号
            if best_break < max_len // 3:  # 如果断点太靠前，尝试其他标点
                for sep in ["，", "；", ",", ";", "、"]:
                    pos = chunk.rfind(sep)
                    if pos > best_break:
                        best_break = pos

            # 优先级3：空格（英文文本）
            if best_break < max_len // 3:
                pos = chunk.rfind(" ")
                if pos > best_break:
                    best_break = pos

            # 最后的退路：强制截断
            if best_break < max_len // 4:
                best_break = max_len - 1

            # 包含断句标点在内
            split_pos = best_break + 1
            segments.append(remaining[:split_pos])
            remaining = remaining[split_pos:]

        return segments
