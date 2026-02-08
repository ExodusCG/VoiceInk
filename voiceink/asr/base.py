"""
VoiceInk ASR - 语音识别抽象基类

定义所有 ASR 后端必须实现的接口，以及通用数据结构。
"""

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, List, Generator
from enum import Enum
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构定义
# ============================================================================

class ASRState(Enum):
    """ASR 引擎状态枚举"""
    UNLOADED = "unloaded"       # 模型未加载
    LOADING = "loading"         # 模型加载中
    READY = "ready"             # 就绪，可以接受音频
    TRANSCRIBING = "transcribing"  # 正在识别中
    ERROR = "error"             # 错误状态


@dataclass
class AudioChunk:
    """
    音频数据块，作为 ASR 识别的最小输入单元。

    Attributes:
        data: 音频 PCM 数据，float32 格式，取值范围 [-1.0, 1.0]，
              采样率应为 16kHz，单声道。
        is_speech: 该音频块是否包含语音（由 VAD 判定）。
        timestamp_ms: 该音频块在录音流中的起始时间戳（毫秒）。
    """
    data: np.ndarray  # float32, 16kHz mono
    is_speech: bool = False
    timestamp_ms: int = 0

    def __post_init__(self):
        """确保音频数据格式正确"""
        if not isinstance(self.data, np.ndarray):
            raise TypeError(f"AudioChunk.data 必须是 np.ndarray，收到 {type(self.data)}")
        if self.data.dtype != np.float32:
            self.data = self.data.astype(np.float32)

    @property
    def duration_ms(self) -> float:
        """该音频块的时长（毫秒），基于 16kHz 采样率计算"""
        return len(self.data) / 16.0  # 16000 samples/s = 16 samples/ms

    @property
    def sample_count(self) -> int:
        """采样点数量"""
        return len(self.data)


@dataclass
class TranscriptionResult:
    """
    识别结果数据结构。

    Attributes:
        text: 识别出的文本内容。
        language: 检测到的语言代码（如 "zh", "en"）。
        confidence: 置信度，0.0 ~ 1.0，-1.0 表示不可用。
        duration_ms: 处理的音频时长（毫秒）。
        is_partial: 是否为流式识别的中间结果（非最终结果）。
        segments: 分段信息列表，每个元素为 (起始秒, 结束秒, 文本)。
    """
    text: str = ""
    language: str = ""
    confidence: float = -1.0
    duration_ms: float = 0.0
    is_partial: bool = False
    segments: List[tuple] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        """识别结果是否为空"""
        return len(self.text.strip()) == 0


# ============================================================================
# ASR 后端抽象基类
# ============================================================================

class ASRBackend(ABC):
    """
    ASR 语音识别后端的抽象基类。

    所有 ASR 后端（Whisper.cpp、faster-whisper 等）都必须继承此类
    并实现全部抽象方法。这保证了上层代码可以无缝切换不同后端。

    典型使用流程:
        1. 创建后端实例（通过工厂函数 create_asr_backend）
        2. 调用 load_model() 加载模型
        3. 可选：调用 set_initial_prompt() 注入自定义词汇
        4. 调用 transcribe() 或 transcribe_stream() 进行识别
        5. 结束时调用 unload() 释放资源
    """

    def __init__(self):
        self._state: ASRState = ASRState.UNLOADED
        self._initial_prompt: str = ""

    # --------------------------------------------------
    # 属性
    # --------------------------------------------------

    @property
    def state(self) -> ASRState:
        """当前后端状态"""
        return self._state

    @property
    def is_ready(self) -> bool:
        """模型是否已加载就绪"""
        return self._state == ASRState.READY

    @property
    def initial_prompt(self) -> str:
        """当前设置的初始提示词"""
        return self._initial_prompt

    # --------------------------------------------------
    # 抽象方法 - 子类必须实现
    # --------------------------------------------------

    @abstractmethod
    def load_model(self) -> None:
        """
        加载 ASR 模型到内存。

        此方法应阻塞直到模型加载完成。加载成功后状态变为 READY，
        加载失败应抛出异常并将状态设为 ERROR。

        Raises:
            FileNotFoundError: 模型文件不存在。
            RuntimeError: 模型加载失败（格式错误、内存不足等）。
        """
        ...

    @abstractmethod
    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        对完整音频进行离线识别。

        适用于录音结束后一次性识别全部音频的场景。

        Args:
            audio: 完整音频数据，float32 格式，16kHz 单声道。

        Returns:
            TranscriptionResult: 识别结果。

        Raises:
            RuntimeError: 模型未加载或识别过程出错。
        """
        ...

    @abstractmethod
    def transcribe_stream(
        self, chunks: Generator[AudioChunk, None, None]
    ) -> Generator[TranscriptionResult, None, None]:
        """
        流式识别：逐块接收音频并实时输出识别结果。

        通过生成器模式实现流式处理：
        - 输入：AudioChunk 生成器，持续产出音频块
        - 输出：TranscriptionResult 生成器，持续产出识别结果

        流式识别的中间结果 is_partial=True，最终结果 is_partial=False。

        Args:
            chunks: AudioChunk 生成器，每次 yield 一个音频块。

        Yields:
            TranscriptionResult: 中间或最终识别结果。

        Raises:
            RuntimeError: 模型未加载或识别过程出错。
        """
        ...

    @abstractmethod
    def unload(self) -> None:
        """
        卸载模型，释放所有资源（内存、GPU 显存等）。

        调用后状态变为 UNLOADED。可以再次调用 load_model() 重新加载。
        多次调用 unload() 应该是安全的（幂等操作）。
        """
        ...

    # --------------------------------------------------
    # 可选覆写方法
    # --------------------------------------------------

    def set_initial_prompt(self, prompt: str) -> None:
        """
        设置初始提示词（initial prompt），用于引导 ASR 识别。

        通过注入自定义词汇表、专有名词等，提高特定领域的识别准确率。
        例如注入 "VoiceInk, ASR, Whisper" 可提高这些词汇的识别率。

        Args:
            prompt: 提示词文本，建议不超过 224 个 token。
                    传入空字符串表示清除提示词。
        """
        self._initial_prompt = prompt
        logger.debug(f"ASR initial prompt 已更新 (长度={len(prompt)})")

    # --------------------------------------------------
    # 辅助方法
    # --------------------------------------------------

    def _ensure_ready(self) -> None:
        """
        检查模型是否就绪，未就绪则抛出异常。

        供子类在 transcribe / transcribe_stream 开头调用。
        """
        if self._state != ASRState.READY:
            raise RuntimeError(
                f"ASR 后端未就绪，当前状态: {self._state.value}。"
                f"请先调用 load_model() 加载模型。"
            )

    def _validate_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        验证并规范化音频数据格式。

        Args:
            audio: 输入音频数据。

        Returns:
            规范化后的 float32 音频数据。

        Raises:
            ValueError: 音频数据无效（空数组、维度错误等）。
        """
        if audio is None or len(audio) == 0:
            raise ValueError("音频数据为空")

        # 确保是一维数组
        if audio.ndim > 1:
            audio = audio.flatten()

        # 确保是 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        # 确保值域在 [-1.0, 1.0] 范围内（int16 转换）
        if audio.max() > 1.0 or audio.min() < -1.0:
            max_abs = max(abs(audio.max()), abs(audio.min()))
            if max_abs > 0:
                audio = audio / max_abs
                logger.debug("音频数据已自动归一化到 [-1.0, 1.0]")

        return audio

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} state={self._state.value}>"
