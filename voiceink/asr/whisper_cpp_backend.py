"""
VoiceInk ASR - Whisper.cpp 后端

基于 pywhispercpp 库实现的 ASR 后端，使用 whisper.cpp 进行推理。
特点：
    - 纯 CPU 推理，无需 GPU
    - 支持 GGML 量化模型（体积小、速度快）
    - 支持流式识别（基于滑动窗口）
    - 支持 initial_prompt 注入自定义词汇
"""

import time
import numpy as np
import logging
from pathlib import Path
from typing import Generator, Optional

from voiceink.config import ASRConfig
from voiceink.asr.base import (
    ASRBackend, ASRState, AudioChunk, TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Whisper 模型的最大上下文窗口：30 秒（480000 个采样点 @ 16kHz）
WHISPER_MAX_AUDIO_SECONDS = 30
WHISPER_MAX_SAMPLES = WHISPER_MAX_AUDIO_SECONDS * 16000

# 模型文件名映射：model_size -> 对应的 GGML 模型文件名前缀
MODEL_SIZE_MAP = {
    "tiny": "tiny",
    "tiny.en": "tiny.en",
    "base": "base",
    "base.en": "base.en",
    "small": "small",
    "small.en": "small.en",
    "medium": "medium",
    "medium.en": "medium.en",
    "large": "large-v3",
    "large-v1": "large-v1",
    "large-v2": "large-v2",
    "large-v3": "large-v3",
    "large-v3-turbo": "large-v3-turbo",
}


class WhisperCppBackend(ASRBackend):
    """
    基于 whisper.cpp (pywhispercpp) 的 ASR 后端实现。

    使用 pywhispercpp 库加载 GGML 格式的 Whisper 模型，
    在 CPU 上进行高效的语音识别推理。

    配置示例 (config.yaml):
        asr:
            backend: whisper_cpp
            model_size: base          # tiny/base/small/medium/large
            model_path: ""            # 自定义模型路径（可选）
            language: auto            # auto/zh/en/ja 等
            n_threads: 4              # CPU 线程数
            stream_interval_seconds: 2.0  # 流式识别间隔
            beam_size: 5              # beam search 宽度
    """

    def __init__(self, config: ASRConfig):
        """
        初始化 Whisper.cpp 后端。

        Args:
            config: ASR 配置，包含模型大小、路径、语言等参数。
        """
        super().__init__()
        self._config = config
        self._model = None  # pywhispercpp.Model 实例

        # 解析语言参数：auto 表示自动检测，传 None 给 whisper.cpp
        self._language = None if config.language == "auto" else config.language

        logger.info(
            f"WhisperCppBackend 已创建: model_size={config.model_size}, "
            f"language={config.language}, n_threads={config.n_threads}"
        )

    # ==================================================================
    # 模型生命周期
    # ==================================================================

    def load_model(self) -> None:
        """
        加载 Whisper GGML 模型。

        pywhispercpp 支持两种模型加载方式：
        1. 传入模型名称（如 "base"），自动从 Hugging Face 下载
        2. 传入本地 .bin 文件的完整路径

        如果 config.model_path 非空，优先使用本地路径。

        Raises:
            FileNotFoundError: 指定的本地模型文件不存在。
            ImportError: pywhispercpp 库未安装。
            RuntimeError: 模型加载失败。
        """
        if self._state == ASRState.READY and self._model is not None:
            logger.warning("模型已加载，跳过重复加载")
            return

        self._state = ASRState.LOADING
        logger.info("正在加载 Whisper.cpp 模型...")

        try:
            # 延迟导入，避免未安装 pywhispercpp 时影响其他模块
            from pywhispercpp.model import Model as WhisperModel
        except ImportError as e:
            self._state = ASRState.ERROR
            raise ImportError(
                "pywhispercpp 未安装。请运行: pip install pywhispercpp\n"
                "详见: https://github.com/absadiki/pywhispercpp"
            ) from e

        try:
            # 确定模型路径或名称
            model_identifier = self._resolve_model_identifier()
            logger.info(f"模型标识: {model_identifier}")

            # 构建模型初始化参数
            init_kwargs = {
                "n_threads": self._config.n_threads,
                "print_realtime": False,
                "print_progress": False,
            }

            # 设置语言（仅在非 auto 模式下）
            if self._language is not None:
                init_kwargs["language"] = self._language

            # 创建模型实例
            start_time = time.time()
            self._model = WhisperModel(model_identifier, **init_kwargs)
            elapsed = time.time() - start_time

            self._state = ASRState.READY
            logger.info(f"Whisper.cpp 模型加载完成，耗时 {elapsed:.2f}s")

        except FileNotFoundError:
            self._state = ASRState.ERROR
            raise
        except Exception as e:
            self._state = ASRState.ERROR
            raise RuntimeError(f"Whisper.cpp 模型加载失败: {e}") from e

    def unload(self) -> None:
        """
        卸载模型并释放内存。

        pywhispercpp 的 Model 对象在垃圾回收时会自动释放 C++ 资源，
        这里显式置为 None 加速释放。
        """
        if self._model is not None:
            logger.info("正在卸载 Whisper.cpp 模型...")
            del self._model
            self._model = None

        self._state = ASRState.UNLOADED
        logger.info("Whisper.cpp 模型已卸载")

    # ==================================================================
    # 转录方法
    # ==================================================================

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        对完整音频进行一次性识别。

        Args:
            audio: 完整音频数据，float32, 16kHz 单声道。

        Returns:
            TranscriptionResult: 包含识别文本、语言、置信度等信息。

        Raises:
            RuntimeError: 模型未加载。
            ValueError: 音频数据无效。
        """
        self._ensure_ready()
        audio = self._validate_audio(audio)

        duration_ms = len(audio) / 16.0  # 16 samples/ms
        logger.debug(f"开始识别，音频时长: {duration_ms:.0f}ms ({len(audio)} samples)")

        self._state = ASRState.TRANSCRIBING
        try:
            # 如果音频超过 30 秒，whisper.cpp 内部会自动分段处理
            # 构建转录参数
            transcribe_kwargs = {}

            # 注入 initial_prompt（如果有）
            if self._initial_prompt:
                transcribe_kwargs["initial_prompt"] = self._initial_prompt

            # 指定语言（避免 auto 模式下短语音误判语言）
            if self._language:
                transcribe_kwargs["language"] = self._language

            # 调用 pywhispercpp 进行转录
            segments = self._model.transcribe(audio, **transcribe_kwargs)

            # 解析结果
            result = self._parse_segments(segments, duration_ms)

            self._state = ASRState.READY
            logger.debug(f"识别完成: '{result.text[:50]}...' " if len(result.text) > 50
                         else f"识别完成: '{result.text}'")
            return result

        except Exception as e:
            self._state = ASRState.READY  # 恢复就绪状态，允许重试
            logger.error(f"Whisper.cpp 识别出错: {e}")
            raise RuntimeError(f"Whisper.cpp 转录失败: {e}") from e

    def transcribe_stream(
        self, chunks: Generator[AudioChunk, None, None]
    ) -> Generator[TranscriptionResult, None, None]:
        """
        流式识别：基于滑动窗口的增量转录。

        实现原理：
        1. 持续接收 AudioChunk，将音频数据追加到缓冲区
        2. 每隔 stream_interval_seconds 秒对缓冲区内容进行一次转录
        3. 使用滑动窗口确保缓冲区不超过 Whisper 的 30 秒上限
        4. 输入结束后对剩余缓冲区做最终转录

        Args:
            chunks: AudioChunk 生成器。

        Yields:
            TranscriptionResult: 中间结果(is_partial=True)和最终结果(is_partial=False)。
        """
        self._ensure_ready()

        # 音频缓冲区：存储所有接收到的音频数据
        audio_buffer = np.array([], dtype=np.float32)
        # 流式识别的时间间隔（转换为采样点数）
        interval_samples = int(self._config.stream_interval_seconds * 16000)
        # 上次转录以来累积的新采样点数
        samples_since_last_transcribe = 0

        self._state = ASRState.TRANSCRIBING

        try:
            for chunk in chunks:
                # 将新音频块追加到缓冲区
                audio_buffer = np.concatenate([audio_buffer, chunk.data])
                samples_since_last_transcribe += len(chunk.data)

                # 滑动窗口：如果缓冲区超过 30 秒，裁剪掉最早的部分
                if len(audio_buffer) > WHISPER_MAX_SAMPLES:
                    overflow = len(audio_buffer) - WHISPER_MAX_SAMPLES
                    audio_buffer = audio_buffer[overflow:]
                    logger.debug(
                        f"滑动窗口裁剪: 丢弃 {overflow} 采样点 "
                        f"({overflow / 16:.0f}ms)"
                    )

                # 判断是否达到转录间隔
                if samples_since_last_transcribe >= interval_samples:
                    samples_since_last_transcribe = 0

                    # 对当前缓冲区进行转录（中间结果）
                    if len(audio_buffer) > 0:
                        result = self._transcribe_buffer(audio_buffer, is_partial=True)
                        if not result.is_empty:
                            yield result

            # 输入结束：对剩余缓冲区进行最终转录
            if len(audio_buffer) > 0:
                final_result = self._transcribe_buffer(audio_buffer, is_partial=False)
                yield final_result

        except Exception as e:
            logger.error(f"流式识别出错: {e}")
            # 即使出错也返回一个空结果
            yield TranscriptionResult(text="", is_partial=False)
        finally:
            self._state = ASRState.READY

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    def _resolve_model_identifier(self) -> str:
        """
        解析模型标识：优先使用自定义路径，否则使用模型名称（自动下载）。

        Returns:
            模型文件路径或模型名称字符串。

        Raises:
            FileNotFoundError: 指定的模型路径不存在。
        """
        # 情况 1：用户指定了具体的模型文件路径
        if self._config.model_path:
            model_path = Path(self._config.model_path)

            # 如果 model_path 是目录（而非文件），尝试在其中查找 ggml-*.bin
            if model_path.is_dir():
                bin_files = sorted(model_path.glob("ggml-*.bin"))
                if bin_files:
                    model_path = bin_files[0]
                    logger.info(
                        "model_path 为目录，自动选择模型文件: %s", model_path
                    )
                else:
                    # 目录中没有模型文件，回退到情况 2（按名称下载）
                    logger.warning(
                        "model_path 为目录但未找到 ggml-*.bin: %s，"
                        "将使用 model_size 名称加载",
                        model_path,
                    )
                    model_path = None

            if model_path is not None:
                if not model_path.exists():
                    raise FileNotFoundError(
                        f"Whisper 模型文件不存在: {model_path}\n"
                        f"请检查配置 asr.model_path 的值是否正确。"
                    )
                return str(model_path)

        # 情况 2：使用模型名称，pywhispercpp 会自动下载到缓存目录
        model_size = self._config.model_size
        if model_size in MODEL_SIZE_MAP:
            model_name = MODEL_SIZE_MAP[model_size]
        else:
            # 直接作为模型名使用（可能是用户自定义的名称）
            model_name = model_size
            logger.warning(
                f"未知的模型大小 '{model_size}'，将尝试直接使用。"
                f"已知模型大小: {list(MODEL_SIZE_MAP.keys())}"
            )

        return model_name

    def _transcribe_buffer(
        self, audio: np.ndarray, is_partial: bool
    ) -> TranscriptionResult:
        """
        对音频缓冲区执行一次转录。

        Args:
            audio: 音频缓冲区数据。
            is_partial: 是否为中间结果。

        Returns:
            TranscriptionResult: 识别结果。
        """
        duration_ms = len(audio) / 16.0

        try:
            transcribe_kwargs = {}
            if self._initial_prompt:
                transcribe_kwargs["initial_prompt"] = self._initial_prompt
            if self._language:
                transcribe_kwargs["language"] = self._language

            segments = self._model.transcribe(audio, **transcribe_kwargs)
            result = self._parse_segments(segments, duration_ms)
            result.is_partial = is_partial
            return result

        except Exception as e:
            logger.warning(f"缓冲区转录失败 (is_partial={is_partial}): {e}")
            return TranscriptionResult(
                text="", is_partial=is_partial, duration_ms=duration_ms
            )

    def _parse_segments(
        self, segments, duration_ms: float
    ) -> TranscriptionResult:
        """
        将 pywhispercpp 返回的 Segment 列表解析为 TranscriptionResult。

        pywhispercpp 的 Segment 对象有以下属性:
            - t0: 起始时间（10ms 为单位的整数）
            - t1: 结束时间（10ms 为单位的整数）
            - text: 识别文本

        Args:
            segments: pywhispercpp 返回的 Segment 对象列表。
            duration_ms: 输入音频的总时长。

        Returns:
            TranscriptionResult: 解析后的识别结果。
        """
        text_parts = []
        parsed_segments = []

        for seg in segments:
            # t0 和 t1 是 10ms 为单位的整数，转换为秒
            start_sec = seg.t0 / 100.0
            end_sec = seg.t1 / 100.0
            text = seg.text.strip()

            if text:
                text_parts.append(text)
                parsed_segments.append((start_sec, end_sec, text))

        full_text = " ".join(text_parts)

        return TranscriptionResult(
            text=full_text,
            language=self._language or "auto",
            confidence=-1.0,  # pywhispercpp 默认不返回置信度
            duration_ms=duration_ms,
            is_partial=False,
            segments=parsed_segments,
        )

    def __repr__(self) -> str:
        return (
            f"<WhisperCppBackend model_size={self._config.model_size} "
            f"language={self._config.language} state={self._state.value}>"
        )
