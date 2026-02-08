"""
VoiceInk ASR - faster-whisper 后端

基于 faster-whisper (CTranslate2) 库实现的 ASR 后端。
特点：
    - 基于 CTranslate2 推理引擎，比原版 Whisper 快 4 倍
    - 支持 int8 量化，大幅降低 CPU 内存占用
    - 内置 VAD（基于 Silero VAD），自动跳过静音段
    - 支持 GPU (CUDA) 和 CPU 推理
    - 支持 initial_prompt 引导识别
"""

import time
import numpy as np
import logging
from pathlib import Path
from typing import Generator, Optional, List

from voiceink.config import ASRConfig
from voiceink.asr.base import (
    ASRBackend, ASRState, AudioChunk, TranscriptionResult,
)

logger = logging.getLogger(__name__)

# Whisper 模型的最大上下文窗口：30 秒
WHISPER_MAX_AUDIO_SECONDS = 30
WHISPER_MAX_SAMPLES = WHISPER_MAX_AUDIO_SECONDS * 16000

# faster-whisper 支持的模型大小列表
SUPPORTED_MODEL_SIZES = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large-v1", "large-v2", "large-v3",
    "turbo",
    "distil-small.en", "distil-medium.en",
    "distil-large-v2", "distil-large-v3",
]


class FasterWhisperBackend(ASRBackend):
    """
    基于 faster-whisper (CTranslate2) 的 ASR 后端实现。

    faster-whisper 使用 CTranslate2 作为推理引擎，对 Whisper 模型进行了
    高度优化。相比原版 PyTorch 实现，在同等精度下速度提升约 4 倍，
    并且支持 int8 量化以降低内存占用。

    默认使用 CPU + int8 量化进行推理，适合没有 GPU 的环境。
    内置 Silero VAD，可自动过滤静音段，提高识别效率。

    配置示例 (config.yaml):
        asr:
            backend: faster_whisper
            model_size: base          # tiny/base/small/medium/large-v3/turbo
            model_path: ""            # 自定义模型路径（可选）
            language: auto            # auto/zh/en/ja 等
            n_threads: 4              # CPU 线程数
            stream_interval_seconds: 2.0  # 流式识别间隔
            beam_size: 5              # beam search 宽度
    """

    def __init__(self, config: ASRConfig):
        """
        初始化 faster-whisper 后端。

        Args:
            config: ASR 配置。
        """
        super().__init__()
        self._config = config
        self._model = None  # faster_whisper.WhisperModel 实例

        # 解析语言参数
        self._language = None if config.language == "auto" else config.language

        # 推理设备和计算精度（默认 CPU + int8 量化）
        self._device = "cpu"
        self._compute_type = "int8"

        logger.info(
            f"FasterWhisperBackend 已创建: model_size={config.model_size}, "
            f"language={config.language}, device={self._device}, "
            f"compute_type={self._compute_type}, n_threads={config.n_threads}"
        )

    # ==================================================================
    # 模型生命周期
    # ==================================================================

    def load_model(self) -> None:
        """
        加载 faster-whisper 模型。

        faster-whisper 支持两种加载方式：
        1. 传入模型名称（如 "base"），自动从 Hugging Face 下载 CTranslate2 格式模型
        2. 传入本地目录路径（已转换为 CTranslate2 格式的模型目录）

        Raises:
            FileNotFoundError: 指定的本地模型路径不存在。
            ImportError: faster-whisper 库未安装。
            RuntimeError: 模型加载失败。
        """
        if self._state == ASRState.READY and self._model is not None:
            logger.warning("模型已加载，跳过重复加载")
            return

        self._state = ASRState.LOADING
        logger.info("正在加载 faster-whisper 模型...")

        try:
            # 延迟导入
            from faster_whisper import WhisperModel
        except ImportError as e:
            self._state = ASRState.ERROR
            raise ImportError(
                "faster-whisper 未安装。请运行: pip install faster-whisper\n"
                "详见: https://github.com/SYSTRAN/faster-whisper"
            ) from e

        try:
            # 确定模型标识
            model_identifier = self._resolve_model_identifier()
            logger.info(
                f"模型标识: {model_identifier}, "
                f"设备: {self._device}, 精度: {self._compute_type}"
            )

            # 创建模型实例
            start_time = time.time()
            self._model = WhisperModel(
                model_identifier,
                device=self._device,
                compute_type=self._compute_type,
                cpu_threads=self._config.n_threads,
            )
            elapsed = time.time() - start_time

            self._state = ASRState.READY
            logger.info(f"faster-whisper 模型加载完成，耗时 {elapsed:.2f}s")

        except FileNotFoundError:
            self._state = ASRState.ERROR
            raise
        except Exception as e:
            self._state = ASRState.ERROR
            raise RuntimeError(f"faster-whisper 模型加载失败: {e}") from e

    def unload(self) -> None:
        """
        卸载模型并释放资源。

        CTranslate2 在 Python 对象销毁时释放内存，
        这里显式置 None 并触发垃圾回收。
        """
        if self._model is not None:
            logger.info("正在卸载 faster-whisper 模型...")
            del self._model
            self._model = None

            # 尝试释放 GPU 缓存（如果使用了 CUDA）
            if self._device == "cuda":
                try:
                    import torch
                    torch.cuda.empty_cache()
                except ImportError:
                    pass

        self._state = ASRState.UNLOADED
        logger.info("faster-whisper 模型已卸载")

    # ==================================================================
    # 转录方法
    # ==================================================================

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        对完整音频进行一次性识别（带 VAD）。

        faster-whisper 的 VAD 过滤会自动跳过静音段，
        只对包含语音的部分进行转录，大幅提升效率。

        Args:
            audio: 完整音频数据，float32, 16kHz 单声道。

        Returns:
            TranscriptionResult: 识别结果。

        Raises:
            RuntimeError: 模型未加载。
            ValueError: 音频数据无效。
        """
        self._ensure_ready()
        audio = self._validate_audio(audio)

        duration_ms = len(audio) / 16.0
        logger.debug(f"开始识别，音频时长: {duration_ms:.0f}ms ({len(audio)} samples)")

        self._state = ASRState.TRANSCRIBING
        try:
            # 构建转录参数
            transcribe_kwargs = self._build_transcribe_kwargs()

            # 调用 faster-whisper 进行转录
            # 注意：segments 是一个生成器，需要消费才会实际执行转录
            segments_gen, info = self._model.transcribe(audio, **transcribe_kwargs)

            # 消费生成器，收集所有段落
            segments = list(segments_gen)

            # 解析结果
            result = self._parse_segments(segments, info, duration_ms)

            self._state = ASRState.READY
            logger.debug(
                f"识别完成: 语言={result.language}, "
                + (f"'{result.text[:50]}...'" if len(result.text) > 50
                   else f"'{result.text}'")
            )
            return result

        except Exception as e:
            self._state = ASRState.READY
            logger.error(f"faster-whisper 识别出错: {e}")
            raise RuntimeError(f"faster-whisper 转录失败: {e}") from e

    def transcribe_stream(
        self, chunks: Generator[AudioChunk, None, None]
    ) -> Generator[TranscriptionResult, None, None]:
        """
        流式识别：基于滑动窗口的增量转录。

        实现原理与 WhisperCppBackend 类似：
        1. 持续接收 AudioChunk，追加到音频缓冲区
        2. 每隔 stream_interval_seconds 秒对缓冲区进行一次转录
        3. 利用 faster-whisper 的内置 VAD 自动跳过静音段
        4. 使用滑动窗口保持缓冲区在 30 秒以内

        Args:
            chunks: AudioChunk 生成器。

        Yields:
            TranscriptionResult: 中间结果和最终结果。
        """
        self._ensure_ready()

        # 音频缓冲区
        audio_buffer = np.array([], dtype=np.float32)
        # 转录间隔（采样点数）
        interval_samples = int(self._config.stream_interval_seconds * 16000)
        # 自上次转录以来的新采样点数
        samples_since_last = 0

        self._state = ASRState.TRANSCRIBING

        try:
            for chunk in chunks:
                # 追加音频数据
                audio_buffer = np.concatenate([audio_buffer, chunk.data])
                samples_since_last += len(chunk.data)

                # 滑动窗口裁剪
                if len(audio_buffer) > WHISPER_MAX_SAMPLES:
                    overflow = len(audio_buffer) - WHISPER_MAX_SAMPLES
                    audio_buffer = audio_buffer[overflow:]
                    logger.debug(
                        f"滑动窗口裁剪: 丢弃 {overflow} 采样点 "
                        f"({overflow / 16:.0f}ms)"
                    )

                # 达到转录间隔 -> 执行中间转录
                if samples_since_last >= interval_samples:
                    samples_since_last = 0

                    if len(audio_buffer) > 0:
                        result = self._transcribe_buffer(
                            audio_buffer, is_partial=True
                        )
                        if not result.is_empty:
                            yield result

            # 输入结束 -> 最终转录
            if len(audio_buffer) > 0:
                final_result = self._transcribe_buffer(
                    audio_buffer, is_partial=False
                )
                yield final_result

        except Exception as e:
            logger.error(f"流式识别出错: {e}")
            yield TranscriptionResult(text="", is_partial=False)
        finally:
            self._state = ASRState.READY

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    def _resolve_model_identifier(self) -> str:
        """
        解析模型标识：优先使用自定义路径，否则使用模型名称。

        faster-whisper 使用的是 CTranslate2 格式的模型，
        可以从 Hugging Face Hub 自动下载（格式为 "base", "small" 等），
        也可以指定本地已转换的模型目录。

        Returns:
            模型目录路径或模型名称。

        Raises:
            FileNotFoundError: 指定路径不存在。
        """
        # 情况 1：用户指定了自定义模型路径
        if self._config.model_path:
            model_path = Path(self._config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"faster-whisper 模型路径不存在: {model_path}\n"
                    f"请检查配置 asr.model_path 的值。\n"
                    f"如果是 CTranslate2 格式，需要指向包含 model.bin 的目录。"
                )
            return str(model_path)

        # 情况 2：使用模型名称，自动从 HuggingFace 下载
        model_size = self._config.model_size

        # 检查是否是已知的模型大小
        if model_size not in SUPPORTED_MODEL_SIZES:
            logger.warning(
                f"未知的模型大小 '{model_size}'，将尝试直接使用。"
                f"已知模型大小: {SUPPORTED_MODEL_SIZES}"
            )

        return model_size

    def _build_transcribe_kwargs(self) -> dict:
        """
        构建 faster-whisper transcribe() 方法的参数字典。

        Returns:
            参数字典。
        """
        kwargs = {
            "beam_size": self._config.beam_size,
            # 启用 VAD 过滤，自动跳过静音段
            "vad_filter": True,
            "vad_parameters": {
                "threshold": 0.5,              # VAD 语音检测阈值
                "min_speech_duration_ms": 250,  # 最短语音段时长
                "min_silence_duration_ms": 1000,  # 最短静音段时长（用于分段）
            },
            # 基于前文条件生成，提高连续语句的连贯性
            "condition_on_previous_text": True,
            # 不输出无语音段的标记
            "without_timestamps": False,
        }

        # 设置语言
        if self._language is not None:
            kwargs["language"] = self._language

        # 注入 initial_prompt
        if self._initial_prompt:
            kwargs["initial_prompt"] = self._initial_prompt

        return kwargs

    def _transcribe_buffer(
        self, audio: np.ndarray, is_partial: bool
    ) -> TranscriptionResult:
        """
        对音频缓冲区执行一次转录。

        Args:
            audio: 音频缓冲区。
            is_partial: 是否为中间结果。

        Returns:
            TranscriptionResult: 识别结果。
        """
        duration_ms = len(audio) / 16.0

        try:
            transcribe_kwargs = self._build_transcribe_kwargs()
            segments_gen, info = self._model.transcribe(audio, **transcribe_kwargs)
            segments = list(segments_gen)

            result = self._parse_segments(segments, info, duration_ms)
            result.is_partial = is_partial
            return result

        except Exception as e:
            logger.warning(f"缓冲区转录失败 (is_partial={is_partial}): {e}")
            return TranscriptionResult(
                text="", is_partial=is_partial, duration_ms=duration_ms
            )

    def _parse_segments(
        self, segments, info, duration_ms: float
    ) -> TranscriptionResult:
        """
        将 faster-whisper 的返回结果解析为 TranscriptionResult。

        faster-whisper 的 Segment 对象有以下属性:
            - start: 起始时间（秒，float）
            - end: 结束时间（秒，float）
            - text: 识别文本
            - avg_logprob: 平均对数概率（越接近 0 越自信）
            - no_speech_prob: 非语音概率

        info 对象有以下属性:
            - language: 检测到的语言代码
            - language_probability: 语言检测置信度

        Args:
            segments: faster-whisper 返回的 Segment 列表。
            info: 转录信息对象。
            duration_ms: 输入音频时长。

        Returns:
            TranscriptionResult: 解析后的结果。
        """
        text_parts = []
        parsed_segments = []
        total_logprob = 0.0
        segment_count = 0

        for seg in segments:
            text = seg.text.strip()
            if text:
                text_parts.append(text)
                parsed_segments.append((seg.start, seg.end, text))
                total_logprob += seg.avg_logprob
                segment_count += 1

        full_text = " ".join(text_parts)

        # 计算平均置信度：将 avg_logprob 转换为 [0, 1] 范围的置信度
        # avg_logprob 通常在 [-1, 0] 范围内，越接近 0 表示越自信
        confidence = -1.0
        if segment_count > 0:
            avg_logprob = total_logprob / segment_count
            # 使用 exp 将 logprob 转为概率，作为近似置信度
            confidence = min(1.0, max(0.0, np.exp(avg_logprob)))

        # 获取检测到的语言
        detected_language = self._language or ""
        if info and hasattr(info, "language") and info.language:
            detected_language = info.language

        return TranscriptionResult(
            text=full_text,
            language=detected_language,
            confidence=confidence,
            duration_ms=duration_ms,
            is_partial=False,
            segments=parsed_segments,
        )

    def __repr__(self) -> str:
        return (
            f"<FasterWhisperBackend model_size={self._config.model_size} "
            f"language={self._config.language} device={self._device} "
            f"compute_type={self._compute_type} state={self._state.value}>"
        )
