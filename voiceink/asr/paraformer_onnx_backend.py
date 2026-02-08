"""
VoiceInk ASR - Paraformer ONNX 后端

基于 funasr-onnx 库实现的 ASR 后端，使用阿里达摩院 Paraformer-zh 模型。
特点：
    - 非自回归架构，推理速度快
    - 中文识别精度远优于 Whisper-base
    - 基于 ONNX Runtime 推理，不依赖 PyTorch
    - 支持量化模型（quantize=True），降低内存和延迟
    - CPU 推理即可达到实时速度
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

# Paraformer 支持的最大音频时长约 20 秒，超过则需分段
# 但实际 funasr-onnx 内部会自动处理长音频，此处保守设 60 秒
PARAFORMER_MAX_AUDIO_SECONDS = 60
PARAFORMER_MAX_SAMPLES = PARAFORMER_MAX_AUDIO_SECONDS * 16000

# ModelScope 模型标识符映射：model_size -> 对应的 ModelScope 模型 ID
# Use the pre-exported ONNX model (has model_quant.onnx ready to use, ~238MB)
# instead of the PyTorch model (model.pt, ~840MB, requires funasr to export).
_MODEL_MAP = {
    "large": "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-onnx",
}

# Paraformer 语言到模型的映射提示
# Paraformer-zh 是专用中文模型，不支持多语言自动检测
_LANGUAGE_MAP = {
    "auto": "zh",   # Paraformer-zh 默认中文
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
}


class ParaformerOnnxBackend(ASRBackend):
    """
    基于 funasr-onnx (Paraformer) 的 ASR 后端实现。

    Paraformer 是阿里达摩院开发的非自回归端到端语音识别模型，
    在中文语音识别场景下精度显著优于 Whisper-base。

    通过 funasr-onnx 包在 ONNX Runtime 上运行，不需要 PyTorch，
    依赖轻量（仅 onnxruntime + soundfile 等）。

    配置示例 (config.yaml):
        asr:
            backend: paraformer_onnx
            model_size: large              # 目前仅支持 large
            model_path: ""                 # 自定义模型目录路径（可选）
            language: auto                 # auto/zh（Paraformer-zh 专用中文）
            n_threads: 4                   # CPU 线程数
            stream_interval_seconds: 2.0   # 流式识别间隔
            beam_size: 5                   # 未使用（Paraformer 非自回归无 beam search）
    """

    def __init__(self, config: ASRConfig):
        """
        初始化 Paraformer ONNX 后端。

        Args:
            config: ASR 配置，包含模型大小、路径、语言等参数。
        """
        super().__init__()
        self._config = config
        self._model = None  # funasr_onnx.Paraformer 实例

        # 解析模型标识符
        self._model_id = self._resolve_model_identifier()

        # Paraformer-zh 是中文专用模型
        self._language = _LANGUAGE_MAP.get(config.language, config.language)

        logger.info(
            f"ParaformerOnnxBackend created: model_size={config.model_size}, "
            f"model_id={self._model_id}, "
            f"language={self._language}, n_threads={config.n_threads}"
        )

    # ==================================================================
    # Model Lifecycle
    # ==================================================================

    def load_model(self) -> None:
        """
        Load the Paraformer ONNX model.

        funasr-onnx automatically downloads the model from ModelScope
        on first use if model_dir is a ModelScope model ID.
        If model_path is set, it will use the local directory instead.

        Raises:
            ImportError: funasr-onnx is not installed.
            RuntimeError: Model loading failed.
        """
        if self._state == ASRState.READY and self._model is not None:
            logger.warning("Model already loaded, skipping reload")
            return

        self._state = ASRState.LOADING
        logger.info("Loading Paraformer ONNX model...")

        try:
            # funasr_onnx.__init__ imports sensevoice_bin which requires torch.
            # We only need Paraformer (from paraformer_bin) which is ONNX-only.
            # Pre-inject a stub for sensevoice_bin so the import doesn't fail.
            import sys as _sys
            import types as _types
            _sv_key = "funasr_onnx.sensevoice_bin"
            _had_sv = _sv_key in _sys.modules
            if not _had_sv:
                _stub = _types.ModuleType(_sv_key)
                _stub.SenseVoiceSmall = None  # dummy attribute
                _sys.modules[_sv_key] = _stub

            try:
                from funasr_onnx import Paraformer
            finally:
                # Clean up stub only if we injected it
                if not _had_sv and _sv_key in _sys.modules:
                    del _sys.modules[_sv_key]
        except ImportError as e:
            self._state = ASRState.ERROR
            raise ImportError(
                "funasr-onnx is not installed. Please run: pip install funasr-onnx\n"
                "See: https://github.com/modelscope/FunASR"
            ) from e

        try:
            start_time = time.time()

            self._model = Paraformer(
                model_dir=self._model_id,
                batch_size=1,
                quantize=True,       # Use quantized model for faster CPU inference
                device_id=-1,        # CPU
                intra_op_num_threads=self._config.n_threads,
            )

            elapsed = time.time() - start_time
            self._state = ASRState.READY
            logger.info(
                f"Paraformer ONNX model loaded in {elapsed:.2f}s "
                f"(model={self._model_id}, threads={self._config.n_threads})"
            )

        except TypeError as e:
            # funasr_onnx has a bug: `raise "string"` instead of raise Exception(...)
            # which causes TypeError("exceptions must derive from BaseException").
            # Translate to a helpful message pointing to the real cause.
            self._state = ASRState.ERROR
            raise RuntimeError(
                f"Paraformer ONNX model loading failed. "
                f"This is likely caused by a missing dependency (modelscope). "
                f"Please run: pip install modelscope\n"
                f"Original error: {e}"
            ) from e
        except Exception as e:
            self._state = ASRState.ERROR
            raise RuntimeError(f"Paraformer ONNX model loading failed: {e}") from e

    def unload(self) -> None:
        """
        Unload the model and release resources.

        ONNX Runtime session is released when the Paraformer object is
        garbage collected. We explicitly set it to None to accelerate release.
        """
        if self._model is not None:
            logger.info("Unloading Paraformer ONNX model...")
            del self._model
            self._model = None

        self._state = ASRState.UNLOADED
        logger.info("Paraformer ONNX model unloaded")

    # ==================================================================
    # Transcription
    # ==================================================================

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        Transcribe a complete audio segment.

        Args:
            audio: Complete audio data, float32, 16kHz mono.

        Returns:
            TranscriptionResult with recognized text.

        Raises:
            RuntimeError: Model not loaded.
            ValueError: Invalid audio data.
        """
        self._ensure_ready()
        audio = self._validate_audio(audio)

        duration_ms = len(audio) / 16.0  # 16 samples/ms
        logger.debug(
            f"Paraformer transcribe: {len(audio)} samples "
            f"({duration_ms:.0f}ms)"
        )

        self._state = ASRState.TRANSCRIBING
        try:
            # funasr-onnx accepts numpy arrays directly
            # Returns List[dict] with key "preds" containing the text
            results = self._model(audio)

            # Parse result
            text = self._extract_text(results)

            self._state = ASRState.READY
            logger.debug(
                f"Paraformer result: '{text[:80]}'"
                if len(text) > 80 else f"Paraformer result: '{text}'"
            )

            return TranscriptionResult(
                text=text,
                language=self._language,
                confidence=-1.0,  # Paraformer does not expose confidence scores
                duration_ms=duration_ms,
                is_partial=False,
                segments=[],
            )

        except Exception as e:
            self._state = ASRState.READY  # Recover to READY for retry
            logger.error(f"Paraformer transcription error: {e}")
            raise RuntimeError(f"Paraformer transcription failed: {e}") from e

    def transcribe_stream(
        self, chunks: Generator[AudioChunk, None, None]
    ) -> Generator[TranscriptionResult, None, None]:
        """
        Streaming transcription using periodic batch re-transcription.

        Uses the same sliding-window approach as the Whisper backends:
        accumulate audio chunks, periodically transcribe the full buffer,
        and yield partial results.

        A future improvement could use Paraformer-online for true streaming.

        Args:
            chunks: AudioChunk generator.

        Yields:
            TranscriptionResult: partial and final results.
        """
        self._ensure_ready()

        audio_buffer = np.array([], dtype=np.float32)
        interval_samples = int(self._config.stream_interval_seconds * 16000)
        samples_since_last = 0

        self._state = ASRState.TRANSCRIBING

        try:
            for chunk in chunks:
                audio_buffer = np.concatenate([audio_buffer, chunk.data])
                samples_since_last += len(chunk.data)

                # Sliding window: trim if exceeding max
                if len(audio_buffer) > PARAFORMER_MAX_SAMPLES:
                    overflow = len(audio_buffer) - PARAFORMER_MAX_SAMPLES
                    audio_buffer = audio_buffer[overflow:]
                    logger.debug(
                        f"Sliding window trim: dropped {overflow} samples "
                        f"({overflow / 16:.0f}ms)"
                    )

                # Periodic partial transcription
                if samples_since_last >= interval_samples:
                    samples_since_last = 0

                    if len(audio_buffer) > 0:
                        result = self._transcribe_buffer(
                            audio_buffer, is_partial=True
                        )
                        if not result.is_empty:
                            yield result

            # Final transcription of remaining buffer
            if len(audio_buffer) > 0:
                final_result = self._transcribe_buffer(
                    audio_buffer, is_partial=False
                )
                yield final_result

        except Exception as e:
            logger.error(f"Streaming transcription error: {e}")
            yield TranscriptionResult(text="", is_partial=False)
        finally:
            self._state = ASRState.READY

    # ==================================================================
    # Optional Override
    # ==================================================================

    def set_initial_prompt(self, prompt: str) -> None:
        """
        Set initial prompt (hot words).

        Note: The base Paraformer model has limited hot-word support.
        The prompt is stored but cannot be injected into inference the
        same way as Whisper's initial_prompt.
        ContextualParaformer would provide better hot-word support.

        Args:
            prompt: Hot-word text. Stored for potential future use.
        """
        self._initial_prompt = prompt
        logger.debug(
            f"Paraformer initial prompt updated (length={len(prompt)}). "
            "Note: basic Paraformer has limited hot-word support."
        )

    # ==================================================================
    # Internal Helpers
    # ==================================================================

    def _resolve_model_identifier(self) -> str:
        """
        Resolve model identifier: prefer custom path, else use model name.

        Returns:
            ModelScope model ID string or local directory path.

        Raises:
            FileNotFoundError: Specified model_path does not exist.
        """
        # Case 1: User specified a custom model directory
        if self._config.model_path:
            model_path = Path(self._config.model_path)
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Paraformer model path does not exist: {model_path}\n"
                    f"Please check the asr.model_path config value.\n"
                    f"The directory should contain model.onnx and related files."
                )
            return str(model_path)

        # Case 2: Use model_size to look up ModelScope ID
        model_size = self._config.model_size.lower().strip()
        if model_size in _MODEL_MAP:
            return _MODEL_MAP[model_size]

        # Fallback: treat model_size as a direct ModelScope model ID
        logger.warning(
            f"Unknown Paraformer model_size '{model_size}', "
            f"using as direct model identifier. "
            f"Known sizes: {list(_MODEL_MAP.keys())}"
        )
        return model_size

    def _transcribe_buffer(
        self, audio: np.ndarray, is_partial: bool
    ) -> TranscriptionResult:
        """
        Transcribe an audio buffer (used by streaming mode).

        Args:
            audio: Audio buffer data.
            is_partial: Whether this is a partial (intermediate) result.

        Returns:
            TranscriptionResult.
        """
        duration_ms = len(audio) / 16.0

        try:
            results = self._model(audio)
            text = self._extract_text(results)

            return TranscriptionResult(
                text=text,
                language=self._language,
                confidence=-1.0,
                duration_ms=duration_ms,
                is_partial=is_partial,
                segments=[],
            )

        except Exception as e:
            logger.warning(
                f"Buffer transcription failed (is_partial={is_partial}): {e}"
            )
            return TranscriptionResult(
                text="", is_partial=is_partial, duration_ms=duration_ms
            )

    @staticmethod
    def _extract_text(results) -> str:
        """
        Extract text from funasr-onnx inference results.

        funasr-onnx returns List[dict] where each dict has:
          - "preds": str or tuple  (the recognized text)
          - Optional "timestamp", "raw_tokens"

        The ONNX model variant may return tuples (text, token_ids) in "preds"
        instead of plain strings.

        Args:
            results: Raw inference output from Paraformer.__call__().

        Returns:
            Concatenated recognized text.
        """
        if not results:
            return ""

        text_parts = []
        for item in results:
            if isinstance(item, dict):
                pred = item.get("preds", "")
                # ONNX model may return (text, token_ids) tuple
                if isinstance(pred, (list, tuple)):
                    pred = pred[0] if pred else ""
                if isinstance(pred, str) and pred.strip():
                    text_parts.append(pred.strip())
            elif isinstance(item, str):
                # Some versions may return List[str] directly
                if item.strip():
                    text_parts.append(item.strip())
            elif isinstance(item, (list, tuple)):
                # Some versions return List[tuple]
                text = item[0] if item else ""
                if isinstance(text, str) and text.strip():
                    text_parts.append(text.strip())

        return "".join(text_parts)

    def __repr__(self) -> str:
        return (
            f"<ParaformerOnnxBackend model_size={self._config.model_size} "
            f"model_id={self._model_id} "
            f"language={self._language} state={self._state.value}>"
        )
