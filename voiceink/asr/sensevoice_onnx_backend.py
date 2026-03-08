"""
VoiceInk ASR - SenseVoice ONNX 后端

基于 sherpa-onnx 库实现的 ASR 后端，使用阿里达摩院 SenseVoice 模型。
特点：
    - 支持多语言：中文(zh)、英文(en)、日语(ja)、韩语(ko)、粤语(yue)、自动检测(auto)
    - 基于 ONNX Runtime 推理，使用预编译的 sherpa-onnx 包（无需安装 VC++）
    - 使用 int8 量化模型（~170MB），体积小、速度快
    - 支持逆文本标准化（ITN）：自动转换数字、日期等
    - CPU 推理即可达到实时速度
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

# SenseVoice 支持的最大音频时长约 30 秒
SENSEVOICE_MAX_AUDIO_SECONDS = 30
SENSEVOICE_MAX_SAMPLES = SENSEVOICE_MAX_AUDIO_SECONDS * 16000

# 语言代码映射（统一到 SenseVoice 支持的格式）
_LANGUAGE_MAP = {
    "auto": "auto",
    "zh": "zh",
    "cn": "zh",
    "chinese": "zh",
    "en": "en",
    "english": "en",
    "ja": "ja",
    "japanese": "ja",
    "ko": "ko",
    "korean": "ko",
    "yue": "yue",
    "cantonese": "yue",
}

# SenseVoice 在静音/噪声输入时可能产生的无意义前缀
# 这些通常是单个字符后跟标点
_NOISE_PREFIXES = [
    "我.", "我。", "我,", "我，",
    "그.", "그。",  # 韩语
    "我", "그",  # 单字符
    ".", "。", ",", "，",  # 单标点
]

# 模型信息
_MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
_MODEL_FILES = {
    "model": "model.int8.onnx",
    "tokens": "tokens.txt",
}


class SenseVoiceOnnxBackend(ASRBackend):
    """
    基于 sherpa-onnx (SenseVoice) 的 ASR 后端实现。

    SenseVoice 是阿里达摩院开发的多语言语音识别模型，
    支持中文、英文、日语、韩语、粤语以及语言自动检测。

    通过 sherpa-onnx Python 包在 ONNX Runtime 上运行，
    使用预编译的二进制文件，无需安装 Visual C++ 运行时。

    配置示例 (config.yaml):
        asr:
            backend: sensevoice_onnx
            model_path: ""              # 空则自动下载到 models/asr/sensevoice/
            language: auto              # 支持: auto/zh/en/ja/ko/yue
            n_threads: 4                # CPU 线程数
            stream_interval_seconds: 2.0  # 流式识别间隔
    """

    def __init__(self, config: ASRConfig):
        """
        初始化 SenseVoice ONNX 后端。

        Args:
            config: ASR 配置，包含模型路径、语言等参数。
        """
        super().__init__()
        self._config = config
        self._recognizer = None  # sherpa_onnx.OfflineRecognizer 实例

        # 解析语言设置
        self._language = _LANGUAGE_MAP.get(
            config.language.lower().strip(),
            config.language
        )

        logger.info(
            f"SenseVoiceOnnxBackend created: "
            f"language={self._language}, n_threads={config.n_threads}"
        )

    # ==================================================================
    # Model Lifecycle
    # ==================================================================

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        清理识别结果，移除 SenseVoice 在静音输入时产生的无意义前缀。

        SenseVoice 在处理开头的静音或低噪声音频时，可能会输出 "我." "그." 等
        无意义的前缀。这个方法会检测并移除这些前缀。

        Args:
            text: 原始识别文本

        Returns:
            清理后的文本
        """
        if not text:
            return text

        original_text = text

        # 移除开头的无意义前缀
        for prefix in _NOISE_PREFIXES:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
                # 可能有多个前缀，继续检查
                if text:
                    for prefix2 in _NOISE_PREFIXES:
                        if text.startswith(prefix2):
                            text = text[len(prefix2):].strip()
                            break
                break

        # 如果清理后文本变空了，保留原文（可能是用户确实说了 "我"）
        # 但如果原文就是单个前缀，返回空
        if not text and original_text in _NOISE_PREFIXES:
            return ""

        if text != original_text:
            logger.debug(f"Cleaned text: '{original_text}' -> '{text}'")

        return text

    def load_model(self) -> None:
        """
        加载 SenseVoice ONNX 模型。

        如果 model_path 未设置，会自动从默认位置查找模型。
        模型文件包括：model.int8.onnx (~170MB) 和 tokens.txt。

        Raises:
            ImportError: sherpa-onnx 未安装。
            FileNotFoundError: 模型文件不存在。
            RuntimeError: 模型加载失败。
        """
        if self._state == ASRState.READY and self._recognizer is not None:
            logger.warning("Model already loaded, skipping reload")
            return

        self._state = ASRState.LOADING
        logger.info("Loading SenseVoice ONNX model...")

        # 延迟导入 sherpa_onnx
        try:
            import sherpa_onnx
        except ImportError as e:
            self._state = ASRState.ERROR
            raise ImportError(
                "sherpa-onnx is not installed. Please run: pip install sherpa-onnx\n"
                "Note: sherpa-onnx comes with pre-compiled binaries, no VC++ needed.\n"
                "See: https://github.com/k2-fsa/sherpa-onnx"
            ) from e

        try:
            start_time = time.time()

            # 获取模型目录
            model_dir = self._get_model_dir()
            model_path = model_dir / _MODEL_FILES["model"]
            tokens_path = model_dir / _MODEL_FILES["tokens"]

            # 验证模型文件存在
            if not model_path.exists():
                raise FileNotFoundError(
                    f"SenseVoice model file not found: {model_path}\n"
                    f"Please download the model using:\n"
                    f"  python -c \"from voiceink.utils.model_downloader import get_downloader; "
                    f"get_downloader().download_model('sensevoice', 'int8', 'models/asr/sensevoice')\"\n"
                    f"Or manually download from: "
                    f"https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/{_MODEL_NAME}.tar.bz2"
                )
            if not tokens_path.exists():
                raise FileNotFoundError(
                    f"SenseVoice tokens file not found: {tokens_path}"
                )

            logger.info(f"Model path: {model_path}")
            logger.info(f"Tokens path: {tokens_path}")

            # 使用 from_sense_voice 类方法创建 OfflineRecognizer
            self._recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(model_path),
                tokens=str(tokens_path),
                num_threads=self._config.n_threads,
                sample_rate=16000,
                feature_dim=80,
                decoding_method="greedy_search",
                debug=False,
                provider="cpu",
                language=self._language,
                use_itn=True,  # 启用逆文本标准化（数字、日期等）
            )

            elapsed = time.time() - start_time
            self._state = ASRState.READY
            logger.info(
                f"SenseVoice ONNX model loaded in {elapsed:.2f}s "
                f"(language={self._language}, threads={self._config.n_threads})"
            )

        except FileNotFoundError:
            self._state = ASRState.ERROR
            raise
        except Exception as e:
            self._state = ASRState.ERROR
            raise RuntimeError(f"SenseVoice ONNX model loading failed: {e}") from e

    def unload(self) -> None:
        """
        卸载模型，释放资源。

        ONNX Runtime session 会在对象被垃圾回收时释放，
        这里显式设置为 None 以加速释放。
        """
        if self._recognizer is not None:
            logger.info("Unloading SenseVoice ONNX model...")
            del self._recognizer
            self._recognizer = None

        self._state = ASRState.UNLOADED
        logger.info("SenseVoice ONNX model unloaded")

    # ==================================================================
    # Transcription
    # ==================================================================

    def transcribe(self, audio: np.ndarray) -> TranscriptionResult:
        """
        对完整音频进行离线识别。

        Args:
            audio: 完整音频数据，float32 格式，16kHz 单声道。

        Returns:
            TranscriptionResult: 识别结果。

        Raises:
            RuntimeError: 模型未加载或识别过程出错。
            ValueError: 音频数据无效。
        """
        self._ensure_ready()
        audio = self._validate_audio(audio)

        duration_ms = len(audio) / 16.0  # 16 samples/ms
        logger.debug(
            f"SenseVoice transcribe: {len(audio)} samples "
            f"({duration_ms:.0f}ms)"
        )

        self._state = ASRState.TRANSCRIBING
        try:
            # 创建流并输入音频
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)

            # 解码
            self._recognizer.decode_stream(stream)

            # 获取结果并清理无意义前缀
            raw_text = stream.result.text.strip()
            text = self._clean_text(raw_text)

            self._state = ASRState.READY
            logger.debug(
                f"SenseVoice result: '{text[:80]}'"
                if len(text) > 80 else f"SenseVoice result: '{text}'"
            )

            return TranscriptionResult(
                text=text,
                language=self._language,
                confidence=-1.0,  # SenseVoice 不暴露置信度分数
                duration_ms=duration_ms,
                is_partial=False,
                segments=[],
            )

        except Exception as e:
            self._state = ASRState.READY  # 恢复到 READY 状态以便重试
            logger.error(f"SenseVoice transcription error: {e}")
            raise RuntimeError(f"SenseVoice transcription failed: {e}") from e

    def transcribe_stream(
        self, chunks: Generator[AudioChunk, None, None]
    ) -> Generator[TranscriptionResult, None, None]:
        """
        流式识别：逐块接收音频并实时输出识别结果。

        使用滑动窗口方式：累积音频块，周期性地对整个缓冲区进行识别，
        输出部分结果。

        注意：SenseVoice 是离线模型，这里使用伪流式方法。

        Args:
            chunks: AudioChunk 生成器，每次 yield 一个音频块。

        Yields:
            TranscriptionResult: 中间或最终识别结果。
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

                # 滑动窗口：如果超过最大长度则裁剪
                if len(audio_buffer) > SENSEVOICE_MAX_SAMPLES:
                    overflow = len(audio_buffer) - SENSEVOICE_MAX_SAMPLES
                    audio_buffer = audio_buffer[overflow:]
                    logger.debug(
                        f"Sliding window trim: dropped {overflow} samples "
                        f"({overflow / 16:.0f}ms)"
                    )

                # 周期性部分识别
                if samples_since_last >= interval_samples:
                    samples_since_last = 0

                    if len(audio_buffer) > 0:
                        result = self._transcribe_buffer(
                            audio_buffer, is_partial=True
                        )
                        if not result.is_empty:
                            yield result

            # 最终识别剩余的缓冲区
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
        设置初始提示词（热词）。

        注意：SenseVoice 模型不直接支持热词注入。
        此方法保存提示词以供将来可能的扩展使用。

        Args:
            prompt: 热词文本。
        """
        self._initial_prompt = prompt
        logger.debug(
            f"SenseVoice initial prompt updated (length={len(prompt)}). "
            "Note: SenseVoice does not support hot-word injection."
        )

    # ==================================================================
    # Internal Helpers
    # ==================================================================

    def _get_model_dir(self) -> Path:
        """
        获取模型目录路径。

        优先级：
        1. 配置中指定的 model_path
        2. 项目目录下的 models/asr/sensevoice/{model_name}/
        3. 用户目录下的 ~/.voiceink/models/asr/sensevoice/{model_name}/

        Returns:
            模型目录的 Path 对象。

        Raises:
            FileNotFoundError: 所有位置都找不到模型。
        """
        # Case 1: 用户指定了 model_path
        if self._config.model_path:
            model_path = Path(self._config.model_path)
            if model_path.is_file():
                # 如果指向文件，返回其父目录
                return model_path.parent
            return model_path

        # Case 2: 项目目录下的模型
        project_model_dir = Path("models/asr/sensevoice") / _MODEL_NAME
        if (project_model_dir / _MODEL_FILES["model"]).exists():
            return project_model_dir

        # Case 3: 用户目录下的模型
        user_model_dir = Path.home() / ".voiceink" / "models" / "asr" / "sensevoice" / _MODEL_NAME
        if (user_model_dir / _MODEL_FILES["model"]).exists():
            return user_model_dir

        # 返回默认路径（会在 load_model 中检测到不存在并报错）
        logger.warning(
            f"SenseVoice model not found in standard locations. "
            f"Checked: {project_model_dir}, {user_model_dir}"
        )
        return project_model_dir

    def _transcribe_buffer(
        self, audio: np.ndarray, is_partial: bool
    ) -> TranscriptionResult:
        """
        对音频缓冲区进行识别（用于流式模式）。

        Args:
            audio: 音频缓冲区数据。
            is_partial: 是否为部分（中间）结果。

        Returns:
            TranscriptionResult。
        """
        duration_ms = len(audio) / 16.0

        try:
            stream = self._recognizer.create_stream()
            stream.accept_waveform(16000, audio)
            self._recognizer.decode_stream(stream)
            raw_text = stream.result.text.strip()

            # 清理无意义前缀（但对部分结果保守处理）
            if is_partial:
                text = raw_text
            else:
                text = self._clean_text(raw_text)

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

    def __repr__(self) -> str:
        return (
            f"<SenseVoiceOnnxBackend "
            f"language={self._language} "
            f"n_threads={self._config.n_threads} "
            f"state={self._state.value}>"
        )
