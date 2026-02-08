"""
VoiceInk ASR - 语音识别模块

提供统一的 ASR（自动语音识别）接口，支持多种后端引擎。

支持的后端:
    - whisper_cpp: 基于 whisper.cpp 的 CPU 推理后端（pywhispercpp）
    - faster_whisper: 基于 CTranslate2 的高性能后端（faster-whisper）
    - paraformer_onnx: 基于 funasr-onnx 的 Paraformer-zh 后端（ONNX Runtime）

使用方式:
    from voiceink.asr import create_asr_backend, AudioChunk, TranscriptionResult
    from voiceink.config import ASRConfig

    config = ASRConfig(backend="faster_whisper", model_size="base")
    asr = create_asr_backend(config)
    asr.load_model()

    result = asr.transcribe(audio_data)
    print(result.text)

    asr.unload()
"""

from voiceink.asr.base import (
    ASRBackend,
    ASRState,
    AudioChunk,
    TranscriptionResult,
)
from voiceink.config import ASRConfig

# 后端名称到类的延迟映射（避免在导入时加载不必要的依赖）
_BACKEND_REGISTRY = {
    "whisper_cpp": (
        "voiceink.asr.whisper_cpp_backend",
        "WhisperCppBackend",
    ),
    "faster_whisper": (
        "voiceink.asr.faster_whisper_backend",
        "FasterWhisperBackend",
    ),
    "paraformer_onnx": (
        "voiceink.asr.paraformer_onnx_backend",
        "ParaformerOnnxBackend",
    ),
}


def create_asr_backend(config: ASRConfig) -> ASRBackend:
    """
    ASR 后端工厂函数：根据配置创建对应的 ASR 后端实例。

    此函数使用延迟导入策略，只有在实际创建某个后端时才导入对应的模块，
    避免未安装某个后端库时影响其他后端的使用。

    Args:
        config: ASR 配置对象，其中 config.backend 指定后端类型。
                支持的值: "whisper_cpp", "faster_whisper"

    Returns:
        ASRBackend: 初始化好的 ASR 后端实例（尚未加载模型，需调用 load_model()）。

    Raises:
        ValueError: 不支持的后端类型。
        ImportError: 后端对应的依赖库未安装。

    Example:
        >>> from voiceink.config import ASRConfig
        >>> config = ASRConfig(backend="whisper_cpp", model_size="base", language="zh")
        >>> asr = create_asr_backend(config)
        >>> asr.load_model()
        >>> result = asr.transcribe(audio_array)
        >>> print(result.text)
        >>> asr.unload()
    """
    backend_name = config.backend.lower().strip()

    if backend_name not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"不支持的 ASR 后端: '{backend_name}'\n"
            f"可选后端: {available}\n"
            f"请检查配置文件中 asr.backend 的值。"
        )

    # 延迟导入后端模块
    module_path, class_name = _BACKEND_REGISTRY[backend_name]

    try:
        import importlib
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"ASR 后端 '{backend_name}' 的依赖库未安装: {e}\n"
            f"请根据以下指引安装:\n"
            f"  - whisper_cpp:     pip install pywhispercpp\n"
            f"  - faster_whisper:  pip install faster-whisper\n"
            f"  - paraformer_onnx: pip install funasr-onnx"
        ) from e
    except AttributeError as e:
        raise RuntimeError(
            f"ASR 后端模块 '{module_path}' 中找不到类 '{class_name}': {e}"
        ) from e

    # 创建后端实例
    return backend_class(config)


def list_available_backends() -> dict:
    """
    列出所有已注册的 ASR 后端及其可用状态。

    Returns:
        字典，键为后端名称，值为 (是否可用, 描述信息)。

    Example:
        >>> for name, (available, desc) in list_available_backends().items():
        ...     status = "✓" if available else "✗"
        ...     print(f"  [{status}] {name}: {desc}")
    """
    result = {}

    for name, (module_path, class_name) in _BACKEND_REGISTRY.items():
        try:
            import importlib
            importlib.import_module(module_path)
            result[name] = (True, f"已安装 ({module_path}.{class_name})")
        except ImportError as e:
            result[name] = (False, f"未安装 ({e})")

    return result


# 模块公开接口
__all__ = [
    # 核心类
    "ASRBackend",
    "ASRState",
    "AudioChunk",
    "TranscriptionResult",
    # 工厂函数
    "create_asr_backend",
    "list_available_backends",
]
