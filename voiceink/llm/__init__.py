"""
VoiceInk - LLM 润色模块

提供语音识别文本的润色（后处理）能力，支持以下后端：
- llama_cpp: 本地 llama.cpp 推理（使用 GGUF 模型，纯 CPU，无需联网）
- api: 云端 API 调用（兼容 OpenAI 格式，支持通义千问/DeepSeek 等）

使用方式:
    from voiceink.config import LLMConfig
    from voiceink.llm import create_llm_backend

    config = LLMConfig(backend="llama_cpp", model_name="Qwen3-0.6B-Q8_0.gguf")
    backend = create_llm_backend(config)
    backend.load_model()

    result = backend.polish("今天天气因该挺好的我们出去走走吧")
    print(result.text)  # "今天天气应该挺好的，我们出去走走吧。"

    backend.unload()
"""

from voiceink.llm.base import LLMBackend, PolishResult, POLISH_SYSTEM_PROMPT, build_polish_prompt
from voiceink.config import LLMConfig
from voiceink.utils.logger import logger

# 后端名称到模块/类的延迟映射（避免在导入时加载不必要的依赖）
_BACKEND_REGISTRY = {
    "llama_cpp": ("voiceink.llm.llama_cpp_backend", "LlamaCppBackend"),
    "api": ("voiceink.llm.api_backend", "APIBackend"),
}

# 后端名称别名映射
_BACKEND_ALIASES = {
    "llama_cpp": "llama_cpp",
    "llama-cpp": "llama_cpp",
    "llamacpp": "llama_cpp",
    "local": "llama_cpp",
    "api": "api",
    "openai": "api",
    "cloud": "api",
}


def create_llm_backend(config: LLMConfig) -> LLMBackend:
    """
    工厂函数：根据配置创建对应的 LLM 后端实例

    根据 config.backend 字段选择具体的后端实现：
    - "llama_cpp" → LlamaCppBackend（本地推理）
    - "api"       → APIBackend（云端 API）

    Args:
        config: LLM 配置对象

    Returns:
        LLMBackend 实例（未加载模型，需要调用 load_model()）

    Raises:
        ValueError: 不支持的 backend 类型

    示例:
        >>> config = LLMConfig(backend="llama_cpp")
        >>> backend = create_llm_backend(config)
        >>> backend.load_model()
        >>> result = backend.polish("原始文本")
    """
    backend_name = config.backend.lower().strip()

    # 通过别名映射获取标准名称
    canonical_name = _BACKEND_ALIASES.get(backend_name)
    if canonical_name is None:
        supported = list(_BACKEND_REGISTRY.keys())
        raise ValueError(
            f"不支持的 LLM 后端类型: '{config.backend}'\n"
            f"支持的后端: {', '.join(supported)}"
        )

    # 延迟导入后端模块
    module_path, class_name = _BACKEND_REGISTRY[canonical_name]
    try:
        import importlib
        module = importlib.import_module(module_path)
        backend_class = getattr(module, class_name)
    except ImportError as e:
        raise ImportError(
            f"LLM 后端 '{canonical_name}' 的依赖库未安装: {e}\n"
            f"请根据以下指引安装:\n"
            f"  - llama_cpp:  pip install llama-cpp-python\n"
            f"  - api:        pip install openai"
        ) from e

    logger.info("[LLM] 创建后端: %s (%s.%s)", canonical_name, module_path, class_name)
    return backend_class(config)


# 模块公开接口
__all__ = [
    # 核心类
    "LLMBackend",
    # 数据类
    "PolishResult",
    # 工厂函数
    "create_llm_backend",
    # 提示词工具（供外部自定义使用）
    "POLISH_SYSTEM_PROMPT",
    "build_polish_prompt",
]
