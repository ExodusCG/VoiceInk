"""
VoiceInk - Windows 语音输入软件

基于 Whisper ASR + LLM 润色的离线/在线混合语音输入方案。
支持全局快捷键 Push-to-Talk，识别后自动输入到焦点窗口。

核心模块:
    - voiceink.core:       音频采集、处理流水线、文本输出
    - voiceink.asr:        语音识别后端（Whisper.cpp / faster-whisper）
    - voiceink.llm:        LLM 润色后端（llama.cpp / OpenAI API）
    - voiceink.dictionary:  自定义词典管理
    - voiceink.ui:          系统托盘、悬浮指示器、设置面板
    - voiceink.utils:       日志、快捷键、模型下载

使用方式:
    python -m voiceink.main
"""

__version__ = "0.2.0"
__author__ = "VoiceInk Team"

from voiceink.config import AppConfig, load_config, save_config

# 延迟导入：pipeline 依赖多个第三方库
try:
    from voiceink.core.pipeline import VoiceInkPipeline, PipelineStatus
except ImportError:
    VoiceInkPipeline = None
    PipelineStatus = None

__all__ = [
    "__version__",
    "__author__",
    # 配置
    "AppConfig",
    "load_config",
    "save_config",
    # 核心流水线
    "VoiceInkPipeline",
    "PipelineStatus",
]
