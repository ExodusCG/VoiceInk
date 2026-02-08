"""
VoiceInk - 核心处理模块

提供音频采集、处理流水线、文本输出等核心功能。
"""

# 延迟导入：这些模块依赖第三方库（sounddevice, keyboard 等）
try:
    from voiceink.core.audio_capture import AudioCapture, AudioChunk, ChunkType
except ImportError:
    AudioCapture = None
    AudioChunk = None
    ChunkType = None

try:
    from voiceink.core.pipeline import VoiceInkPipeline, PipelineStatus
except ImportError:
    VoiceInkPipeline = None
    PipelineStatus = None

try:
    from voiceink.core.text_output import TextOutput
except ImportError:
    TextOutput = None

__all__ = [
    "AudioCapture",
    "AudioChunk",
    "ChunkType",
    "VoiceInkPipeline",
    "PipelineStatus",
    "TextOutput",
]
