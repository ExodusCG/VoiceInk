"""
VoiceInk - 工具模块

提供日志、全局快捷键、模型下载等工具功能。
"""

from voiceink.utils.logger import setup_logger, logger

# 延迟导入：这些模块依赖第三方库（keyboard, urllib 等）
try:
    from voiceink.utils.hotkey import HotkeyManager
except ImportError:
    HotkeyManager = None

try:
    from voiceink.utils.model_downloader import ModelDownloader, get_downloader
except ImportError:
    ModelDownloader = None
    get_downloader = None

__all__ = [
    "setup_logger",
    "logger",
    "HotkeyManager",
    "ModelDownloader",
    "get_downloader",
]
