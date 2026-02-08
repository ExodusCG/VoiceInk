"""
VoiceInk - UI 模块

提供系统托盘、悬浮状态指示器、设置面板和词典管理面板。
"""

from voiceink.ui.tray import SystemTray, STATUS_IDLE, STATUS_RECORDING, STATUS_PROCESSING
from voiceink.ui.status_indicator import StatusIndicator
from voiceink.ui.settings_panel import SettingsPanel
from voiceink.ui.dictionary_panel import DictionaryPanel, DictEntry

__all__ = [
    # 系统托盘
    "SystemTray",
    "STATUS_IDLE",
    "STATUS_RECORDING",
    "STATUS_PROCESSING",
    # 悬浮状态指示器
    "StatusIndicator",
    # 设置面板
    "SettingsPanel",
    # 词典管理面板
    "DictionaryPanel",
    "DictEntry",
]
