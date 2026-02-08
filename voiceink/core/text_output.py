"""
VoiceInk - 文本输出模块

将识别后的文本输出到当前焦点窗口。
支持两种输出模式：
  1. keyboard 模式：通过 Win32 SendInput API 逐字符模拟 Unicode 键盘输入
  2. clipboard 模式：复制到剪贴板后模拟 Ctrl+V 粘贴，完成后恢复原剪贴板内容

在非 Windows 平台上自动降级为 clipboard 模式。
"""

import time
import platform
import logging
from typing import Optional

from voiceink.config import OutputConfig

logger = logging.getLogger("voiceink")

# ============================================================
# 平台检测
# ============================================================
IS_WINDOWS = platform.system() == "Windows"

# ============================================================
# Win32 API 常量与结构体定义（仅在 Windows 下加载）
# ============================================================
if IS_WINDOWS:
    import ctypes
    import ctypes.wintypes as wintypes

    # --- 常量 ---
    INPUT_KEYBOARD = 1  # INPUT 结构体的 type 字段：键盘事件

    KEYEVENTF_UNICODE = 0x0004   # 发送 Unicode 字符（wScan 携带字符码）
    KEYEVENTF_KEYUP   = 0x0002   # 按键释放事件

    VK_CONTROL = 0x11  # Ctrl 虚拟键码
    VK_V       = 0x56  # V 虚拟键码

    # --- 结构体 ---
    class KEYBDINPUT(ctypes.Structure):
        """
        Win32 KEYBDINPUT 结构体。
        参见：https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-keybdinput
        """
        _fields_ = [
            ("wVk",         wintypes.WORD),   # 虚拟键码（Unicode 模式下为 0）
            ("wScan",       wintypes.WORD),   # 硬件扫描码 / Unicode 字符码
            ("dwFlags",     wintypes.DWORD),  # 事件标志
            ("time",        wintypes.DWORD),  # 时间戳（0 = 系统自动填充）
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),  # 附加信息
        ]

    class MOUSEINPUT(ctypes.Structure):
        """Win32 MOUSEINPUT 占位结构体（用于 INPUT union）。"""
        _fields_ = [
            ("dx",          wintypes.LONG),
            ("dy",          wintypes.LONG),
            ("mouseData",   wintypes.DWORD),
            ("dwFlags",     wintypes.DWORD),
            ("time",        wintypes.DWORD),
            ("dwExtraInfo", ctypes.POINTER(ctypes.c_ulong)),
        ]

    class HARDWAREINPUT(ctypes.Structure):
        """Win32 HARDWAREINPUT 占位结构体（用于 INPUT union）。"""
        _fields_ = [
            ("uMsg",    wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        ]

    class _INPUT_UNION(ctypes.Union):
        """INPUT 结构体内部的匿名联合体。"""
        _fields_ = [
            ("ki", KEYBDINPUT),
            ("mi", MOUSEINPUT),
            ("hi", HARDWAREINPUT),
        ]

    class INPUT(ctypes.Structure):
        """
        Win32 INPUT 结构体。
        参见：https://learn.microsoft.com/en-us/windows/win32/api/winuser/ns-winuser-input
        """
        _fields_ = [
            ("type", wintypes.DWORD),
            ("_input", _INPUT_UNION),
        ]

    # SendInput 函数原型
    _SendInput = ctypes.windll.user32.SendInput
    _SendInput.argtypes = [
        ctypes.c_uint,                  # nInputs
        ctypes.POINTER(INPUT),          # pInputs
        ctypes.c_int,                   # cbSize
    ]
    _SendInput.restype = ctypes.c_uint  # 返回成功注入的事件数


class TextOutput:
    """
    文本输出器 —— 将文本发送到当前焦点窗口。

    使用方式：
        output = TextOutput(config.output)
        output.type_text("你好，世界！Hello 🌍")

    Args:
        config: OutputConfig 实例，包含 method / typing_delay_ms / paste_delay_ms
    """

    def __init__(self, config: Optional[OutputConfig] = None):
        # 若未传入配置，使用默认值
        self._config = config or OutputConfig()

        # 决定实际使用的输出模式
        requested_method = self._config.method.lower()
        if requested_method == "keyboard" and not IS_WINDOWS:
            logger.warning(
                "当前平台 (%s) 不支持 keyboard 模式，自动降级为 clipboard 模式",
                platform.system(),
            )
            self._method = "clipboard"
        else:
            self._method = requested_method

        logger.info("TextOutput 初始化完成，输出模式: %s", self._method)

    # ============================================================
    # 公共 API
    # ============================================================

    def type_text(self, text: str) -> None:
        """
        将文本输出到当前焦点窗口。

        根据初始化时确定的模式，自动选择 keyboard 或 clipboard 方式。

        Args:
            text: 要输出的文本（支持中英文、标点、emoji 等任意 Unicode 字符）
        """
        if not text:
            logger.debug("type_text 收到空文本，跳过")
            return

        logger.debug("type_text: 输出 %d 个字符，模式=%s", len(text), self._method)

        if self._method == "keyboard":
            self._type_via_keyboard(text)
        else:
            self._type_via_clipboard(text)

    # ============================================================
    # keyboard 模式 —— Win32 SendInput Unicode
    # ============================================================

    def _type_via_keyboard(self, text: str) -> None:
        """
        通过 Win32 SendInput API 逐字符发送 Unicode 键盘事件。

        对于基本多文种平面（BMP, U+0000 ~ U+FFFF）的字符，直接发送一对
        KEYDOWN + KEYUP 事件。对于补充平面字符（如 emoji），拆分为 UTF-16
        代理对（surrogate pair），分别发送高代理和低代理。

        每个字符之间插入 typing_delay_ms 毫秒的延迟，避免某些应用丢字。
        """
        delay_sec = self._config.typing_delay_ms / 1000.0

        for char in text:
            code_point = ord(char)

            if code_point <= 0xFFFF:
                # BMP 字符：直接发送
                self._send_unicode_char(code_point)
            else:
                # 补充平面字符（如 emoji）：拆分为 UTF-16 代理对
                # 高代理 = 0xD800 + ((code_point - 0x10000) >> 10)
                # 低代理 = 0xDC00 + ((code_point - 0x10000) & 0x3FF)
                high_surrogate = 0xD800 + ((code_point - 0x10000) >> 10)
                low_surrogate  = 0xDC00 + ((code_point - 0x10000) & 0x3FF)
                self._send_unicode_char(high_surrogate)
                self._send_unicode_char(low_surrogate)

            # 字符间延迟
            if delay_sec > 0:
                time.sleep(delay_sec)

        logger.debug("keyboard 模式输出完成，共 %d 个字符", len(text))

    @staticmethod
    def _send_unicode_char(char_code: int) -> None:
        """
        发送单个 Unicode 字符的 KEYDOWN + KEYUP 事件。

        Args:
            char_code: Unicode 码点（或 UTF-16 代理值），范围 0x0000 ~ 0xFFFF
        """
        # 构造 KEYDOWN 事件
        key_down = INPUT()
        key_down.type = INPUT_KEYBOARD
        key_down._input.ki.wVk = 0                 # Unicode 模式下虚拟键码为 0
        key_down._input.ki.wScan = char_code        # wScan 承载 Unicode 字符码
        key_down._input.ki.dwFlags = KEYEVENTF_UNICODE
        key_down._input.ki.time = 0
        key_down._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))

        # 构造 KEYUP 事件
        key_up = INPUT()
        key_up.type = INPUT_KEYBOARD
        key_up._input.ki.wVk = 0
        key_up._input.ki.wScan = char_code
        key_up._input.ki.dwFlags = KEYEVENTF_UNICODE | KEYEVENTF_KEYUP
        key_up._input.ki.time = 0
        key_up._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))

        # 一次性发送 KEYDOWN + KEYUP
        inputs = (INPUT * 2)(key_down, key_up)
        sent = _SendInput(2, inputs, ctypes.sizeof(INPUT))

        if sent != 2:
            logger.warning(
                "SendInput 返回 %d（期望 2），字符码 0x%04X 可能未成功发送",
                sent, char_code,
            )

    # ============================================================
    # clipboard 模式 —— 剪贴板 + Ctrl+V
    # ============================================================

    def _type_via_clipboard(self, text: str) -> None:
        """
        通过剪贴板粘贴文本：
        1. 备份当前剪贴板内容
        2. 将目标文本写入剪贴板
        3. 模拟 Ctrl+V 粘贴
        4. 等待粘贴完成后恢复原剪贴板内容
        """
        import pyperclip

        # 1. 备份原剪贴板内容
        original_clipboard = ""
        try:
            original_clipboard = pyperclip.paste()
        except Exception:
            # 剪贴板可能为空或包含非文本内容，忽略
            logger.debug("备份剪贴板内容失败（可能为空或非文本），继续执行")

        try:
            # 2. 写入目标文本到剪贴板
            pyperclip.copy(text)

            # 短暂延迟，确保剪贴板数据就绪
            time.sleep(0.05)

            # 3. 模拟 Ctrl+V 粘贴
            self._simulate_paste()

            # 4. 等待粘贴操作完成
            paste_delay_sec = self._config.paste_delay_ms / 1000.0
            time.sleep(paste_delay_sec)

            logger.debug("clipboard 模式输出完成，共 %d 个字符", len(text))

        finally:
            # 5. 恢复原剪贴板内容
            try:
                pyperclip.copy(original_clipboard)
                logger.debug("已恢复原剪贴板内容")
            except Exception:
                logger.warning("恢复原剪贴板内容失败")

    def _simulate_paste(self) -> None:
        """
        模拟 Ctrl+V 粘贴快捷键。

        Windows：使用 SendInput 发送虚拟键码。
        非 Windows：使用 keyboard 库作为后备方案。
        """
        if IS_WINDOWS:
            self._simulate_paste_win32()
        else:
            self._simulate_paste_fallback()

    @staticmethod
    def _simulate_paste_win32() -> None:
        """
        Windows 平台：通过 SendInput 模拟 Ctrl+V。

        事件顺序：Ctrl↓ → V↓ → V↑ → Ctrl↑
        """
        events = []

        # Ctrl 按下
        ctrl_down = INPUT()
        ctrl_down.type = INPUT_KEYBOARD
        ctrl_down._input.ki.wVk = VK_CONTROL
        ctrl_down._input.ki.dwFlags = 0
        ctrl_down._input.ki.time = 0
        ctrl_down._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
        events.append(ctrl_down)

        # V 按下
        v_down = INPUT()
        v_down.type = INPUT_KEYBOARD
        v_down._input.ki.wVk = VK_V
        v_down._input.ki.dwFlags = 0
        v_down._input.ki.time = 0
        v_down._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
        events.append(v_down)

        # V 释放
        v_up = INPUT()
        v_up.type = INPUT_KEYBOARD
        v_up._input.ki.wVk = VK_V
        v_up._input.ki.dwFlags = KEYEVENTF_KEYUP
        v_up._input.ki.time = 0
        v_up._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
        events.append(v_up)

        # Ctrl 释放
        ctrl_up = INPUT()
        ctrl_up.type = INPUT_KEYBOARD
        ctrl_up._input.ki.wVk = VK_CONTROL
        ctrl_up._input.ki.dwFlags = KEYEVENTF_KEYUP
        ctrl_up._input.ki.time = 0
        ctrl_up._input.ki.dwExtraInfo = ctypes.pointer(ctypes.c_ulong(0))
        events.append(ctrl_up)

        # 一次性发送所有事件
        input_array = (INPUT * len(events))(*events)
        sent = _SendInput(len(events), input_array, ctypes.sizeof(INPUT))

        if sent != len(events):
            logger.warning("模拟 Ctrl+V 时 SendInput 返回 %d（期望 %d）", sent, len(events))

    @staticmethod
    def _simulate_paste_fallback() -> None:
        """
        非 Windows 平台的粘贴后备方案：使用 keyboard 库发送 Ctrl+V。
        """
        try:
            import keyboard as kb
            kb.press_and_release("ctrl+v")
        except ImportError:
            logger.error(
                "非 Windows 平台需要安装 keyboard 库来模拟粘贴: pip install keyboard"
            )
            raise
        except Exception as e:
            logger.error("模拟 Ctrl+V 失败: %s", e)
            raise

    # ============================================================
    # 配置热更新
    # ============================================================

    def update_config(self, config: OutputConfig) -> None:
        """
        动态更新输出配置。

        Args:
            config: 新的 OutputConfig 实例
        """
        old_method = self._method
        self._config = config

        requested_method = config.method.lower()
        if requested_method == "keyboard" and not IS_WINDOWS:
            self._method = "clipboard"
        else:
            self._method = requested_method

        if old_method != self._method:
            logger.info("输出模式已切换: %s -> %s", old_method, self._method)

    @property
    def method(self) -> str:
        """当前实际使用的输出模式。"""
        return self._method
