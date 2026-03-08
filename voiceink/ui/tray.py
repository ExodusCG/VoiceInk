"""
VoiceInk - 系统托盘模块

使用 pystray 实现系统托盘图标和右键菜单。
支持：
  - 动态生成麦克风图标（Pillow 绘制，无需外部图片）
  - ASR 引擎切换（单选子菜单）
  - LLM 后端切换（单选子菜单）
  - 打开词典面板、设置面板
  - 状态变化时自动更新图标颜色
"""

import threading
from typing import Callable, Optional

from PIL import Image, ImageDraw
from pystray import Icon, Menu, MenuItem

from voiceink.utils.logger import logger


# ---------------------------------------------------------------------------
# 状态枚举常量
# ---------------------------------------------------------------------------
STATUS_IDLE = "idle"          # 空闲 → 灰色
STATUS_RECORDING = "recording"  # 录音中 → 红色
STATUS_PROCESSING = "processing"  # 处理中 → 黄色

# 状态 → 颜色映射
_STATUS_COLORS = {
    STATUS_IDLE: "#888888",
    STATUS_RECORDING: "#E53935",
    STATUS_PROCESSING: "#FBC02D",
}


def _create_microphone_icon(color: str = "#888888", size: int = 64) -> Image.Image:
    """
    使用 Pillow 动态绘制一个简单的麦克风形状图标。

    Args:
        color: 麦克风主体颜色（十六进制）
        size:  图标尺寸（正方形）

    Returns:
        PIL Image 对象
    """
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    cx = size // 2  # 中心 x
    # ---- 麦克风头部（椭圆） ----
    head_w = size * 0.30  # 半宽
    head_h = size * 0.28  # 半高
    head_top = size * 0.10
    head_bottom = head_top + head_h * 2
    draw.ellipse(
        [cx - head_w, head_top, cx + head_w, head_bottom],
        fill=color,
    )

    # ---- 麦克风柄（矩形） ----
    bar_w = size * 0.12
    bar_top = head_bottom - head_h * 0.3  # 稍微嵌入头部
    bar_bottom = head_bottom + size * 0.10
    draw.rectangle(
        [cx - bar_w, bar_top, cx + bar_w, bar_bottom],
        fill=color,
    )

    # ---- 弧形支架（用圆弧表示） ----
    arc_w = size * 0.38
    arc_top = head_top + head_h * 0.5
    arc_bottom = head_bottom + size * 0.08
    draw.arc(
        [cx - arc_w, arc_top, cx + arc_w, arc_bottom + size * 0.10],
        start=0, end=180,
        fill=color, width=max(2, size // 16),
    )

    # ---- 底座竖线 ----
    stand_top = arc_bottom + size * 0.04
    stand_bottom = stand_top + size * 0.14
    draw.line(
        [(cx, stand_top), (cx, stand_bottom)],
        fill=color, width=max(2, size // 16),
    )

    # ---- 底座横线 ----
    base_w = size * 0.22
    draw.line(
        [(cx - base_w, stand_bottom), (cx + base_w, stand_bottom)],
        fill=color, width=max(2, size // 16),
    )

    return img


class SystemTray:
    """
    系统托盘管理器。

    提供右键菜单、状态图标切换以及各种回调。

    用法::

        tray = SystemTray(
            on_asr_change=lambda backend: ...,
            on_llm_change=lambda backend: ...,
            on_open_settings=lambda: ...,
            on_open_dictionary=lambda: ...,
            on_quit=lambda: ...,
        )
        tray.run()  # 阻塞，通常在独立线程中调用
    """

    def __init__(
        self,
        on_asr_change: Optional[Callable[[str], None]] = None,
        on_llm_change: Optional[Callable[[str], None]] = None,
        on_open_settings: Optional[Callable[[], None]] = None,
        on_open_dictionary: Optional[Callable[[], None]] = None,
        on_quit: Optional[Callable[[], None]] = None,
        initial_asr: str = "whisper_cpp",
        initial_llm: str = "llama_cpp",
    ):
        """
        初始化系统托盘。

        Args:
            on_asr_change:      ASR 引擎变更回调，参数为引擎名称
            on_llm_change:      LLM 后端变更回调，参数为后端名称
            on_open_settings:   打开设置面板回调
            on_open_dictionary: 打开词典面板回调
            on_quit:            退出回调
            initial_asr:        初始 ASR 后端
            initial_llm:        初始 LLM 后端
        """
        # 回调
        self._on_asr_change = on_asr_change
        self._on_llm_change = on_llm_change
        self._on_open_settings = on_open_settings
        self._on_open_dictionary = on_open_dictionary
        self._on_quit = on_quit

        # 当前状态
        self._status: str = STATUS_IDLE
        self._asr_backend: str = initial_asr
        self._llm_backend: str = initial_llm

        # 创建托盘图标
        self._icon: Optional[Icon] = None
        self._build_icon()

        logger.info("SystemTray 初始化完成 (ASR=%s, LLM=%s)", self._asr_backend, self._llm_backend)

    # ------------------------------------------------------------------
    # ASR / LLM 选择辅助
    # ------------------------------------------------------------------

    def _set_asr(self, backend: str):
        """设置 ASR 引擎并触发回调"""
        def handler(icon, item):
            self._asr_backend = backend
            logger.info("ASR 引擎切换为: %s", backend)
            if self._on_asr_change:
                self._on_asr_change(backend)
            # 刷新菜单以更新单选状态
            icon.update_menu()
        return handler

    def _is_asr(self, backend: str):
        """返回一个可调用对象，用于判断当前 ASR 是否为指定后端"""
        def checker(item):
            return self._asr_backend == backend
        return checker

    def _set_llm(self, backend: str):
        """设置 LLM 后端并触发回调"""
        def handler(icon, item):
            self._llm_backend = backend
            logger.info("LLM 后端切换为: %s", backend)
            if self._on_llm_change:
                self._on_llm_change(backend)
            icon.update_menu()
        return handler

    def _is_llm(self, backend: str):
        """返回一个可调用对象，用于判断当前 LLM 是否为指定后端"""
        def checker(item):
            return self._llm_backend == backend
        return checker

    # ------------------------------------------------------------------
    # 菜单构建
    # ------------------------------------------------------------------

    def _build_menu(self) -> Menu:
        """构建右键上下文菜单"""
        return Menu(
            # ---- ASR 引擎子菜单 ----
            MenuItem(
                "ASR 引擎",
                Menu(
                    MenuItem(
                        "SenseVoice (推荐)",
                        self._set_asr("sensevoice_onnx"),
                        checked=self._is_asr("sensevoice_onnx"),
                        radio=True,
                    ),
                    MenuItem(
                        "Whisper.cpp",
                        self._set_asr("whisper_cpp"),
                        checked=self._is_asr("whisper_cpp"),
                        radio=True,
                    ),
                    MenuItem(
                        "Faster-Whisper",
                        self._set_asr("faster_whisper"),
                        checked=self._is_asr("faster_whisper"),
                        radio=True,
                    ),
                ),
            ),

            # ---- LLM 后端子菜单 ----
            MenuItem(
                "LLM 后端",
                Menu(
                    MenuItem(
                        "本地 llama.cpp",
                        self._set_llm("llama_cpp"),
                        checked=self._is_llm("llama_cpp"),
                        radio=True,
                    ),
                    MenuItem(
                        "云端 API",
                        self._set_llm("api"),
                        checked=self._is_llm("api"),
                        radio=True,
                    ),
                    MenuItem(
                        "禁用",
                        self._set_llm("disabled"),
                        checked=self._is_llm("disabled"),
                        radio=True,
                    ),
                ),
            ),

            # ---- 词典 ----
            MenuItem("自定义词典...", self._handle_open_dictionary),

            # ---- 设置 ----
            MenuItem("设置...", self._handle_open_settings),

            Menu.SEPARATOR,

            # ---- 退出 ----
            MenuItem("退出", self._handle_quit),
        )

    def _build_icon(self):
        """创建 / 重建 pystray.Icon 实例"""
        color = _STATUS_COLORS.get(self._status, _STATUS_COLORS[STATUS_IDLE])
        image = _create_microphone_icon(color)
        menu = self._build_menu()
        self._icon = Icon("VoiceInk", image, "VoiceInk 语音输入", menu)

    # ------------------------------------------------------------------
    # 菜单回调处理
    # ------------------------------------------------------------------

    def _handle_open_settings(self, icon, item):
        """处理"设置"菜单项点击"""
        logger.info("用户点击: 打开设置面板")
        if self._on_open_settings:
            self._on_open_settings()

    def _handle_open_dictionary(self, icon, item):
        """处理"自定义词典"菜单项点击"""
        logger.info("用户点击: 打开词典面板")
        if self._on_open_dictionary:
            self._on_open_dictionary()

    def _handle_quit(self, icon, item):
        """处理"退出"菜单项点击"""
        logger.info("用户点击: 退出")
        icon.stop()
        if self._on_quit:
            self._on_quit()

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def set_status(self, status: str):
        """
        更新托盘图标状态（改变颜色）。

        Args:
            status: STATUS_IDLE / STATUS_RECORDING / STATUS_PROCESSING
        """
        if status == self._status:
            return
        self._status = status
        color = _STATUS_COLORS.get(status, _STATUS_COLORS[STATUS_IDLE])
        new_image = _create_microphone_icon(color)
        if self._icon:
            self._icon.icon = new_image
        logger.debug("托盘图标状态更新: %s (颜色: %s)", status, color)

    def set_tooltip(self, text: str):
        """更新托盘图标的 tooltip 文字"""
        if self._icon:
            self._icon.title = text

    def run(self):
        """
        启动托盘图标（阻塞）。

        通常在独立线程中调用::

            tray_thread = threading.Thread(target=tray.run, daemon=True)
            tray_thread.start()
        """
        logger.info("系统托盘启动")
        if self._icon:
            self._icon.run()

    def run_detached(self):
        """
        在后台线程中启动托盘图标（非阻塞）。

        Returns:
            启动的线程对象
        """
        thread = threading.Thread(target=self.run, daemon=True, name="SystemTray")
        thread.start()
        return thread

    def stop(self):
        """停止托盘图标"""
        if self._icon:
            self._icon.stop()
            logger.info("系统托盘已停止")

    @property
    def status(self) -> str:
        """当前状态"""
        return self._status

    @property
    def asr_backend(self) -> str:
        """当前 ASR 后端"""
        return self._asr_backend

    @property
    def llm_backend(self) -> str:
        """当前 LLM 后端"""
        return self._llm_backend
