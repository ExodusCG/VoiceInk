"""
VoiceInk - 悬浮状态指示器

使用 tkinter Toplevel 实现无边框、置顶、半透明的悬浮窗口。
在独立线程中运行 tkinter mainloop，通过线程安全队列与主线程通信。

功能：
  - 显示当前状态图标和文字（录音中 / 识别中 / 润色中 / 完成）
  - 显示流式识别的实时文本预览
  - 支持鼠标拖拽移动
  - 可配置显示位置（top_right / top_left / bottom_right / bottom_left / center）
"""

import queue
import threading
import tkinter as tk
from tkinter import font as tkfont
from typing import Optional

from voiceink.config import StatusIndicatorConfig
from voiceink.utils.logger import logger


# ---------------------------------------------------------------------------
# 状态定义与对应的显示配置
# ---------------------------------------------------------------------------
STATUS_RECORDING = "recording"      # 录音中
STATUS_RECOGNIZING = "recognizing"  # 识别中
STATUS_POLISHING = "polishing"      # 润色中
STATUS_DONE = "done"                # 完成

_STATUS_DISPLAY = {
    STATUS_RECORDING:   {"text": "🎙 录音中...",   "color": "#E53935", "bg": "#FFF3F3"},
    STATUS_RECOGNIZING: {"text": "🔍 识别中...",   "color": "#FB8C00", "bg": "#FFF8E1"},
    STATUS_POLISHING:   {"text": "✨ 润色中...",   "color": "#7B1FA2", "bg": "#F3E5F5"},
    STATUS_DONE:        {"text": "✅ 完成",        "color": "#388E3C", "bg": "#E8F5E9"},
}

# 队列中的命令类型
_CMD_SHOW = "show"
_CMD_HIDE = "hide"
_CMD_UPDATE_TEXT = "update_text"
_CMD_UPDATE_STATUS = "update_status"
_CMD_DESTROY = "destroy"


class StatusIndicator:
    """
    悬浮状态指示器。

    在独立线程中运行 tkinter 窗口，主线程通过 show() / hide() / update_text() 等
    方法安全地控制 UI。

    用法::

        indicator = StatusIndicator(config)
        indicator.start()                          # 启动后台线程
        indicator.show("准备录音...", "recording")  # 显示
        indicator.update_text("你好世界")           # 更新文本
        indicator.hide()                            # 隐藏
        indicator.destroy()                         # 销毁
    """

    def __init__(self, config: Optional[StatusIndicatorConfig] = None):
        """
        初始化状态指示器。

        Args:
            config: StatusIndicatorConfig 配置对象，None 则使用默认值
        """
        if config is None:
            config = StatusIndicatorConfig()

        self._config = config
        self._enabled = config.enabled

        # 线程间通信队列
        self._cmd_queue: queue.Queue = queue.Queue()

        # tkinter 相关（在 UI 线程中初始化）
        self._root: Optional[tk.Tk] = None
        self._window: Optional[tk.Toplevel] = None
        self._status_label: Optional[tk.Label] = None
        self._text_label: Optional[tk.Label] = None
        self._frame: Optional[tk.Frame] = None

        # 拖拽状态
        self._drag_start_x: int = 0
        self._drag_start_y: int = 0

        # 线程
        self._thread: Optional[threading.Thread] = None
        self._running = False

        logger.info(
            "StatusIndicator 初始化 (enabled=%s, position=%s, opacity=%.2f)",
            config.enabled, config.position, config.opacity,
        )

    # ------------------------------------------------------------------
    # 公共 API（主线程调用，线程安全）
    # ------------------------------------------------------------------

    def start(self):
        """启动 UI 线程"""
        if not self._enabled:
            logger.info("StatusIndicator 已禁用，跳过启动")
            return
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(
            target=self._ui_thread_main, daemon=True, name="StatusIndicator"
        )
        self._thread.start()
        logger.info("StatusIndicator UI 线程已启动")

    def show(self, text: str = "", status: str = STATUS_RECORDING):
        """
        显示悬浮窗口。

        Args:
            text:   要显示的文本内容
            status: 状态（recording / recognizing / polishing / done）
        """
        if not self._enabled:
            return
        self._cmd_queue.put((_CMD_SHOW, {"text": text, "status": status}))

    def hide(self):
        """隐藏悬浮窗口"""
        if not self._enabled:
            return
        self._cmd_queue.put((_CMD_HIDE, {}))

    def update_text(self, text: str):
        """
        更新流式文本预览。

        Args:
            text: 最新的识别文本
        """
        if not self._enabled:
            return
        self._cmd_queue.put((_CMD_UPDATE_TEXT, {"text": text}))

    def update_status(self, status: str):
        """
        仅更新状态（不改变文本）。

        Args:
            status: 新状态
        """
        if not self._enabled:
            return
        self._cmd_queue.put((_CMD_UPDATE_STATUS, {"status": status}))

    def destroy(self):
        """销毁 UI 线程和所有窗口"""
        self._running = False
        self._cmd_queue.put((_CMD_DESTROY, {}))
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=3)
        logger.info("StatusIndicator 已销毁")

    # ------------------------------------------------------------------
    # UI 线程（内部）
    # ------------------------------------------------------------------

    def _ui_thread_main(self):
        """UI 线程入口：创建窗口并运行 mainloop"""
        try:
            self._root = tk.Tk()
            self._root.withdraw()  # 隐藏主窗口

            # 创建悬浮窗口
            self._create_window()

            # 启动命令队列轮询
            self._poll_commands()

            # 进入主循环
            self._root.mainloop()
        except Exception as e:
            logger.error("StatusIndicator UI 线程异常: %s", e, exc_info=True)
        finally:
            self._running = False
            logger.debug("StatusIndicator UI 线程退出")

    def _create_window(self):
        """创建悬浮窗口及其子控件"""
        win = tk.Toplevel(self._root)
        self._window = win

        # 无边框、置顶
        win.overrideredirect(True)
        win.attributes("-topmost", True)

        # 半透明（Windows 支持 -alpha）
        try:
            win.attributes("-alpha", self._config.opacity)
        except tk.TclError:
            pass  # 某些平台不支持

        # 尺寸
        width = self._config.width
        height = self._config.height

        # 计算初始位置
        x, y = self._calculate_position(width, height)
        win.geometry(f"{width}x{height}+{x}+{y}")

        # 圆角模拟：使用带圆角的 Frame 背景
        win.configure(bg="#FFFFFF")

        # 主容器
        frame = tk.Frame(win, bg="#FFFFFF", padx=12, pady=8)
        frame.pack(fill=tk.BOTH, expand=True)
        self._frame = frame

        # 状态行标签
        status_font = tkfont.Font(family="Microsoft YaHei UI", size=12, weight="bold")
        self._status_label = tk.Label(
            frame,
            text="🎙 录音中...",
            font=status_font,
            fg="#E53935",
            bg="#FFFFFF",
            anchor="w",
        )
        self._status_label.pack(fill=tk.X, pady=(0, 2))

        # 文本预览标签
        text_font = tkfont.Font(family="Microsoft YaHei UI", size=10)
        self._text_label = tk.Label(
            frame,
            text="",
            font=text_font,
            fg="#555555",
            bg="#FFFFFF",
            anchor="w",
            justify="left",
            wraplength=width - 30,
        )
        self._text_label.pack(fill=tk.BOTH, expand=True)

        # ---- 拖拽绑定 ----
        for widget in (win, frame, self._status_label, self._text_label):
            widget.bind("<Button-1>", self._on_drag_start)
            widget.bind("<B1-Motion>", self._on_drag_motion)

        # 初始隐藏
        win.withdraw()

        logger.debug("StatusIndicator 悬浮窗口已创建 (位置: %d, %d)", x, y)

    def _calculate_position(self, width: int, height: int):
        """
        根据配置的 position 计算窗口坐标。

        支持: top_right, top_left, bottom_right, bottom_left, center
        """
        # 获取屏幕尺寸
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()

        margin = 20  # 边距

        positions = {
            "top_right":    (screen_w - width - margin, margin),
            "top_left":     (margin, margin),
            "bottom_right": (screen_w - width - margin, screen_h - height - margin - 40),
            "bottom_left":  (margin, screen_h - height - margin - 40),
            "center":       ((screen_w - width) // 2, (screen_h - height) // 2),
        }

        return positions.get(self._config.position, positions["top_right"])

    # ------------------------------------------------------------------
    # 拖拽支持
    # ------------------------------------------------------------------

    def _on_drag_start(self, event):
        """记录拖拽起始位置"""
        self._drag_start_x = event.x_root - self._window.winfo_x()
        self._drag_start_y = event.y_root - self._window.winfo_y()

    def _on_drag_motion(self, event):
        """拖拽移动窗口"""
        x = event.x_root - self._drag_start_x
        y = event.y_root - self._drag_start_y
        self._window.geometry(f"+{x}+{y}")

    # ------------------------------------------------------------------
    # 命令队列轮询
    # ------------------------------------------------------------------

    def _poll_commands(self):
        """从队列中取出命令并在 UI 线程中执行"""
        try:
            while not self._cmd_queue.empty():
                cmd, data = self._cmd_queue.get_nowait()

                if cmd == _CMD_SHOW:
                    self._do_show(data.get("text", ""), data.get("status", STATUS_RECORDING))
                elif cmd == _CMD_HIDE:
                    self._do_hide()
                elif cmd == _CMD_UPDATE_TEXT:
                    self._do_update_text(data.get("text", ""))
                elif cmd == _CMD_UPDATE_STATUS:
                    self._do_update_status(data.get("status", STATUS_RECORDING))
                elif cmd == _CMD_DESTROY:
                    self._do_destroy()
                    return  # 不再轮询
        except Exception as e:
            logger.error("StatusIndicator 命令处理异常: %s", e)

        # 每 50ms 轮询一次
        if self._running and self._root:
            self._root.after(50, self._poll_commands)

    # ------------------------------------------------------------------
    # 实际 UI 操作（仅在 UI 线程中调用）
    # ------------------------------------------------------------------

    def _do_show(self, text: str, status: str):
        """显示窗口并设置内容"""
        if not self._window:
            return

        # 更新状态
        self._apply_status(status)

        # 更新文本
        if self._text_label:
            display_text = text if text else ""
            self._text_label.config(text=display_text)

        # 显示窗口
        self._window.deiconify()
        self._window.lift()
        logger.debug("StatusIndicator 显示 (status=%s)", status)

    def _do_hide(self):
        """隐藏窗口"""
        if self._window:
            self._window.withdraw()
            logger.debug("StatusIndicator 隐藏")

    def _do_update_text(self, text: str):
        """更新文本预览"""
        if self._text_label:
            # 截断过长文本，保留最后部分
            max_chars = 200
            if len(text) > max_chars:
                text = "..." + text[-max_chars:]
            self._text_label.config(text=text)

    def _do_update_status(self, status: str):
        """更新状态显示"""
        self._apply_status(status)

    def _apply_status(self, status: str):
        """应用状态样式到 UI"""
        display = _STATUS_DISPLAY.get(status, _STATUS_DISPLAY[STATUS_RECORDING])

        if self._status_label:
            self._status_label.config(
                text=display["text"],
                fg=display["color"],
            )

        # 更新背景色
        bg = display["bg"]
        if self._window:
            self._window.configure(bg=bg)
        if self._frame:
            self._frame.configure(bg=bg)
        if self._status_label:
            self._status_label.configure(bg=bg)
        if self._text_label:
            self._text_label.configure(bg=bg)

    def _do_destroy(self):
        """销毁所有窗口并退出 mainloop"""
        try:
            if self._window:
                self._window.destroy()
                self._window = None
            if self._root:
                self._root.quit()
                self._root.destroy()
                self._root = None
        except Exception:
            pass
        self._running = False
