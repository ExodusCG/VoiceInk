"""
VoiceInk - 设置面板

使用 tkinter 实现标签页式设置界面。
标签页包含：常规、音频、ASR、LLM、输出。
支持保存 / 取消 / 恢复默认操作。
"""

import copy
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from dataclasses import asdict, fields
from typing import Optional, Callable

from voiceink.config import (
    AppConfig, AudioConfig, ASRConfig, LLMConfig, APIConfig,
    OutputConfig, StatusIndicatorConfig,
)
from voiceink.utils.logger import logger


class SettingsPanel:
    """
    设置面板。

    显示一个模态对话框，接受 AppConfig 实例，用户修改后可保存返回新配置。

    用法::

        panel = SettingsPanel(current_config, on_save=lambda cfg: save_config(cfg))
        panel.show()
    """

    def __init__(
        self,
        config: AppConfig,
        on_save: Optional[Callable[[AppConfig], None]] = None,
        parent: Optional[tk.Tk] = None,
    ):
        """
        初始化设置面板。

        Args:
            config:  当前 AppConfig 配置
            on_save: 保存回调，参数为修改后的 AppConfig
            parent:  父窗口，None 则自动创建
        """
        self._original_config = config
        # 深拷贝一份用于编辑
        self._config = copy.deepcopy(config)
        self._on_save = on_save
        self._parent = parent

        # tkinter 变量（延迟初始化）
        self._vars: dict = {}
        self._window: Optional[tk.Toplevel] = None

        logger.info("SettingsPanel 初始化")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def show(self):
        """显示设置面板窗口"""
        # 如果没有父窗口，创建一个隐藏的根窗口
        if self._parent is None:
            self._parent = tk.Tk()
            self._parent.withdraw()

        self._create_window()

    # ------------------------------------------------------------------
    # 窗口创建
    # ------------------------------------------------------------------

    def _create_window(self):
        """创建设置面板主窗口"""
        win = tk.Toplevel(self._parent)
        self._window = win
        win.title("VoiceInk 设置")
        win.geometry("620x520")
        win.resizable(False, False)
        win.grab_set()  # 模态

        # 尝试居中
        win.update_idletasks()
        x = (win.winfo_screenwidth() - 620) // 2
        y = (win.winfo_screenheight() - 520) // 2
        win.geometry(f"+{x}+{y}")

        # ---- 标签页 ----
        notebook = ttk.Notebook(win)
        notebook.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 0))

        # 创建各标签页
        self._create_general_tab(notebook)
        self._create_audio_tab(notebook)
        self._create_asr_tab(notebook)
        self._create_llm_tab(notebook)
        self._create_output_tab(notebook)

        # ---- 底部按钮 ----
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(btn_frame, text="恢复默认", command=self._on_reset_defaults).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="取消", command=self._on_cancel).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="保存", command=self._on_save_click).pack(side=tk.RIGHT)

    # ------------------------------------------------------------------
    # 辅助方法：创建带标签的输入控件
    # ------------------------------------------------------------------

    def _add_row(self, parent, row: int, label_text: str, var, widget_type="entry",
                 options=None, width=30, tooltip=""):
        """
        在 grid 布局中添加一行：标签 + 控件。

        Args:
            parent:      父容器
            row:         行号
            label_text:  标签文字
            var:         tkinter 变量
            widget_type: entry / combo / check / spin / file
            options:     combo 的选项列表
            width:       控件宽度
            tooltip:     提示文字

        Returns:
            创建的控件
        """
        label = ttk.Label(parent, text=label_text)
        label.grid(row=row, column=0, sticky="w", padx=(8, 4), pady=4)

        widget = None
        if widget_type == "entry":
            widget = ttk.Entry(parent, textvariable=var, width=width)
        elif widget_type == "combo":
            widget = ttk.Combobox(parent, textvariable=var, values=options or [],
                                  state="readonly", width=width - 2)
        elif widget_type == "check":
            widget = ttk.Checkbutton(parent, variable=var, text="")
        elif widget_type == "spin":
            widget = ttk.Spinbox(parent, textvariable=var, from_=options[0],
                                 to=options[1], width=10)
        elif widget_type == "file":
            frame = tk.Frame(parent)
            entry = ttk.Entry(frame, textvariable=var, width=width - 6)
            entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
            btn = ttk.Button(
                frame, text="浏览...", width=6,
                command=lambda: var.set(
                    filedialog.askopenfilename(title=f"选择{label_text}") or var.get()
                ),
            )
            btn.pack(side=tk.LEFT, padx=(4, 0))
            widget = frame

        if widget:
            widget.grid(row=row, column=1, sticky="we", padx=(4, 8), pady=4)

        # 提示文字
        if tooltip:
            tip_label = ttk.Label(parent, text=tooltip, foreground="gray")
            tip_label.grid(row=row, column=2, sticky="w", padx=(0, 4))

        return widget

    def _make_var(self, key: str, value, var_type="str"):
        """创建 tkinter 变量并注册到 _vars"""
        if var_type == "str":
            var = tk.StringVar(value=str(value))
        elif var_type == "int":
            var = tk.IntVar(value=int(value))
        elif var_type == "float":
            var = tk.DoubleVar(value=float(value))
        elif var_type == "bool":
            var = tk.BooleanVar(value=bool(value))
        else:
            var = tk.StringVar(value=str(value))
        self._vars[key] = var
        return var

    # ------------------------------------------------------------------
    # 标签页：常规
    # ------------------------------------------------------------------

    def _create_general_tab(self, notebook: ttk.Notebook):
        """创建"常规"标签页"""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="  常规  ")

        frame.columnconfigure(1, weight=1)

        row = 0
        # 语言
        v_lang = self._make_var("language", self._config.language)
        self._add_row(frame, row, "识别语言:", v_lang, "combo",
                      options=["auto", "zh", "en", "ja", "ko", "fr", "de", "es"],
                      tooltip="auto=自动检测")
        row += 1

        # 按住说话快捷键
        v_ptt = self._make_var("hotkey_push_to_talk", self._config.hotkey_push_to_talk)
        self._add_row(frame, row, "按住说话快捷键:", v_ptt, tooltip="例: ctrl+shift+space")
        row += 1

        # 切换快捷键
        v_toggle = self._make_var("hotkey_toggle", self._config.hotkey_toggle)
        self._add_row(frame, row, "切换快捷键:", v_toggle, tooltip="例: ctrl+shift+v")
        row += 1

        # 自动启动
        v_auto = self._make_var("auto_start", self._config.auto_start, "bool")
        self._add_row(frame, row, "开机自启:", v_auto, "check")
        row += 1

        # 日志级别
        v_log = self._make_var("log_level", self._config.log_level)
        self._add_row(frame, row, "日志级别:", v_log, "combo",
                      options=["DEBUG", "INFO", "WARNING", "ERROR"])

    # ------------------------------------------------------------------
    # 标签页：音频
    # ------------------------------------------------------------------

    def _create_audio_tab(self, notebook: ttk.Notebook):
        """创建"音频"标签页"""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="  音频  ")

        frame.columnconfigure(1, weight=1)
        cfg = self._config.audio

        row = 0
        # 设备
        v_dev = self._make_var("audio.device", cfg.device)
        self._add_row(frame, row, "输入设备:", v_dev, tooltip="default=系统默认")
        row += 1

        # 采样率
        v_sr = self._make_var("audio.sample_rate", cfg.sample_rate, "int")
        self._add_row(frame, row, "采样率:", v_sr, "combo",
                      options=["8000", "16000", "22050", "44100", "48000"],
                      tooltip="推荐 16000")
        row += 1

        # VAD 阈值
        v_vad = self._make_var("audio.vad_threshold", cfg.vad_threshold, "float")
        self._add_row(frame, row, "VAD 阈值:", v_vad, "entry", tooltip="0.0~1.0")
        row += 1

        # 静音持续时间
        v_silence = self._make_var("audio.silence_duration_ms", cfg.silence_duration_ms, "int")
        self._add_row(frame, row, "静音判定(ms):", v_silence, "spin",
                      options=[100, 5000], tooltip="静音多久后停止录音")
        row += 1

        # 最大录音时间
        v_max = self._make_var("audio.max_recording_seconds", cfg.max_recording_seconds, "int")
        self._add_row(frame, row, "最大录音时长(秒):", v_max, "spin",
                      options=[10, 600])
        row += 1

        # chunk 时长
        v_chunk = self._make_var("audio.chunk_duration_ms", cfg.chunk_duration_ms, "int")
        self._add_row(frame, row, "Chunk 时长(ms):", v_chunk, "spin",
                      options=[100, 2000], tooltip="音频块大小")

    # ------------------------------------------------------------------
    # 标签页：ASR
    # ------------------------------------------------------------------

    def _create_asr_tab(self, notebook: ttk.Notebook):
        """创建"ASR"标签页"""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="  ASR  ")

        frame.columnconfigure(1, weight=1)
        cfg = self._config.asr

        row = 0
        # 后端
        v_backend = self._make_var("asr.backend", cfg.backend)
        self._add_row(frame, row, "ASR 后端:", v_backend, "combo",
                      options=["whisper_cpp", "faster_whisper"])
        row += 1

        # 模型大小
        v_model_size = self._make_var("asr.model_size", cfg.model_size)
        self._add_row(frame, row, "模型大小:", v_model_size, "combo",
                      options=["tiny", "base", "small", "medium", "large-v2", "large-v3"],
                      tooltip="越大越准，但越慢")
        row += 1

        # 模型路径
        v_model_path = self._make_var("asr.model_path", cfg.model_path)
        self._add_row(frame, row, "模型路径:", v_model_path, "file")
        row += 1

        # 语言
        v_lang = self._make_var("asr.language", cfg.language)
        self._add_row(frame, row, "语言:", v_lang, "combo",
                      options=["auto", "zh", "en", "ja", "ko", "fr", "de"],
                      tooltip="auto=自动检测")
        row += 1

        # 线程数
        v_threads = self._make_var("asr.n_threads", cfg.n_threads, "int")
        self._add_row(frame, row, "线程数:", v_threads, "spin", options=[1, 32])
        row += 1

        # 流式间隔
        v_interval = self._make_var("asr.stream_interval_seconds", cfg.stream_interval_seconds, "float")
        self._add_row(frame, row, "流式间隔(秒):", v_interval, "entry",
                      tooltip="流式识别的时间间隔")
        row += 1

        # Beam size
        v_beam = self._make_var("asr.beam_size", cfg.beam_size, "int")
        self._add_row(frame, row, "Beam Size:", v_beam, "spin", options=[1, 20])

    # ------------------------------------------------------------------
    # 标签页：LLM
    # ------------------------------------------------------------------

    def _create_llm_tab(self, notebook: ttk.Notebook):
        """创建"LLM"标签页"""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="  LLM  ")

        frame.columnconfigure(1, weight=1)
        cfg = self._config.llm

        row = 0
        # 后端
        v_backend = self._make_var("llm.backend", cfg.backend)
        self._add_row(frame, row, "LLM 后端:", v_backend, "combo",
                      options=["llama_cpp", "api", "disabled"])
        row += 1

        # ---- 本地模型设置 ----
        sep1 = ttk.Separator(frame, orient="horizontal")
        sep1.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        lbl_local = ttk.Label(frame, text="── 本地 llama.cpp ──", foreground="gray")
        lbl_local.grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1

        # 模型路径
        v_model_path = self._make_var("llm.model_path", cfg.model_path)
        self._add_row(frame, row, "模型路径:", v_model_path, "file")
        row += 1

        # 模型名称
        v_model_name = self._make_var("llm.model_name", cfg.model_name)
        self._add_row(frame, row, "模型文件名:", v_model_name, tooltip=".gguf 文件名")
        row += 1

        # 上下文长度
        v_ctx = self._make_var("llm.n_ctx", cfg.n_ctx, "int")
        self._add_row(frame, row, "上下文长度:", v_ctx, "spin", options=[512, 32768])
        row += 1

        # 线程数
        v_threads = self._make_var("llm.n_threads", cfg.n_threads, "int")
        self._add_row(frame, row, "线程数:", v_threads, "spin", options=[1, 32])
        row += 1

        # 温度
        v_temp = self._make_var("llm.temperature", cfg.temperature, "float")
        self._add_row(frame, row, "温度:", v_temp, "entry", tooltip="0.0~2.0，越低越确定")
        row += 1

        # top_p
        v_top_p = self._make_var("llm.top_p", cfg.top_p, "float")
        self._add_row(frame, row, "Top-P:", v_top_p, "entry", tooltip="0.0~1.0")
        row += 1

        # ---- 云端 API 设置 ----
        sep2 = ttk.Separator(frame, orient="horizontal")
        sep2.grid(row=row, column=0, columnspan=3, sticky="we", pady=8)
        row += 1

        lbl_api = ttk.Label(frame, text="── 云端 API ──", foreground="gray")
        lbl_api.grid(row=row, column=0, columnspan=2, sticky="w", padx=8)
        row += 1

        # API Base URL
        v_base_url = self._make_var("llm.api.base_url", cfg.api.base_url)
        self._add_row(frame, row, "API Base URL:", v_base_url)
        row += 1

        # API Key
        v_api_key = self._make_var("llm.api.api_key", cfg.api.api_key)
        self._add_row(frame, row, "API Key:", v_api_key, tooltip="留空则从环境变量读取")
        row += 1

        # 模型名
        v_api_model = self._make_var("llm.api.model", cfg.api.model)
        self._add_row(frame, row, "模型名称:", v_api_model, tooltip="如 gpt-4o-mini")

    # ------------------------------------------------------------------
    # 标签页：输出
    # ------------------------------------------------------------------

    def _create_output_tab(self, notebook: ttk.Notebook):
        """创建"输出"标签页"""
        frame = ttk.Frame(notebook, padding=10)
        notebook.add(frame, text="  输出  ")

        frame.columnconfigure(1, weight=1)
        cfg = self._config.output

        row = 0
        # 输出方式
        v_method = self._make_var("output.method", cfg.method)
        self._add_row(frame, row, "输出方式:", v_method, "combo",
                      options=["keyboard", "clipboard", "both"],
                      tooltip="keyboard=模拟键盘输入")
        row += 1

        # 键入延迟
        v_typing = self._make_var("output.typing_delay_ms", cfg.typing_delay_ms, "int")
        self._add_row(frame, row, "键入延迟(ms):", v_typing, "spin",
                      options=[0, 100], tooltip="每个字符间延迟")
        row += 1

        # 粘贴延迟
        v_paste = self._make_var("output.paste_delay_ms", cfg.paste_delay_ms, "int")
        self._add_row(frame, row, "粘贴延迟(ms):", v_paste, "spin",
                      options=[0, 1000], tooltip="粘贴前等待时间")

    # ------------------------------------------------------------------
    # 按钮事件
    # ------------------------------------------------------------------

    def _collect_config(self) -> AppConfig:
        """从 UI 变量收集配置值，生成新的 AppConfig"""
        cfg = copy.deepcopy(self._config)

        # 辅助函数：安全获取变量值
        def get_str(key: str, default: str = "") -> str:
            v = self._vars.get(key)
            return v.get() if v else default

        def get_int(key: str, default: int = 0) -> int:
            v = self._vars.get(key)
            if v is None:
                return default
            try:
                return int(v.get())
            except (ValueError, tk.TclError):
                return default

        def get_float(key: str, default: float = 0.0) -> float:
            v = self._vars.get(key)
            if v is None:
                return default
            try:
                return float(v.get())
            except (ValueError, tk.TclError):
                return default

        def get_bool(key: str, default: bool = False) -> bool:
            v = self._vars.get(key)
            if v is None:
                return default
            try:
                return bool(v.get())
            except (ValueError, tk.TclError):
                return default

        # -- 常规 --
        cfg.language = get_str("language", cfg.language)
        cfg.hotkey_push_to_talk = get_str("hotkey_push_to_talk", cfg.hotkey_push_to_talk)
        cfg.hotkey_toggle = get_str("hotkey_toggle", cfg.hotkey_toggle)
        cfg.auto_start = get_bool("auto_start", cfg.auto_start)
        cfg.log_level = get_str("log_level", cfg.log_level)

        # -- 音频 --
        cfg.audio.device = get_str("audio.device", cfg.audio.device)
        cfg.audio.sample_rate = get_int("audio.sample_rate", cfg.audio.sample_rate)
        cfg.audio.vad_threshold = get_float("audio.vad_threshold", cfg.audio.vad_threshold)
        cfg.audio.silence_duration_ms = get_int("audio.silence_duration_ms", cfg.audio.silence_duration_ms)
        cfg.audio.max_recording_seconds = get_int("audio.max_recording_seconds", cfg.audio.max_recording_seconds)
        cfg.audio.chunk_duration_ms = get_int("audio.chunk_duration_ms", cfg.audio.chunk_duration_ms)

        # -- ASR --
        cfg.asr.backend = get_str("asr.backend", cfg.asr.backend)
        cfg.asr.model_size = get_str("asr.model_size", cfg.asr.model_size)
        cfg.asr.model_path = get_str("asr.model_path", cfg.asr.model_path)
        cfg.asr.language = get_str("asr.language", cfg.asr.language)
        cfg.asr.n_threads = get_int("asr.n_threads", cfg.asr.n_threads)
        cfg.asr.stream_interval_seconds = get_float("asr.stream_interval_seconds", cfg.asr.stream_interval_seconds)
        cfg.asr.beam_size = get_int("asr.beam_size", cfg.asr.beam_size)

        # -- LLM --
        cfg.llm.backend = get_str("llm.backend", cfg.llm.backend)
        cfg.llm.model_path = get_str("llm.model_path", cfg.llm.model_path)
        cfg.llm.model_name = get_str("llm.model_name", cfg.llm.model_name)
        cfg.llm.n_ctx = get_int("llm.n_ctx", cfg.llm.n_ctx)
        cfg.llm.n_threads = get_int("llm.n_threads", cfg.llm.n_threads)
        cfg.llm.temperature = get_float("llm.temperature", cfg.llm.temperature)
        cfg.llm.top_p = get_float("llm.top_p", cfg.llm.top_p)
        cfg.llm.api.base_url = get_str("llm.api.base_url", cfg.llm.api.base_url)
        cfg.llm.api.api_key = get_str("llm.api.api_key", cfg.llm.api.api_key)
        cfg.llm.api.model = get_str("llm.api.model", cfg.llm.api.model)

        # -- 输出 --
        cfg.output.method = get_str("output.method", cfg.output.method)
        cfg.output.typing_delay_ms = get_int("output.typing_delay_ms", cfg.output.typing_delay_ms)
        cfg.output.paste_delay_ms = get_int("output.paste_delay_ms", cfg.output.paste_delay_ms)

        return cfg

    def _on_save_click(self):
        """保存按钮点击"""
        try:
            new_config = self._collect_config()
            logger.info("设置已保存")
            if self._on_save:
                self._on_save(new_config)
            if self._window:
                self._window.destroy()
        except Exception as e:
            logger.error("保存设置失败: %s", e, exc_info=True)
            messagebox.showerror("保存失败", f"保存设置时出错:\n{e}")

    def _on_cancel(self):
        """取消按钮点击"""
        if self._window:
            self._window.destroy()
        logger.info("设置面板已取消")

    def _on_reset_defaults(self):
        """恢复默认按钮点击"""
        result = messagebox.askyesno("恢复默认", "确定要将所有设置恢复为默认值吗？")
        if not result:
            return

        # 使用默认配置重新填充
        default_cfg = AppConfig()
        self._config = default_cfg
        self._update_vars_from_config(default_cfg)
        logger.info("设置已恢复默认值")

    def _update_vars_from_config(self, cfg: AppConfig):
        """根据 AppConfig 更新所有 tkinter 变量"""
        mapping = {
            "language": cfg.language,
            "hotkey_push_to_talk": cfg.hotkey_push_to_talk,
            "hotkey_toggle": cfg.hotkey_toggle,
            "auto_start": cfg.auto_start,
            "log_level": cfg.log_level,
            "audio.device": cfg.audio.device,
            "audio.sample_rate": cfg.audio.sample_rate,
            "audio.vad_threshold": cfg.audio.vad_threshold,
            "audio.silence_duration_ms": cfg.audio.silence_duration_ms,
            "audio.max_recording_seconds": cfg.audio.max_recording_seconds,
            "audio.chunk_duration_ms": cfg.audio.chunk_duration_ms,
            "asr.backend": cfg.asr.backend,
            "asr.model_size": cfg.asr.model_size,
            "asr.model_path": cfg.asr.model_path,
            "asr.language": cfg.asr.language,
            "asr.n_threads": cfg.asr.n_threads,
            "asr.stream_interval_seconds": cfg.asr.stream_interval_seconds,
            "asr.beam_size": cfg.asr.beam_size,
            "llm.backend": cfg.llm.backend,
            "llm.model_path": cfg.llm.model_path,
            "llm.model_name": cfg.llm.model_name,
            "llm.n_ctx": cfg.llm.n_ctx,
            "llm.n_threads": cfg.llm.n_threads,
            "llm.temperature": cfg.llm.temperature,
            "llm.top_p": cfg.llm.top_p,
            "llm.api.base_url": cfg.llm.api.base_url,
            "llm.api.api_key": cfg.llm.api.api_key,
            "llm.api.model": cfg.llm.api.model,
            "output.method": cfg.output.method,
            "output.typing_delay_ms": cfg.output.typing_delay_ms,
            "output.paste_delay_ms": cfg.output.paste_delay_ms,
        }
        for key, value in mapping.items():
            if key in self._vars:
                try:
                    self._vars[key].set(value)
                except (ValueError, tk.TclError):
                    pass
