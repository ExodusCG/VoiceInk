"""
VoiceInk - 应用主入口

VoiceInkApp 类作为整个应用的入口和协调者，负责：
  - 加载配置
  - 初始化各子系统（StatusIndicator、Pipeline、HotkeyManager、SystemTray）
  - 协调系统托盘菜单的回调（ASR/LLM 切换、设置面板、词典面板）
  - 启动时检查并下载所需模型
  - 信号处理与安全退出

使用方式:
    python -m voiceink.main
    或
    python voiceink/main.py
"""

import os
import sys
import json
import signal
import logging
import threading
import tempfile
import atexit
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# 单实例检测 - 在导入其他模块前执行
# ---------------------------------------------------------------------------
def _check_single_instance():
    """
    检查是否已有 VoiceInk 实例在运行。
    使用锁文件 + PID 验证的方式。
    """
    lock_file = Path(tempfile.gettempdir()) / "voiceink.lock"

    # 检查锁文件
    if lock_file.exists():
        try:
            old_pid = int(lock_file.read_text().strip())
            # 检查进程是否存在
            if sys.platform == "win32":
                import ctypes
                kernel32 = ctypes.windll.kernel32
                handle = kernel32.OpenProcess(0x1000, False, old_pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if handle:
                    kernel32.CloseHandle(handle)
                    print(f"[INFO] VoiceInk is already running (PID: {old_pid})")
                    print("       Please check the system tray icon.")
                    sys.exit(0)
            else:
                # Unix: 检查进程是否存在
                try:
                    os.kill(old_pid, 0)
                    print(f"[INFO] VoiceInk is already running (PID: {old_pid})")
                    sys.exit(0)
                except OSError:
                    pass  # 进程不存在
        except (ValueError, FileNotFoundError):
            pass  # 锁文件损坏或不存在

    # 创建锁文件
    lock_file.write_text(str(os.getpid()))

    # 注册退出时清理
    def cleanup_lock():
        try:
            if lock_file.exists():
                lock_file.unlink()
        except Exception:
            pass

    atexit.register(cleanup_lock)

# 执行单实例检测
_check_single_instance()

# ---------------------------------------------------------------------------
# 确保项目根目录在 sys.path 中，以便直接运行此文件时能正确导入
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# VoiceInk 内部模块导入
# ---------------------------------------------------------------------------
from voiceink.config import load_config, save_config, AppConfig, PROJECT_ROOT
from voiceink.utils.logger import setup_logger, logger
from voiceink.utils.model_downloader import ModelDownloader, get_downloader
from voiceink.utils.hotkey import HotkeyManager
from voiceink.ui.status_indicator import StatusIndicator
from voiceink.ui.tray import SystemTray, STATUS_IDLE, STATUS_RECORDING, STATUS_PROCESSING
from voiceink.ui.settings_panel import SettingsPanel
from voiceink.ui.dictionary_panel import DictionaryPanel


# ============================================================
# VoiceInkApp 主应用类
# ============================================================

class VoiceInkApp:
    """
    VoiceInk 应用主入口类。

    作为全局协调者，管理所有子系统的生命周期和交互。

    生命周期::

        app = VoiceInkApp()  # 初始化（加载配置、初始化子系统）
        app.run()             # 启动主循环（阻塞）
        # ... 用户使用中 ...
        app.quit()            # 安全退出

    子系统交互关系::

        [SystemTray]  --回调-->  [VoiceInkApp]  --控制-->  [Pipeline]
                                      |                         |
                                      |                  [AudioCapture]
                                      |                  [ASR Backend]
                                      |                  [LLM Backend]
                                      |                  [TextOutput]
                                      |
                                 [HotkeyManager]  --按键-->  [Pipeline.start/stop]
                                      |
                              [StatusIndicator]  <--状态更新--  [Pipeline 回调]
    """

    def __init__(self):
        """
        初始化 VoiceInk 应用。

        初始化流程：
        1. 加载配置文件
        2. 设置日志级别
        3. 检查/下载模型文件
        4. 初始化 StatusIndicator（悬浮状态窗口）
        5. 初始化 Pipeline（核心处理流水线）— 延迟导入
        6. 连接 Pipeline 回调到 StatusIndicator
        7. 注册全局快捷键（Push-to-Talk）
        8. 初始化 SystemTray（系统托盘）
        """
        logger.info("=" * 60)
        logger.info("VoiceInk 正在启动...")
        logger.info("=" * 60)

        # ---- 1. 加载配置 ----
        self._config: AppConfig = load_config()
        logger.info("配置已加载 (ASR=%s, LLM=%s)", self._config.asr.backend, self._config.llm.backend)

        # ---- 2. 设置日志级别 ----
        log_level = self._config.log_level.upper()
        logging.getLogger("voiceink").setLevel(getattr(logging, log_level, logging.INFO))
        logger.info("日志级别: %s", log_level)

        # ---- 3. 检查/下载模型文件 ----
        self._model_downloader = get_downloader()
        self._ensure_models()

        # ---- 4. 初始化 StatusIndicator（悬浮状态指示器） ----
        self._status_indicator = StatusIndicator(self._config.ui.status_indicator)
        self._status_indicator.start()
        logger.info("StatusIndicator 已启动")

        # ---- 5. 初始化 Pipeline（核心流水线） ----
        # Pipeline 可能还在开发中，使用 try/except 延迟导入
        self._pipeline = None
        self._init_pipeline()

        # ---- 6. 连接 Pipeline 回调到 StatusIndicator ----
        self._connect_pipeline_callbacks()

        # ---- 7. 注册全局快捷键 ----
        self._hotkey_manager = HotkeyManager()
        self._register_hotkeys()

        # ---- 8. 初始化 SystemTray（系统托盘） ----
        self._tray = SystemTray(
            on_asr_change=self.on_asr_change,
            on_llm_change=self.on_llm_change,
            on_open_settings=self.on_open_settings,
            on_open_dictionary=self.on_open_dictionary,
            on_quit=self.on_quit,
            initial_asr=self._config.asr.backend,
            initial_llm=self._config.llm.backend,
        )

        # ---- 退出标志 ----
        self._running = True

        logger.info("VoiceInk 初始化完成 ✓")

    # ==================================================================
    # 模型管理
    # ==================================================================

    def _ensure_models(self):
        """
        检查所需模型文件是否存在，不存在则提示用户下载。

        检查的模型：
        - ASR 模型：根据 config.asr.backend 和 config.asr.model_size 确定
          - sensevoice_onnx: 使用 sensevoice/int8 模型
          - whisper_cpp/faster_whisper: 使用 asr/{model_size} 模型
        - LLM 模型：根据 config.llm.model_name 确定（仅当 LLM 后端不是 "disabled" 或 "api" 时）
        """
        default_model_dir = self._model_downloader.get_default_model_dir()

        # ---- 检查 ASR 模型 ----
        asr_backend = self._config.asr.backend.lower()

        # 根据后端类型确定模型类型和名称
        if asr_backend == "sensevoice_onnx":
            # SenseVoice 使用独立的模型类型
            asr_model_type = "sensevoice"
            asr_model_name = "int8"
            default_asr_dir = str(Path(default_model_dir) / "asr" / "sensevoice")
        else:
            # Whisper 系列使用 asr 模型类型
            asr_model_type = "asr"
            asr_model_name = self._config.asr.model_size  # 如 "base"
            default_asr_dir = str(Path(default_model_dir) / "asr")

        # model_path 可能是目录、文件路径、已损坏的路径或空字符串
        raw_asr_path = self._config.asr.model_path
        asr_model_dir = None  # None 表示已确定文件路径，无需进一步处理

        if raw_asr_path:
            p = Path(raw_asr_path)
            if p.is_file():
                # 已经是完整的、确实存在的模型文件 → 直接使用
                self._config.asr.model_path = str(p)
                logger.info("ASR 模型已就绪（文件路径）: %s", p)
            elif p.is_dir():
                # 是一个存在的目录 → 作为 target_dir
                asr_model_dir = str(p)
            else:
                # 路径不存在，使用默认目录
                logger.warning(
                    "ASR model_path 不存在: %s，使用默认目录: %s",
                    raw_asr_path, default_asr_dir,
                )
                asr_model_dir = default_asr_dir
        else:
            asr_model_dir = default_asr_dir

        if asr_model_dir is not None:
            if not self._model_downloader.check_model_exists(asr_model_type, asr_model_name, asr_model_dir):
                logger.info("ASR 模型不存在，需要下载: %s/%s", asr_model_type, asr_model_name)
                self._download_model_with_progress(asr_model_type, asr_model_name, asr_model_dir)

            # 更新配置中的模型路径
            model_filepath = self._model_downloader.get_model_filepath(asr_model_type, asr_model_name, asr_model_dir)
            if model_filepath:
                self._config.asr.model_path = model_filepath
                logger.info("ASR 模型路径: %s", model_filepath)
            else:
                # 对于多文件模型（如 SenseVoice），路径是目录
                info = self._model_downloader.get_model_info(asr_model_type, asr_model_name)
                if info:
                    model_dir_path = Path(asr_model_dir) / info["filename"]
                    self._config.asr.model_path = str(model_dir_path)
                    logger.info("ASR 模型已就绪: %s", model_dir_path)
                else:
                    logger.info("ASR 模型已就绪: %s (%s)", asr_model_name, asr_model_dir)

        # ---- 检查 LLM 模型（仅本地后端需要） ----
        llm_backend = self._config.llm.backend.lower()
        if llm_backend not in ("disabled", "api"):
            # 从配置的 model_name 中推断模型注册名
            # config.llm.model_name 格式如 "Qwen3-0.6B-Q8_0.gguf"
            # 注册表中的名称不带 .gguf 后缀，且为小写
            llm_model_name_cfg = self._config.llm.model_name
            llm_model_key = llm_model_name_cfg.replace(".gguf", "").lower()
            llm_model_dir = self._config.llm.model_path or str(Path(default_model_dir) / "llm")

            if self._model_downloader.get_model_info("llm", llm_model_key):
                # 模型在注册表中
                if not self._model_downloader.check_model_exists("llm", llm_model_key, llm_model_dir):
                    logger.info("LLM 模型不存在，需要下载: %s", llm_model_key)
                    self._download_model_with_progress("llm", llm_model_key, llm_model_dir)
                    # 更新配置中的模型路径
                    self._config.llm.model_path = llm_model_dir
                    logger.info("LLM 模型路径已更新: %s", llm_model_dir)
                else:
                    logger.info("LLM 模型已就绪: %s (%s)", llm_model_key, llm_model_dir)
            else:
                # 模型不在注册表中（用户自定义模型），仅检查文件是否存在
                custom_path = Path(llm_model_dir) / llm_model_name_cfg
                if not custom_path.exists():
                    logger.warning(
                        "LLM 模型文件不存在: %s (非预定义模型，无法自动下载)",
                        custom_path,
                    )
                else:
                    logger.info("LLM 自定义模型已就绪: %s", custom_path)

    def _download_model_with_progress(
        self,
        model_type: str,
        model_name: str,
        target_dir: str,
    ):
        """
        下载模型文件，在控制台显示下载进度。

        如果下载失败，记录警告但不中断启动流程——用户仍然可以
        在设置面板中手动指定模型路径。

        Args:
            model_type: 模型类型（"asr" / "llm"）
            model_name: 模型名称
            target_dir: 目标目录
        """
        info = self._model_downloader.get_model_info(model_type, model_name)
        if info is None:
            logger.warning("无法下载未知模型: %s/%s", model_type, model_name)
            return

        print(f"\n{'='*60}")
        print(f"  VoiceInk 模型下载")
        print(f"  类型: {model_type.upper()}")
        print(f"  模型: {model_name}")
        print(f"  描述: {info.get('description', '')}")
        print(f"  大小: {self._format_size(info.get('size', 0))}")
        print(f"  目标: {target_dir}")
        print(f"{'='*60}")

        def progress_callback(downloaded: int, total: int, speed: float):
            """在控制台打印下载进度条。"""
            if total > 0:
                percent = (downloaded / total) * 100
                bar_len = 40
                filled = int(bar_len * downloaded / total)
                bar = "█" * filled + "░" * (bar_len - filled)
                print(
                    f"\r  [{bar}] {percent:5.1f}% "
                    f"({self._format_size(downloaded)}/{self._format_size(total)}) "
                    f"{speed:.0f} KB/s",
                    end="", flush=True,
                )
            else:
                print(
                    f"\r  已下载: {self._format_size(downloaded)} ({speed:.0f} KB/s)",
                    end="", flush=True,
                )

        try:
            filepath = self._model_downloader.download_model(
                model_type=model_type,
                model_name=model_name,
                target_dir=target_dir,
                progress_callback=progress_callback,
            )
            print()  # 换行
            print(f"  ✓ 下载完成: {filepath}")
            print()
        except Exception as e:
            print()  # 换行
            print(f"  ✗ 下载失败: {e}")
            print(f"  提示: 可以手动下载模型文件放入 {target_dir}/")
            print()
            logger.error("模型下载失败 (%s/%s): %s", model_type, model_name, e)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """将字节数格式化为人类可读的字符串。"""
        if size_bytes <= 0:
            return "未知大小"
        for unit in ("B", "KB", "MB", "GB", "TB"):
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"

    # ==================================================================
    # Pipeline 初始化
    # ==================================================================

    def _init_pipeline(self):
        """
        尝试初始化核心处理流水线。

        Pipeline 模块可能尚未完成开发，因此使用 try/except 进行保护。
        如果导入失败，应用仍可启动（托盘菜单和设置面板可用），
        但语音输入功能将不可用。
        """
        try:
            from voiceink.core.pipeline import VoiceInkPipeline
            self._pipeline = VoiceInkPipeline(self._config, register_hotkeys=False)
            logger.info("Pipeline 初始化成功")

            # 注意：模型加载延迟到 run() 中启动，避免与 keyboard/pystray 初始化竞争
        except ImportError as e:
            logger.warning(
                "Pipeline 模块尚未完成，语音输入功能暂不可用: %s", e
            )
            self._pipeline = None
        except Exception as e:
            logger.error("Pipeline 初始化失败: %s", e, exc_info=True)
            self._pipeline = None

    def _connect_pipeline_callbacks(self):
        """
        将 Pipeline 的状态回调连接到 StatusIndicator。

        当 Pipeline 状态变化时（录音中、识别中、润色中、完成），
        自动更新悬浮窗口的显示。
        """
        if self._pipeline is None:
            logger.debug("Pipeline 不可用，跳过回调连接")
            return

        # 尝试注册 Pipeline 的回调
        # 预期 Pipeline 提供以下回调注册接口：
        #   pipeline.on_recording_start  -> callable
        #   pipeline.on_recording_stop   -> callable
        #   pipeline.on_transcription    -> callable(text: str)
        #   pipeline.on_polishing        -> callable
        #   pipeline.on_result           -> callable(text: str)
        #   pipeline.on_error            -> callable(error: str)

        try:
            # 录音开始 → 显示悬浮窗（录音状态）
            if hasattr(self._pipeline, 'on_recording_start'):
                self._pipeline.on_recording_start = lambda: (
                    self._status_indicator.show("", "recording"),
                    self._tray.set_status(STATUS_RECORDING) if self._tray else None,
                )

            # 录音结束 → 切换到识别状态
            if hasattr(self._pipeline, 'on_recording_stop'):
                self._pipeline.on_recording_stop = lambda: (
                    self._status_indicator.update_status("recognizing"),
                    self._tray.set_status(STATUS_PROCESSING) if self._tray else None,
                )

            # 流式识别文本更新
            if hasattr(self._pipeline, 'on_transcription'):
                self._pipeline.on_transcription = lambda text: (
                    self._status_indicator.update_text(text),
                )

            # 润色开始
            if hasattr(self._pipeline, 'on_polishing'):
                self._pipeline.on_polishing = lambda: (
                    self._status_indicator.update_status("polishing"),
                )

            # 最终结果输出
            if hasattr(self._pipeline, 'on_result'):
                self._pipeline.on_result = lambda text: (
                    self._status_indicator.show(text, "done"),
                    self._tray.set_status(STATUS_IDLE) if self._tray else None,
                    # 延迟 2 秒后隐藏悬浮窗
                    self._delayed_hide_indicator(2.0),
                )

            # 错误处理
            if hasattr(self._pipeline, 'on_error'):
                self._pipeline.on_error = lambda err: (
                    logger.error("Pipeline 错误: %s", err),
                    self._status_indicator.hide(),
                    self._tray.set_status(STATUS_IDLE) if self._tray else None,
                )

            logger.info("Pipeline 回调已连接到 StatusIndicator")

        except Exception as e:
            logger.error("连接 Pipeline 回调失败: %s", e, exc_info=True)

    def _delayed_hide_indicator(self, delay_seconds: float):
        """
        延迟隐藏状态指示器。

        Args:
            delay_seconds: 延迟秒数
        """
        def _hide():
            import time
            time.sleep(delay_seconds)
            self._status_indicator.hide()

        t = threading.Thread(target=_hide, daemon=True, name="HideIndicator")
        t.start()

    # ==================================================================
    # 快捷键注册
    # ==================================================================

    def _register_hotkeys(self):
        """
        注册全局快捷键。

        当前注册的快捷键：
        - Push-to-Talk (hotkey_push_to_talk): 按住开始录音，释放停止录音
        - Toggle (hotkey_toggle): 按一次开始录音，再按一次停止录音
        """
        ptt_key = self._config.hotkey_push_to_talk
        toggle_key = self._config.hotkey_toggle

        # ---- 注册 Push-to-Talk 快捷键 ----
        if ptt_key:
            if self._pipeline is not None:
                success = self._hotkey_manager.register_push_to_talk(
                    key=ptt_key,
                    on_press=self._on_ptt_press,
                    on_release=self._on_ptt_release,
                )
            else:
                success = self._hotkey_manager.register_push_to_talk(
                    key=ptt_key,
                    on_press=lambda: logger.info("Push-to-Talk 按下 (Pipeline 未就绪)"),
                    on_release=lambda: logger.info("Push-to-Talk 释放 (Pipeline 未就绪)"),
                )

            if success:
                logger.info("Push-to-Talk 快捷键已注册: %s", ptt_key)
            else:
                logger.error("Push-to-Talk 快捷键注册失败: %s", ptt_key)

        # ---- 注册 Toggle 快捷键 ----
        if toggle_key:
            if self._pipeline is not None:
                success = self._hotkey_manager.register_toggle(
                    key=toggle_key,
                    on_start=self._on_toggle_start,
                    on_stop=self._on_toggle_stop,
                )
            else:
                success = self._hotkey_manager.register_toggle(
                    key=toggle_key,
                    on_start=lambda: logger.info("Toggle 开始 (Pipeline 未就绪)"),
                    on_stop=lambda: logger.info("Toggle 停止 (Pipeline 未就绪)"),
                )

            if success:
                logger.info("Toggle 快捷键已注册: %s", toggle_key)
            else:
                logger.error("Toggle 快捷键注册失败: %s", toggle_key)

    def _on_ptt_press(self):
        """
        Push-to-Talk 按下回调：开始录音。
        """
        if self._pipeline is None:
            logger.warning("Pipeline 未初始化，无法开始录音")
            return

        logger.debug("Push-to-Talk 按下 → 开始录音")
        try:
            if hasattr(self._pipeline, 'start_recording'):
                self._pipeline.start_recording()
            elif hasattr(self._pipeline, 'start'):
                self._pipeline.start()
        except Exception as e:
            logger.error("开始录音失败: %s", e, exc_info=True)

    def _on_ptt_release(self):
        """
        Push-to-Talk 释放回调：停止录音并开始处理。
        """
        if self._pipeline is None:
            logger.warning("Pipeline 未初始化，无法停止录音")
            return

        logger.debug("Push-to-Talk 释放 → 停止录音")
        try:
            if hasattr(self._pipeline, 'stop_recording'):
                self._pipeline.stop_recording()
            elif hasattr(self._pipeline, 'stop'):
                self._pipeline.stop()
        except Exception as e:
            logger.error("停止录音失败: %s", e, exc_info=True)

    def _on_toggle_start(self):
        """
        Toggle 模式开始回调：开始录音。
        """
        if self._pipeline is None:
            logger.warning("Pipeline 未初始化，无法开始录音")
            return

        logger.debug("Toggle 开始 → 开始录音")
        try:
            if hasattr(self._pipeline, 'start_recording'):
                self._pipeline.start_recording()
            elif hasattr(self._pipeline, 'start'):
                self._pipeline.start()
        except Exception as e:
            logger.error("开始录音失败: %s", e, exc_info=True)

    def _on_toggle_stop(self):
        """
        Toggle 模式停止回调：停止录音并开始处理。
        """
        if self._pipeline is None:
            logger.warning("Pipeline 未初始化，无法停止录音")
            return

        logger.debug("Toggle 停止 → 停止录音")
        try:
            if hasattr(self._pipeline, 'stop_recording'):
                self._pipeline.stop_recording()
            elif hasattr(self._pipeline, 'stop'):
                self._pipeline.stop()
        except Exception as e:
            logger.error("停止录音失败: %s", e, exc_info=True)

    # ==================================================================
    # 系统托盘回调处理
    # ==================================================================

    def on_asr_change(self, backend_name: str):
        """
        ASR 后端切换回调。

        当用户在系统托盘菜单中切换 ASR 引擎时调用。
        更新配置并通知 Pipeline 重新加载 ASR 后端。

        Args:
            backend_name: 新的 ASR 后端名称（"whisper_cpp" / "faster_whisper"）
        """
        logger.info("切换 ASR 后端: %s -> %s", self._config.asr.backend, backend_name)
        self._config.asr.backend = backend_name
        save_config(self._config)

        # 通知 Pipeline 重新加载 ASR 后端
        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, 'set_asr_backend'):
                    self._pipeline.set_asr_backend(backend_name)
                elif hasattr(self._pipeline, 'reload_asr'):
                    self._pipeline.reload_asr()
                logger.info("ASR 后端已切换为: %s", backend_name)
            except Exception as e:
                logger.error("切换 ASR 后端失败: %s", e, exc_info=True)

    def on_llm_change(self, backend_name: str):
        """
        LLM 后端切换回调。

        当用户在系统托盘菜单中切换 LLM 后端时调用。
        更新配置并通知 Pipeline 重新加载 LLM 后端。

        Args:
            backend_name: 新的 LLM 后端名称（"llama_cpp" / "api" / "disabled"）
        """
        logger.info("切换 LLM 后端: %s -> %s", self._config.llm.backend, backend_name)
        self._config.llm.backend = backend_name
        save_config(self._config)

        # 通知 Pipeline 重新加载 LLM 后端
        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, 'set_llm_backend'):
                    self._pipeline.set_llm_backend(backend_name)
                elif hasattr(self._pipeline, 'reload_llm'):
                    self._pipeline.reload_llm()
                logger.info("LLM 后端已切换为: %s", backend_name)
            except Exception as e:
                logger.error("切换 LLM 后端失败: %s", e, exc_info=True)

    def on_open_settings(self):
        """
        打开设置面板回调。

        在新线程中打开设置面板窗口（因为 tkinter 需要在独立线程中运行，
        避免与系统托盘的主循环冲突）。
        """
        logger.info("打开设置面板")

        def _show_settings():
            """在独立线程中显示设置面板。"""
            try:
                panel = SettingsPanel(
                    config=self._config,
                    on_save=self._on_settings_saved,
                )
                panel.show()
            except Exception as e:
                logger.error("打开设置面板失败: %s", e, exc_info=True)

        thread = threading.Thread(target=_show_settings, daemon=True, name="SettingsPanel")
        thread.start()

    def _on_settings_saved(self, new_config: AppConfig):
        """
        设置面板保存回调。

        当用户在设置面板中点击"保存"时调用。
        更新内存中的配置，保存到文件，并通知各子系统。

        Args:
            new_config: 修改后的新配置
        """
        old_config = self._config
        self._config = new_config

        # 保存配置到文件
        save_config(new_config)
        logger.info("设置已保存")

        # 更新日志级别
        new_level = new_config.log_level.upper()
        logging.getLogger("voiceink").setLevel(getattr(logging, new_level, logging.INFO))

        # 如果 ASR 后端或模型变更了，通知 Pipeline
        if (old_config.asr.backend != new_config.asr.backend
                or old_config.asr.model_size != new_config.asr.model_size):
            self.on_asr_change(new_config.asr.backend)

        # 如果 LLM 后端变更了，通知 Pipeline
        if old_config.llm.backend != new_config.llm.backend:
            self.on_llm_change(new_config.llm.backend)

        # 如果快捷键变更了，重新注册
        hotkey_changed = (
            old_config.hotkey_push_to_talk != new_config.hotkey_push_to_talk
            or old_config.hotkey_toggle != new_config.hotkey_toggle
        )
        if hotkey_changed:
            logger.info("快捷键已变更，重新注册...")
            self._hotkey_manager.unregister_all()
            self._register_hotkeys()

    def on_open_dictionary(self):
        """
        打开词典面板回调。

        加载自定义词典数据，然后在新线程中打开词典管理面板。
        """
        logger.info("打开词典面板")

        def _show_dictionary():
            """在独立线程中显示词典面板。"""
            try:
                # 加载已有词典数据
                entries = self._load_dictionary_entries()

                panel = DictionaryPanel(
                    entries=entries,
                    on_save=self._on_dictionary_saved,
                )
                panel.show()
            except Exception as e:
                logger.error("打开词典面板失败: %s", e, exc_info=True)

        thread = threading.Thread(target=_show_dictionary, daemon=True, name="DictionaryPanel")
        thread.start()

    def _load_dictionary_entries(self) -> list:
        """
        从文件加载词典数据。

        Returns:
            词条字典列表（转换为 DictionaryPanel 需要的格式）
        """
        dict_path = self._config.dictionary.path
        # 支持相对路径（相对于项目根目录）
        if not os.path.isabs(dict_path):
            dict_path = str(PROJECT_ROOT / dict_path)

        if not os.path.exists(dict_path):
            logger.debug("词典文件不存在: %s", dict_path)
            return []

        try:
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 提取词条列表
            raw_entries = []
            if isinstance(data, list):
                raw_entries = data
            elif isinstance(data, dict):
                # 支持 "terms" 或 "entries" 键
                raw_entries = data.get("terms") or data.get("entries") or []

            # 转换为 DictionaryPanel 需要的格式
            entries = []
            for item in raw_entries:
                if isinstance(item, dict):
                    # 兼容两种格式：term/word, aliases 可能是列表或字符串
                    word = item.get("term") or item.get("word") or ""
                    aliases = item.get("aliases", "")
                    if isinstance(aliases, list):
                        aliases = ", ".join(aliases)
                    entries.append({
                        "word": word,
                        "aliases": aliases,
                        "category": item.get("category", "通用"),
                        "enabled": item.get("enabled", True),
                    })

            return entries
        except Exception as e:
            logger.error("加载词典文件失败: %s", e)
            return []

    def _on_dictionary_saved(self, entries: list):
        """
        词典保存回调。

        Args:
            entries: 修改后的词条列表（DictionaryPanel 格式）
        """
        dict_path = self._config.dictionary.path
        if not os.path.isabs(dict_path):
            dict_path = str(PROJECT_ROOT / dict_path)

        try:
            # 确保目录存在
            Path(dict_path).parent.mkdir(parents=True, exist_ok=True)

            # 转换为原始词典格式
            terms = []
            for item in entries:
                aliases = item.get("aliases", "")
                if isinstance(aliases, str):
                    aliases = [a.strip() for a in aliases.split(",") if a.strip()]
                terms.append({
                    "term": item.get("word", ""),
                    "aliases": aliases,
                    "category": item.get("category", "通用"),
                    "pinyin": "",
                    "enabled": item.get("enabled", True),
                })

            # 保存为带版本信息的格式
            data = {
                "version": "1.0",
                "terms": terms,
            }

            with open(dict_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info("词典已保存: %d 条词条 -> %s", len(terms), dict_path)

            # 通知 Pipeline 重新加载词典（如果 Pipeline 支持）
            if self._pipeline is not None and hasattr(self._pipeline, 'reload_dictionary'):
                try:
                    self._pipeline.reload_dictionary()
                except Exception as e:
                    logger.error("重新加载词典失败: %s", e)

        except Exception as e:
            logger.error("保存词典失败: %s", e, exc_info=True)

    def on_quit(self):
        """
        退出回调。

        安全释放所有资源并退出应用：
        1. 停止 Pipeline（释放音频设备、卸载模型）
        2. 注销所有快捷键
        3. 销毁状态指示器
        4. 保存配置
        """
        logger.info("正在退出 VoiceInk...")
        self._running = False

        # 1. 停止 Pipeline
        if self._pipeline is not None:
            try:
                if hasattr(self._pipeline, 'shutdown'):
                    self._pipeline.shutdown()
                elif hasattr(self._pipeline, 'stop'):
                    self._pipeline.stop()
                logger.info("Pipeline 已停止")
            except Exception as e:
                logger.error("停止 Pipeline 时出错: %s", e)

        # 2. 关闭快捷键管理器（注销所有热键 + 停止消息泵线程）
        try:
            self._hotkey_manager.shutdown()
            logger.info("快捷键管理器已关闭")
        except Exception as e:
            logger.error("注销快捷键时出错: %s", e)

        # 3. 销毁状态指示器
        try:
            self._status_indicator.destroy()
            logger.info("StatusIndicator 已销毁")
        except Exception as e:
            logger.error("销毁 StatusIndicator 时出错: %s", e)

        # 4. 保存最终配置
        try:
            save_config(self._config)
            logger.info("配置已保存")
        except Exception as e:
            logger.error("保存配置时出错: %s", e)

        logger.info("VoiceInk 已退出。再见！")

    # ==================================================================
    # 主循环
    # ==================================================================

    def run(self):
        """
        启动 VoiceInk 主循环。

        此方法会阻塞当前线程，直到用户退出应用。
        系统托盘图标在此方法中运行。

        注意：
        - 在 Windows 上，pystray 的 run() 会阻塞主线程
        - 使用 Ctrl+C 可以安全退出
        """
        # 注册信号处理器（Ctrl+C 安全退出）
        self._setup_signal_handlers()

        logger.info("VoiceInk 主循环启动")
        print()
        print("  VoiceInk 语音输入已启动！")
        print(f"  按住 [{self._config.hotkey_push_to_talk}] 说话")
        print("  右键系统托盘图标可打开设置")
        print("  按 Ctrl+C 退出")
        print()

        # 在主线程初始化全部完成后，才启动后台模型加载
        # （避免 llama.cpp 的 C 层内存操作与 keyboard.hook / pystray 竞争导致崩溃）
        if self._pipeline is not None:
            self._pipeline.load_models()
            logger.info("Pipeline 模型加载已启动（后台线程）")

        try:
            # 系统托盘的 run() 是阻塞调用
            # 当用户点击"退出"时，tray.run() 会返回
            self._tray.run()
        except KeyboardInterrupt:
            logger.info("收到 Ctrl+C 信号")
            self.on_quit()
        except Exception as e:
            logger.error("主循环异常: %s", e, exc_info=True)
            self.on_quit()

    def _setup_signal_handlers(self):
        """
        注册操作系统信号处理器。

        处理 SIGINT (Ctrl+C) 和 SIGTERM 信号，确保安全退出。
        """
        def _signal_handler(signum, frame):
            """信号处理回调。"""
            sig_name = signal.Signals(signum).name if hasattr(signal, 'Signals') else str(signum)
            logger.info("收到信号: %s", sig_name)
            # 停止系统托盘（这会导致 tray.run() 返回）
            if hasattr(self, '_tray') and self._tray is not None:
                self._tray.stop()
            self.on_quit()

        # 注册 SIGINT 处理
        signal.signal(signal.SIGINT, _signal_handler)

        # 在非 Windows 平台上注册 SIGTERM
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, _signal_handler)

        logger.debug("信号处理器已注册")


# ============================================================
# 程序入口
# ============================================================

def main():
    """
    VoiceInk 应用入口函数。

    创建 VoiceInkApp 实例并启动主循环。
    """
    try:
        app = VoiceInkApp()
        app.run()
    except KeyboardInterrupt:
        print("\n已退出。")
    except Exception as e:
        logger.critical("VoiceInk 启动失败: %s", e, exc_info=True)
        print(f"\n[错误] VoiceInk 启动失败: {e}")
        print("请查看日志文件获取详细信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
