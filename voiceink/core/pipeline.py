"""
VoiceInk - 核心流水线控制器

统一协调以下子模块，实现从"按下按键"到"文本输出"的完整流水线：
  AudioCapture → ASR → CustomDictionary → LLM → TextOutput

主要职责：
  1. 子模块的创建与生命周期管理
  2. Push-to-Talk 录音控制
  3. 流式识别（录音过程中实时出中间结果）
  4. 段落分割（VAD 静音超阈值时自动切分）
  5. LLM 润色与上下文管理
  6. 热键注册 / 注销
  7. 配置热更新（切换 ASR / LLM 后端）
  8. 回调接口（供 UI 层使用）

线程模型：
  ┌──────────┐    queue     ┌──────────────┐    callback    ┌────┐
  │ 录音线程  │ ──────────► │ 流式识别线程  │ ────────────► │ UI │
  │(sounddev)│  AudioChunk  │ (ASR + 后处理) │  on_partial   │    │
  └──────────┘              └──────────────┘               └────┘
                                   │
                                   ▼ (release 后)
                            ┌──────────────┐
                            │ 最终处理线程  │ → Dict → LLM → TextOutput
                            └──────────────┘
"""

import time
import re
import threading
import queue
import numpy as np
from typing import Optional, Callable, List

from voiceink.config import AppConfig
from voiceink.core.audio_capture import (
    AudioCapture,
    AudioChunk as CoreAudioChunk,
    ChunkType,
)
from voiceink.asr import create_asr_backend, ASRBackend, TranscriptionResult
from voiceink.asr.base import AudioChunk as ASRAudioChunk, ASRState
from voiceink.llm import create_llm_backend
from voiceink.llm.base import LLMBackend, PolishResult, build_polish_prompt
from voiceink.dictionary import CustomDictionary
from voiceink.core.text_output import TextOutput
from voiceink.utils.hotkey import HotkeyManager
from voiceink.utils.logger import setup_logger

# 模块级 logger
logger = setup_logger("voiceink.pipeline")


# ============================================================================
# 流水线状态枚举
# ============================================================================

class PipelineStatus:
    """流水线状态常量（用字符串表示，方便 UI 直接显示）"""
    IDLE = "idle"                       # 空闲，等待用户操作
    LOADING = "loading"                 # 模型加载中
    READY = "ready"                     # 模型已就绪，等待录音
    RECORDING = "recording"             # 正在录音
    RECOGNIZING = "recognizing"         # 正在识别（录音已停止，等待最终结果）
    POLISHING = "polishing"             # LLM 润色中
    OUTPUTTING = "outputting"           # 正在输出文本到焦点窗口
    ERROR = "error"                     # 发生错误


# ============================================================================
# 数据转换工具
# ============================================================================

def _core_chunk_to_asr_chunk(core_chunk: CoreAudioChunk) -> ASRAudioChunk:
    """
    将 core 模块的 AudioChunk 转换为 asr 模块的 AudioChunk。

    core.AudioChunk 字段较丰富（chunk_type, energy, sequence_num 等），
    asr.AudioChunk 只需要 data, is_speech, timestamp_ms 三个字段。

    Args:
        core_chunk: core.audio_capture.AudioChunk 实例

    Returns:
        asr.base.AudioChunk 实例
    """
    return ASRAudioChunk(
        data=core_chunk.data.astype(np.float32) if core_chunk.data.size > 0 else core_chunk.data,
        is_speech=core_chunk.is_speech,
        timestamp_ms=int(core_chunk.timestamp * 1000),  # monotonic 秒 → 毫秒
    )


# ============================================================================
# 核心流水线控制器
# ============================================================================

class VoiceInkPipeline:
    """
    VoiceInk 核心流水线控制器。

    协调音频采集、语音识别、词典纠错、LLM 润色、文本输出等模块，
    为 UI 层提供简洁的高层接口。

    使用示例::

        config = load_config()
        pipeline = VoiceInkPipeline(config)

        # 注册回调
        pipeline.on_status_change = lambda s, t: print(f"[{s}] {t}")
        pipeline.on_partial_result = lambda t: print(f"中间: {t}")
        pipeline.on_final_result = lambda t: print(f"最终: {t}")

        # 加载模型
        pipeline.load_models()

        # Push-to-Talk
        pipeline.on_push_to_talk_press()   # 按下：开始录音
        ...
        pipeline.on_push_to_talk_release() # 松开：停止录音，完成识别→润色→输出

        # 关闭
        pipeline.shutdown()
    """

    def __init__(self, config: AppConfig, register_hotkeys: bool = True):
        """
        初始化流水线控制器，创建所有子模块实例。

        Args:
            config: 全局应用配置对象
            register_hotkeys: 是否由 Pipeline 自行注册快捷键。
                              当由 main.py 管理快捷键时传 False。
        """
        self._config = config
        self._skip_hotkey_registration = not register_hotkeys

        # ──────────────────────────────────────────────
        # 回调函数（UI 层通过赋值注册）
        # ──────────────────────────────────────────────
        self.on_status_change: Optional[Callable[[str, str], None]] = None
        self.on_partial_result: Optional[Callable[[str], None]] = None
        self.on_final_result: Optional[Callable[[str], None]] = None

        # ──────────────────────────────────────────────
        # main.py 兼容回调（由 VoiceInkApp._connect_pipeline_callbacks 设置）
        # 这些回调在 on_push_to_talk_press/release 等流程中被自动调用
        # ──────────────────────────────────────────────
        self.on_recording_start: Optional[Callable[[], None]] = None
        self.on_recording_stop: Optional[Callable[[], None]] = None
        self.on_transcription: Optional[Callable[[str], None]] = None
        self.on_polishing: Optional[Callable[[], None]] = None
        self.on_result: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[str], None]] = None

        # ──────────────────────────────────────────────
        # 流水线状态
        # ──────────────────────────────────────────────
        self._status: str = PipelineStatus.IDLE
        self._status_lock = threading.Lock()

        # ──────────────────────────────────────────────
        # 子模块实例
        # ──────────────────────────────────────────────

        # 1) 音频采集
        self._audio: AudioCapture = AudioCapture(config.audio)

        # 2) ASR 语音识别后端（延迟到 load_models 时才 load_model）
        self._asr: Optional[ASRBackend] = None

        # 3) LLM 润色后端（backend="disabled" 时为 None）
        self._llm: Optional[LLMBackend] = None

        # 4) 自定义词典
        self._dictionary: CustomDictionary = CustomDictionary(config.dictionary)
        self._dictionary.load()

        # 5) 文本输出
        self._text_output: TextOutput = TextOutput(config.output)

        # 6) 全局快捷键管理器
        self._hotkey: HotkeyManager = HotkeyManager()

        # ──────────────────────────────────────────────
        # 流式识别相关
        # ──────────────────────────────────────────────
        # 录音线程往此队列送入 CoreAudioChunk，流式识别线程从中消费
        self._chunk_queue: queue.Queue = queue.Queue(maxsize=1000)

        # 控制录音/识别线程的停止信号
        self._recording_active = threading.Event()     # 正在录音中（set=录音中）
        self._stream_thread: Optional[threading.Thread] = None   # 流式识别线程
        self._collect_thread: Optional[threading.Thread] = None  # 音频收集线程

        # ──────────────────────────────────────────────
        # 段落管理与上下文缓冲
        # ──────────────────────────────────────────────
        # 当前段落累积的音频块（每遇到 BOUNDARY 重置）
        self._segment_chunks: List[CoreAudioChunk] = []
        self._segment_lock = threading.Lock()

        # 已完成的段落文本列表（一次 PTT 会话可能跨多个段落）
        self._completed_segments: List[str] = []
        self._completed_lock = threading.Lock()

        # 上下文缓冲区：保存最近输出的文本，传给 LLM 以保持连贯
        self._context_buffer: str = ""
        self._context_lock = threading.Lock()
        self._max_context_length: int = 600  # 上下文最大字符数

        # ──────────────────────────────────────────────
        # 全局关闭标志
        # ──────────────────────────────────────────────
        self._shutdown_flag = threading.Event()

        # ──────────────────────────────────────────────
        # ASR 转录互斥锁（防止流式线程和最终线程并发调用 transcribe）
        # ──────────────────────────────────────────────
        self._transcribe_lock = threading.Lock()

        logger.info(
            "VoiceInkPipeline 初始化完成 | "
            f"ASR={config.asr.backend}, LLM={config.llm.backend}, "
            f"输出={config.output.method}"
        )

    # ================================================================
    #  状态管理
    # ================================================================

    def _set_status(self, status: str, detail: str = "") -> None:
        """
        更新流水线状态并触发回调通知 UI。

        Args:
            status: PipelineStatus 常量
            detail: 额外说明文本（可选）
        """
        with self._status_lock:
            self._status = status

        logger.debug(f"状态变更: {status} | {detail}")

        # 安全回调
        if self.on_status_change is not None:
            try:
                self.on_status_change(status, detail)
            except Exception as e:
                logger.error(f"on_status_change 回调异常: {e}")

    @property
    def status(self) -> str:
        """当前流水线状态"""
        with self._status_lock:
            return self._status

    # ================================================================
    #  模型加载
    # ================================================================

    def load_models(self) -> bool:
        """
        异步加载 ASR 和 LLM 模型。

        在后台线程中依次加载 ASR 模型和 LLM 模型，加载完成后状态变为 READY。
        加载过程中状态为 LOADING，失败则状态变为 ERROR。

        Returns:
            True 表示已启动加载线程，不代表加载已完成
        """
        self._set_status(PipelineStatus.LOADING, "正在加载模型...")

        def _load_worker():
            """模型加载工作线程"""
            try:
                # ---- 1. 加载 ASR 模型 ----
                self._set_status(PipelineStatus.LOADING, "正在加载 ASR 模型...")
                logger.info(f"开始加载 ASR 模型: backend={self._config.asr.backend}")

                self._asr = create_asr_backend(self._config.asr)
                self._asr.load_model()

                # 注入自定义词典的热词到 ASR 的 initial_prompt
                asr_prompt = self._dictionary.get_asr_prompt()
                if asr_prompt:
                    self._asr.set_initial_prompt(asr_prompt)
                    logger.info(f"已注入 ASR 热词 prompt（{len(asr_prompt)} 字符）")

                logger.info("ASR 模型加载完成")

                # ---- 2. 加载 LLM 模型（如果未禁用） ----
                llm_backend_name = self._config.llm.backend.lower().strip()
                if llm_backend_name not in ("disabled", "none", "off", ""):
                    self._set_status(PipelineStatus.LOADING, "正在加载 LLM 模型...")
                    logger.info(f"开始加载 LLM 模型: backend={self._config.llm.backend}")

                    self._llm = create_llm_backend(self._config.llm)
                    self._llm.load_model()

                    logger.info("LLM 模型加载完成")
                else:
                    self._llm = None
                    logger.info("LLM 润色已禁用，跳过模型加载")

                # ---- 3. 注册热键（仅独立运行时；由 main.py 调用时跳过） ----
                if not self._skip_hotkey_registration:
                    self._register_hotkeys()

                # ---- 加载完成 ----
                self._set_status(PipelineStatus.READY, "模型加载完成，就绪")
                logger.info("所有模型加载完成，流水线就绪")

            except Exception as e:
                logger.error(f"模型加载失败: {e}", exc_info=True)
                self._set_status(PipelineStatus.ERROR, f"模型加载失败: {e}")

        load_thread = threading.Thread(
            target=_load_worker,
            name="VoiceInk-ModelLoader",
            daemon=True,
        )
        load_thread.start()
        return True

    def load_models_sync(self) -> bool:
        """
        同步加载模型（阻塞当前线程直到完成）。

        Returns:
            True 表示加载成功，False 表示失败
        """
        try:
            self._set_status(PipelineStatus.LOADING, "正在加载 ASR 模型...")

            # 加载 ASR
            self._asr = create_asr_backend(self._config.asr)
            self._asr.load_model()

            asr_prompt = self._dictionary.get_asr_prompt()
            if asr_prompt:
                self._asr.set_initial_prompt(asr_prompt)

            # 加载 LLM
            llm_backend_name = self._config.llm.backend.lower().strip()
            if llm_backend_name not in ("disabled", "none", "off", ""):
                self._set_status(PipelineStatus.LOADING, "正在加载 LLM 模型...")
                self._llm = create_llm_backend(self._config.llm)
                self._llm.load_model()
            else:
                self._llm = None

            if not self._skip_hotkey_registration:
                self._register_hotkeys()
            self._set_status(PipelineStatus.READY, "模型加载完成，就绪")
            return True

        except Exception as e:
            logger.error(f"同步模型加载失败: {e}", exc_info=True)
            self._set_status(PipelineStatus.ERROR, f"模型加载失败: {e}")
            return False

    # ================================================================
    #  热键注册
    # ================================================================

    def _register_hotkeys(self) -> None:
        """注册 Push-to-Talk 和其他全局快捷键"""
        ptt_key = self._config.hotkey_push_to_talk
        if ptt_key:
            success = self._hotkey.register_push_to_talk(
                key=ptt_key,
                on_press=self.on_push_to_talk_press,
                on_release=self.on_push_to_talk_release,
            )
            if success:
                logger.info(f"Push-to-Talk 热键已注册: {ptt_key}")
            else:
                logger.warning(f"Push-to-Talk 热键注册失败: {ptt_key}")

    # ================================================================
    #  兼容接口（main.py 使用这些方法名）
    # ================================================================

    def start_recording(self) -> None:
        """开始录音（main.py 兼容接口，等同于 on_push_to_talk_press）"""
        self.on_push_to_talk_press()

    def stop_recording(self) -> None:
        """停止录音（main.py 兼容接口，等同于 on_push_to_talk_release）"""
        self.on_push_to_talk_release()

    def set_asr_backend(self, backend_name: str) -> bool:
        """切换 ASR 后端（main.py 兼容接口，等同于 switch_asr_backend）"""
        return self.switch_asr_backend(backend_name)

    def set_llm_backend(self, backend_name: str) -> bool:
        """切换 LLM 后端（main.py 兼容接口，等同于 switch_llm_backend）"""
        return self.switch_llm_backend(backend_name)

    # ================================================================
    #  Push-to-Talk：按下 → 开始录音 + 流式识别
    # ================================================================

    def on_push_to_talk_press(self) -> None:
        """
        Push-to-Talk 按下回调。

        开始录音，同时启动：
          1. 音频收集线程：从 AudioCapture 获取音频块，放入内部队列
          2. 流式识别线程：从队列消费音频块，定期做中间识别
        """
        # 检查前置条件
        if self._asr is None or not self._asr.is_ready:
            logger.warning("ASR 模型未就绪，忽略 Push-to-Talk 按下")
            self._set_status(PipelineStatus.ERROR, "ASR 模型未就绪")
            return

        if self._recording_active.is_set():
            logger.warning("已在录音中，忽略重复按下")
            return

        logger.info("Push-to-Talk 按下 → 开始录音")

        # 重置会话状态
        self._reset_session()

        # 标记录音活动
        self._recording_active.set()
        self._set_status(PipelineStatus.RECORDING, "正在录音...")

        # 开始音频采集
        if not self._audio.start_recording():
            logger.error("音频采集启动失败")
            self._recording_active.clear()
            self._set_status(PipelineStatus.ERROR, "无法打开麦克风")
            self._fire_error("无法打开麦克风")
            return

        # 触发录音开始回调（供 main.py StatusIndicator 使用）
        self._fire_recording_start()

        # 启动音频收集线程
        self._collect_thread = threading.Thread(
            target=self._audio_collect_worker,
            name="VoiceInk-AudioCollect",
            daemon=True,
        )
        self._collect_thread.start()

        # 启动流式识别线程
        self._stream_thread = threading.Thread(
            target=self._stream_recognition_worker,
            name="VoiceInk-StreamRecog",
            daemon=True,
        )
        self._stream_thread.start()

    def _reset_session(self) -> None:
        """
        重置一次 PTT 会话的临时状态。

        在每次按下 Push-to-Talk 时调用，清理上一次会话的残留。
        """
        # 清空队列
        while not self._chunk_queue.empty():
            try:
                self._chunk_queue.get_nowait()
            except queue.Empty:
                break

        # 清空段落缓冲
        with self._segment_lock:
            self._segment_chunks.clear()

        with self._completed_lock:
            self._completed_segments.clear()

    # ================================================================
    #  Push-to-Talk：松开 → 停止录音 + 最终识别 + 后处理 + 输出
    # ================================================================

    def on_push_to_talk_release(self) -> None:
        """
        Push-to-Talk 松开回调。

        停止录音，等待收集线程结束，然后在后台线程中：
          1. 对最后一段音频做最终 ASR 识别
          2. 词典后处理纠错
          3. LLM 润色（如果启用）
          4. 键盘输出最终文本
        """
        if not self._recording_active.is_set():
            logger.debug("非录音状态下松开，忽略")
            return

        logger.info("Push-to-Talk 松开 → 停止录音")

        # 标记录音停止
        self._recording_active.clear()

        # 触发录音停止回调（供 main.py StatusIndicator 使用）
        self._fire_recording_stop()

        # 停止音频采集（这会向 AudioCapture 的内部队列发送 END 标记）
        self._audio.stop_recording()

        # 在后台线程中执行最终处理（不阻塞热键回调线程）
        finalize_thread = threading.Thread(
            target=self._finalize_worker,
            name="VoiceInk-Finalize",
            daemon=True,
        )
        finalize_thread.start()

    # ================================================================
    #  音频收集线程
    # ================================================================

    def _audio_collect_worker(self) -> None:
        """
        音频收集工作线程。

        从 AudioCapture.stream_chunks() 逐块读取音频数据，
        将有效音频块放入 _chunk_queue 供流式识别线程消费。
        遇到 END 标记或录音停止时退出。
        """
        logger.debug("音频收集线程启动")

        try:
            for chunk in self._audio.stream_chunks(timeout=0.1):
                # 检查全局关闭
                if self._shutdown_flag.is_set():
                    logger.debug("收到关闭信号，音频收集线程退出")
                    break

                if chunk.chunk_type == ChunkType.START:
                    # 录音开始标记，转发到队列
                    self._chunk_queue.put(chunk, timeout=1.0)
                    continue

                elif chunk.chunk_type == ChunkType.END:
                    # 录音结束标记，转发后退出
                    self._chunk_queue.put(chunk, timeout=1.0)
                    logger.debug("收到 END 标记，音频收集线程退出")
                    break

                elif chunk.chunk_type == ChunkType.BOUNDARY:
                    # 段落边界标记
                    self._chunk_queue.put(chunk, timeout=1.0)
                    continue

                else:
                    # 普通音频块或静音块，放入队列
                    try:
                        self._chunk_queue.put(chunk, timeout=0.5)
                    except queue.Full:
                        logger.warning("音频队列已满，丢弃一块数据")

        except Exception as e:
            logger.error(f"音频收集线程异常: {e}", exc_info=True)

        logger.debug("音频收集线程结束")

    # ================================================================
    #  流式识别线程
    # ================================================================

    def _stream_recognition_worker(self) -> None:
        """
        流式识别工作线程。

        从 _chunk_queue 消费音频块，在以下时机触发 ASR 中间识别：
          1. 累积音频时长达到 stream_interval_seconds
          2. 遇到 BOUNDARY（段落边界）→ 对当前段落做最终识别并保存

        中间识别结果通过 on_partial_result 回调通知 UI。
        """
        logger.debug("流式识别线程启动")

        # 流式识别的时间间隔（秒）
        stream_interval = self._config.asr.stream_interval_seconds
        last_recognize_time = time.monotonic()

        # 当前段落累积的音频块
        current_segment_chunks: List[CoreAudioChunk] = []
        # 累积的音频时长（毫秒）
        accumulated_duration_ms: float = 0.0

        try:
            while True:
                # 检查全局关闭
                if self._shutdown_flag.is_set():
                    break

                # 从队列取音频块
                try:
                    chunk: CoreAudioChunk = self._chunk_queue.get(timeout=0.1)
                except queue.Empty:
                    # 队列空，但如果录音已停止且队列确实空了，退出
                    if not self._recording_active.is_set() and self._chunk_queue.empty():
                        logger.debug(
                            "流式识别: 录音已停止且队列空，保存 %d 块后退出",
                            len(current_segment_chunks),
                        )
                        # 保存当前段落供 finalize 使用（与 END 标记同逻辑）
                        with self._segment_lock:
                            self._segment_chunks = list(current_segment_chunks)
                        break
                    continue

                # ---- 处理特殊标记 ----
                if chunk.chunk_type == ChunkType.START:
                    logger.debug("流式识别: 收到 START 标记")
                    continue

                if chunk.chunk_type == ChunkType.END:
                    logger.debug("流式识别: 收到 END 标记")
                    # 将当前段落的音频块存入共享缓冲供 finalize 使用
                    with self._segment_lock:
                        self._segment_chunks = list(current_segment_chunks)
                    break

                if chunk.chunk_type == ChunkType.BOUNDARY:
                    # ---- 段落边界：对当前段落做一次完整识别 ----
                    logger.info("流式识别: 检测到段落边界，处理当前段落")
                    if current_segment_chunks:
                        segment_text = self._recognize_chunks(current_segment_chunks, is_partial=False)
                        if segment_text.strip():
                            # 应用词典纠错
                            segment_text = self._apply_dictionary(segment_text)
                            with self._completed_lock:
                                self._completed_segments.append(segment_text)
                            # 通知 UI 段落识别完成
                            self._fire_partial(segment_text)
                            logger.info(f"段落识别完成: {segment_text[:50]}...")

                    # 重置当前段落
                    current_segment_chunks.clear()
                    accumulated_duration_ms = 0.0
                    last_recognize_time = time.monotonic()
                    continue

                # ---- 普通音频块 / 静音块 ----
                if chunk.data.size > 0:
                    current_segment_chunks.append(chunk)
                    accumulated_duration_ms += chunk.duration_ms

                # ---- 定期触发中间识别 ----
                now = time.monotonic()
                if (now - last_recognize_time) >= stream_interval and current_segment_chunks:
                    partial_text = self._recognize_chunks(current_segment_chunks, is_partial=True)
                    if partial_text.strip():
                        # 组合已完成段落 + 当前段落的中间结果
                        with self._completed_lock:
                            all_segments = list(self._completed_segments)
                        combined = "".join(all_segments) + partial_text
                        self._fire_partial(combined)
                        self._fire_transcription(combined)

                    last_recognize_time = now

        except Exception as e:
            logger.error(f"流式识别线程异常: {e}", exc_info=True)

        logger.debug("流式识别线程结束")

    # ================================================================
    #  最终处理线程
    # ================================================================

    def _finalize_worker(self) -> None:
        """
        最终处理工作线程（在 Push-to-Talk 松开后执行）。

        流程：
          1. 等待音频收集线程和流式识别线程结束
          2. 对最后一个段落做最终 ASR 识别
          3. 合并所有段落文本
          4. 应用词典纠错
          5. LLM 润色（如果启用）
          6. 键盘输出
        """
        logger.info("最终处理线程启动")

        try:
            # ---- 1. 等待收集线程和识别线程结束 ----
            if self._collect_thread is not None and self._collect_thread.is_alive():
                self._collect_thread.join(timeout=5.0)
            if self._stream_thread is not None and self._stream_thread.is_alive():
                self._stream_thread.join(timeout=5.0)

            self._set_status(PipelineStatus.RECOGNIZING, "正在识别...")

            # ---- 2. 对最后一个段落做最终识别 ----
            with self._segment_lock:
                final_segment_chunks = list(self._segment_chunks)
                self._segment_chunks.clear()

            logger.info(
                f"最终段落: {len(final_segment_chunks)} 块"
                + (f", 总采样 {sum(c.data.size for c in final_segment_chunks)}"
                   if final_segment_chunks else "")
            )

            last_segment_text = ""
            if final_segment_chunks:
                last_segment_text = self._recognize_chunks(final_segment_chunks, is_partial=False)
                if last_segment_text.strip():
                    last_segment_text = self._apply_dictionary(last_segment_text)

            # ---- 3. 合并所有段落文本 ----
            with self._completed_lock:
                all_segments = list(self._completed_segments)

            logger.info(
                f"已完成段落: {len(all_segments)}, "
                f"最终段落文本: '{last_segment_text[:60]}'"
            )

            if last_segment_text.strip():
                all_segments.append(last_segment_text)

            full_text = "".join(all_segments)

            # 过滤 Whisper 特殊标记（非实际语音内容）
            full_text = self._clean_asr_artifacts(full_text)

            if not full_text.strip():
                logger.info("识别结果为空，跳过输出")
                self._set_status(PipelineStatus.READY, "就绪（无识别内容）")
                return

            logger.info(f"ASR 识别完成: {full_text[:80]}...")

            # ---- 4. LLM 润色 ----
            final_text = full_text
            if self._llm is not None and self._llm.is_loaded():
                self._set_status(PipelineStatus.POLISHING, "正在润色...")
                self._fire_polishing()
                final_text = self._polish_text(full_text)
            else:
                logger.debug("LLM 未启用或未加载，跳过润色")

            # ---- 5. 键盘输出 ----
            self._set_status(PipelineStatus.OUTPUTTING, "正在输出...")
            self._text_output.type_text(final_text)

            # ---- 6. 更新上下文缓冲 ----
            self._update_context(final_text)

            # ---- 7. 通知 UI 最终结果 ----
            self._fire_final(final_text)
            self._fire_result(final_text)

            logger.info(f"输出完成: {final_text[:80]}...")
            self._set_status(PipelineStatus.READY, "就绪")

        except Exception as e:
            logger.error(f"最终处理线程异常: {e}", exc_info=True)
            self._set_status(PipelineStatus.ERROR, f"处理失败: {e}")
            self._fire_error(f"处理失败: {e}")

    # ================================================================
    #  ASR 识别辅助方法
    # ================================================================

    def _recognize_chunks(
        self,
        chunks: List[CoreAudioChunk],
        is_partial: bool = False,
    ) -> str:
        """
        将一组音频块合并后送入 ASR 进行识别。

        对于中间识别（is_partial=True），使用 transcribe() 对合并后的完整音频识别。
        对于最终识别（is_partial=False），同样使用 transcribe() 以获得最佳准确率。

        Args:
            chunks:     core.AudioChunk 列表
            is_partial: 是否为中间结果（影响日志，不影响识别逻辑）

        Returns:
            识别出的文本
        """
        if not chunks:
            return ""

        if self._asr is None or not self._asr.is_ready:
            logger.warning("ASR 未就绪，无法识别")
            return ""

        try:
            # 合并所有音频块数据为一个 numpy 数组
            audio_arrays = [
                chunk.data for chunk in chunks
                if chunk.data.size > 0
            ]
            if not audio_arrays:
                logger.warning("所有音频块数据为空，无法识别")
                return ""

            merged_audio = np.concatenate(audio_arrays)
            rms = float(np.sqrt(np.mean(merged_audio ** 2)))
            peak = float(np.max(np.abs(merged_audio)))

            tag = "中间" if is_partial else "最终"
            logger.info(
                f"ASR {tag}输入: {len(merged_audio)} 采样 "
                f"({len(merged_audio)/16000:.2f}s), "
                f"RMS={rms:.4f}, Peak={peak:.4f}"
            )

            # 调用 ASR transcribe（加锁防止流式线程和最终线程并发调用）
            with self._transcribe_lock:
                result: TranscriptionResult = self._asr.transcribe(merged_audio)

            if result.text.strip():
                logger.info(f"ASR {tag}识别结果: '{result.text[:80]}'")
            else:
                logger.info(f"ASR {tag}识别结果为空")

            return result.text

        except Exception as e:
            logger.error(f"ASR 识别异常: {e}", exc_info=True)
            return ""

    # Whisper / Paraformer 模型可能输出的特殊标记（非实际语音内容），需在后续处理前过滤
    _ASR_ARTIFACT_PATTERNS = re.compile(
        r"\[BLANK_AUDIO\]"
        r"|\[MUSIC\]"
        r"|\[APPLAUSE\]"
        r"|\[LAUGHTER\]"
        r"|\[NOISE\]"
        r"|\[SILENCE\]"
        r"|\(BLANK_AUDIO\)"
        r"|\*BLANK_AUDIO\*"
        # Paraformer artifacts
        r"|<\|[\w]+\|>"          # e.g. <|zh|>, <|en|>, <|NOISE|>
        r"|<sil>"                # Paraformer silence token
        r"|<blank>",             # Paraformer blank token
        re.IGNORECASE,
    )

    @classmethod
    def _clean_asr_artifacts(cls, text: str) -> str:
        """
        清除 ASR 输出中的特殊标记。

        Whisper 在静音或非语音段会输出 [BLANK_AUDIO]、[MUSIC] 等标记，
        Paraformer 可能输出 <sil>、<blank>、<|zh|> 等标记。
        这些不应作为文本发送给 LLM 润色或键盘输出。

        Args:
            text: ASR 原始输出文本

        Returns:
            清理后的文本（移除特殊标记后 strip）
        """
        if not text:
            return text
        cleaned = cls._ASR_ARTIFACT_PATTERNS.sub("", text)
        return cleaned.strip()

    # ================================================================
    #  词典后处理
    # ================================================================

    def _apply_dictionary(self, text: str) -> str:
        """
        使用自定义词典对文本做后处理纠错。

        Args:
            text: ASR 输出的原始文本

        Returns:
            纠错后的文本
        """
        if not text:
            return text

        try:
            corrected = self._dictionary.apply_corrections(text)
            if corrected != text:
                logger.debug(f"词典纠错: '{text[:40]}...' → '{corrected[:40]}...'")
            return corrected
        except Exception as e:
            logger.error(f"词典纠错异常: {e}")
            return text  # 出错时返回原文

    # ================================================================
    #  LLM 润色
    # ================================================================

    def _polish_text(self, raw_text: str) -> str:
        """
        使用 LLM 对文本进行润色。

        如果润色失败，回退到原文（保证不丢失用户输入）。

        Args:
            raw_text: 待润色的文本

        Returns:
            润色后的文本（失败则返回原文）
        """
        if not raw_text or not raw_text.strip():
            return raw_text

        if self._llm is None or not self._llm.is_loaded():
            return raw_text

        try:
            # 获取上下文
            with self._context_lock:
                context = self._context_buffer if self._context_buffer else None

            # 获取词典中的自定义词汇列表（供 LLM 参考）
            custom_terms = self._dictionary.get_active_terms()

            # 调用 LLM 润色（使用安全方法，失败自动回退原文）
            result: PolishResult = self._llm.polish_or_passthrough(
                raw_text=raw_text,
                context=context,
                custom_terms=custom_terms if custom_terms else None,
            )

            if result.success:
                logger.debug(f"LLM 润色成功: '{raw_text[:30]}...' → '{result.text[:30]}...'")
            else:
                logger.warning(f"LLM 润色失败（已回退原文）: {result.error}")

            return result.text

        except Exception as e:
            logger.error(f"LLM 润色异常: {e}", exc_info=True)
            return raw_text  # 异常时返回原文

    # ================================================================
    #  上下文管理
    # ================================================================

    def _update_context(self, text: str) -> None:
        """
        将新输出的文本追加到上下文缓冲区。

        缓冲区有最大长度限制，超出时截断最早的内容。

        Args:
            text: 新输出的文本
        """
        if not text:
            return

        with self._context_lock:
            self._context_buffer += text

            # 截断：只保留最后 max_context_length 个字符
            if len(self._context_buffer) > self._max_context_length:
                self._context_buffer = self._context_buffer[
                    -self._max_context_length:
                ]

    def clear_context(self) -> None:
        """清空上下文缓冲区（例如用户切换话题时调用）"""
        with self._context_lock:
            self._context_buffer = ""
        logger.info("上下文缓冲区已清空")

    # ================================================================
    #  回调触发辅助方法
    # ================================================================

    def _fire_partial(self, text: str) -> None:
        """安全触发中间结果回调"""
        if self.on_partial_result is not None:
            try:
                self.on_partial_result(text)
            except Exception as e:
                logger.error(f"on_partial_result 回调异常: {e}")

    def _fire_final(self, text: str) -> None:
        """安全触发最终结果回调"""
        if self.on_final_result is not None:
            try:
                self.on_final_result(text)
            except Exception as e:
                logger.error(f"on_final_result 回调异常: {e}")

    def _fire_recording_start(self) -> None:
        """安全触发录音开始回调（main.py 兼容）"""
        if self.on_recording_start is not None:
            try:
                self.on_recording_start()
            except Exception as e:
                logger.error(f"on_recording_start 回调异常: {e}")

    def _fire_recording_stop(self) -> None:
        """安全触发录音停止回调（main.py 兼容）"""
        if self.on_recording_stop is not None:
            try:
                self.on_recording_stop()
            except Exception as e:
                logger.error(f"on_recording_stop 回调异常: {e}")

    def _fire_transcription(self, text: str) -> None:
        """安全触发转录文本回调（main.py 兼容）"""
        if self.on_transcription is not None:
            try:
                self.on_transcription(text)
            except Exception as e:
                logger.error(f"on_transcription 回调异常: {e}")

    def _fire_polishing(self) -> None:
        """安全触发润色开始回调（main.py 兼容）"""
        if self.on_polishing is not None:
            try:
                self.on_polishing()
            except Exception as e:
                logger.error(f"on_polishing 回调异常: {e}")

    def _fire_result(self, text: str) -> None:
        """安全触发最终结果回调（main.py 兼容）"""
        if self.on_result is not None:
            try:
                self.on_result(text)
            except Exception as e:
                logger.error(f"on_result 回调异常: {e}")

    def _fire_error(self, error: str) -> None:
        """安全触发错误回调（main.py 兼容）"""
        if self.on_error is not None:
            try:
                self.on_error(error)
            except Exception as e:
                logger.error(f"on_error 回调异常: {e}")

    # ================================================================
    #  配置热更新
    # ================================================================

    def switch_asr_backend(self, new_backend: Optional[str] = None) -> bool:
        """
        热切换 ASR 后端。

        卸载当前 ASR 模型，使用新配置重新创建并加载。
        切换过程中流水线暂时不可用（状态为 LOADING）。

        Args:
            new_backend: 新的 ASR 后端名称（如 "whisper_cpp"、"faster_whisper"）。
                         为 None 时使用配置文件中的当前值。

        Returns:
            True 表示切换成功，False 表示失败
        """
        # 确保不在录音中
        if self._recording_active.is_set():
            logger.warning("正在录音中，无法切换 ASR 后端")
            return False

        self._set_status(PipelineStatus.LOADING, "正在切换 ASR 后端...")

        try:
            # 卸载旧模型
            if self._asr is not None:
                try:
                    self._asr.unload()
                except Exception as e:
                    logger.warning(f"卸载旧 ASR 模型时出错: {e}")

            # 更新配置
            if new_backend is not None:
                self._config.asr.backend = new_backend

            # 创建并加载新模型
            self._asr = create_asr_backend(self._config.asr)
            self._asr.load_model()

            # 重新注入热词
            asr_prompt = self._dictionary.get_asr_prompt()
            if asr_prompt:
                self._asr.set_initial_prompt(asr_prompt)

            self._set_status(PipelineStatus.READY, f"ASR 后端已切换为 {self._config.asr.backend}")
            logger.info(f"ASR 后端切换完成: {self._config.asr.backend}")
            return True

        except Exception as e:
            logger.error(f"切换 ASR 后端失败: {e}", exc_info=True)
            self._set_status(PipelineStatus.ERROR, f"切换 ASR 失败: {e}")
            return False

    def switch_llm_backend(self, new_backend: Optional[str] = None) -> bool:
        """
        热切换 LLM 后端。

        卸载当前 LLM 模型，使用新配置重新创建并加载。
        设置为 "disabled" 时卸载模型并禁用润色。

        Args:
            new_backend: 新的 LLM 后端名称（如 "llama_cpp"、"api"、"disabled"）。
                         为 None 时使用配置文件中的当前值。

        Returns:
            True 表示切换成功，False 表示失败
        """
        # 确保不在录音中
        if self._recording_active.is_set():
            logger.warning("正在录音中，无法切换 LLM 后端")
            return False

        self._set_status(PipelineStatus.LOADING, "正在切换 LLM 后端...")

        try:
            # 卸载旧模型
            if self._llm is not None:
                try:
                    self._llm.unload()
                except Exception as e:
                    logger.warning(f"卸载旧 LLM 模型时出错: {e}")
                self._llm = None

            # 更新配置
            if new_backend is not None:
                self._config.llm.backend = new_backend

            # 判断是否禁用
            llm_backend_name = self._config.llm.backend.lower().strip()
            if llm_backend_name in ("disabled", "none", "off", ""):
                self._llm = None
                self._set_status(PipelineStatus.READY, "LLM 润色已禁用")
                logger.info("LLM 润色已禁用")
                return True

            # 创建并加载新模型
            self._llm = create_llm_backend(self._config.llm)
            self._llm.load_model()

            self._set_status(PipelineStatus.READY, f"LLM 后端已切换为 {self._config.llm.backend}")
            logger.info(f"LLM 后端切换完成: {self._config.llm.backend}")
            return True

        except Exception as e:
            logger.error(f"切换 LLM 后端失败: {e}", exc_info=True)
            self._llm = None  # 确保失败后不会使用损坏的实例
            self._set_status(PipelineStatus.ERROR, f"切换 LLM 失败: {e}")
            return False

    def reload_dictionary(self) -> None:
        """重新加载自定义词典，并更新 ASR 的 initial_prompt"""
        try:
            self._dictionary.load()
            logger.info("自定义词典重新加载完成")

            # 更新 ASR 热词
            if self._asr is not None and self._asr.is_ready:
                asr_prompt = self._dictionary.get_asr_prompt()
                self._asr.set_initial_prompt(asr_prompt)
                logger.info(f"ASR 热词已更新（{len(asr_prompt)} 字符）")

        except Exception as e:
            logger.error(f"重新加载词典失败: {e}")

    # ================================================================
    #  属性访问
    # ================================================================

    @property
    def config(self) -> AppConfig:
        """当前应用配置"""
        return self._config

    @property
    def audio(self) -> AudioCapture:
        """音频采集器实例"""
        return self._audio

    @property
    def asr(self) -> Optional[ASRBackend]:
        """ASR 后端实例"""
        return self._asr

    @property
    def llm(self) -> Optional[LLMBackend]:
        """LLM 后端实例"""
        return self._llm

    @property
    def dictionary(self) -> CustomDictionary:
        """自定义词典实例"""
        return self._dictionary

    @property
    def text_output(self) -> TextOutput:
        """文本输出器实例"""
        return self._text_output

    @property
    def hotkey(self) -> HotkeyManager:
        """热键管理器实例"""
        return self._hotkey

    @property
    def is_recording(self) -> bool:
        """是否正在录音"""
        return self._recording_active.is_set()

    @property
    def is_ready(self) -> bool:
        """流水线是否就绪（ASR 模型已加载）"""
        return (
            self._asr is not None
            and self._asr.is_ready
            and self.status == PipelineStatus.READY
        )

    @property
    def context_buffer(self) -> str:
        """当前上下文缓冲内容"""
        with self._context_lock:
            return self._context_buffer

    # ================================================================
    #  资源管理
    # ================================================================

    def shutdown(self) -> None:
        """
        关闭流水线，释放所有资源。

        释放顺序：
          1. 设置关闭标志，停止录音
          2. 等待所有工作线程结束
          3. 注销所有热键
          4. 卸载 LLM 模型
          5. 卸载 ASR 模型
          6. 关闭音频采集器
        """
        logger.info("VoiceInkPipeline 开始关闭...")

        # ---- 1. 设置全局关闭标志 ----
        self._shutdown_flag.set()

        # 停止录音（如果正在进行）
        if self._recording_active.is_set():
            self._recording_active.clear()
            try:
                self._audio.stop_recording()
            except Exception as e:
                logger.warning(f"关闭时停止录音出错: {e}")

        # ---- 2. 等待工作线程结束 ----
        for thread, name in [
            (self._collect_thread, "AudioCollect"),
            (self._stream_thread, "StreamRecog"),
        ]:
            if thread is not None and thread.is_alive():
                logger.debug(f"等待 {name} 线程结束...")
                thread.join(timeout=3.0)
                if thread.is_alive():
                    logger.warning(f"{name} 线程未在超时内结束")

        # ---- 3. 注销热键 ----
        try:
            self._hotkey.unregister_all()
            logger.debug("所有热键已注销")
        except Exception as e:
            logger.warning(f"注销热键时出错: {e}")

        # ---- 4. 卸载 LLM 模型 ----
        if self._llm is not None:
            try:
                self._llm.unload()
                logger.debug("LLM 模型已卸载")
            except Exception as e:
                logger.warning(f"卸载 LLM 模型时出错: {e}")
            self._llm = None

        # ---- 5. 卸载 ASR 模型 ----
        if self._asr is not None:
            try:
                self._asr.unload()
                logger.debug("ASR 模型已卸载")
            except Exception as e:
                logger.warning(f"卸载 ASR 模型时出错: {e}")
            self._asr = None

        # ---- 6. 关闭音频采集器 ----
        try:
            self._audio.close()
            logger.debug("音频采集器已关闭")
        except Exception as e:
            logger.warning(f"关闭音频采集器时出错: {e}")

        self._set_status(PipelineStatus.IDLE, "已关闭")
        logger.info("VoiceInkPipeline 已完全关闭")

    def __enter__(self):
        """支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 语句退出时自动关闭"""
        self.shutdown()
        return False

    def __del__(self):
        """析构时确保资源释放"""
        try:
            if not self._shutdown_flag.is_set():
                self.shutdown()
        except Exception:
            pass


# ============================================================================
# 模块公开接口
# ============================================================================

__all__ = [
    "VoiceInkPipeline",
    "Pipeline",
    "PipelineStatus",
]

# 兼容别名：main.py 使用 `from voiceink.core.pipeline import Pipeline`
Pipeline = VoiceInkPipeline
