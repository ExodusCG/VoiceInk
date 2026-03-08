"""
VoiceInk - 音频采集模块

使用 sounddevice 进行实时音频采集，支持：
- Push-to-Talk 模式（手动控制开始/停止录音）
- 基于能量的 VAD（语音活动检测），自动检测语音停顿并标记段落边界
- 流式音频块输出（generator 模式）
- 线程安全的并发操作
- 音频设备枚举与选择
"""

import time
import threading
import queue
import numpy as np
import sounddevice as sd
from dataclasses import dataclass, field
from typing import Optional, Generator, List, Dict, Any
from enum import Enum, auto

from voiceink.config import AudioConfig
from voiceink.utils.logger import setup_logger

# 模块级 logger
logger = setup_logger("voiceink.audio_capture")


# ============================================================
#  音频工具函数
# ============================================================

def _resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """
    使用线性插值将音频从 src_rate 重采样到 dst_rate。

    此实现纯 numpy，无需 scipy，适合音频回调中的低延迟场景。
    对于语音信号（<8kHz 带宽），线性插值的质量完全满足 ASR 输入需求。

    Args:
        audio:    1-D float32 音频数据
        src_rate: 源采样率 (Hz)
        dst_rate: 目标采样率 (Hz)

    Returns:
        重采样后的 1-D float32 音频数据
    """
    if src_rate == dst_rate or len(audio) == 0:
        return audio

    # 计算输出长度
    duration = len(audio) / src_rate
    out_len = int(duration * dst_rate)
    if out_len == 0:
        return np.array([], dtype=np.float32)

    # numpy 线性插值
    x_old = np.linspace(0, 1, len(audio), endpoint=False)
    x_new = np.linspace(0, 1, out_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


# ============================================================
#  数据结构定义
# ============================================================

class ChunkType(Enum):
    """音频块类型枚举"""
    AUDIO = auto()           # 普通音频数据
    SILENCE = auto()         # 静音块
    BOUNDARY = auto()        # 段落边界（VAD 检测到语音停顿超过阈值）
    START = auto()           # 录音开始标记
    END = auto()             # 录音结束标记


@dataclass
class AudioChunk:
    """
    音频块数据结构

    每个 AudioChunk 代表一小段音频数据及其元信息，
    用于在音频采集与下游 ASR 之间传递数据。

    Attributes:
        data:           原始音频数据，numpy float32 数组，值域 [-1.0, 1.0]
        timestamp:      该块被采集到的时间戳（time.monotonic()）
        chunk_type:     块类型（音频 / 静音 / 段落边界 / 开始 / 结束）
        sample_rate:    采样率（Hz）
        channels:       声道数
        duration_ms:    该块时长（毫秒）
        energy:         该块的 RMS 能量值（用于 VAD 判断）
        is_speech:      VAD 判定：该块是否包含语音
        sequence_num:   块序列号（从 0 开始递增，每次录音重置）
    """
    data: np.ndarray                        # 音频采样数据 (float32)
    timestamp: float = 0.0                  # 采集时间戳
    chunk_type: ChunkType = ChunkType.AUDIO # 块类型
    sample_rate: int = 16000                # 采样率
    channels: int = 1                       # 声道数
    duration_ms: float = 0.0                # 块时长（毫秒）
    energy: float = 0.0                     # RMS 能量
    is_speech: bool = False                 # 是否为语音
    sequence_num: int = 0                   # 序列号


# ============================================================
#  VAD：基于能量的语音活动检测器
# ============================================================

class EnergyVAD:
    """
    基于能量的语音活动检测器（Energy-based Voice Activity Detector）

    工作原理：
    1. 计算每个音频块的 RMS（均方根）能量
    2. 维护一个自适应的背景噪声估计（使用指数移动平均）
    3. 当能量超过 "噪声底 × 倍数 + 固定阈值" 时判定为语音
    4. 使用 hangover 机制避免语音中短暂停顿导致的误判

    参数说明：
    - energy_threshold:    固定能量阈值，低于此值一定判定为静音
    - noise_multiplier:    噪声倍数，能量需要高于 噪声底 × 此倍数 才判定为语音
    - noise_smooth_factor: 噪声估计的平滑因子（指数移动平均的 alpha）
    - hangover_chunks:     语音结束后的挂起块数（防止短暂停顿误判）
    """

    def __init__(
        self,
        energy_threshold: float = 0.01,
        noise_multiplier: float = 3.0,
        noise_smooth_factor: float = 0.05,
        hangover_chunks: int = 3,
    ):
        self._energy_threshold = energy_threshold
        self._noise_multiplier = noise_multiplier
        self._noise_smooth_factor = noise_smooth_factor
        self._hangover_chunks = hangover_chunks

        # 内部状态
        self._noise_estimate: float = 0.0       # 背景噪声能量估计
        self._is_initialized: bool = False       # 是否已完成初始化校准
        self._init_frames: List[float] = []      # 初始化阶段收集的能量值
        self._init_frame_count: int = 10         # 初始化需要的帧数
        self._hangover_counter: int = 0          # 挂起计数器
        self._is_speech: bool = False             # 当前是否处于语音状态

        # 锁（保护内部状态的线程安全）
        self._lock = threading.Lock()

    def reset(self) -> None:
        """重置 VAD 状态，用于新一轮录音开始时"""
        with self._lock:
            self._noise_estimate = 0.0
            self._is_initialized = False
            self._init_frames.clear()
            self._hangover_counter = 0
            self._is_speech = False

    @staticmethod
    def compute_rms_energy(audio_data: np.ndarray) -> float:
        """
        计算音频数据的 RMS（均方根）能量

        Args:
            audio_data: float32 音频数组

        Returns:
            RMS 能量值
        """
        if audio_data.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(audio_data.astype(np.float64) ** 2)))

    def process(self, audio_data: np.ndarray) -> tuple:
        """
        对一块音频数据进行 VAD 判定

        Args:
            audio_data: float32 音频数组

        Returns:
            (is_speech, energy) - 是否为语音，以及该块的 RMS 能量
        """
        energy = self.compute_rms_energy(audio_data)

        with self._lock:
            # ---- 初始化阶段：收集前几帧用于估计背景噪声 ----
            if not self._is_initialized:
                self._init_frames.append(energy)
                if len(self._init_frames) >= self._init_frame_count:
                    # 取中位数作为初始噪声估计（比均值更鲁棒）
                    self._noise_estimate = float(np.median(self._init_frames))
                    self._is_initialized = True
                    logger.debug(
                        f"VAD 初始化完成，背景噪声估计: {self._noise_estimate:.6f}"
                    )
                # 初始化阶段保守地判定为无语音
                return False, energy

            # ---- 正常检测阶段 ----
            # 动态阈值 = max(固定阈值, 噪声底 × 倍数)
            dynamic_threshold = max(
                self._energy_threshold,
                self._noise_estimate * self._noise_multiplier,
            )

            if energy > dynamic_threshold:
                # 能量超过阈值 → 判定为语音
                self._is_speech = True
                self._hangover_counter = self._hangover_chunks
            else:
                # 能量低于阈值
                if self._hangover_counter > 0:
                    # 挂起期间仍判定为语音（防止短暂停顿误判）
                    self._hangover_counter -= 1
                else:
                    self._is_speech = False

                # 仅在非语音状态下更新噪声估计（避免语音污染噪声底）
                if not self._is_speech:
                    self._noise_estimate = (
                        (1 - self._noise_smooth_factor) * self._noise_estimate
                        + self._noise_smooth_factor * energy
                    )

            return self._is_speech, energy

    @property
    def noise_estimate(self) -> float:
        """返回当前噪声估计值"""
        with self._lock:
            return self._noise_estimate

    @property
    def is_initialized(self) -> bool:
        """是否已完成初始化校准"""
        with self._lock:
            return self._is_initialized


# ============================================================
#  音频设备工具函数
# ============================================================

def list_audio_devices() -> List[Dict[str, Any]]:
    """
    枚举系统中所有可用的音频输入设备

    Returns:
        设备信息列表，每个元素是一个字典，包含：
        - index:             设备索引号
        - name:              设备名称
        - max_input_channels: 最大输入声道数
        - default_samplerate: 默认采样率
        - is_default:        是否为系统默认输入设备
    """
    devices = sd.query_devices()
    default_input = sd.default.device[0]  # 默认输入设备索引

    input_devices = []
    for i, dev in enumerate(devices):
        # 只列出有输入能力的设备（max_input_channels > 0）
        if dev["max_input_channels"] > 0:
            input_devices.append({
                "index": i,
                "name": dev["name"],
                "max_input_channels": dev["max_input_channels"],
                "default_samplerate": dev["default_samplerate"],
                "is_default": (i == default_input),
            })

    return input_devices


def get_default_input_device() -> Optional[Dict[str, Any]]:
    """
    获取系统默认的音频输入设备信息

    Returns:
        设备信息字典，如果没有输入设备则返回 None
    """
    try:
        default_idx = sd.default.device[0]
        if default_idx is None or default_idx < 0:
            return None
        dev = sd.query_devices(default_idx)
        if dev["max_input_channels"] <= 0:
            return None
        return {
            "index": default_idx,
            "name": dev["name"],
            "max_input_channels": dev["max_input_channels"],
            "default_samplerate": dev["default_samplerate"],
            "is_default": True,
        }
    except Exception as e:
        logger.warning(f"获取默认输入设备失败: {e}")
        return None


# ============================================================
#  核心：音频采集器
# ============================================================

class AudioCapture:
    """
    音频采集器 —— VoiceInk 的核心音频输入组件

    功能特性：
    1. Push-to-Talk 模式：通过 start_recording() / stop_recording() 控制录音
    2. 内置 Energy-based VAD，自动检测语音/静音，并在停顿超过阈值时标记段落边界
    3. 流式输出：通过 stream_chunks() generator 逐块获取音频数据
    4. 线程安全：所有状态读写均有锁保护，支持多线程并发操作
    5. 音频设备选择：支持指定设备或使用系统默认设备

    使用示例::

        config = AudioConfig(sample_rate=16000, channels=1)
        capture = AudioCapture(config)

        # 开始录音
        capture.start_recording()

        # 流式获取音频块
        for chunk in capture.stream_chunks():
            if chunk.chunk_type == ChunkType.BOUNDARY:
                print("检测到段落边界")
            elif chunk.chunk_type == ChunkType.END:
                print("录音结束")
                break
            else:
                process_audio(chunk.data)

        # 或者手动停止
        capture.stop_recording()
    """

    def __init__(self, config: Optional[AudioConfig] = None):
        """
        初始化音频采集器

        Args:
            config: 音频配置，为 None 时使用默认配置
        """
        self._config = config or AudioConfig()

        # ---- 核心参数 ----
        self._sample_rate: int = self._config.sample_rate           # 采样率
        self._channels: int = self._config.channels                 # 声道数
        self._chunk_duration_ms: int = self._config.chunk_duration_ms  # 每块时长（毫秒）
        self._silence_duration_ms: int = self._config.silence_duration_ms  # 静音判定阈值（毫秒）
        self._max_recording_seconds: int = self._config.max_recording_seconds  # 最大录音时长

        # 每块的采样点数 = 采样率 × 块时长 / 1000
        self._chunk_samples: int = int(
            self._sample_rate * self._chunk_duration_ms / 1000
        )

        # ---- 设备选择 ----
        self._device: Optional[int] = self._resolve_device(self._config.device)

        # ---- VAD 实例 ----
        self._vad = EnergyVAD(
            energy_threshold=self._config.vad_threshold * 0.02,
            # 将配置中的 vad_threshold (0~1) 映射为合理的绝对能量阈值
            noise_multiplier=3.0,
            noise_smooth_factor=0.05,
            hangover_chunks=3,
        )

        # ---- 音频流 & 线程同步 ----
        self._stream: Optional[sd.InputStream] = None        # sounddevice 输入流
        self._audio_queue: queue.Queue = queue.Queue(maxsize=500)  # 音频块队列
        self._is_recording: bool = False                      # 是否正在录音
        self._should_stop: bool = False                       # 是否应停止录音
        self._sequence_num: int = 0                           # 块序列号

        # ---- 静音追踪（用于段落边界检测） ----
        self._silence_start_time: Optional[float] = None      # 静音开始时间
        self._in_speech_segment: bool = False                  # 是否处于语音段中
        self._boundary_emitted: bool = False                   # 当前静音段是否已发出边界标记

        # ---- 录音开始时间（用于最大时长限制） ----
        self._recording_start_time: float = 0.0

        # ---- 采样率自适应（设备原生采样率 ≠ 目标 16kHz 时启用重采样） ----
        self._actual_sample_rate: int = self._sample_rate  # 设备实际采样率
        self._needs_resample: bool = False                  # 是否需要重采样

        # ---- 线程锁 ----
        self._lock = threading.RLock()  # 可重入锁，保护所有状态变量

        logger.info(
            f"AudioCapture 初始化完成: "
            f"采样率={self._sample_rate}Hz, "
            f"声道={self._channels}, "
            f"块时长={self._chunk_duration_ms}ms, "
            f"块采样点={self._chunk_samples}, "
            f"静音阈值={self._silence_duration_ms}ms, "
            f"最大录音={self._max_recording_seconds}s"
        )

    # ============================================================
    #  设备管理
    # ============================================================

    def _resolve_device(self, device_str: str) -> Optional[int]:
        """
        解析设备配置字符串，返回 sounddevice 需要的设备索引

        Args:
            device_str: 设备标识，可以是：
                - "default" → 使用系统默认设备（返回 None）
                - "auto"    → 自动探测可用的输入设备
                - 纯数字字符串 → 按索引选择
                - 设备名称子串 → 模糊匹配

        Returns:
            设备索引，None 表示使用系统默认
        """
        if device_str == "auto":
            logger.info("自动探测输入设备...")
            return self._find_working_device()

        if device_str == "default" or not device_str:
            logger.info("使用系统默认输入设备")
            return None

        # 尝试按索引解析
        try:
            idx = int(device_str)
            dev = sd.query_devices(idx)
            if dev["max_input_channels"] > 0:
                logger.info(f"使用指定设备 [{idx}]: {dev['name']}")
                return idx
            else:
                logger.warning(f"设备 [{idx}] 不支持输入，回退到默认设备")
                return None
        except (ValueError, sd.PortAudioError):
            pass

        # 尝试按名称模糊匹配
        devices = list_audio_devices()
        for dev in devices:
            if device_str.lower() in dev["name"].lower():
                logger.info(f"匹配到设备 [{dev['index']}]: {dev['name']}")
                return dev["index"]

        logger.warning(f"未找到设备 '{device_str}'，回退到默认设备")
        return None

    @staticmethod
    def _get_device_priority(dev_name: str) -> int:
        """
        根据设备名称返回优先级分数（越高越优先）

        优先级策略：
        - Windows 声音映射器最高优先级 (110) - 它跟随系统默认设置
        - 麦克风设备 (100)
        - USB 麦克风 (90)
        - 蓝牙耳机麦克风 (80)
        - 主声音捕获驱动程序 (75)
        - 耳机 (70)
        - 默认设备 (50)
        - 线路输入/立体声混音最低优先级 (10)
        - 其他设备 (30)

        Args:
            dev_name: 设备名称

        Returns:
            优先级分数
        """
        name_lower = dev_name.lower()

        # 排除的设备类型（通常不是用于语音输入的）
        exclude_keywords = [
            "line input", "line in", "线路输入",
            "stereo mix", "立体声混音",
            "what u hear", "loopback",
            "spdif", "digital",
        ]
        for kw in exclude_keywords:
            if kw in name_lower:
                return 10  # 最低优先级

        # Windows 声音映射器 - 跟随系统默认设置
        if "声音映射器" in dev_name or "sound mapper" in name_lower:
            return 110  # 最高优先级

        # 优先的麦克风设备
        mic_keywords = [
            "microphone", "mic input", "麦克风", "话筒",
        ]
        for kw in mic_keywords:
            if kw in name_lower:
                return 100

        # USB 麦克风
        if "usb" in name_lower and ("mic" in name_lower or "audio" in name_lower):
            return 90

        # 蓝牙耳机/麦克风
        if "bluetooth" in name_lower or "bt " in name_lower or "蓝牙" in name_lower:
            return 80

        # 主声音捕获驱动程序
        if "主声音捕获" in dev_name or "primary sound capture" in name_lower:
            return 75

        # 耳机（可能有麦克风）
        if "headset" in name_lower or "headphone" in name_lower or "耳机" in name_lower:
            return 70

        # 摄像头内置麦克风
        if "webcam" in name_lower or "camera" in name_lower or "摄像" in name_lower:
            return 60

        # 其他设备
        return 30

    def _find_working_device(self) -> Optional[int]:
        """
        探测系统中实际可用的输入设备。

        智能优先级策略：
        1. 收集所有可用输入设备
        2. 按设备类型排序（麦克风 > USB > 蓝牙 > 默认 > 线路输入）
        3. 逐个验证设备可用性

        对每个候选设备，尝试以目标采样率打开 InputStream 并验证数据可达。
        如果目标采样率不支持，再用设备默认采样率重试。

        Returns:
            可用设备的索引，None 表示所有设备均不可用
        """
        # 收集所有输入设备及其信息
        all_devices: list[tuple[int, str, int]] = []  # (idx, name, priority)

        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    dev_name = dev["name"]
                    priority = self._get_device_priority(dev_name)
                    all_devices.append((i, dev_name, priority))
        except Exception as e:
            logger.error(f"枚举音频设备失败: {e}")
            return None

        if not all_devices:
            logger.error("自动探测: 未找到任何输入设备")
            return None

        # 按优先级排序（高优先级在前）
        all_devices.sort(key=lambda x: x[2], reverse=True)

        logger.info("自动探测: 发现 %d 个输入设备，按优先级排序:", len(all_devices))
        for idx, name, priority in all_devices[:5]:  # 只显示前5个
            logger.info("  [%d] %s (优先级=%d)", idx, name, priority)

        # 逐个探测
        for idx, dev_name, priority in all_devices:
            try:
                dev_info = sd.query_devices(idx)
            except Exception:
                continue

            # 尝试用目标采样率（16kHz）
            if self._try_open_device(idx, self._sample_rate):
                logger.info(
                    "自动探测: 选择设备 [%d] %s (采样率=%dHz, 优先级=%d)",
                    idx, dev_name, self._sample_rate, priority,
                )
                return idx

            # 目标采样率不行，尝试设备默认采样率
            native_rate = int(dev_info.get("default_samplerate", 0))
            if native_rate > 0 and native_rate != self._sample_rate:
                if self._try_open_device(idx, native_rate):
                    logger.info(
                        "自动探测: 选择设备 [%d] %s (原生采样率=%dHz, 优先级=%d)",
                        idx, dev_name, native_rate, priority,
                    )
                    return idx

        logger.error("自动探测: 未找到任何可用的输入设备")
        return None

    @staticmethod
    def _try_open_device(
        device_idx: int, samplerate: int, timeout: float = 0.5
    ) -> bool:
        """
        尝试打开指定设备并验证是否有音频数据到达。

        仅能打开但无数据的设备（如断连的蓝牙设备）会被判定为不可用。

        Args:
            device_idx: 设备索引
            samplerate: 采样率
            timeout:    等待数据的超时时间（秒）

        Returns:
            True 表示设备可正常打开且有数据到达
        """
        got_data = threading.Event()

        def _probe_callback(indata, frames, time_info, status):
            if frames > 0:
                got_data.set()

        try:
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=1,
                dtype="float32",
                blocksize=1024,
                device=device_idx,
                callback=_probe_callback,
            )
            stream.start()
            got_data.wait(timeout=timeout)
            stream.stop()
            stream.close()
            return got_data.is_set()
        except Exception:
            return False

    # ============================================================
    #  sounddevice 回调
    # ============================================================

    def _audio_callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """
        sounddevice 音频回调函数（在音频线程中被调用）

        每当音频硬件产生一块新数据时，此回调被触发。
        回调中只做最少的工作：将数据复制一份放入队列。
        如果设备采样率 ≠ 目标采样率，则在此处实时重采样。

        Args:
            indata:    输入音频数据 (numpy 数组, shape: [frames, channels])
            frames:    帧数
            time_info: 时间信息（由 PortAudio 提供）
            status:    状态标志（溢出/欠载等）
        """
        if status:
            # 记录音频流异常（通常是缓冲区溢出）
            logger.warning(f"音频流状态异常: {status}")

        # 只在录音状态下才处理
        if not self._is_recording:
            return

        try:
            # 复制数据（回调中的 indata 是临时缓冲区，必须复制）
            audio_data = indata[:, 0].copy() if self._channels == 1 else indata.copy()

            # 采样率自适应：如果设备采样率 ≠ 目标采样率，进行重采样
            if self._needs_resample:
                audio_data = _resample(
                    audio_data, self._actual_sample_rate, self._sample_rate
                )

            # 将数据放入队列（非阻塞，队列满则丢弃旧数据）
            try:
                self._audio_queue.put_nowait(audio_data)
            except queue.Full:
                # 队列满了，丢弃最旧的块，放入新块
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._audio_queue.put_nowait(audio_data)
                except queue.Full:
                    logger.warning("音频队列持续溢出，丢弃数据块")
        except Exception as e:
            logger.error(f"音频回调异常: {e}")

    # ============================================================
    #  录音控制（Push-to-Talk 接口）
    # ============================================================

    def start_recording(self) -> bool:
        """
        开始录音（Push-to-Talk 按下时调用）

        如果配置的设备/默认设备无法打开，会自动探测可用设备。
        如果目标采样率（16kHz）不被设备支持，自动使用设备原生采样率
        录制并在回调中实时重采样到目标采样率。

        Returns:
            是否成功开始录音
        """
        with self._lock:
            if self._is_recording:
                logger.warning("录音已在进行中，忽略重复调用")
                return False

            # 重置状态
            self._reset_state()

            # 尝试打开音频流：先用配置设备，失败后自动探测
            stream = self._try_open_stream(self._device, self._sample_rate)
            if stream is None:
                logger.warning(
                    "配置的音频设备不可用 (device=%s), 自动探测可用设备...",
                    self._device,
                )
                stream = self._auto_open_stream()

            if stream is None:
                logger.error("无法打开任何音频输入设备")
                return False

            try:
                self._stream = stream
                self._stream.start()

                self._is_recording = True
                self._should_stop = False
                self._recording_start_time = time.monotonic()

                # 向队列发送录音开始标记
                start_chunk = AudioChunk(
                    data=np.array([], dtype=np.float32),
                    timestamp=time.monotonic(),
                    chunk_type=ChunkType.START,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    duration_ms=0.0,
                    energy=0.0,
                    is_speech=False,
                    sequence_num=0,
                )
                self._audio_queue.put_nowait(start_chunk)

                if self._needs_resample:
                    logger.info(
                        "录音开始 (设备采样率=%dHz -> 重采样到%dHz)",
                        self._actual_sample_rate, self._sample_rate,
                    )
                else:
                    logger.info("录音开始")
                return True

            except Exception as e:
                logger.error(f"启动音频流失败: {e}")
                self._is_recording = False
                if self._stream:
                    try:
                        self._stream.close()
                    except Exception:
                        pass
                    self._stream = None
                return False

    def _try_open_stream(
        self, device: Optional[int], samplerate: int
    ) -> Optional[sd.InputStream]:
        """
        尝试用指定设备和采样率创建 InputStream。

        Args:
            device:     设备索引（None = 系统默认）
            samplerate: 采样率

        Returns:
            InputStream 对象，失败返回 None
        """
        # 计算该采样率对应的 blocksize
        blocksize = int(samplerate * self._chunk_duration_ms / 1000)
        try:
            stream = sd.InputStream(
                samplerate=samplerate,
                channels=self._channels,
                dtype="float32",
                blocksize=blocksize,
                device=device,
                callback=self._audio_callback,
            )
            self._actual_sample_rate = samplerate
            self._needs_resample = (samplerate != self._sample_rate)
            return stream
        except Exception:
            return None

    def _auto_open_stream(self) -> Optional[sd.InputStream]:
        """
        自动探测可用设备并打开音频流。

        使用智能优先级策略：麦克风 > USB > 蓝牙 > 默认 > 线路输入

        Returns:
            InputStream 对象，或所有设备都不可用时返回 None
        """
        # 收集所有输入设备及其信息
        all_devices: list[tuple[int, str, int]] = []  # (idx, name, priority)

        try:
            for i, dev in enumerate(sd.query_devices()):
                if dev["max_input_channels"] > 0:
                    dev_name = dev["name"]
                    priority = self._get_device_priority(dev_name)
                    all_devices.append((i, dev_name, priority))
        except Exception as e:
            logger.error(f"枚举音频设备失败: {e}")
            return None

        if not all_devices:
            return None

        # 按优先级排序（高优先级在前）
        all_devices.sort(key=lambda x: x[2], reverse=True)

        for idx, dev_name, priority in all_devices:
            try:
                dev_info = sd.query_devices(idx)
            except Exception:
                continue

            # 先试目标采样率
            stream = self._try_open_stream(idx, self._sample_rate)
            if stream is not None:
                logger.info(
                    "自动选择设备 [%d] %s (%dHz, 优先级=%d)",
                    idx, dev_name, self._sample_rate, priority,
                )
                self._device = idx
                return stream

            # 再试设备原生采样率
            native_rate = int(dev_info.get("default_samplerate", 0))
            if native_rate > 0 and native_rate != self._sample_rate:
                stream = self._try_open_stream(idx, native_rate)
                if stream is not None:
                    logger.info(
                        "自动选择设备 [%d] %s (原生%dHz, 将重采样到%dHz, 优先级=%d)",
                        idx, dev_name, native_rate, self._sample_rate, priority,
                    )
                    self._device = idx
                    return stream

        return None

    def stop_recording(self) -> bool:
        """
        停止录音（Push-to-Talk 松开时调用）

        Returns:
            是否成功停止录音
        """
        with self._lock:
            if not self._is_recording:
                logger.debug("未在录音中，忽略停止调用")
                return False

            logger.info("停止录音...")
            self._should_stop = True
            self._is_recording = False

            # 停止并关闭音频流
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception as e:
                    logger.warning(f"关闭音频流时出错: {e}")
                finally:
                    self._stream = None

            # 向队列发送录音结束标记
            end_chunk = AudioChunk(
                data=np.array([], dtype=np.float32),
                timestamp=time.monotonic(),
                chunk_type=ChunkType.END,
                sample_rate=self._sample_rate,
                channels=self._channels,
                duration_ms=0.0,
                energy=0.0,
                is_speech=False,
                sequence_num=self._sequence_num,
            )
            try:
                self._audio_queue.put(end_chunk, timeout=1.0)
            except queue.Full:
                logger.warning("无法发送录音结束标记（队列满）")

            logger.info("录音已停止")
            return True

    def _reset_state(self) -> None:
        """重置所有内部状态（在新一轮录音开始前调用）"""
        # 清空队列
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

        # 重置序列号和 VAD
        self._sequence_num = 0
        self._vad.reset()

        # 重置静音追踪
        self._silence_start_time = None
        self._in_speech_segment = False
        self._boundary_emitted = False

        # 重置停止标志
        self._should_stop = False

    # ============================================================
    #  流式输出（Generator 接口）
    # ============================================================

    def stream_chunks(
        self,
        timeout: float = 0.1,
    ) -> Generator[AudioChunk, None, None]:
        """
        流式输出音频块的 Generator

        这是音频采集器的主要消费接口。调用方通过 for 循环迭代此 generator，
        逐块获取音频数据。当录音停止时（收到 END 标记），generator 结束。

        Args:
            timeout: 从队列获取数据的超时时间（秒），超时则 yield None 并继续等待

        Yields:
            AudioChunk 实例，可能的类型包括：
            - START:    录音开始标记
            - AUDIO:    包含语音的音频块
            - SILENCE:  静音块
            - BOUNDARY: 段落边界标记（VAD 检测到停顿超过 silence_duration_ms）
            - END:      录音结束标记

        使用示例::

            for chunk in capture.stream_chunks():
                if chunk.chunk_type == ChunkType.END:
                    break
                if chunk.chunk_type == ChunkType.BOUNDARY:
                    # 处理段落分界
                    finalize_segment()
                elif chunk.is_speech:
                    # 处理语音数据
                    buffer.append(chunk.data)
        """
        logger.debug("stream_chunks() generator 启动")

        while True:
            try:
                # 从队列中取出数据
                raw = self._audio_queue.get(timeout=timeout)
            except queue.Empty:
                # 超时：检查是否应该结束
                with self._lock:
                    if self._should_stop and not self._is_recording:
                        # 确认队列确实为空后再结束
                        if self._audio_queue.empty():
                            logger.debug("stream_chunks() 超时且录音已停止，结束")
                            return
                    # 检查最大录音时长
                    if self._is_recording and self._max_recording_seconds > 0:
                        elapsed = time.monotonic() - self._recording_start_time
                        if elapsed >= self._max_recording_seconds:
                            logger.warning(
                                f"达到最大录音时长 {self._max_recording_seconds}s，自动停止"
                            )
                            # 在锁外执行 stop 以避免死锁
                            threading.Thread(
                                target=self.stop_recording, daemon=True
                            ).start()
                continue

            # ---- 处理已经封装好的 AudioChunk（START / END 标记） ----
            if isinstance(raw, AudioChunk):
                if raw.chunk_type == ChunkType.START:
                    logger.debug("输出: 录音开始标记")
                    yield raw
                    continue
                elif raw.chunk_type == ChunkType.END:
                    logger.debug("输出: 录音结束标记")
                    yield raw
                    return  # generator 结束
                else:
                    yield raw
                    continue

            # ---- 处理原始音频数据 (numpy array) ----
            audio_data: np.ndarray = raw
            now = time.monotonic()

            # VAD 处理
            is_speech, energy = self._vad.process(audio_data)

            # 计算块时长
            duration_ms = len(audio_data) / self._sample_rate * 1000

            # 确定块类型并处理段落边界逻辑
            chunk_type, should_emit_boundary = self._update_speech_state(
                is_speech, now
            )

            # 递增序列号
            with self._lock:
                seq = self._sequence_num
                self._sequence_num += 1

            # 如果需要先发出段落边界标记
            if should_emit_boundary:
                boundary_chunk = AudioChunk(
                    data=np.array([], dtype=np.float32),
                    timestamp=now,
                    chunk_type=ChunkType.BOUNDARY,
                    sample_rate=self._sample_rate,
                    channels=self._channels,
                    duration_ms=0.0,
                    energy=0.0,
                    is_speech=False,
                    sequence_num=seq,
                )
                logger.debug(
                    f"输出: 段落边界标记 (seq={seq})"
                )
                yield boundary_chunk

            # 构建并输出音频块
            chunk = AudioChunk(
                data=audio_data,
                timestamp=now,
                chunk_type=chunk_type,
                sample_rate=self._sample_rate,
                channels=self._channels,
                duration_ms=duration_ms,
                energy=energy,
                is_speech=is_speech,
                sequence_num=seq,
            )

            yield chunk

    def _update_speech_state(
        self, is_speech: bool, current_time: float
    ) -> tuple:
        """
        更新语音/静音状态，检测是否需要发出段落边界

        状态转换逻辑：
        - 从静音进入语音 → 标记进入语音段
        - 从语音进入静音 → 开始计时
        - 持续静音超过 silence_duration_ms → 发出段落边界

        Args:
            is_speech:    当前块是否为语音
            current_time: 当前时间戳

        Returns:
            (chunk_type, should_emit_boundary)
            - chunk_type: AUDIO 或 SILENCE
            - should_emit_boundary: 是否应在此块之前发出 BOUNDARY 标记
        """
        with self._lock:
            should_emit_boundary = False

            if is_speech:
                # ---- 当前是语音 ----
                chunk_type = ChunkType.AUDIO
                self._in_speech_segment = True
                self._silence_start_time = None   # 重置静音计时
                self._boundary_emitted = False    # 重置边界标记
            else:
                # ---- 当前是静音 ----
                chunk_type = ChunkType.SILENCE

                if self._in_speech_segment:
                    # 之前在语音段中，现在进入静音
                    if self._silence_start_time is None:
                        # 刚刚从语音转为静音，开始计时
                        self._silence_start_time = current_time

                    # 检查静音持续时长
                    silence_elapsed_ms = (
                        (current_time - self._silence_start_time) * 1000
                    )

                    if (
                        silence_elapsed_ms >= self._silence_duration_ms
                        and not self._boundary_emitted
                    ):
                        # 静音时间超过阈值，发出段落边界
                        should_emit_boundary = True
                        self._boundary_emitted = True
                        self._in_speech_segment = False  # 当前语音段结束
                        logger.debug(
                            f"VAD 检测到段落边界: 静音持续 {silence_elapsed_ms:.0f}ms "
                            f"> 阈值 {self._silence_duration_ms}ms"
                        )

            return chunk_type, should_emit_boundary

    # ============================================================
    #  一次性获取全部录音数据
    # ============================================================

    def get_all_audio(self) -> Optional[np.ndarray]:
        """
        获取当前队列中所有音频数据（合并为一个 numpy 数组）

        此方法会清空队列中所有原始音频数据块并拼接返回。
        通常在 stop_recording() 之后调用，用于获取完整录音。

        Returns:
            合并后的 float32 音频数组，如果无数据则返回 None
        """
        chunks = []

        while not self._audio_queue.empty():
            try:
                raw = self._audio_queue.get_nowait()
                if isinstance(raw, np.ndarray):
                    chunks.append(raw)
                elif isinstance(raw, AudioChunk) and raw.data.size > 0:
                    chunks.append(raw.data)
            except queue.Empty:
                break

        if not chunks:
            return None

        return np.concatenate(chunks)

    # ============================================================
    #  属性访问
    # ============================================================

    @property
    def is_recording(self) -> bool:
        """是否正在录音"""
        with self._lock:
            return self._is_recording

    @property
    def sample_rate(self) -> int:
        """当前采样率"""
        return self._sample_rate

    @property
    def channels(self) -> int:
        """当前声道数"""
        return self._channels

    @property
    def config(self) -> AudioConfig:
        """当前音频配置"""
        return self._config

    @property
    def vad(self) -> EnergyVAD:
        """VAD 实例（可用于外部监控 VAD 状态）"""
        return self._vad

    @property
    def queue_size(self) -> int:
        """当前队列中待处理的块数"""
        return self._audio_queue.qsize()

    # ============================================================
    #  资源释放
    # ============================================================

    def close(self) -> None:
        """
        释放所有资源

        停止录音（如果正在进行），关闭音频流，清空队列。
        """
        logger.info("AudioCapture 关闭，释放资源...")

        with self._lock:
            if self._is_recording:
                # 先解锁再调 stop_recording（因为它也要获取锁）
                pass

        # stop_recording 自己会获取锁
        if self._is_recording:
            self.stop_recording()

        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None

            # 清空队列
            while not self._audio_queue.empty():
                try:
                    self._audio_queue.get_nowait()
                except queue.Empty:
                    break

        logger.info("AudioCapture 资源已释放")

    def __enter__(self):
        """支持 with 语句"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """with 语句退出时自动释放资源"""
        self.close()
        return False

    def __del__(self):
        """析构时确保资源释放"""
        try:
            self.close()
        except Exception:
            pass


# ============================================================
#  模块自测（直接运行此文件时执行）
# ============================================================

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("  VoiceInk 音频采集模块 - 自测")
    print("=" * 60)

    # 1. 列举音频设备
    print("\n【可用音频输入设备】")
    devices = list_audio_devices()
    if not devices:
        print("  ❌ 未发现任何音频输入设备！")
        sys.exit(1)

    for dev in devices:
        marker = " ★ (默认)" if dev["is_default"] else ""
        print(
            f"  [{dev['index']}] {dev['name']} "
            f"(ch={dev['max_input_channels']}, "
            f"rate={dev['default_samplerate']:.0f}Hz){marker}"
        )

    # 2. Push-to-Talk 测试
    print("\n【Push-to-Talk 测试】")
    print("  按 Enter 开始录音，再按 Enter 停止录音...")

    config = AudioConfig(
        sample_rate=16000,
        channels=1,
        chunk_duration_ms=500,
        silence_duration_ms=800,
    )

    capture = AudioCapture(config)

    input("  → 按 Enter 开始录音... ")
    capture.start_recording()

    # 在后台线程中消费音频块
    total_chunks = 0
    speech_chunks = 0
    boundary_count = 0
    total_duration_ms = 0.0
    stop_event = threading.Event()

    def consumer():
        """消费者线程：从 stream_chunks 读取音频块"""
        global total_chunks, speech_chunks, boundary_count, total_duration_ms
        for chunk in capture.stream_chunks():
            if chunk.chunk_type == ChunkType.START:
                print("  🎙️  录音开始")
                continue
            elif chunk.chunk_type == ChunkType.END:
                print("  🛑 录音结束")
                break
            elif chunk.chunk_type == ChunkType.BOUNDARY:
                boundary_count += 1
                print(f"  📍 段落边界 #{boundary_count}")
                continue

            total_chunks += 1
            total_duration_ms += chunk.duration_ms
            if chunk.is_speech:
                speech_chunks += 1

            # 每 2 秒显示一次状态
            if total_chunks % 4 == 0:
                noise = capture.vad.noise_estimate
                print(
                    f"  📊 块 #{chunk.sequence_num:4d} | "
                    f"能量={chunk.energy:.6f} | "
                    f"语音={'✅' if chunk.is_speech else '❌'} | "
                    f"噪底={noise:.6f} | "
                    f"类型={chunk.chunk_type.name}"
                )

        stop_event.set()

    consumer_thread = threading.Thread(target=consumer, daemon=True)
    consumer_thread.start()

    input("  → 按 Enter 停止录音... ")
    capture.stop_recording()

    # 等待消费者线程结束
    stop_event.wait(timeout=3.0)

    # 3. 输出统计
    print(f"\n【录音统计】")
    print(f"  总块数:       {total_chunks}")
    print(f"  语音块数:     {speech_chunks}")
    print(f"  静音块数:     {total_chunks - speech_chunks}")
    print(f"  段落边界:     {boundary_count}")
    print(f"  总时长:       {total_duration_ms / 1000:.2f}s")
    if total_chunks > 0:
        print(f"  语音占比:     {speech_chunks / total_chunks * 100:.1f}%")

    capture.close()
    print("\n✅ 自测完成！")
