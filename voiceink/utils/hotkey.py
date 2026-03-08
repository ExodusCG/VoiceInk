"""
VoiceInk - 全局快捷键管理模块

混合双引擎方案：
  - 组合键（如 ctrl+space, ctrl+shift+v）：使用 Windows RegisterHotKey API
  - 单独修饰键（如 right alt, left shift）：使用 GetAsyncKeyState 轮询

不使用低级键盘钩子（SetWindowsHookEx），避免 Windows 因回调超时自动摘除钩子。

支持：
  - Push-to-Talk（按住说话）：按下触发、释放触发
  - Toggle（切换）：按一次开始，再按一次停止
  - 快捷键冲突检测
  - 线程安全的注册 / 注销
"""

import ctypes
import ctypes.wintypes
import queue
import threading
import logging
import time
from typing import Callable, Dict, Optional, Set, Tuple

logger = logging.getLogger("voiceink")

# ----------------------------------------------------------------
# Windows API
# ----------------------------------------------------------------
user32 = ctypes.windll.user32
kernel32 = ctypes.windll.kernel32

# 虚拟键码映射
_VK_MAP = {
    # 修饰键
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "menu": 0x12,
    "shift": 0x10,
    "win": 0x5B, "windows": 0x5B, "super": 0x5B, "cmd": 0x5B,
    "left ctrl": 0xA2, "right ctrl": 0xA3,
    "left alt": 0xA4, "right alt": 0xA5,
    "left shift": 0xA0, "right shift": 0xA1,
    # 功能键
    "f1": 0x70, "f2": 0x71, "f3": 0x72, "f4": 0x73,
    "f5": 0x74, "f6": 0x75, "f7": 0x76, "f8": 0x77,
    "f9": 0x78, "f10": 0x79, "f11": 0x7A, "f12": 0x7B,
    # 特殊键
    "space": 0x20, "enter": 0x0D, "return": 0x0D,
    "tab": 0x09, "escape": 0x1B, "esc": 0x1B,
    "backspace": 0x08, "delete": 0x2E, "del": 0x2E,
    "insert": 0x2D, "ins": 0x2D,
    "home": 0x24, "end": 0x23,
    "page up": 0x21, "pageup": 0x21,
    "page down": 0x22, "pagedown": 0x22,
    "up": 0x26, "down": 0x28, "left": 0x25, "right": 0x27,
    "caps lock": 0x14, "capslock": 0x14,
    "num lock": 0x90, "numlock": 0x90,
    "scroll lock": 0x91, "scrolllock": 0x91,
    "print screen": 0x2C, "printscreen": 0x2C,
    "pause": 0x13,
    # 数字键盘
    "numpad 0": 0x60, "numpad 1": 0x61, "numpad 2": 0x62,
    "numpad 3": 0x63, "numpad 4": 0x64, "numpad 5": 0x65,
    "numpad 6": 0x66, "numpad 7": 0x67, "numpad 8": 0x68,
    "numpad 9": 0x69,
    "numpad +": 0x6B, "numpad -": 0x6D,
    "numpad *": 0x6A, "numpad /": 0x6F,
    "numpad .": 0x6E, "numpad enter": 0x0D,
    # 标点
    ";": 0xBA, "=": 0xBB, ",": 0xBC, "-": 0xBD,
    ".": 0xBE, "/": 0xBF, "`": 0xC0,
    "[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}

# RegisterHotKey 修饰键标志
MOD_ALT = 0x0001
MOD_CONTROL = 0x0002
MOD_SHIFT = 0x0004
MOD_WIN = 0x0008
MOD_NOREPEAT = 0x4000

WM_HOTKEY = 0x0312
WM_QUIT = 0x0012

_MOD_MAP = {
    "ctrl": MOD_CONTROL, "control": MOD_CONTROL,
    "alt": MOD_ALT, "menu": MOD_ALT,
    "shift": MOD_SHIFT,
    "win": MOD_WIN, "windows": MOD_WIN, "super": MOD_WIN,
    "cmd": MOD_WIN, "command": MOD_WIN,
}

# 带方向的修饰键（单独使用时走轮询引擎）
_SIDE_MODIFIER_MAP = {
    "left ctrl": ("ctrl", 0xA2),
    "right ctrl": ("ctrl", 0xA3),
    "left alt": ("alt", 0xA4),
    "right alt": ("alt", 0xA5),
    "left shift": ("shift", 0xA0),
    "right shift": ("shift", 0xA1),
}

# 通用修饰键（单独使用时也走轮询引擎）
_GENERIC_MODIFIER_MAP = {
    "ctrl": 0x11, "control": 0x11,
    "alt": 0x12, "menu": 0x12,
    "shift": 0x10,
    "win": 0x5B, "windows": 0x5B, "super": 0x5B, "cmd": 0x5B,
}

# 轮询间隔
_POLL_INTERVAL_S = 0.030  # 30ms

# 消抖计数：按键必须连续 N 个轮询周期保持一致状态才触发
# 2 次 × 30ms = 60ms，足够过滤电气噪声，不影响手感
_DEBOUNCE_COUNT = 2


def _key_name_to_vk(name: str) -> int:
    """将键名转换为 Windows 虚拟键码。"""
    name_lower = name.strip().lower()
    if name_lower in _VK_MAP:
        return _VK_MAP[name_lower]
    if len(name_lower) == 1:
        ch = name_lower.upper()
        if 'A' <= ch <= 'Z':
            return ord(ch)
        if '0' <= ch <= '9':
            return ord(ch)
    raise ValueError(f"无法识别的键名: '{name}'")


def _is_solo_modifier(hotkey_str: str) -> bool:
    """判断是否是单独的修饰键（如 'right alt', 'ctrl'）。"""
    key = hotkey_str.strip().lower()
    return key in _SIDE_MODIFIER_MAP or key in _GENERIC_MODIFIER_MAP


def _parse_hotkey(hotkey_str: str) -> Tuple[int, int, int]:
    """
    解析快捷键字符串为 (modifiers, vk_code, trigger_vk)。

    对于单独修饰键，返回 (0, vk, vk) —— 不走 RegisterHotKey。
    对于组合键，返回 (MOD_xxx | MOD_NOREPEAT, vk, vk)。
    """
    parts = [p.strip().lower() for p in hotkey_str.split("+")]
    parts = [p for p in parts if p]

    # 单独修饰键：不走 RegisterHotKey
    reassembled = " ".join(parts)
    if reassembled in _SIDE_MODIFIER_MAP:
        _, vk = _SIDE_MODIFIER_MAP[reassembled]
        return (0, vk, vk)  # modifiers=0 表示走轮询引擎

    # 通用修饰键单独使用：也走轮询引擎
    if len(parts) == 1 and parts[0] in _GENERIC_MODIFIER_MAP:
        vk = _GENERIC_MODIFIER_MAP[parts[0]]
        return (0, vk, vk)

    # 组合键
    modifiers = 0
    trigger_key = None

    for p in parts:
        if p in _MOD_MAP:
            modifiers |= _MOD_MAP[p]
        elif p in _SIDE_MODIFIER_MAP:
            mod_name, _ = _SIDE_MODIFIER_MAP[p]
            modifiers |= _MOD_MAP[mod_name]
            logger.warning(
                "组合键 '%s' 中使用了方向修饰键 '%s'，"
                "RegisterHotKey 不支持区分左右，将视为通用 '%s'",
                hotkey_str, p, mod_name,
            )
        else:
            if trigger_key is not None:
                raise ValueError(
                    f"快捷键 '{hotkey_str}' 含有多个非修饰键: "
                    f"'{trigger_key}' 和 '{p}'"
                )
            trigger_key = p

    if trigger_key is None:
        raise ValueError(
            f"快捷键 '{hotkey_str}' 缺少触发键（非修饰键部分）"
        )

    vk = _key_name_to_vk(trigger_key)
    modifiers |= MOD_NOREPEAT
    return (modifiers, vk, vk)


class HotkeyManager:
    """
    全局快捷键管理器（混合双引擎）。

    - 组合键：RegisterHotKey + 消息泵线程
    - 单独修饰键：GetAsyncKeyState 轮询线程

    所有公共方法线程安全。
    """

    _WM_EXECUTE_TASK = 0x0400  # WM_USER

    def __init__(self):
        self._lock = threading.Lock()
        self._next_id: int = 1

        self._hotkeys: Dict[int, dict] = {}
        self._key_to_id: Dict[str, int] = {}
        self._registered_keys: Set[str] = set()

        # 跨线程任务队列
        self._task_queue: queue.Queue = queue.Queue()

        # 消息泵线程（RegisterHotKey 引擎）
        self._msg_thread: Optional[threading.Thread] = None
        self._msg_thread_id: Optional[int] = None
        self._msg_thread_ready = threading.Event()

        # 轮询线程（GetAsyncKeyState 引擎）
        self._poll_thread: Optional[threading.Thread] = None
        # _poll_keys 的 info["was_pressed"] / info["debounce_counter"] 仅由轮询线程写入，
        # 注册/注销操作通过 _lock 保护 _poll_keys 字典本身的增删。
        self._poll_keys: Dict[int, dict] = {}  # hotkey_id -> poll info

        self._running = False

        self._start_message_pump()
        self._start_poll_thread()

        logger.info("HotkeyManager 初始化完成 (双引擎: RegisterHotKey + GetAsyncKeyState)")

    # ============================================================
    # 公共 API
    # ============================================================

    def register_push_to_talk(
        self,
        key: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> bool:
        """注册 Push-to-Talk 快捷键（按住说话，松开停止）。"""
        return self._register(key, "ptt",
                              on_press=on_press, on_release=on_release)

    def register_toggle(
        self,
        key: str,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
    ) -> bool:
        """注册 Toggle 快捷键（按一次开始，再按停止）。"""
        return self._register(key, "toggle",
                              on_start=on_start, on_stop=on_stop)

    def register_hotkey(
        self,
        key: str,
        callback: Callable[[], None],
    ) -> bool:
        """注册普通快捷键（按下触发）。"""
        return self._register(key, "simple", callback=callback)

    def unregister_hotkey(self, key: str) -> bool:
        """注销指定快捷键。"""
        normalized = self._normalize_key(key)
        with self._lock:
            if normalized not in self._registered_keys:
                logger.warning("尝试注销未注册的快捷键: %s", normalized)
                return False

            hotkey_id = self._key_to_id.get(normalized)
            if hotkey_id is not None:
                info = self._hotkeys.get(hotkey_id, {})
                if info.get("engine") == "poll":
                    del self._poll_keys[hotkey_id]
                else:
                    self._unregister_in_pump(hotkey_id)
                del self._hotkeys[hotkey_id]
                del self._key_to_id[normalized]

            self._registered_keys.discard(normalized)
            logger.info("快捷键已注销: %s", normalized)
            return True

    def unregister_all(self) -> None:
        """注销所有已注册的快捷键。"""
        with self._lock:
            for hotkey_id, info in list(self._hotkeys.items()):
                if info.get("engine") != "poll":
                    self._unregister_in_pump(hotkey_id)
            self._hotkeys.clear()
            self._key_to_id.clear()
            self._registered_keys.clear()
            self._poll_keys.clear()
            logger.info("所有快捷键已注销")

    def is_registered(self, key: str) -> bool:
        normalized = self._normalize_key(key)
        with self._lock:
            return normalized in self._registered_keys

    @property
    def registered_keys(self) -> Set[str]:
        with self._lock:
            return self._registered_keys.copy()

    def shutdown(self) -> None:
        """关闭热键管理器。先停止线程，再注销热键。"""
        if not self._running:
            return
        # 1. 停止所有线程
        self._running = False
        # 2. 等待轮询线程退出（它检查 self._running）
        if self._poll_thread and self._poll_thread.is_alive():
            self._poll_thread.join(timeout=2.0)
        # 3. 轮询线程已停止，安全注销所有热键
        self.unregister_all()
        # 4. 停止消息泵线程
        self._stop_message_pump()
        logger.info("HotkeyManager 已关闭")

    # ============================================================
    # 内部注册逻辑
    # ============================================================

    def _register(self, key: str, htype: str, **callbacks) -> bool:
        """统一注册入口，自动选择引擎。"""
        normalized = self._normalize_key(key)

        with self._lock:
            if normalized in self._registered_keys:
                logger.warning("快捷键冲突：'%s' 已被注册", normalized)
                return False

            try:
                modifiers, vk, trigger_vk = _parse_hotkey(normalized)
            except ValueError as e:
                logger.error("解析快捷键失败 [%s]: %s", normalized, e)
                return False

            hotkey_id = self._next_id
            self._next_id += 1

            use_poll = (modifiers == 0)  # 单独修饰键走轮询

            info = {
                "key": normalized,
                "type": htype,
                "engine": "poll" if use_poll else "register",
                "modifiers": modifiers,
                "vk": vk,
                "trigger_vk": trigger_vk,
            }

            # 包装回调
            if htype == "ptt":
                info["on_press"] = self._safe_callback(
                    callbacks["on_press"], f"PTT 按下 [{normalized}]")
                info["on_release"] = self._safe_callback(
                    callbacks["on_release"], f"PTT 释放 [{normalized}]")
            elif htype == "toggle":
                info["on_start"] = self._safe_callback(
                    callbacks["on_start"], f"Toggle 开始 [{normalized}]")
                info["on_stop"] = self._safe_callback(
                    callbacks["on_stop"], f"Toggle 停止 [{normalized}]")
                info["toggle_state"] = False
            elif htype == "simple":
                info["callback"] = self._safe_callback(
                    callbacks["callback"], f"快捷键 [{normalized}]")

            if use_poll:
                # 轮询引擎
                info["was_pressed"] = False
                info["debounce_counter"] = 0  # 消抖计数器
                self._poll_keys[hotkey_id] = info
                engine_name = "GetAsyncKeyState 轮询"
            else:
                # RegisterHotKey 引擎
                if not self._register_in_pump(hotkey_id, modifiers, vk):
                    logger.error("RegisterHotKey 失败 [%s]", normalized)
                    return False
                engine_name = "RegisterHotKey"

            self._hotkeys[hotkey_id] = info
            self._key_to_id[normalized] = hotkey_id
            self._registered_keys.add(normalized)

            logger.info("%s 快捷键已注册: %s (id=%d, 引擎=%s)",
                        htype.upper(), normalized, hotkey_id, engine_name)
            return True

    # ============================================================
    # 引擎 1: RegisterHotKey 消息泵
    # ============================================================

    def _start_message_pump(self) -> None:
        self._running = True
        self._msg_thread = threading.Thread(
            target=self._message_pump_worker,
            name="VoiceInk-HotkeyPump",
            daemon=True,
        )
        self._msg_thread.start()
        if not self._msg_thread_ready.wait(timeout=5.0):
            logger.error("消息泵线程启动超时")

    def _stop_message_pump(self) -> None:
        if self._msg_thread_id is not None:
            user32.PostThreadMessageW(self._msg_thread_id, WM_QUIT, 0, 0)
        if self._msg_thread and self._msg_thread.is_alive():
            self._msg_thread.join(timeout=3.0)

    def _message_pump_worker(self) -> None:
        self._msg_thread_id = kernel32.GetCurrentThreadId()
        msg = ctypes.wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)
        self._msg_thread_ready.set()
        logger.debug("消息泵线程已启动 (tid=%d)", self._msg_thread_id)

        while self._running:
            result = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
            if result == 0:
                break
            elif result == -1:
                logger.error("GetMessageW 返回错误")
                break

            if msg.message == WM_HOTKEY:
                self._on_hotkey_triggered(msg.wParam)
            elif msg.message == self._WM_EXECUTE_TASK:
                self._drain_task_queue()

            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))

        logger.debug("消息泵线程已退出")

    def _drain_task_queue(self) -> None:
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()()
            except queue.Empty:
                break
            except Exception as e:
                logger.error("跨线程任务异常: %s", e, exc_info=True)

    def _register_in_pump(self, hotkey_id: int, modifiers: int, vk: int) -> bool:
        if not self._msg_thread_ready.is_set():
            return False

        if kernel32.GetCurrentThreadId() == self._msg_thread_id:
            return self._do_register(hotkey_id, modifiers, vk)

        result = {"ok": False}
        done = threading.Event()

        def task():
            result["ok"] = self._do_register(hotkey_id, modifiers, vk)
            done.set()

        self._task_queue.put(task)
        user32.PostThreadMessageW(self._msg_thread_id, self._WM_EXECUTE_TASK, 0, 0)
        return result["ok"] if done.wait(timeout=5.0) else False

    def _unregister_in_pump(self, hotkey_id: int) -> None:
        if kernel32.GetCurrentThreadId() == self._msg_thread_id:
            self._do_unregister(hotkey_id)
            return
        if self._msg_thread_id is None or not self._msg_thread_ready.is_set():
            self._do_unregister(hotkey_id)
            return

        done = threading.Event()

        def task():
            self._do_unregister(hotkey_id)
            done.set()

        self._task_queue.put(task)
        user32.PostThreadMessageW(self._msg_thread_id, self._WM_EXECUTE_TASK, 0, 0)
        done.wait(timeout=3.0)

    def _do_register(self, hotkey_id, modifiers, vk):
        ok = user32.RegisterHotKey(None, hotkey_id, modifiers, vk)
        if not ok:
            logger.error("RegisterHotKey 失败: id=%d mod=0x%X vk=0x%X err=%d",
                         hotkey_id, modifiers, vk, ctypes.GetLastError())
        return bool(ok)

    def _do_unregister(self, hotkey_id):
        try:
            user32.UnregisterHotKey(None, hotkey_id)
        except Exception as e:
            logger.debug("UnregisterHotKey 异常 (id=%d): %s", hotkey_id, e)

    # ============================================================
    # 引擎 2: GetAsyncKeyState 轮询
    # ============================================================

    def _start_poll_thread(self) -> None:
        self._poll_thread = threading.Thread(
            target=self._poll_worker,
            name="VoiceInk-HotkeyPoll",
            daemon=True,
        )
        self._poll_thread.start()

    def _poll_worker(self) -> None:
        """
        轮询线程：以 30ms 间隔检测单独修饰键的按下/释放状态。

        带消抖机制：按键状态必须连续 _DEBOUNCE_COUNT 个周期保持一致
        才会触发状态变化，防止电气噪声或短暂误触导致的假触发。
        """
        logger.debug("轮询线程已启动")

        while self._running:
            with self._lock:
                keys_snapshot = list(self._poll_keys.items())

            for hotkey_id, info in keys_snapshot:
                vk = info["trigger_vk"]
                state = user32.GetAsyncKeyState(vk)
                is_pressed = bool(state & 0x8000)
                was_pressed = info.get("was_pressed", False)

                if is_pressed != was_pressed:
                    # 状态与上次不同，递增消抖计数器
                    info["debounce_counter"] = info.get("debounce_counter", 0) + 1

                    if info["debounce_counter"] >= _DEBOUNCE_COUNT:
                        # 消抖通过，确认状态变化
                        info["debounce_counter"] = 0

                        if is_pressed and not was_pressed:
                            # 按下沿
                            info["was_pressed"] = True
                            self._on_poll_press(info)
                        elif not is_pressed and was_pressed:
                            # 释放沿
                            info["was_pressed"] = False
                            self._on_poll_release(info)
                else:
                    # 状态与上次一致，重置消抖计数器
                    info["debounce_counter"] = 0

            time.sleep(_POLL_INTERVAL_S)

        logger.debug("轮询线程已退出")

    def _on_poll_press(self, info: dict) -> None:
        """轮询引擎检测到按下。"""
        htype = info["type"]
        key_name = info["key"]
        logger.debug("轮询检测按下: %s (type=%s)", key_name, htype)

        if htype == "ptt":
            info["on_press"]()
        elif htype == "toggle":
            self._handle_toggle(info)
        elif htype == "simple":
            info["callback"]()

    def _on_poll_release(self, info: dict) -> None:
        """轮询引擎检测到释放。"""
        htype = info["type"]
        key_name = info["key"]
        logger.debug("轮询检测释放: %s (type=%s)", key_name, htype)

        if htype == "ptt":
            info["on_release"]()
        # toggle 和 simple 不需要释放事件

    # ============================================================
    # RegisterHotKey 引擎的事件处理
    # ============================================================

    def _on_hotkey_triggered(self, hotkey_id: int) -> None:
        with self._lock:
            info = self._hotkeys.get(hotkey_id)
            if info is None:
                return
            htype = info["type"]

        logger.debug("WM_HOTKEY: %s (type=%s, id=%d)", info["key"], htype, hotkey_id)

        if htype == "ptt":
            info["on_press"]()
            # 启动释放检测线程
            threading.Thread(
                target=self._poll_key_release,
                args=(info["trigger_vk"], info["on_release"], info["key"]),
                name=f"PTTRelease-{info['key']}",
                daemon=True,
            ).start()
        elif htype == "toggle":
            self._handle_toggle(info)
        elif htype == "simple":
            info["callback"]()

    def _poll_key_release(self, trigger_vk, on_release, key_name):
        """RegisterHotKey 引擎的 PTT 释放检测（短期轮询）。"""
        time.sleep(_POLL_INTERVAL_S)
        for _ in range(1000):  # 最多 30 秒
            if not self._running:
                break
            if not (user32.GetAsyncKeyState(trigger_vk) & 0x8000):
                logger.debug("PTT 释放: %s", key_name)
                on_release()
                return
            time.sleep(_POLL_INTERVAL_S)
        logger.warning("PTT 释放超时: %s", key_name)
        on_release()

    # ============================================================
    # 共用逻辑
    # ============================================================

    def _handle_toggle(self, info: dict) -> None:
        with self._lock:
            current = info.get("toggle_state", False)
            info["toggle_state"] = not current

        if not current:
            logger.debug("Toggle 激活: %s", info["key"])
            info["on_start"]()
        else:
            logger.debug("Toggle 停止: %s", info["key"])
            info["on_stop"]()

    # ============================================================
    # 工具方法
    # ============================================================

    @staticmethod
    def _normalize_key(key: str) -> str:
        key = key.strip().lower()
        # 带方向的修饰键直接返回
        if key in _SIDE_MODIFIER_MAP:
            return key
        # 通用修饰键单独使用时直接返回（规范化别名）
        if key in _GENERIC_MODIFIER_MAP and "+" not in key:
            # 规范化别名：control→ctrl, menu→alt, windows/super/cmd→win
            alias_map = {
                "control": "ctrl", "menu": "alt",
                "windows": "win", "super": "win", "cmd": "win",
            }
            return alias_map.get(key, key)

        modifier_order = {
            "ctrl": 0, "control": 0, "alt": 1, "menu": 1,
            "shift": 2, "win": 3, "windows": 3, "super": 3,
            "cmd": 3, "command": 3,
        }

        parts = [p.strip() for p in key.split("+")]
        if not parts:
            return key

        modifiers = []
        trigger = []
        for p in parts:
            if p in modifier_order:
                modifiers.append(p)
            else:
                trigger.append(p)

        normalized_mods = []
        for m in modifiers:
            if m in ("control",):
                m = "ctrl"
            elif m in ("menu",):
                m = "alt"
            elif m in ("windows", "super", "cmd", "command"):
                m = "win"
            normalized_mods.append(m)

        seen = set()
        unique = []
        for m in normalized_mods:
            if m not in seen:
                seen.add(m)
                unique.append(m)
        unique.sort(key=lambda x: modifier_order.get(x, 99))

        return "+".join(unique + trigger)

    @staticmethod
    def _safe_callback(callback, description):
        def wrapped():
            try:
                callback()
            except Exception as e:
                logger.error("%s 回调异常: %s", description, e, exc_info=True)
        return wrapped

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.shutdown()
        return False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass
