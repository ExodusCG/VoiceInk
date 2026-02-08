"""
VoiceInk - 全局快捷键管理模块

基于 keyboard 库实现系统级全局快捷键监听。
支持：
  - Push-to-Talk（按住说话）模式：按下触发、释放触发
  - 普通快捷键：按下触发回调
  - 快捷键冲突检测
  - 线程安全的注册 / 注销
"""

import threading
import logging
from typing import Callable, Dict, Optional, Set

import keyboard

logger = logging.getLogger("voiceink")


class HotkeyManager:
    """
    全局快捷键管理器。

    所有公共方法均为线程安全。内部使用 threading.Lock 保护共享状态。

    使用方式：
        manager = HotkeyManager()

        # 注册 Push-to-Talk 快捷键
        manager.register_push_to_talk(
            key="ctrl+shift+space",
            on_press=lambda: print("开始录音"),
            on_release=lambda: print("停止录音"),
        )

        # 注册普通快捷键
        manager.register_hotkey(
            key="ctrl+shift+v",
            callback=lambda: print("切换模式"),
        )

        # 注销所有快捷键
        manager.unregister_all()
    """

    def __init__(self):
        # ---- 线程锁 ----
        self._lock = threading.Lock()

        # ---- 已注册的普通快捷键 ----
        # 结构: { 标准化键名: keyboard 库返回的 hook/hotkey 句柄 }
        self._hotkeys: Dict[str, object] = {}

        # ---- 已注册的 Push-to-Talk 快捷键 ----
        # 结构: { 标准化键名: {"on_press_hook": hook, "on_release_hook": hook} }
        self._ptt_hotkeys: Dict[str, Dict[str, object]] = {}

        # ---- 所有已占用的键名集合（用于冲突检测）----
        self._registered_keys: Set[str] = set()

        logger.info("HotkeyManager 初始化完成")

    # ============================================================
    # 公共 API
    # ============================================================

    def register_push_to_talk(
        self,
        key: str,
        on_press: Callable[[], None],
        on_release: Callable[[], None],
    ) -> bool:
        """
        注册 Push-to-Talk（按住说话）快捷键。

        按下 key 时触发 on_press 回调，释放 key 时触发 on_release 回调。

        Args:
            key:        快捷键组合字符串，如 "ctrl+shift+space"
            on_press:   按下时的回调函数（无参数）
            on_release: 释放时的回调函数（无参数）

        Returns:
            True 表示注册成功，False 表示快捷键冲突
        """
        normalized = self._normalize_key(key)

        with self._lock:
            # 冲突检测
            if normalized in self._registered_keys:
                logger.warning(
                    "快捷键冲突：'%s' 已被注册，无法重复注册为 Push-to-Talk",
                    normalized,
                )
                return False

            try:
                # 使用 keyboard.add_hotkey 分别监听按下和释放
                # trigger_on_release=False：按下时触发
                on_press_hook = keyboard.add_hotkey(
                    normalized,
                    callback=self._safe_callback(on_press, f"PTT 按下 [{normalized}]"),
                    suppress=False,
                    trigger_on_release=False,
                )

                # 监听释放事件：使用 on_release 钩子
                # keyboard 库的 add_hotkey 不直接支持 release 回调，
                # 我们通过 hook 监听按键事件来实现精确的释放检测
                on_release_hook = self._register_ptt_release(
                    normalized,
                    self._safe_callback(on_release, f"PTT 释放 [{normalized}]"),
                )

                self._ptt_hotkeys[normalized] = {
                    "on_press_hook": on_press_hook,
                    "on_release_hook": on_release_hook,
                }
                self._registered_keys.add(normalized)

                logger.info("Push-to-Talk 快捷键已注册: %s", normalized)
                return True

            except Exception as e:
                logger.error("注册 Push-to-Talk 快捷键失败 [%s]: %s", normalized, e)
                # 回滚：清理可能已部分注册的钩子
                self._cleanup_ptt(normalized)
                return False

    def register_hotkey(
        self,
        key: str,
        callback: Callable[[], None],
    ) -> bool:
        """
        注册普通全局快捷键。

        按下 key 组合时触发 callback。

        Args:
            key:      快捷键组合字符串，如 "ctrl+shift+v"
            callback: 触发时的回调函数（无参数）

        Returns:
            True 表示注册成功，False 表示快捷键冲突
        """
        normalized = self._normalize_key(key)

        with self._lock:
            # 冲突检测
            if normalized in self._registered_keys:
                logger.warning(
                    "快捷键冲突：'%s' 已被注册，无法重复注册",
                    normalized,
                )
                return False

            try:
                hook = keyboard.add_hotkey(
                    normalized,
                    callback=self._safe_callback(callback, f"快捷键 [{normalized}]"),
                    suppress=False,
                    trigger_on_release=False,
                )

                self._hotkeys[normalized] = hook
                self._registered_keys.add(normalized)

                logger.info("快捷键已注册: %s", normalized)
                return True

            except Exception as e:
                logger.error("注册快捷键失败 [%s]: %s", normalized, e)
                return False

    def unregister_hotkey(self, key: str) -> bool:
        """
        注销指定快捷键。

        Args:
            key: 快捷键组合字符串

        Returns:
            True 表示注销成功，False 表示该快捷键未注册
        """
        normalized = self._normalize_key(key)

        with self._lock:
            if normalized not in self._registered_keys:
                logger.warning("尝试注销未注册的快捷键: %s", normalized)
                return False

            # 注销普通快捷键
            if normalized in self._hotkeys:
                try:
                    keyboard.remove_hotkey(self._hotkeys[normalized])
                except Exception as e:
                    logger.warning("注销快捷键 [%s] 时出错: %s", normalized, e)
                del self._hotkeys[normalized]

            # 注销 Push-to-Talk 快捷键
            if normalized in self._ptt_hotkeys:
                self._cleanup_ptt(normalized)

            self._registered_keys.discard(normalized)
            logger.info("快捷键已注销: %s", normalized)
            return True

    def unregister_all(self) -> None:
        """
        注销所有已注册的快捷键。

        此方法会安全地移除所有普通快捷键和 Push-to-Talk 快捷键的钩子。
        """
        with self._lock:
            # 注销所有普通快捷键
            for key_name, hook in list(self._hotkeys.items()):
                try:
                    keyboard.remove_hotkey(hook)
                    logger.debug("已注销快捷键: %s", key_name)
                except Exception as e:
                    logger.warning("注销快捷键 [%s] 时出错: %s", key_name, e)

            # 注销所有 Push-to-Talk 快捷键
            for key_name in list(self._ptt_hotkeys.keys()):
                self._cleanup_ptt(key_name)

            # 清空所有记录
            self._hotkeys.clear()
            self._ptt_hotkeys.clear()
            self._registered_keys.clear()

            logger.info("所有快捷键已注销")

    def is_registered(self, key: str) -> bool:
        """
        检查某个快捷键是否已注册。

        Args:
            key: 快捷键组合字符串

        Returns:
            True 表示已注册
        """
        normalized = self._normalize_key(key)
        with self._lock:
            return normalized in self._registered_keys

    @property
    def registered_keys(self) -> Set[str]:
        """返回当前所有已注册快捷键的集合（副本）。"""
        with self._lock:
            return self._registered_keys.copy()

    # ============================================================
    # 内部方法
    # ============================================================

    def _register_ptt_release(
        self,
        hotkey_str: str,
        release_callback: Callable[[], None],
    ) -> object:
        """
        为 Push-to-Talk 注册释放回调。

        通过解析组合键中的各个按键，监听最后一个非修饰键（触发键）的释放事件。
        只有当组合键处于"已按下"状态时，释放触发键才会触发回调。

        Args:
            hotkey_str:       标准化后的快捷键字符串
            release_callback: 释放时的回调

        Returns:
            keyboard 库的 hook 对象，用于后续注销
        """
        # 解析组合键中的各个键名
        parts = [p.strip() for p in hotkey_str.split("+")]
        # 最后一个键视为"触发键"，前面的都是修饰键
        trigger_key = parts[-1] if parts else hotkey_str

        # 用于跟踪按下状态的标志（线程安全由 GIL 保证 bool 赋值的原子性）
        state = {"pressed": False}

        def on_key_event(event: keyboard.KeyboardEvent):
            """监听所有键盘事件，检测组合键的按下和释放。"""
            if event.event_type == keyboard.KEY_DOWN:
                # 检测组合键是否完全按下
                if keyboard.is_pressed(hotkey_str):
                    state["pressed"] = True

            elif event.event_type == keyboard.KEY_UP:
                # 触发键释放且之前处于"已按下"状态
                if state["pressed"] and event.name == trigger_key:
                    state["pressed"] = False
                    release_callback()

        hook = keyboard.hook(on_key_event)
        return hook

    def _cleanup_ptt(self, key_name: str) -> None:
        """
        清理 Push-to-Talk 快捷键的所有钩子。

        注意：此方法在已持有 _lock 的上下文中调用，不应再获取锁。

        Args:
            key_name: 标准化后的快捷键名
        """
        if key_name not in self._ptt_hotkeys:
            return

        hooks = self._ptt_hotkeys[key_name]

        # 注销按下回调（通过 add_hotkey 注册的）
        if "on_press_hook" in hooks and hooks["on_press_hook"] is not None:
            try:
                keyboard.remove_hotkey(hooks["on_press_hook"])
            except Exception as e:
                logger.debug("注销 PTT 按下钩子 [%s] 时出错: %s", key_name, e)

        # 注销释放回调（通过 keyboard.hook 注册的）
        if "on_release_hook" in hooks and hooks["on_release_hook"] is not None:
            try:
                keyboard.unhook(hooks["on_release_hook"])
            except Exception as e:
                logger.debug("注销 PTT 释放钩子 [%s] 时出错: %s", key_name, e)

        del self._ptt_hotkeys[key_name]
        logger.debug("已清理 Push-to-Talk 快捷键: %s", key_name)

    @staticmethod
    def _normalize_key(key: str) -> str:
        """
        标准化快捷键字符串，确保一致的比较和查找。

        处理规则：
        1. 转为小写
        2. 按 '+' 分割后去除各部分的空白
        3. 将修饰键排序（ctrl < alt < shift < win），触发键保持在末尾
        4. 重新用 '+' 拼接

        示例：
            "Shift+Ctrl+A"  -> "ctrl+shift+a"
            " ctrl + shift + space " -> "ctrl+shift+space"

        Args:
            key: 原始快捷键字符串

        Returns:
            标准化后的快捷键字符串
        """
        # 修饰键的排序权重（越小越靠前）
        modifier_order = {
            "ctrl": 0, "control": 0,
            "alt": 1, "menu": 1,
            "shift": 2,
            "win": 3, "windows": 3, "super": 3, "cmd": 3, "command": 3,
        }

        parts = [p.strip().lower() for p in key.split("+")]
        if not parts:
            return key.lower().strip()

        # 分离修饰键和触发键
        modifiers = []
        trigger = []
        for p in parts:
            if p in modifier_order:
                modifiers.append(p)
            else:
                trigger.append(p)

        # 标准化修饰键名称
        normalized_modifiers = []
        for m in modifiers:
            # 统一别名：control -> ctrl, menu -> alt, windows/super/cmd/command -> win
            if m in ("control",):
                m = "ctrl"
            elif m in ("menu",):
                m = "alt"
            elif m in ("windows", "super", "cmd", "command"):
                m = "win"
            normalized_modifiers.append(m)

        # 修饰键去重并按权重排序
        seen = set()
        unique_modifiers = []
        for m in normalized_modifiers:
            if m not in seen:
                seen.add(m)
                unique_modifiers.append(m)
        unique_modifiers.sort(key=lambda x: modifier_order.get(x, 99))

        # 拼接：修饰键 + 触发键
        all_parts = unique_modifiers + trigger
        return "+".join(all_parts)

    @staticmethod
    def _safe_callback(
        callback: Callable[[], None],
        description: str,
    ) -> Callable[[], None]:
        """
        将回调包装为安全版本，捕获并记录所有异常。

        快捷键回调中的未捕获异常会导致 keyboard 库的监听线程崩溃，
        因此所有回调都必须经过包装。

        Args:
            callback:    原始回调函数
            description: 回调的描述信息（用于日志）

        Returns:
            包装后的安全回调函数
        """
        def wrapped():
            try:
                callback()
            except Exception as e:
                logger.error("%s 回调执行异常: %s", description, e, exc_info=True)

        return wrapped

    # ============================================================
    # 上下文管理器支持
    # ============================================================

    def __enter__(self):
        """支持 with 语句，返回自身。"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出 with 语句时自动注销所有快捷键。"""
        self.unregister_all()
        return False

    def __del__(self):
        """析构时尝试注销所有快捷键（最后的安全网）。"""
        try:
            self.unregister_all()
        except Exception:
            pass
