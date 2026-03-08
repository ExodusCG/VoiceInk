"""
VoiceInk - 词典管理面板

使用 tkinter 实现自定义词典的管理界面。
功能：
  - Treeview 表格显示词条（词汇、别名、分类、启用状态）
  - 添加 / 编辑 / 删除词条
  - 搜索过滤
  - 导入 / 导出 JSON 文件
  - 启用 / 禁用切换
"""

import json
import copy
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
from typing import Optional, Callable, List, Dict

from voiceink.utils.logger import logger


# ---------------------------------------------------------------------------
# 词条数据结构
# ---------------------------------------------------------------------------

class DictEntry:
    """单条词典词条"""

    def __init__(
        self,
        word: str = "",
        aliases: str = "",
        category: str = "通用",
        enabled: bool = True,
    ):
        self.word = word          # 词汇
        self.aliases = aliases    # 别名（逗号分隔）
        self.category = category  # 分类
        self.enabled = enabled    # 是否启用

    def to_dict(self) -> dict:
        return {
            "word": self.word,
            "aliases": self.aliases,
            "category": self.category,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "DictEntry":
        return cls(
            word=data.get("word", ""),
            aliases=data.get("aliases", ""),
            category=data.get("category", "通用"),
            enabled=data.get("enabled", True),
        )


class DictionaryPanel:
    """
    词典管理面板。

    用法::

        panel = DictionaryPanel(
            entries=existing_entries,
            on_save=lambda entries: save_to_file(entries),
        )
        panel.show()
    """

    # 默认分类列表
    CATEGORIES = ["通用", "技术", "医学", "法律", "金融", "人名", "地名", "其他"]

    def __init__(
        self,
        entries: Optional[List[Dict]] = None,
        on_save: Optional[Callable[[List[Dict]], None]] = None,
        parent: Optional[tk.Tk] = None,
    ):
        """
        初始化词典面板。

        Args:
            entries: 词条列表（字典格式），None 则为空
            on_save: 保存回调，参数为修改后的词条列表
            parent:  父窗口
        """
        # 将字典列表转为 DictEntry 对象
        self._entries: List[DictEntry] = []
        if entries:
            for e in entries:
                if isinstance(e, dict):
                    self._entries.append(DictEntry.from_dict(e))
                elif isinstance(e, DictEntry):
                    self._entries.append(copy.deepcopy(e))

        self._on_save = on_save
        self._parent = parent

        # UI 引用
        self._window: Optional[tk.Toplevel] = None
        self._tree: Optional[ttk.Treeview] = None
        self._search_var: Optional[tk.StringVar] = None

        logger.info("DictionaryPanel 初始化 (%d 条词条)", len(self._entries))

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def show(self):
        """显示词典面板"""
        if self._parent is None:
            self._parent = tk.Tk()
            self._parent.withdraw()

        self._create_window()

        # 启动 tkinter 事件循环
        self._parent.mainloop()

    # ------------------------------------------------------------------
    # 窗口构建
    # ------------------------------------------------------------------

    def _create_window(self):
        """创建主窗口"""
        win = tk.Toplevel(self._parent)
        self._window = win
        win.title("VoiceInk 自定义词典")
        win.geometry("700x500")
        win.resizable(True, True)
        win.minsize(500, 350)

        # 窗口关闭时退出 mainloop
        win.protocol("WM_DELETE_WINDOW", self._on_close)

        # 居中
        win.update_idletasks()
        x = (win.winfo_screenwidth() - 700) // 2
        y = (win.winfo_screenheight() - 500) // 2
        win.geometry(f"+{x}+{y}")

        # ---- 顶部工具栏 ----
        toolbar = tk.Frame(win)
        toolbar.pack(fill=tk.X, padx=8, pady=(8, 4))

        # 搜索
        ttk.Label(toolbar, text="搜索:").pack(side=tk.LEFT)
        self._search_var = tk.StringVar(master=self._parent)
        self._search_var.trace_add("write", self._on_search_changed)
        search_entry = ttk.Entry(toolbar, textvariable=self._search_var, width=25)
        search_entry.pack(side=tk.LEFT, padx=(4, 12))

        # 操作按钮
        ttk.Button(toolbar, text="添加", command=self._on_add).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="编辑", command=self._on_edit).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="删除", command=self._on_delete).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="切换启用", command=self._on_toggle_enable).pack(side=tk.LEFT, padx=2)

        ttk.Separator(toolbar, orient="vertical").pack(side=tk.LEFT, fill=tk.Y, padx=8)

        ttk.Button(toolbar, text="导入", command=self._on_import).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="导出", command=self._on_export).pack(side=tk.LEFT, padx=2)

        # 词条计数
        self._count_label = ttk.Label(toolbar, text="", foreground="gray")
        self._count_label.pack(side=tk.RIGHT)

        # ---- Treeview 表格 ----
        tree_frame = tk.Frame(win)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        columns = ("word", "aliases", "category", "enabled")
        self._tree = ttk.Treeview(
            tree_frame,
            columns=columns,
            show="headings",
            selectmode="extended",
        )

        # 列定义
        self._tree.heading("word", text="词汇", command=lambda: self._sort_by("word"))
        self._tree.heading("aliases", text="别名", command=lambda: self._sort_by("aliases"))
        self._tree.heading("category", text="分类", command=lambda: self._sort_by("category"))
        self._tree.heading("enabled", text="启用", command=lambda: self._sort_by("enabled"))

        self._tree.column("word", width=160, minwidth=80)
        self._tree.column("aliases", width=220, minwidth=80)
        self._tree.column("category", width=100, minwidth=60)
        self._tree.column("enabled", width=60, minwidth=40, anchor="center")

        # 双击编辑
        self._tree.bind("<Double-1>", lambda e: self._on_edit())

        # 滚动条
        scrollbar = ttk.Scrollbar(tree_frame, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=scrollbar.set)

        self._tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # ---- 底部按钮 ----
        btn_frame = tk.Frame(win)
        btn_frame.pack(fill=tk.X, padx=8, pady=8)

        ttk.Button(btn_frame, text="关闭", command=self._on_close).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="保存", command=self._on_save_click).pack(side=tk.RIGHT)

        # ---- 排序状态 ----
        self._sort_column = "word"
        self._sort_reverse = False

        # 初始加载数据
        self._refresh_tree()

    # ------------------------------------------------------------------
    # 表格操作
    # ------------------------------------------------------------------

    def _refresh_tree(self, filter_text: str = ""):
        """
        刷新 Treeview 内容。

        Args:
            filter_text: 搜索过滤文本（为空则显示全部）
        """
        if not self._tree:
            return

        # 清空
        for item in self._tree.get_children():
            self._tree.delete(item)

        # 过滤
        filter_lower = filter_text.lower().strip()
        visible_count = 0

        for idx, entry in enumerate(self._entries):
            # 搜索过滤：匹配词汇、别名、分类
            if filter_lower:
                searchable = f"{entry.word} {entry.aliases} {entry.category}".lower()
                if filter_lower not in searchable:
                    continue

            enabled_text = "✓" if entry.enabled else "✗"
            tag = "enabled" if entry.enabled else "disabled"

            self._tree.insert(
                "",
                "end",
                iid=str(idx),
                values=(entry.word, entry.aliases, entry.category, enabled_text),
                tags=(tag,),
            )
            visible_count += 1

        # 设置标签样式
        self._tree.tag_configure("disabled", foreground="gray")
        self._tree.tag_configure("enabled", foreground="black")

        # 更新计数
        self._update_count(visible_count)

    def _update_count(self, visible: Optional[int] = None):
        """更新词条计数标签"""
        total = len(self._entries)
        if visible is not None and visible != total:
            text = f"显示 {visible} / 共 {total} 条"
        else:
            text = f"共 {total} 条"
        if self._count_label:
            self._count_label.config(text=text)

    def _get_selected_indices(self) -> List[int]:
        """获取当前选中的词条索引"""
        selection = self._tree.selection() if self._tree else ()
        indices = []
        for item_id in selection:
            try:
                indices.append(int(item_id))
            except ValueError:
                pass
        return indices

    def _sort_by(self, column: str):
        """按列排序"""
        if self._sort_column == column:
            self._sort_reverse = not self._sort_reverse
        else:
            self._sort_column = column
            self._sort_reverse = False

        key_map = {
            "word": lambda e: e.word.lower(),
            "aliases": lambda e: e.aliases.lower(),
            "category": lambda e: e.category.lower(),
            "enabled": lambda e: (0 if e.enabled else 1),
        }

        key_func = key_map.get(column, key_map["word"])
        self._entries.sort(key=key_func, reverse=self._sort_reverse)
        self._refresh_tree(self._search_var.get() if self._search_var else "")

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _on_search_changed(self, *args):
        """搜索框文本变化"""
        text = self._search_var.get() if self._search_var else ""
        self._refresh_tree(text)

    def _on_add(self):
        """添加词条"""
        dialog = _EntryDialog(self._window, title="添加词条", categories=self.CATEGORIES)
        if dialog.result:
            self._entries.append(dialog.result)
            self._refresh_tree(self._search_var.get() if self._search_var else "")
            logger.info("添加词条: %s", dialog.result.word)

    def _on_edit(self):
        """编辑选中词条"""
        indices = self._get_selected_indices()
        if not indices:
            messagebox.showwarning("提示", "请先选择要编辑的词条")
            return

        idx = indices[0]
        entry = self._entries[idx]
        dialog = _EntryDialog(
            self._window,
            title="编辑词条",
            categories=self.CATEGORIES,
            entry=entry,
        )
        if dialog.result:
            self._entries[idx] = dialog.result
            self._refresh_tree(self._search_var.get() if self._search_var else "")
            logger.info("编辑词条: %s", dialog.result.word)

    def _on_delete(self):
        """删除选中词条"""
        indices = self._get_selected_indices()
        if not indices:
            messagebox.showwarning("提示", "请先选择要删除的词条")
            return

        count = len(indices)
        result = messagebox.askyesno("确认删除", f"确定要删除选中的 {count} 条词条吗？")
        if not result:
            return

        # 从后往前删除以避免索引偏移
        for idx in sorted(indices, reverse=True):
            if 0 <= idx < len(self._entries):
                deleted = self._entries.pop(idx)
                logger.info("删除词条: %s", deleted.word)

        self._refresh_tree(self._search_var.get() if self._search_var else "")

    def _on_toggle_enable(self):
        """切换选中词条的启用/禁用状态"""
        indices = self._get_selected_indices()
        if not indices:
            messagebox.showwarning("提示", "请先选择词条")
            return

        for idx in indices:
            if 0 <= idx < len(self._entries):
                self._entries[idx].enabled = not self._entries[idx].enabled

        self._refresh_tree(self._search_var.get() if self._search_var else "")

    def _on_import(self):
        """从 JSON 文件导入词条"""
        filepath = filedialog.askopenfilename(
            title="导入词典",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not filepath:
            return

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                raise ValueError("JSON 文件应包含一个列表")

            imported = 0
            for item in data:
                if isinstance(item, dict) and "word" in item:
                    self._entries.append(DictEntry.from_dict(item))
                    imported += 1

            self._refresh_tree(self._search_var.get() if self._search_var else "")
            messagebox.showinfo("导入成功", f"成功导入 {imported} 条词条")
            logger.info("从 %s 导入 %d 条词条", filepath, imported)

        except Exception as e:
            logger.error("导入词典失败: %s", e, exc_info=True)
            messagebox.showerror("导入失败", f"导入词典时出错:\n{e}")

    def _on_export(self):
        """导出词条到 JSON 文件"""
        filepath = filedialog.asksaveasfilename(
            title="导出词典",
            defaultextension=".json",
            filetypes=[("JSON 文件", "*.json"), ("所有文件", "*.*")],
        )
        if not filepath:
            return

        try:
            data = [entry.to_dict() for entry in self._entries]
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            messagebox.showinfo("导出成功", f"已导出 {len(data)} 条词条到:\n{filepath}")
            logger.info("导出 %d 条词条到 %s", len(data), filepath)

        except Exception as e:
            logger.error("导出词典失败: %s", e, exc_info=True)
            messagebox.showerror("导出失败", f"导出词典时出错:\n{e}")

    def _on_save_click(self):
        """保存按钮"""
        data = [entry.to_dict() for entry in self._entries]
        if self._on_save:
            self._on_save(data)
        logger.info("词典已保存 (%d 条)", len(data))
        if self._window:
            self._window.destroy()
        if self._parent:
            self._parent.quit()

    def _on_close(self):
        """关闭按钮"""
        if self._window:
            self._window.destroy()
        if self._parent:
            self._parent.quit()


# ---------------------------------------------------------------------------
# 词条编辑对话框
# ---------------------------------------------------------------------------

class _EntryDialog:
    """
    添加/编辑词条的模态对话框。

    完成后 self.result 为 DictEntry 或 None（取消）。
    """

    def __init__(
        self,
        parent: tk.Toplevel,
        title: str = "词条",
        categories: Optional[List[str]] = None,
        entry: Optional[DictEntry] = None,
    ):
        self.result: Optional[DictEntry] = None

        self._dialog = tk.Toplevel(parent)
        self._dialog.title(title)
        self._dialog.geometry("400x260")
        self._dialog.resizable(False, False)
        self._dialog.grab_set()
        self._dialog.transient(parent)

        # 居中于父窗口
        self._dialog.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - 400) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 260) // 2
        self._dialog.geometry(f"+{px}+{py}")

        frame = ttk.Frame(self._dialog, padding=15)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(1, weight=1)

        # ---- 词汇 ----
        ttk.Label(frame, text="词汇:").grid(row=0, column=0, sticky="w", pady=4)
        self._word_var = tk.StringVar(value=entry.word if entry else "")
        ttk.Entry(frame, textvariable=self._word_var, width=35).grid(
            row=0, column=1, sticky="we", pady=4,
        )

        # ---- 别名 ----
        ttk.Label(frame, text="别名:").grid(row=1, column=0, sticky="w", pady=4)
        self._aliases_var = tk.StringVar(value=entry.aliases if entry else "")
        ttk.Entry(frame, textvariable=self._aliases_var, width=35).grid(
            row=1, column=1, sticky="we", pady=4,
        )
        ttk.Label(frame, text="多个别名用逗号分隔", foreground="gray").grid(
            row=2, column=1, sticky="w",
        )

        # ---- 分类 ----
        ttk.Label(frame, text="分类:").grid(row=3, column=0, sticky="w", pady=4)
        self._category_var = tk.StringVar(value=entry.category if entry else "通用")
        cat_combo = ttk.Combobox(
            frame,
            textvariable=self._category_var,
            values=categories or ["通用"],
            width=15,
        )
        cat_combo.grid(row=3, column=1, sticky="w", pady=4)

        # ---- 启用 ----
        self._enabled_var = tk.BooleanVar(value=entry.enabled if entry else True)
        ttk.Checkbutton(frame, text="启用此词条", variable=self._enabled_var).grid(
            row=4, column=0, columnspan=2, sticky="w", pady=8,
        )

        # ---- 按钮 ----
        btn_frame = tk.Frame(frame)
        btn_frame.grid(row=5, column=0, columnspan=2, sticky="e", pady=(8, 0))

        ttk.Button(btn_frame, text="取消", command=self._on_cancel).pack(side=tk.RIGHT, padx=(4, 0))
        ttk.Button(btn_frame, text="确定", command=self._on_ok).pack(side=tk.RIGHT)

        # 等待窗口关闭
        self._dialog.wait_window()

    def _on_ok(self):
        """确定按钮"""
        word = self._word_var.get().strip()
        if not word:
            messagebox.showwarning("提示", "词汇不能为空", parent=self._dialog)
            return

        self.result = DictEntry(
            word=word,
            aliases=self._aliases_var.get().strip(),
            category=self._category_var.get().strip() or "通用",
            enabled=self._enabled_var.get(),
        )
        self._dialog.destroy()

    def _on_cancel(self):
        """取消按钮"""
        self.result = None
        self._dialog.destroy()
