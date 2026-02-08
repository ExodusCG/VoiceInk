"""
VoiceInk - 自定义词典管理模块

提供自定义词典的加载、保存、增删改查、ASR prompt 生成、
别名纠错替换、批量导入导出等功能。

线程安全：所有对词典数据的读写操作均通过 threading.Lock 保护。
"""

import json
import csv
import logging
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ============================================================
# 尝试导入 pypinyin，不可用时优雅降级
# ============================================================
try:
    from pypinyin import pinyin, Style

    _HAS_PYPINYIN = True
except ImportError:
    _HAS_PYPINYIN = False
    logger.debug("pypinyin 未安装，将跳过自动拼音生成")

# ============================================================
# 当前词典文件格式版本号
# ============================================================
DICT_FORMAT_VERSION = "1.0"


# ============================================================
# 词条数据类
# ============================================================
@dataclass
class DictEntry:
    """
    单条自定义词汇。

    Attributes:
        term:     正确词汇（用于 ASR prompt 及最终输出）
        aliases:  常见错误形式列表（用于后处理纠错替换）
        category: 词汇分类，如 "人名"、"地名"、"专业术语" 等
        pinyin:   词汇拼音（自动生成或手动指定）
        enabled:  是否启用（禁用后不参与 prompt 与纠错）
    """

    term: str
    aliases: list[str] = field(default_factory=list)
    category: str = ""
    pinyin: str = ""
    enabled: bool = True

    # ----------------------------------------------------------
    # 序列化 / 反序列化
    # ----------------------------------------------------------
    def to_dict(self) -> dict:
        """将词条转换为可 JSON 序列化的字典"""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "DictEntry":
        """从字典创建词条实例"""
        return cls(
            term=data.get("term", ""),
            aliases=data.get("aliases", []),
            category=data.get("category", ""),
            pinyin=data.get("pinyin", ""),
            enabled=data.get("enabled", True),
        )


# ============================================================
# 拼音工具函数
# ============================================================
def _generate_pinyin(text: str) -> str:
    """
    为中文文本生成拼音字符串。

    如果 pypinyin 不可用或文本不含中文字符，则返回空字符串。

    Args:
        text: 需要生成拼音的文本

    Returns:
        以空格分隔的拼音字符串，例如 "zhōng guó"
    """
    if not _HAS_PYPINYIN:
        return ""

    # 仅当文本包含中文字符时才生成拼音
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return ""

    try:
        result = pinyin(text, style=Style.TONE)
        return " ".join(item[0] for item in result)
    except Exception as exc:
        logger.warning("生成拼音失败: %s -> %s", text, exc)
        return ""


# ============================================================
# 自定义词典管理器
# ============================================================
class CustomDictionary:
    """
    自定义词典管理器。

    负责词典 JSON 文件的读写，以及对词条的增删改查操作。
    所有公共方法均为线程安全。

    典型用法::

        from voiceink.config import load_config
        config = load_config()
        dictionary = CustomDictionary(config.dictionary)
        dictionary.load()

        # 添加词汇
        dictionary.add_term("VoiceInk", aliases=["voiceing", "voice ink"], category="产品名")

        # 生成 ASR 热词 prompt
        prompt = dictionary.get_asr_prompt()

        # 对识别结果做后处理纠错
        corrected = dictionary.apply_corrections(raw_text)
    """

    def __init__(self, config=None):
        """
        初始化自定义词典。

        Args:
            config: DictionaryConfig 实例；为 None 时使用默认值
        """
        # ----------------------------------------------------------
        # 配置参数
        # ----------------------------------------------------------
        if config is not None:
            self._dict_path = Path(config.path)
            self._max_prompt_terms = config.max_asr_prompt_terms
            self._enabled = config.enabled
        else:
            self._dict_path = Path("custom_dictionary.json")
            self._max_prompt_terms = 50
            self._enabled = True

        # 如果路径是相对路径，则基于本包上级目录解析
        if not self._dict_path.is_absolute():
            self._dict_path = Path(__file__).parent.parent / self._dict_path

        # ----------------------------------------------------------
        # 内部数据：term -> DictEntry 的有序映射
        # ----------------------------------------------------------
        self._entries: dict[str, DictEntry] = {}

        # ----------------------------------------------------------
        # 线程锁
        # ----------------------------------------------------------
        self._lock = threading.Lock()

        logger.info("自定义词典初始化完成，路径: %s", self._dict_path)

    # ==========================================================
    # 文件读写
    # ==========================================================

    def load(self) -> None:
        """
        从 JSON 文件加载词典数据。

        如果文件不存在或解析失败，将初始化为空词典并记录警告日志。
        """
        with self._lock:
            self._entries.clear()

            if not self._dict_path.exists():
                logger.info("词典文件不存在，将使用空词典: %s", self._dict_path)
                return

            try:
                with open(self._dict_path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)

                version = data.get("version", "unknown")
                logger.info("加载词典文件，版本: %s", version)

                for item in data.get("terms", []):
                    entry = DictEntry.from_dict(item)
                    if entry.term:
                        self._entries[entry.term] = entry

                logger.info("成功加载 %d 条词汇", len(self._entries))

            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("加载词典文件失败: %s", exc)

    def save(self) -> None:
        """
        将当前词典数据保存到 JSON 文件。

        文件格式包含 version 字段和 terms 列表。
        """
        with self._lock:
            data = {
                "version": DICT_FORMAT_VERSION,
                "terms": [entry.to_dict() for entry in self._entries.values()],
            }

            # 确保父目录存在
            self._dict_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                with open(self._dict_path, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, ensure_ascii=False, indent=2)
                logger.info("词典已保存到 %s（共 %d 条）", self._dict_path, len(self._entries))
            except OSError as exc:
                logger.error("保存词典文件失败: %s", exc)

    # ==========================================================
    # 增删改查
    # ==========================================================

    def add_term(
        self,
        term: str,
        aliases: Optional[list[str]] = None,
        category: str = "",
        pinyin: str = "",
        enabled: bool = True,
    ) -> bool:
        """
        添加新词汇到词典。

        如果词汇已存在，则返回 False 并记录警告。
        如果未提供拼音，将尝试自动生成中文拼音。

        Args:
            term:     正确词汇
            aliases:  常见错误形式列表
            category: 词汇分类
            pinyin:   拼音（留空则自动生成）
            enabled:  是否启用

        Returns:
            添加成功返回 True，词汇已存在返回 False
        """
        if not term or not term.strip():
            logger.warning("词汇不能为空")
            return False

        term = term.strip()

        with self._lock:
            if term in self._entries:
                logger.warning("词汇已存在: %s", term)
                return False

            # 自动生成拼音
            if not pinyin:
                pinyin = _generate_pinyin(term)

            entry = DictEntry(
                term=term,
                aliases=aliases or [],
                category=category,
                pinyin=pinyin,
                enabled=enabled,
            )
            self._entries[term] = entry
            logger.info("添加词汇: %s (分类: %s)", term, category or "无")
            return True

    def remove_term(self, term: str) -> bool:
        """
        从词典中删除指定词汇。

        Args:
            term: 要删除的词汇

        Returns:
            删除成功返回 True，词汇不存在返回 False
        """
        with self._lock:
            if term in self._entries:
                del self._entries[term]
                logger.info("删除词汇: %s", term)
                return True
            else:
                logger.warning("要删除的词汇不存在: %s", term)
                return False

    def update_term(
        self,
        term: str,
        aliases: Optional[list[str]] = None,
        category: Optional[str] = None,
        pinyin: Optional[str] = None,
        enabled: Optional[bool] = None,
    ) -> bool:
        """
        更新已有词汇的属性。

        仅更新传入的非 None 参数，其余保持不变。

        Args:
            term:     目标词汇（不可更改）
            aliases:  新的别名列表
            category: 新的分类
            pinyin:   新的拼音
            enabled:  新的启用状态

        Returns:
            更新成功返回 True，词汇不存在返回 False
        """
        with self._lock:
            entry = self._entries.get(term)
            if entry is None:
                logger.warning("要更新的词汇不存在: %s", term)
                return False

            if aliases is not None:
                entry.aliases = aliases
            if category is not None:
                entry.category = category
            if pinyin is not None:
                entry.pinyin = pinyin
            if enabled is not None:
                entry.enabled = enabled

            logger.info("更新词汇: %s", term)
            return True

    def get_term(self, term: str) -> Optional[DictEntry]:
        """
        获取指定词汇的词条。

        Args:
            term: 词汇名称

        Returns:
            DictEntry 实例，不存在则返回 None
        """
        with self._lock:
            return self._entries.get(term)

    def get_all_terms(self) -> list[DictEntry]:
        """
        获取所有词条列表。

        Returns:
            所有 DictEntry 的列表（副本）
        """
        with self._lock:
            return list(self._entries.values())

    def get_active_terms(self) -> list[str]:
        """
        获取所有启用的词汇名列表。

        Returns:
            启用词汇的 term 字符串列表
        """
        with self._lock:
            return [
                entry.term
                for entry in self._entries.values()
                if entry.enabled
            ]

    def search(self, keyword: str) -> list[DictEntry]:
        """
        根据关键词搜索词汇。

        在 term、aliases、category、pinyin 中进行模糊匹配。

        Args:
            keyword: 搜索关键词（不区分大小写）

        Returns:
            匹配的 DictEntry 列表
        """
        if not keyword:
            return self.get_all_terms()

        keyword_lower = keyword.lower()
        results: list[DictEntry] = []

        with self._lock:
            for entry in self._entries.values():
                # 在 term 中搜索
                if keyword_lower in entry.term.lower():
                    results.append(entry)
                    continue

                # 在别名中搜索
                if any(keyword_lower in alias.lower() for alias in entry.aliases):
                    results.append(entry)
                    continue

                # 在分类中搜索
                if keyword_lower in entry.category.lower():
                    results.append(entry)
                    continue

                # 在拼音中搜索
                if entry.pinyin and keyword_lower in entry.pinyin.lower():
                    results.append(entry)
                    continue

        return results

    # ==========================================================
    # ASR Prompt 生成
    # ==========================================================

    def get_asr_prompt(self) -> str:
        """
        生成 ASR initial_prompt 字符串。

        将所有启用词汇用逗号连接，数量不超过 max_asr_prompt_terms。
        这些词汇将作为 Whisper 的 initial_prompt，引导 ASR 正确识别专有名词。

        Returns:
            逗号分隔的词汇字符串，例如 "VoiceInk,语音输入,人工智能"
        """
        if not self._enabled:
            return ""

        with self._lock:
            active = [
                entry.term
                for entry in self._entries.values()
                if entry.enabled
            ]

        # 截取不超过上限数量
        active = active[: self._max_prompt_terms]
        return ",".join(active)

    # ==========================================================
    # 后处理纠错
    # ==========================================================

    def apply_corrections(self, text: str) -> str:
        """
        基于别名表对文本做字符串替换后处理。

        遍历所有启用词汇的 aliases，将文本中出现的错误形式
        替换为正确的 term。替换按别名长度从长到短排序，
        以避免短别名误匹配长别名的子串。

        Args:
            text: 需要纠错的原始文本

        Returns:
            纠错后的文本
        """
        if not self._enabled or not text:
            return text

        # 收集所有 (alias -> term) 的替换对
        replacements: list[tuple[str, str]] = []

        with self._lock:
            for entry in self._entries.values():
                if not entry.enabled:
                    continue
                for alias in entry.aliases:
                    if alias:  # 忽略空别名
                        replacements.append((alias, entry.term))

        # 按别名长度降序排序，优先替换更长的匹配
        replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

        # 逐一执行替换
        for alias, term in replacements:
            text = text.replace(alias, term)

        return text

    # ==========================================================
    # 批量导入 / 导出
    # ==========================================================

    def import_terms(self, file_path: str) -> int:
        """
        从文件批量导入词汇。

        支持以下格式：
        - JSON 文件（.json）：与词典文件相同的格式
        - CSV 文件（.csv）：列顺序为 term, aliases(分号分隔), category
        - 文本文件（.txt）：每行一个词汇

        Args:
            file_path: 导入文件路径

        Returns:
            成功导入的词汇数量
        """
        path = Path(file_path)

        if not path.exists():
            logger.error("导入文件不存在: %s", file_path)
            return 0

        suffix = path.suffix.lower()
        imported_count = 0

        try:
            if suffix == ".json":
                imported_count = self._import_from_json(path)
            elif suffix == ".csv":
                imported_count = self._import_from_csv(path)
            elif suffix == ".txt":
                imported_count = self._import_from_txt(path)
            else:
                logger.error("不支持的导入格式: %s", suffix)
                return 0
        except Exception as exc:
            logger.error("导入失败: %s", exc)
            return 0

        logger.info("从 %s 导入了 %d 条词汇", file_path, imported_count)
        return imported_count

    def _import_from_json(self, path: Path) -> int:
        """从 JSON 文件导入词汇"""
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        count = 0
        for item in data.get("terms", []):
            entry = DictEntry.from_dict(item)
            if entry.term and self.add_term(
                term=entry.term,
                aliases=entry.aliases,
                category=entry.category,
                pinyin=entry.pinyin,
                enabled=entry.enabled,
            ):
                count += 1
        return count

    def _import_from_csv(self, path: Path) -> int:
        """
        从 CSV 文件导入词汇。

        CSV 列格式：term, aliases(分号分隔), category
        """
        count = 0
        with open(path, "r", encoding="utf-8") as fh:
            reader = csv.reader(fh)
            for row in reader:
                if not row or not row[0].strip():
                    continue
                # 跳过表头行
                if row[0].strip().lower() == "term":
                    continue

                term = row[0].strip()
                aliases = [a.strip() for a in row[1].split(";")] if len(row) > 1 and row[1] else []
                category = row[2].strip() if len(row) > 2 else ""

                if self.add_term(term=term, aliases=aliases, category=category):
                    count += 1

        return count

    def _import_from_txt(self, path: Path) -> int:
        """从纯文本文件导入词汇（每行一个词汇）"""
        count = 0
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                term = line.strip()
                if term and self.add_term(term=term):
                    count += 1
        return count

    def export_terms(self, file_path: str) -> bool:
        """
        导出词典到文件。

        支持以下格式：
        - JSON 文件（.json）：包含完整词条信息
        - CSV 文件（.csv）：列顺序为 term, aliases, category, pinyin, enabled

        Args:
            file_path: 导出文件路径

        Returns:
            导出成功返回 True，失败返回 False
        """
        path = Path(file_path)
        suffix = path.suffix.lower()

        try:
            # 确保父目录存在
            path.parent.mkdir(parents=True, exist_ok=True)

            if suffix == ".json":
                return self._export_to_json(path)
            elif suffix == ".csv":
                return self._export_to_csv(path)
            else:
                logger.error("不支持的导出格式: %s", suffix)
                return False
        except Exception as exc:
            logger.error("导出失败: %s", exc)
            return False

    def _export_to_json(self, path: Path) -> bool:
        """导出为 JSON 格式"""
        with self._lock:
            data = {
                "version": DICT_FORMAT_VERSION,
                "terms": [entry.to_dict() for entry in self._entries.values()],
            }

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)

        logger.info("词典已导出到 %s（JSON 格式）", path)
        return True

    def _export_to_csv(self, path: Path) -> bool:
        """导出为 CSV 格式"""
        with self._lock:
            entries = list(self._entries.values())

        with open(path, "w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            # 写入表头
            writer.writerow(["term", "aliases", "category", "pinyin", "enabled"])
            for entry in entries:
                writer.writerow([
                    entry.term,
                    ";".join(entry.aliases),
                    entry.category,
                    entry.pinyin,
                    str(entry.enabled),
                ])

        logger.info("词典已导出到 %s（CSV 格式）", path)
        return True

    # ==========================================================
    # 实用属性和方法
    # ==========================================================

    @property
    def count(self) -> int:
        """词典中的词条总数"""
        with self._lock:
            return len(self._entries)

    @property
    def active_count(self) -> int:
        """启用的词条数"""
        with self._lock:
            return sum(1 for e in self._entries.values() if e.enabled)

    @property
    def dict_path(self) -> Path:
        """词典文件路径"""
        return self._dict_path

    @property
    def enabled(self) -> bool:
        """词典是否启用"""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool) -> None:
        """设置词典启用状态"""
        self._enabled = value
        logger.info("词典启用状态: %s", value)

    def clear(self) -> None:
        """清空所有词条"""
        with self._lock:
            self._entries.clear()
            logger.info("词典已清空")

    def __repr__(self) -> str:
        return (
            f"CustomDictionary(path={self._dict_path!r}, "
            f"total={self.count}, active={self.active_count})"
        )

    def __len__(self) -> int:
        return self.count
