"""
VoiceInk - 自定义词典模块

提供自定义词典管理功能，包括词条的增删改查、
ASR prompt 生成、别名纠错替换、批量导入导出等。

主要导出：
- DictEntry:        单条词汇的数据类
- CustomDictionary: 自定义词典管理器
"""

from .custom_dict import DictEntry, CustomDictionary, DICT_FORMAT_VERSION

__all__ = [
    "DictEntry",
    "CustomDictionary",
    "DICT_FORMAT_VERSION",
]
