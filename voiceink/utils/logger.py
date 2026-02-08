"""
VoiceInk - 日志工具
"""

import logging
import sys
from pathlib import Path


def setup_logger(name: str = "voiceink", level: str = "INFO") -> logging.Logger:
    """
    设置并返回 logger。

    Args:
        name: logger 名称
        level: 日志级别

    Returns:
        配置好的 Logger 实例
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # 子 logger 不要重复输出到父 logger 的 handler
    if "." in name:
        logger.propagate = False

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s.%(module)s: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(console_fmt)
    logger.addHandler(console_handler)

    # 文件输出
    log_dir = Path.home() / ".voiceink" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(
        log_dir / "voiceink.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-7s %(name)s.%(module)s:%(lineno)d: %(message)s"
    )
    file_handler.setFormatter(file_fmt)
    logger.addHandler(file_handler)

    return logger


# 全局 logger
logger = setup_logger()
