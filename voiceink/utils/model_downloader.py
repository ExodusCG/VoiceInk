"""
VoiceInk - 模型下载工具

从 HuggingFace 仓库下载 ASR（Whisper）和 LLM（Qwen）模型文件。
支持：
  - 预定义模型信息（名称、URL、大小、SHA256）
  - 断点续传（检查已下载大小，使用 Range 头继续下载）
  - 下载进度回调
  - SHA256 校验（可选）
  - 推荐模型组合查询

模型来源：
  - Whisper GGML 模型: https://huggingface.co/ggml-org/whisper-{size}
  - Qwen GGUF 模型:    https://huggingface.co/Qwen/

使用方式::

    from voiceink.utils.model_downloader import ModelDownloader

    downloader = ModelDownloader()

    # 检查模型是否已存在
    if not downloader.check_model_exists("asr", "base", "./models"):
        # 下载模型（带进度回调）
        downloader.download_model(
            model_type="asr",
            model_name="base",
            target_dir="./models",
            progress_callback=lambda downloaded, total, speed: print(
                f"\\r下载进度: {downloaded}/{total} ({speed:.1f} KB/s)", end=""
            ),
        )
"""

import os
import hashlib
import logging
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional, Callable, Dict, Tuple

logger = logging.getLogger("voiceink")


# ============================================================
# 预定义模型信息
# ============================================================
# 结构: {
#     "模型类型": {
#         "模型名称": {
#             "filename": "文件名",
#             "url": "下载地址",
#             "size": 预计大小（字节），0 表示未知,
#             "sha256": "校验哈希" 或 ""（空表示跳过校验）,
#             "description": "描述",
#         }
#     }
# }

# HuggingFace 基础 URL
_HF_BASE = "https://huggingface.co"

# ---------------------------------------------------------------------------
# ASR 模型：Whisper GGML 格式（用于 whisper.cpp / pywhispercpp）
# 来源: https://huggingface.co/ggerganov/whisper.cpp
# ---------------------------------------------------------------------------
_ASR_MODELS: Dict[str, dict] = {
    "tiny": {
        "filename": "ggml-tiny.bin",
        "url": f"{_HF_BASE}/ggerganov/whisper.cpp/resolve/main/ggml-tiny.bin",
        "size": 77_691_713,       # ~74 MB
        "sha256": "",
        "description": "Whisper tiny - 最快速度，适合实时场景（精度较低）",
    },
    "base": {
        "filename": "ggml-base.bin",
        "url": f"{_HF_BASE}/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "size": 147_951_465,      # ~141 MB
        "sha256": "",
        "description": "Whisper base - 速度与精度的平衡（推荐）",
    },
    "small": {
        "filename": "ggml-small.bin",
        "url": f"{_HF_BASE}/ggerganov/whisper.cpp/resolve/main/ggml-small.bin",
        "size": 487_601_967,      # ~465 MB
        "sha256": "",
        "description": "Whisper small - 更高精度，需要较多内存",
    },
    "medium": {
        "filename": "ggml-medium.bin",
        "url": f"{_HF_BASE}/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin",
        "size": 1_533_774_848,    # ~1.5 GB
        "sha256": "",
        "description": "Whisper medium - 高精度，需要大量内存",
    },
    "large-v3": {
        "filename": "ggml-large-v3.bin",
        "url": f"{_HF_BASE}/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin",
        "size": 3_094_623_744,    # ~2.9 GB
        "sha256": "",
        "description": "Whisper large-v3 - 最高精度，需要大量资源",
    },
}

# ---------------------------------------------------------------------------
# LLM 模型：Qwen GGUF 格式（用于 llama.cpp）
# 来源: https://huggingface.co/Qwen/
# ---------------------------------------------------------------------------
_LLM_MODELS: Dict[str, dict] = {
    "qwen3-0.6b-q8_0": {
        "filename": "Qwen3-0.6B-Q8_0.gguf",
        "url": f"{_HF_BASE}/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf",
        "size": 639_446_688,      # ~610 MB
        "sha256": "",
        "description": "Qwen3-0.6B Q8_0 - 超轻量润色模型（推荐，CPU 友好）",
    },
    "qwen2.5-1.5b-instruct-q4_k_m": {
        "filename": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "url": f"{_HF_BASE}/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf",
        "size": 1_049_927_808,    # ~1.0 GB
        "sha256": "",
        "description": "Qwen2.5-1.5B Q4_K_M - 较高润色质量，需要更多内存",
    },
    "qwen2.5-3b-instruct-q4_k_m": {
        "filename": "qwen2.5-3b-instruct-q4_k_m.gguf",
        "url": f"{_HF_BASE}/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf",
        "size": 2_058_878_976,    # ~1.9 GB
        "sha256": "",
        "description": "Qwen2.5-3B Q4_K_M - 高质量润色，需要较多资源",
    },
}

# 合并所有模型信息，按类型分组
MODEL_REGISTRY: Dict[str, Dict[str, dict]] = {
    "asr": _ASR_MODELS,
    "llm": _LLM_MODELS,
}


# ============================================================
# 进度回调类型定义
# ============================================================
# progress_callback(downloaded_bytes: int, total_bytes: int, speed_kbps: float)
ProgressCallback = Callable[[int, int, float], None]


# ============================================================
# 工具函数
# ============================================================

def _format_size(size_bytes: int) -> str:
    """
    将字节数格式化为人类可读的字符串。

    Args:
        size_bytes: 字节数

    Returns:
        格式化后的字符串，如 "141.2 MB"
    """
    if size_bytes <= 0:
        return "未知大小"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def _compute_sha256(filepath: str, chunk_size: int = 8192) -> str:
    """
    计算文件的 SHA256 哈希值。

    Args:
        filepath:   文件路径
        chunk_size: 读取块大小

    Returns:
        小写十六进制 SHA256 字符串
    """
    sha256 = hashlib.sha256()
    with open(filepath, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            sha256.update(chunk)
    return sha256.hexdigest()


# ============================================================
# ModelDownloader 类
# ============================================================

class ModelDownloader:
    """
    模型下载管理器。

    负责从远程仓库下载 ASR 和 LLM 模型文件到本地目录。
    支持断点续传、进度回调和 SHA256 校验。

    用法::

        downloader = ModelDownloader()

        # 列出所有可用模型
        for model_type, models in downloader.list_models().items():
            for name, info in models.items():
                print(f"[{model_type}] {name}: {info['description']}")

        # 下载 ASR 模型
        downloader.download_model("asr", "base", "./models/asr")

        # 下载 LLM 模型
        downloader.download_model("llm", "qwen3-0.6b-q8_0", "./models/llm")
    """

    # 下载缓冲区大小（8 KB）
    CHUNK_SIZE = 8192

    # HTTP 请求超时（秒）
    TIMEOUT = 30

    # User-Agent（部分 CDN 要求提供有效的 UA 才允许下载）
    USER_AGENT = "VoiceInk-ModelDownloader/1.0"

    def __init__(self):
        """初始化模型下载器。"""
        logger.info("ModelDownloader 初始化完成")

    # ------------------------------------------------------------------
    # 公共 API
    # ------------------------------------------------------------------

    def list_models(self) -> Dict[str, Dict[str, dict]]:
        """
        列出所有预定义的可下载模型。

        Returns:
            嵌套字典 {模型类型: {模型名称: 模型信息}}
        """
        return MODEL_REGISTRY

    def get_model_info(self, model_type: str, model_name: str) -> Optional[dict]:
        """
        获取指定模型的详细信息。

        Args:
            model_type: 模型类型，"asr" 或 "llm"
            model_name: 模型名称（如 "base"、"qwen3-0.6b-q8_0"）

        Returns:
            模型信息字典，不存在则返回 None
        """
        return MODEL_REGISTRY.get(model_type, {}).get(model_name)

    def check_model_exists(
        self,
        model_type: str,
        model_name: str,
        target_dir: str,
    ) -> bool:
        """
        检查模型文件是否已存在于目标目录中。

        检查逻辑：
        1. 文件是否存在
        2. 文件大小是否与预期一致（如果预期大小 > 0）

        Args:
            model_type: 模型类型（"asr" / "llm"）
            model_name: 模型名称
            target_dir: 目标目录路径

        Returns:
            True 表示模型文件已存在且大小正确
        """
        info = self.get_model_info(model_type, model_name)
        if info is None:
            logger.warning(
                "未找到模型信息: type=%s, name=%s",
                model_type, model_name,
            )
            return False

        filepath = Path(target_dir) / info["filename"]

        if not filepath.exists():
            return False

        # 如果有预期大小，检查文件大小是否匹配
        expected_size = info.get("size", 0)
        if expected_size > 0:
            actual_size = filepath.stat().st_size
            if actual_size != expected_size:
                logger.warning(
                    "模型文件大小不匹配: %s (预期 %s, 实际 %s)",
                    filepath.name,
                    _format_size(expected_size),
                    _format_size(actual_size),
                )
                return False

        logger.debug("模型文件已存在: %s", filepath)
        return True

    def download_model(
        self,
        model_type: str,
        model_name: str,
        target_dir: str,
        progress_callback: Optional[ProgressCallback] = None,
        verify_sha256: bool = True,
    ) -> str:
        """
        下载指定模型到目标目录。

        支持断点续传：如果目标文件已部分下载（.part 临时文件存在），
        将从已下载位置继续下载。

        Args:
            model_type:        模型类型（"asr" / "llm"）
            model_name:        模型名称
            target_dir:        目标目录路径（不存在则自动创建）
            progress_callback: 进度回调函数，签名: (downloaded_bytes, total_bytes, speed_kbps) -> None
            verify_sha256:     下载完成后是否进行 SHA256 校验（仅在模型信息中提供了哈希时生效）

        Returns:
            下载完成后的文件完整路径

        Raises:
            ValueError: 模型类型或名称无效
            ConnectionError: 网络连接失败
            IOError: 文件写入失败
            RuntimeError: SHA256 校验失败
        """
        # ---- 1. 验证模型信息 ----
        info = self.get_model_info(model_type, model_name)
        if info is None:
            available = list(MODEL_REGISTRY.get(model_type, {}).keys())
            raise ValueError(
                f"未知的模型: type='{model_type}', name='{model_name}'\n"
                f"可用的 {model_type} 模型: {available}"
            )

        url = info["url"]
        filename = info["filename"]
        expected_size = info.get("size", 0)
        expected_sha256 = info.get("sha256", "")
        description = info.get("description", "")

        logger.info(
            "准备下载模型: %s (%s) -> %s",
            model_name, _format_size(expected_size), target_dir,
        )
        logger.info("  描述: %s", description)
        logger.info("  URL: %s", url)

        # ---- 2. 创建目标目录 ----
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        # 最终文件路径
        final_path = target_path / filename
        # 临时文件路径（用于断点续传）
        part_path = target_path / f"{filename}.part"

        # ---- 3. 检查是否已完成下载 ----
        if final_path.exists():
            actual_size = final_path.stat().st_size
            if expected_size <= 0 or actual_size == expected_size:
                logger.info("模型文件已存在，跳过下载: %s", final_path)
                return str(final_path)
            else:
                # 文件大小不匹配，删除后重新下载
                logger.warning(
                    "已存在的模型文件大小不正确 (预期 %s, 实际 %s)，将重新下载",
                    _format_size(expected_size), _format_size(actual_size),
                )
                final_path.unlink()

        # ---- 4. 断点续传：检查临时文件的已下载大小 ----
        downloaded_size = 0
        if part_path.exists():
            downloaded_size = part_path.stat().st_size
            logger.info(
                "发现未完成的下载，将从 %s 处继续 (已下载 %s)",
                _format_size(downloaded_size), _format_size(downloaded_size),
            )

        # ---- 5. 构建 HTTP 请求 ----
        request = urllib.request.Request(url)
        request.add_header("User-Agent", self.USER_AGENT)

        # 如果有已下载的部分，添加 Range 头实现断点续传
        if downloaded_size > 0:
            request.add_header("Range", f"bytes={downloaded_size}-")
            logger.debug("设置 Range 头: bytes=%d-", downloaded_size)

        # ---- 6. 执行下载 ----
        import time
        try:
            response = urllib.request.urlopen(request, timeout=self.TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code == 416:
                # 416 Range Not Satisfiable：文件已完全下载
                logger.info("服务器返回 416，文件已完全下载")
                if part_path.exists():
                    part_path.rename(final_path)
                return str(final_path)
            raise ConnectionError(
                f"下载模型失败 (HTTP {e.code}): {e.reason}\nURL: {url}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"无法连接到下载服务器: {e.reason}\n"
                f"URL: {url}\n"
                f"请检查网络连接，或尝试配置代理。"
            ) from e

        # 获取响应中的总大小信息
        content_length = response.headers.get("Content-Length")
        if content_length:
            total_size = downloaded_size + int(content_length)
        elif expected_size > 0:
            total_size = expected_size
        else:
            total_size = 0

        # 检查服务器是否支持断点续传
        content_range = response.headers.get("Content-Range")
        if downloaded_size > 0 and not content_range:
            # 服务器不支持 Range 请求，从头开始下载
            logger.warning("服务器不支持断点续传，将从头开始下载")
            downloaded_size = 0
            if part_path.exists():
                part_path.unlink()

        logger.info(
            "开始下载: 总大小 %s, 已下载 %s",
            _format_size(total_size), _format_size(downloaded_size),
        )

        # ---- 7. 写入文件（追加模式用于断点续传） ----
        open_mode = "ab" if downloaded_size > 0 else "wb"
        start_time = time.time()
        last_callback_time = start_time

        try:
            with open(str(part_path), open_mode) as f:
                while True:
                    chunk = response.read(self.CHUNK_SIZE)
                    if not chunk:
                        break

                    f.write(chunk)
                    downloaded_size += len(chunk)

                    # 进度回调（每 0.3 秒最多触发一次，避免过于频繁）
                    now = time.time()
                    if progress_callback and (now - last_callback_time >= 0.3):
                        elapsed = now - start_time
                        speed_kbps = (downloaded_size / 1024.0) / max(elapsed, 0.001)
                        progress_callback(downloaded_size, total_size, speed_kbps)
                        last_callback_time = now

        except Exception as e:
            logger.error("下载中断: %s (已保存 %s)", e, _format_size(downloaded_size))
            raise IOError(
                f"下载模型时发生错误: {e}\n"
                f"已下载 {_format_size(downloaded_size)}，可重新运行以继续下载。"
            ) from e
        finally:
            response.close()

        # ---- 8. 最终进度回调 ----
        if progress_callback:
            elapsed = time.time() - start_time
            speed_kbps = (downloaded_size / 1024.0) / max(elapsed, 0.001)
            progress_callback(downloaded_size, total_size, speed_kbps)

        logger.info(
            "下载完成: %s (大小: %s, 耗时: %.1f 秒)",
            filename, _format_size(downloaded_size), time.time() - start_time,
        )

        # ---- 9. SHA256 校验（可选） ----
        if verify_sha256 and expected_sha256:
            logger.info("正在校验 SHA256...")
            actual_sha256 = _compute_sha256(str(part_path))
            if actual_sha256 != expected_sha256.lower():
                # 校验失败，删除文件
                part_path.unlink(missing_ok=True)
                raise RuntimeError(
                    f"SHA256 校验失败！\n"
                    f"  预期: {expected_sha256}\n"
                    f"  实际: {actual_sha256}\n"
                    f"文件可能已损坏，已删除。请重新下载。"
                )
            logger.info("SHA256 校验通过: %s", actual_sha256[:16] + "...")

        # ---- 10. 重命名临时文件为最终文件 ----
        part_path.rename(final_path)
        logger.info("模型已保存: %s", final_path)

        return str(final_path)

    def get_recommended_models(self) -> dict:
        """
        返回推荐的模型组合。

        针对不同使用场景（轻量、平衡、高质量）提供推荐配置。

        Returns:
            字典，键为场景名称，值为推荐的模型配置::

                {
                    "轻量": {
                        "asr": {"name": "tiny", "description": "..."},
                        "llm": {"name": "qwen3-0.6b-q8_0", "description": "..."},
                        "total_size": "~453 MB",
                        "description": "适合低配电脑，速度最快",
                    },
                    ...
                }
        """
        return {
            "轻量（推荐低配电脑）": {
                "asr": {
                    "name": "tiny",
                    "info": _ASR_MODELS["tiny"],
                },
                "llm": {
                    "name": "qwen3-0.6b-q8_0",
                    "info": _LLM_MODELS["qwen3-0.6b-q8_0"],
                },
                "total_size": _format_size(
                    _ASR_MODELS["tiny"]["size"]
                    + _LLM_MODELS["qwen3-0.6b-q8_0"]["size"]
                ),
                "description": "速度最快，资源占用最低。适合实时语音输入、低配电脑。",
            },
            "平衡（推荐大多数用户）": {
                "asr": {
                    "name": "base",
                    "info": _ASR_MODELS["base"],
                },
                "llm": {
                    "name": "qwen3-0.6b-q8_0",
                    "info": _LLM_MODELS["qwen3-0.6b-q8_0"],
                },
                "total_size": _format_size(
                    _ASR_MODELS["base"]["size"]
                    + _LLM_MODELS["qwen3-0.6b-q8_0"]["size"]
                ),
                "description": "识别精度与速度的最佳平衡。推荐大多数用户使用。",
            },
            "高质量": {
                "asr": {
                    "name": "small",
                    "info": _ASR_MODELS["small"],
                },
                "llm": {
                    "name": "qwen2.5-1.5b-instruct-q4_k_m",
                    "info": _LLM_MODELS["qwen2.5-1.5b-instruct-q4_k_m"],
                },
                "total_size": _format_size(
                    _ASR_MODELS["small"]["size"]
                    + _LLM_MODELS["qwen2.5-1.5b-instruct-q4_k_m"]["size"]
                ),
                "description": "更高精度的识别和润色。需要 4GB+ 可用内存。",
            },
        }

    def get_default_model_dir(self) -> str:
        """
        获取默认模型存储目录。

        Returns:
            默认模型目录路径（~/.voiceink/models/）
        """
        model_dir = Path.home() / ".voiceink" / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        return str(model_dir)

    def get_model_filepath(
        self,
        model_type: str,
        model_name: str,
        target_dir: Optional[str] = None,
    ) -> Optional[str]:
        """
        获取模型文件的完整路径（不检查是否存在）。

        Args:
            model_type: 模型类型（"asr" / "llm"）
            model_name: 模型名称
            target_dir: 目标目录，None 则使用默认目录

        Returns:
            文件完整路径字符串，模型信息不存在则返回 None
        """
        info = self.get_model_info(model_type, model_name)
        if info is None:
            return None

        if target_dir is None:
            target_dir = self.get_default_model_dir()

        return str(Path(target_dir) / info["filename"])


# ============================================================
# 模块级便捷函数
# ============================================================

# 全局单例（延迟初始化）
_downloader: Optional[ModelDownloader] = None


def get_downloader() -> ModelDownloader:
    """获取全局 ModelDownloader 单例。"""
    global _downloader
    if _downloader is None:
        _downloader = ModelDownloader()
    return _downloader
