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
import tarfile
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

# HuggingFace 镜像（中国大陆可访问）
_HF_MIRROR = "https://hf-mirror.com"

# GitHub Releases 基础 URL (用于 sherpa-onnx 模型)
_GITHUB_RELEASES_BASE = "https://github.com/k2-fsa/sherpa-onnx/releases/download"

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

# ---------------------------------------------------------------------------
# SenseVoice 模型：sherpa-onnx 格式（用于 sensevoice_onnx 后端）
# 来源: HuggingFace 镜像 (中国大陆可访问)
# 注意：SenseVoice 模型是目录形式，包含 model.int8.onnx 和 tokens.txt
# ---------------------------------------------------------------------------
_SENSEVOICE_MODEL_NAME = "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-int8-2024-07-17"
_SENSEVOICE_HF_REPO = "csukuangfj/sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17"

# SenseVoice 模型文件列表（需要下载的文件）
_SENSEVOICE_FILES = [
    {
        "filename": "model.int8.onnx",
        "url": f"{_HF_MIRROR}/{_SENSEVOICE_HF_REPO}/resolve/main/model.int8.onnx",
        "size": 231_000_000,  # ~220 MB
    },
    {
        "filename": "tokens.txt",
        "url": f"{_HF_MIRROR}/{_SENSEVOICE_HF_REPO}/resolve/main/tokens.txt",
        "size": 315_000,  # ~307 KB
    },
]

_SENSEVOICE_MODELS: Dict[str, dict] = {
    "int8": {
        "filename": _SENSEVOICE_MODEL_NAME,  # 目录名
        "files": _SENSEVOICE_FILES,  # 需要下载的文件列表
        "size": 231_000_000,  # 主模型文件大小（用于显示）
        "sha256": "",
        "description": "SenseVoice int8 量化 - 多语言 ASR (zh/en/ja/ko/yue)，体积小速度快",
        "is_multi_file": True,  # 标记为多文件模型
    },
}

# 合并所有模型信息，按类型分组
MODEL_REGISTRY: Dict[str, Dict[str, dict]] = {
    "asr": _ASR_MODELS,
    "llm": _LLM_MODELS,
    "sensevoice": _SENSEVOICE_MODELS,  # SenseVoice 单独分类（因为是目录形式）
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


def _extract_tar_bz2(archive_path: str, target_dir: str) -> str:
    """
    解压 tar.bz2 压缩包。

    Args:
        archive_path: 压缩包路径
        target_dir:   解压目标目录

    Returns:
        解压后的目录/文件路径
    """
    logger.info(f"正在解压: {archive_path}")
    with tarfile.open(archive_path, "r:bz2") as tar:
        # 获取压缩包内的顶级目录名
        members = tar.getmembers()
        if not members:
            raise RuntimeError(f"压缩包为空: {archive_path}")

        # 提取顶级目录名（假设所有文件都在一个目录下）
        top_dir = members[0].name.split("/")[0]

        # 解压到目标目录
        tar.extractall(target_dir)

    extracted_path = Path(target_dir) / top_dir
    logger.info(f"解压完成: {extracted_path}")
    return str(extracted_path)


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
        1. 单文件模型：文件是否存在且大小正确
        2. 多文件模型：目录是否存在且包含所有必要文件

        Args:
            model_type: 模型类型（"asr" / "llm" / "sensevoice"）
            model_name: 模型名称
            target_dir: 目标目录路径

        Returns:
            True 表示模型文件已存在且完整
        """
        info = self.get_model_info(model_type, model_name)
        if info is None:
            logger.warning(
                "未找到模型信息: type=%s, name=%s",
                model_type, model_name,
            )
            return False

        filepath = Path(target_dir) / info["filename"]

        # 检查是否为多文件模型（如 SenseVoice）
        is_multi_file = info.get("is_multi_file", False)
        if is_multi_file:
            # 检查目录是否存在
            if not filepath.is_dir():
                return False

            # 检查所有必要文件是否存在
            files = info.get("files", [])
            for file_info in files:
                file_path = filepath / file_info["filename"]
                if not file_path.exists():
                    logger.warning("多文件模型不完整，缺少: %s", file_path)
                    return False

            logger.debug("多文件模型已完整存在: %s", filepath)
            return True

        # 单文件模型检查
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

        支持：
        - 单文件模型（直接下载）
        - 多文件模型（如 SenseVoice，下载多个文件到子目录）
        - 断点续传

        Args:
            model_type:        模型类型（"asr" / "llm" / "sensevoice"）
            model_name:        模型名称
            target_dir:        目标目录路径（不存在则自动创建）
            progress_callback: 进度回调函数，签名: (downloaded_bytes, total_bytes, speed_kbps) -> None
            verify_sha256:     下载完成后是否进行 SHA256 校验（仅在模型信息中提供了哈希时生效）

        Returns:
            下载完成后的文件/目录完整路径

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

        filename = info["filename"]
        expected_size = info.get("size", 0)
        description = info.get("description", "")
        is_multi_file = info.get("is_multi_file", False)

        logger.info(
            "准备下载模型: %s (%s) -> %s",
            model_name, _format_size(expected_size), target_dir,
        )
        logger.info("  描述: %s", description)

        # ---- 2. 创建目标目录 ----
        target_path = Path(target_dir)
        target_path.mkdir(parents=True, exist_ok=True)

        # ---- 3. 多文件模型下载（如 SenseVoice） ----
        if is_multi_file:
            return self._download_multi_file_model(
                info=info,
                model_type=model_type,
                target_dir=target_dir,
                progress_callback=progress_callback,
            )

        # ---- 4. 单文件模型下载 ----
        url = info.get("url", "")
        if not url:
            raise ValueError(f"模型信息中缺少 URL: {model_type}/{model_name}")

        logger.info("  URL: %s", url)
        expected_sha256 = info.get("sha256", "")

        final_path = target_path / filename
        part_path = target_path / f"{filename}.part"

        # 检查是否已存在
        if final_path.exists():
            actual_size = final_path.stat().st_size
            if expected_size <= 0 or actual_size == expected_size:
                logger.info("模型文件已存在，跳过下载: %s", final_path)
                return str(final_path)
            else:
                logger.warning(
                    "已存在的模型文件大小不正确 (预期 %s, 实际 %s)，将重新下载",
                    _format_size(expected_size), _format_size(actual_size),
                )
                final_path.unlink()

        # 下载文件
        self._download_single_file(
            url=url,
            target_path=final_path,
            part_path=part_path,
            expected_size=expected_size,
            expected_sha256=expected_sha256 if verify_sha256 else "",
            progress_callback=progress_callback,
        )

        return str(final_path)

    def _download_multi_file_model(
        self,
        info: dict,
        model_type: str,
        target_dir: str,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> str:
        """
        下载多文件模型（如 SenseVoice）。

        Args:
            info: 模型信息字典
            model_type: 模型类型
            target_dir: 目标目录
            progress_callback: 进度回调

        Returns:
            模型目录路径
        """
        import time

        files = info.get("files", [])
        if not files:
            raise ValueError(f"多文件模型信息中缺少 files 列表: {info.get('filename')}")

        # 创建模型子目录
        model_dir_name = info["filename"]
        model_dir = Path(target_dir) / model_dir_name
        model_dir.mkdir(parents=True, exist_ok=True)

        # 检查是否已完整下载
        all_exist = True
        for file_info in files:
            file_path = model_dir / file_info["filename"]
            if not file_path.exists():
                all_exist = False
                break

        if all_exist:
            logger.info("多文件模型已完整存在，跳过下载: %s", model_dir)
            return str(model_dir)

        # 计算总大小
        total_size = sum(f.get("size", 0) for f in files)
        total_downloaded = 0

        logger.info("开始下载多文件模型 (%d 个文件, 总计 %s)", len(files), _format_size(total_size))

        start_time = time.time()

        for i, file_info in enumerate(files):
            file_name = file_info["filename"]
            file_url = file_info["url"]
            file_size = file_info.get("size", 0)

            file_path = model_dir / file_name
            part_path = model_dir / f"{file_name}.part"

            logger.info("  [%d/%d] 下载: %s", i + 1, len(files), file_name)

            # 如果文件已存在且大小正确，跳过
            if file_path.exists():
                actual_size = file_path.stat().st_size
                if file_size <= 0 or actual_size == file_size:
                    logger.info("    已存在，跳过")
                    total_downloaded += actual_size
                    continue
                else:
                    file_path.unlink()

            # 创建内部进度回调，累加到总进度
            def make_file_progress_callback(base_downloaded: int):
                def inner_callback(downloaded: int, file_total: int, speed: float):
                    if progress_callback:
                        progress_callback(base_downloaded + downloaded, total_size, speed)
                return inner_callback

            # 下载单个文件
            self._download_single_file(
                url=file_url,
                target_path=file_path,
                part_path=part_path,
                expected_size=file_size,
                expected_sha256="",
                progress_callback=make_file_progress_callback(total_downloaded),
            )

            total_downloaded += file_size if file_size > 0 else file_path.stat().st_size

        elapsed = time.time() - start_time
        logger.info(
            "多文件模型下载完成: %s (耗时 %.1f 秒)",
            model_dir, elapsed,
        )

        return str(model_dir)

    def _download_single_file(
        self,
        url: str,
        target_path: Path,
        part_path: Path,
        expected_size: int = 0,
        expected_sha256: str = "",
        progress_callback: Optional[ProgressCallback] = None,
    ) -> None:
        """
        下载单个文件，支持断点续传。

        Args:
            url: 下载 URL
            target_path: 最终文件路径
            part_path: 临时文件路径（用于断点续传）
            expected_size: 预期文件大小
            expected_sha256: 预期 SHA256（空则跳过校验）
            progress_callback: 进度回调
        """
        import time

        # 断点续传：检查临时文件
        downloaded_size = 0
        if part_path.exists():
            downloaded_size = part_path.stat().st_size
            logger.debug("发现未完成的下载，从 %s 处继续", _format_size(downloaded_size))

        # 构建 HTTP 请求
        request = urllib.request.Request(url)
        request.add_header("User-Agent", self.USER_AGENT)

        if downloaded_size > 0:
            request.add_header("Range", f"bytes={downloaded_size}-")

        # 执行下载
        try:
            response = urllib.request.urlopen(request, timeout=self.TIMEOUT)
        except urllib.error.HTTPError as e:
            if e.code == 416:
                # 文件已完全下载
                if part_path.exists():
                    part_path.rename(target_path)
                return
            raise ConnectionError(
                f"下载失败 (HTTP {e.code}): {e.reason}\nURL: {url}"
            ) from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"无法连接到下载服务器: {e.reason}\nURL: {url}"
            ) from e

        # 获取总大小
        content_length = response.headers.get("Content-Length")
        if content_length:
            total_size = downloaded_size + int(content_length)
        elif expected_size > 0:
            total_size = expected_size
        else:
            total_size = 0

        # 检查断点续传支持
        content_range = response.headers.get("Content-Range")
        if downloaded_size > 0 and not content_range:
            logger.warning("服务器不支持断点续传，从头开始下载")
            downloaded_size = 0
            if part_path.exists():
                part_path.unlink()

        # 写入文件
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

                    now = time.time()
                    if progress_callback and (now - last_callback_time >= 0.3):
                        elapsed = now - start_time
                        speed_kbps = (downloaded_size / 1024.0) / max(elapsed, 0.001)
                        progress_callback(downloaded_size, total_size, speed_kbps)
                        last_callback_time = now

        except Exception as e:
            logger.error("下载中断: %s", e)
            raise IOError(f"下载文件时发生错误: {e}") from e
        finally:
            response.close()

        # 最终进度回调
        if progress_callback:
            elapsed = time.time() - start_time
            speed_kbps = (downloaded_size / 1024.0) / max(elapsed, 0.001)
            progress_callback(downloaded_size, total_size, speed_kbps)

        # SHA256 校验
        if expected_sha256:
            actual_sha256 = _compute_sha256(str(part_path))
            if actual_sha256 != expected_sha256.lower():
                part_path.unlink(missing_ok=True)
                raise RuntimeError(f"SHA256 校验失败: {target_path.name}")
            logger.debug("SHA256 校验通过: %s", actual_sha256[:16] + "...")

        # 重命名为最终文件
        part_path.rename(target_path)
        logger.debug("文件已保存: %s", target_path)

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


def download_sensevoice_model(
    target_dir: Optional[str] = None,
    progress_callback: Optional[ProgressCallback] = None,
) -> str:
    """
    下载 SenseVoice 多语言 ASR 模型（便捷函数）。

    SenseVoice 是阿里达摩院的多语言语音识别模型，
    支持中文、英文、日语、韩语、粤语和自动语言检测。

    Args:
        target_dir: 目标目录路径（默认: models/asr/sensevoice/）
        progress_callback: 进度回调函数

    Returns:
        下载完成后的模型目录路径

    Example:
        >>> from voiceink.utils.model_downloader import download_sensevoice_model
        >>> model_dir = download_sensevoice_model()
        >>> print(f"Model downloaded to: {model_dir}")
    """
    if target_dir is None:
        target_dir = "models/asr/sensevoice"

    downloader = get_downloader()
    return downloader.download_model(
        model_type="sensevoice",
        model_name="int8",
        target_dir=target_dir,
        progress_callback=progress_callback,
    )
