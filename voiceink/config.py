"""
VoiceInk - 全局配置管理

从 config.yaml 加载配置，提供类型安全的配置访问。
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


# 项目根目录
PROJECT_ROOT = Path(__file__).parent.parent
DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"


@dataclass
class AudioConfig:
    device: str = "auto"
    sample_rate: int = 16000
    channels: int = 1
    chunk_duration_ms: int = 500
    vad_threshold: float = 0.5
    silence_duration_ms: int = 800
    max_recording_seconds: int = 300


@dataclass
class ASRConfig:
    backend: str = "whisper_cpp"
    model_size: str = "base"
    model_path: str = ""
    language: str = "auto"
    n_threads: int = 4
    stream_interval_seconds: float = 2.0
    beam_size: int = 5


@dataclass
class APIConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"


@dataclass
class LLMConfig:
    backend: str = "llama_cpp"
    model_path: str = ""
    model_name: str = "Qwen3-0.6B-Q8_0.gguf"
    n_ctx: int = 2048
    n_threads: int = 4
    temperature: float = 0.1
    top_p: float = 0.9
    max_polish_length: int = 500
    api: APIConfig = field(default_factory=APIConfig)


@dataclass
class OutputConfig:
    method: str = "keyboard"
    typing_delay_ms: int = 5
    paste_delay_ms: int = 100


@dataclass
class DictionaryConfig:
    path: str = "custom_dictionary.json"
    max_asr_prompt_terms: int = 50
    enabled: bool = True


@dataclass
class StatusIndicatorConfig:
    enabled: bool = True
    position: str = "top_right"
    opacity: float = 0.85
    width: int = 350
    height: int = 80


@dataclass
class UIConfig:
    status_indicator: StatusIndicatorConfig = field(default_factory=StatusIndicatorConfig)


@dataclass
class AppConfig:
    language: str = "auto"
    hotkey_push_to_talk: str = "right alt"
    hotkey_toggle: str = "ctrl+shift+v"
    auto_start: bool = False
    log_level: str = "INFO"

    audio: AudioConfig = field(default_factory=AudioConfig)
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    dictionary: DictionaryConfig = field(default_factory=DictionaryConfig)
    ui: UIConfig = field(default_factory=UIConfig)


def _dict_to_dataclass(cls, data: dict):
    """递归地将字典转换为 dataclass 实例"""
    if data is None:
        return cls()

    fieldtypes = {f.name: f.type for f in cls.__dataclass_fields__.values()}
    init_kwargs = {}

    for key, value in data.items():
        if key in fieldtypes:
            field_type = fieldtypes[key]
            # 检查是否是嵌套 dataclass
            if isinstance(value, dict) and hasattr(field_type, '__dataclass_fields__'):
                init_kwargs[key] = _dict_to_dataclass(field_type, value)
            else:
                # 处理字符串类型注解
                actual_type = cls.__dataclass_fields__[key].type
                if isinstance(actual_type, str):
                    # 解析字符串类型
                    type_map = {
                        'AudioConfig': AudioConfig,
                        'ASRConfig': ASRConfig,
                        'LLMConfig': LLMConfig,
                        'APIConfig': APIConfig,
                        'OutputConfig': OutputConfig,
                        'DictionaryConfig': DictionaryConfig,
                        'UIConfig': UIConfig,
                        'StatusIndicatorConfig': StatusIndicatorConfig,
                    }
                    if actual_type in type_map and isinstance(value, dict):
                        init_kwargs[key] = _dict_to_dataclass(type_map[actual_type], value)
                    else:
                        init_kwargs[key] = value
                else:
                    init_kwargs[key] = value

    return cls(**init_kwargs)


def load_config(config_path: Optional[str] = None) -> AppConfig:
    """
    加载配置文件。

    Args:
        config_path: 配置文件路径，None 则使用默认路径

    Returns:
        AppConfig 实例
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)

    if not os.path.exists(config_path):
        return AppConfig()

    with open(config_path, 'r', encoding='utf-8') as f:
        raw = yaml.safe_load(f) or {}

    # 提取顶层 app 配置
    app_data = raw.get('app', {})
    app_data['audio'] = raw.get('audio', {})
    app_data['asr'] = raw.get('asr', {})
    app_data['llm'] = raw.get('llm', {})
    app_data['output'] = raw.get('output', {})
    app_data['dictionary'] = raw.get('dictionary', {})
    app_data['ui'] = raw.get('ui', {})

    return _dict_to_dataclass(AppConfig, app_data)


def save_config(config: AppConfig, config_path: Optional[str] = None):
    """
    保存配置到文件。

    Args:
        config: AppConfig 实例
        config_path: 保存路径，None 则保存到用户目录
    """
    if config_path is None:
        config_path = str(DEFAULT_CONFIG_PATH)

    from dataclasses import asdict
    data = asdict(config)

    # 重组为 YAML 的顶层结构
    yaml_data = {
        'app': {
            'language': data['language'],
            'hotkey_push_to_talk': data['hotkey_push_to_talk'],
            'hotkey_toggle': data['hotkey_toggle'],
            'auto_start': data['auto_start'],
            'log_level': data['log_level'],
        },
        'audio': data['audio'],
        'asr': data['asr'],
        'llm': data['llm'],
        'output': data['output'],
        'dictionary': data['dictionary'],
        'ui': data['ui'],
    }

    # 确保父目录存在
    Path(config_path).parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, 'w', encoding='utf-8') as f:
        yaml.dump(yaml_data, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
