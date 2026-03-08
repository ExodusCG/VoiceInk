# VoiceInk

**Windows 本地语音输入工具 — 说话、识别、润色、输入**

**Windows local voice input tool — Speak, Recognize, Polish, Type**

---

[English](#english) | [中文](#中文)

---

<a name="中文"></a>
## 中文

按住快捷键说话，松开后自动识别。语音通过本地 ASR 引擎识别，本地 LLM 润色，然后输入到任意应用程序 — 全程本地运行，无需联网。

### 功能特点

- **按住说话 (Push-to-Talk)** — 按住 Right Alt 录音，松开后处理
- **本地 ASR** — SenseVoice ONNX（多语言：中/英/日/韩/粤）/ Whisper.cpp / Faster Whisper
- **本地 LLM 润色** — Qwen3 0.6B 自动修正标点、口语词和语法
- **云端 API 支持** — 可选 OpenAI 兼容 API 作为 LLM 后端
- **自定义词典** — 自动修正专业术语和 ASR 识别错误
- **系统托盘 UI** — 状态指示器、设置面板、词典编辑器
- **实时流式识别** — 说话时即可看到部分识别结果
- **能量 VAD** — 智能语音活动检测，自动忽略静音
- **零云端依赖** — 完全本地运行，语音数据不会离开您的电脑
- **单实例保护** — 防止重复启动多个实例

### 快速开始

#### 系统要求

- Windows 10 / 11 (64位)
- Python 3.10+（[下载](https://www.python.org/downloads/) — 安装时勾选 **"Add Python to PATH"**）
- 可用的麦克风
- ~2 GB 磁盘空间（用于模型和依赖）

#### 安装与运行

```bash
# 1. 克隆仓库
git clone https://github.com/wolaiye945/VoiceInk.git
cd VoiceInk

# 2. 运行安装脚本（创建虚拟环境、安装依赖、检查模型）
setup.bat

# 3. 启动
start.bat
```

首次启动时，VoiceInk 会自动下载：
- **SenseVoice ASR 模型**（~230 MB，从 HuggingFace 镜像下载）
- **Qwen3 0.6B LLM 模型**（~700 MB，GGUF 格式）

#### 使用方法

| 操作 | 快捷键 |
|------|--------|
| 按住说话（录音） | `Right Alt` |
| 切换语音输入开关 | `Ctrl+Shift+V` |

右键系统托盘图标可打开 **设置**、**词典** 和 **退出**。

### 配置

默认配置：`voiceink/config.yaml`
用户配置：`%USERPROFILE%\.voiceink\config.yaml`

#### 主要设置

```yaml
app:
  hotkey_push_to_talk: right alt    # 按住说话键
  hotkey_toggle: ctrl+shift+v       # 切换开关

asr:
  backend: sensevoice_onnx          # sensevoice_onnx | whisper_cpp | faster_whisper
  language: zh                      # zh | en | ja | ko | yue | auto

llm:
  backend: llama_cpp                # llama_cpp | api
  temperature: 0.1
  api:                              # 仅当 backend: api 时使用
    base_url: https://api.openai.com/v1
    api_key: ''
    model: gpt-4o-mini

output:
  method: keyboard                  # 通过 Win32 SendInput 模拟按键
  typing_delay_ms: 5
```

### 自定义词典

编辑 `voiceink/custom_dictionary.json` 或使用托盘菜单 → 词典：

```json
{
  "terms": [
    {"wrong": "wei xin", "correct": "微信"},
    {"wrong": "bo ke",   "correct": "博客"}
  ]
}
```

### 故障排除

| 问题 | 解决方案 |
|------|----------|
| "Python not found" | 安装 Python 3.10+，安装时勾选 "Add to PATH" |
| 没有音频输入 | 在 Windows 声音设置中设置默认麦克风 |
| 识别效果差 | 添加专业术语到自定义词典 |
| 字符丢失 | 增加配置中的 `output.typing_delay_ms` |
| 模型下载失败 | 检查网络；如需代理，设置 `HTTP_PROXY`/`HTTPS_PROXY` |

日志位置：`%USERPROFILE%\.voiceink\logs\voiceink.log`

---

<a name="english"></a>
## English

Hold a hotkey, speak, release. Your words are recognized by a local ASR engine, polished by a local LLM, and typed into any application — all running on your machine, no cloud required.

### Features

- **Push-to-Talk** — Hold Right Alt to record, release to process
- **Local ASR** — SenseVoice ONNX (multilingual: zh/en/ja/ko/yue) / Whisper.cpp / Faster Whisper
- **Local LLM Polish** — Qwen3 0.6B cleans up punctuation, filler words, and grammar
- **Cloud API Support** — Optional OpenAI-compatible API backend for LLM
- **Custom Dictionary** — Auto-correct domain-specific terms and ASR mistakes
- **System Tray UI** — Status indicator, settings panel, dictionary editor
- **Real-time Streaming** — See partial recognition results as you speak
- **Energy VAD** — Smart voice activity detection, ignores silence
- **Zero Cloud Dependency** — Everything runs locally, your voice data never leaves your machine
- **Single Instance Protection** — Prevents multiple instances from running

### Quick Start

#### Requirements

- Windows 10 / 11 (64-bit)
- Python 3.10+ ([download](https://www.python.org/downloads/) — check **"Add Python to PATH"**)
- A working microphone
- ~2 GB disk space (for models and dependencies)

#### Install & Run

```bash
# 1. Clone the repo
git clone https://github.com/wolaiye945/VoiceInk.git
cd VoiceInk

# 2. Run setup (creates venv, installs deps, checks models)
setup.bat

# 3. Launch
start.bat
```

On first launch, VoiceInk automatically downloads:
- **SenseVoice ASR model** (~230 MB, from HuggingFace mirror)
- **Qwen3 0.6B LLM model** (~700 MB, GGUF format)

#### Usage

| Action | Hotkey |
|--------|--------|
| Push-to-Talk (hold to record) | `Right Alt` |
| Toggle voice input on/off | `Ctrl+Shift+V` |

Right-click the system tray icon for **Settings**, **Dictionary**, and **Exit**.

### Configuration

Default config: `voiceink/config.yaml`
User override: `%USERPROFILE%\.voiceink\config.yaml`

#### Key Settings

```yaml
app:
  hotkey_push_to_talk: right alt    # Push-to-Talk key
  hotkey_toggle: ctrl+shift+v       # Toggle on/off

asr:
  backend: sensevoice_onnx          # sensevoice_onnx | whisper_cpp | faster_whisper
  language: zh                      # zh | en | ja | ko | yue | auto

llm:
  backend: llama_cpp                # llama_cpp | api
  temperature: 0.1
  api:                              # Only used when backend: api
    base_url: https://api.openai.com/v1
    api_key: ''
    model: gpt-4o-mini

output:
  method: keyboard                  # Simulated keystrokes via Win32 SendInput
  typing_delay_ms: 5
```

### Custom Dictionary

Edit `voiceink/custom_dictionary.json` or use the tray menu → Dictionary:

```json
{
  "terms": [
    {"wrong": "wei xin", "correct": "WeChat"},
    {"wrong": "bo ke",   "correct": "blog"}
  ]
}
```

### Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python 3.10+ with "Add to PATH" checked |
| No audio captured | Set microphone as default in Windows Sound settings |
| Poor recognition | Add domain terms to custom dictionary |
| Characters dropped | Increase `output.typing_delay_ms` in config |
| Model download fails | Check network; set `HTTP_PROXY`/`HTTPS_PROXY` if behind proxy |

Logs: `%USERPROFILE%\.voiceink\logs\voiceink.log`

---

## Architecture / 架构

```
[Push-to-Talk] → AudioCapture (sounddevice + EnergyVAD)
    → Stream Recognition Thread (real-time partials)
    → Finalize: ASR → Dictionary Correction → LLM Polish
    → TextOutput (Win32 SendInput)
```

## Project Structure / 项目结构

```
VoiceInk/
├── setup.bat                    # One-click install / 一键安装
├── start.bat                    # One-click launch / 一键启动
├── requirements.txt
├── scripts/
│   └── check_deps.py            # Dependency checker / 依赖检查
└── voiceink/
    ├── main.py                  # Entry point / 入口点
    ├── config.py                # Configuration / 配置
    ├── config.yaml              # Default config / 默认配置
    ├── custom_dictionary.json   # Dictionary data / 词典数据
    ├── asr/                     # ASR backends / ASR 后端
    ├── core/                    # AudioCapture, Pipeline, TextOutput
    ├── dictionary/              # Dictionary correction / 词典修正
    ├── llm/                     # LLM backends / LLM 后端
    ├── ui/                      # UI components / UI 组件
    └── utils/                   # Utilities / 工具
```

## License / 许可证

[MIT](LICENSE)
