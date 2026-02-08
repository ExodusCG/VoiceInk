# VoiceInk

**Windows local voice input tool — Speak, Recognize, Polish, Type.**

Hold a hotkey, speak, release. Your words are recognized by a local ASR engine, polished by a local LLM, and typed into any application — all running on your machine, no cloud required.

## Features

- **Push-to-Talk** — Hold Right Alt to record, release to process
- **Local ASR** — Paraformer ONNX (Chinese-optimized) / Whisper.cpp / Faster Whisper
- **Local LLM Polish** — Qwen3 0.6B cleans up punctuation, filler words, and grammar
- **Cloud API Support** — Optional OpenAI-compatible API backend for LLM
- **Custom Dictionary** — Auto-correct domain-specific terms and ASR mistakes
- **System Tray UI** — Status indicator, settings panel, dictionary editor
- **Real-time Streaming** — See partial recognition results as you speak
- **Energy VAD** — Smart voice activity detection, ignores silence
- **Zero Cloud Dependency** — Everything runs locally, your voice data never leaves your machine

## Quick Start

### Requirements

- Windows 10 / 11 (64-bit)
- Python 3.10+  ([download](https://www.python.org/downloads/) — check **"Add Python to PATH"**)
- A working microphone
- ~2 GB disk space (for models and dependencies)

### Install & Run

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
- **Paraformer ASR model** (~200 MB, from ModelScope)
- **Qwen3 0.6B LLM model** (~700 MB, GGUF format)

### Usage

| Action | Hotkey |
|--------|--------|
| Push-to-Talk (hold to record) | `Right Alt` |
| Toggle voice input on/off | `Ctrl+Shift+V` |

Right-click the system tray icon for **Settings**, **Dictionary**, and **Exit**.

## Architecture

```
[Push-to-Talk] → AudioCapture (sounddevice + EnergyVAD)
    → Stream Recognition Thread (real-time partials)
    → Finalize: ASR → Dictionary Correction → LLM Polish
    → TextOutput (Win32 SendInput)
```

### Pipeline

The core pipeline (`voiceink/core/pipeline.py`) orchestrates three threads:
1. **Recording thread** — captures audio via `sounddevice`, runs energy-based VAD
2. **Stream recognition thread** — feeds audio chunks to ASR for real-time partial results
3. **Finalize thread** — on key release, runs full ASR → dictionary → LLM → keyboard output

### Backend Factory Pattern

Both ASR and LLM use lazy-import factories:

```python
from voiceink.asr import create_asr_backend
from voiceink.llm import create_llm_backend

asr = create_asr_backend(config)   # → ParaformerOnnxBackend / WhisperCppBackend / FasterWhisperBackend
llm = create_llm_backend(config)   # → LlamaCppBackend / APIBackend
```

Backends are only imported when instantiated, so optional dependencies don't cause errors.

## Configuration

Default config: `voiceink/config.yaml`
User override: `%USERPROFILE%\.voiceink\config.yaml`

### Key Settings

```yaml
app:
  hotkey_push_to_talk: right alt    # Push-to-Talk key
  hotkey_toggle: ctrl+shift+v       # Toggle on/off

asr:
  backend: paraformer_onnx          # paraformer_onnx | whisper_cpp | faster_whisper
  language: zh                      # zh | en | auto

llm:
  backend: llama_cpp                # llama_cpp | api
  temperature: 0.1
  api:                              # Only used when backend: api
    base_url: https://api.openai.com/v1
    api_key: ''                     # Set via env var OPENAI_API_KEY or here
    model: gpt-4o-mini

output:
  method: keyboard                  # Simulated keystrokes via Win32 SendInput
  typing_delay_ms: 5
```

## Custom Dictionary

Edit `voiceink/custom_dictionary.json` or use the tray menu → Dictionary:

```json
{
  "terms": [
    {"wrong": "wei xin", "correct": "WeChat"},
    {"wrong": "bo ke",   "correct": "blog"}
  ]
}
```

The dictionary corrects common ASR mistakes before LLM polishing.

## Project Structure

```
VoiceInk/
├── setup.bat                    # One-click install
├── start.bat                    # One-click launch
├── requirements.txt
├── scripts/
│   └── check_deps.py            # Dependency checker
└── voiceink/
    ├── main.py                  # Entry point & tray app
    ├── config.py                # Nested dataclass config
    ├── config.yaml              # Default configuration
    ├── custom_dictionary.json   # Dictionary data
    ├── asr/                     # ASR backends (Paraformer, Whisper.cpp, Faster Whisper)
    ├── core/                    # AudioCapture, Pipeline, TextOutput
    ├── dictionary/              # Dictionary correction engine
    ├── llm/                     # LLM backends (llama.cpp, OpenAI API)
    ├── ui/                      # Tray, Settings, Status Indicator, Dictionary Panel
    └── utils/                   # Hotkey, Logger, Model Downloader
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Install Python 3.10+ with "Add to PATH" checked |
| No audio captured | Set microphone as default in Windows Sound settings |
| Poor recognition | Use Paraformer for Chinese; add terms to custom dictionary |
| Characters dropped | Increase `output.typing_delay_ms` in config |
| Model download fails | Check network; set `HTTP_PROXY`/`HTTPS_PROXY` if behind proxy |

Logs: `%USERPROFILE%\.voiceink\logs\voiceink.log`

## License

[MIT](LICENSE)
