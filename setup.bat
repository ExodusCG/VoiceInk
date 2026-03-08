@echo off
chcp 65001 >nul 2>&1
setlocal enabledelayedexpansion

echo ============================================
echo    VoiceInk - Setup Script
echo    Local Voice Input + AI Polish
echo ============================================
echo.

:: Check Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found. Please install Python 3.10+ and add to PATH
    echo Download: https://www.python.org/downloads/
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo         Python %PYVER% OK
echo.

:: Create virtual environment
echo [2/5] Creating virtual environment...
if not exist ".venv" (
    python -m venv .venv
    echo         Virtual environment created OK
) else (
    echo         Virtual environment already exists OK
)
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Upgrade pip
echo [3/5] Upgrading pip...
python -m pip install --upgrade pip -q
echo         pip updated OK
echo.

:: Install dependencies
echo [4/5] Installing dependencies...
echo     Installing core dependencies...
pip install sounddevice numpy pyyaml -q
echo     Installing UI dependencies...
pip install pystray Pillow keyboard pyperclip -q
echo     Installing ASR engine (SenseVoice - recommended)...
pip install sherpa-onnx -q
echo     Installing ASR engine (Whisper.cpp - optional)...
pip install pywhispercpp -q
echo     Installing LLM inference (llama.cpp)...
pip install llama-cpp-python -q
echo     Installing utilities...
pip install pypinyin webrtcvad -q
echo     Installing API client (optional)...
pip install openai -q
echo     Installing dev tools...
pip install pytest -q
echo         All dependencies installed OK
echo.

:: Download models (based on config)
echo [5/5] Checking / downloading models...
python -c "import yaml; from pathlib import Path; cfg = yaml.safe_load(open('voiceink/config.yaml', encoding='utf-8')) or {}; asr_be = cfg.get('asr', {}).get('backend', 'whisper_cpp'); from voiceink.utils.model_downloader import get_downloader; dl = get_downloader(); print('    ASR backend:', asr_be); asr_ok = dl.check_model_exists('asr', cfg.get('asr', {}).get('model_size', 'base'), 'models/asr') if asr_be in ('whisper_cpp', 'faster_whisper') else True; print('    ASR model OK' if asr_ok else '    ASR model not found, will download on first run'); print('    Checking Qwen3 0.6B model...'); llm_ok = dl.check_model_exists('llm', 'qwen3-0.6b-q8_0', 'models/llm'); print('    Qwen3 0.6B model already exists OK' if llm_ok else '    Qwen3 0.6B model not found, will download on first run')"
echo.

echo ============================================
echo    Setup complete!
echo ============================================
echo.
echo To start: double-click start.bat
echo Default hotkey: Right Alt (hold to talk)
echo.
pause
