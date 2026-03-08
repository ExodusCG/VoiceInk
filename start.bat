@echo off
chcp 65001 >nul 2>&1

echo ============================================
echo    VoiceInk - Starting...
echo ============================================
echo.

:: Check virtual environment
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo        Please run setup.bat first.
    pause
    exit /b 1
)

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Check dependencies
echo [CHECK] Verifying dependencies...
python scripts\check_deps.py
if errorlevel 1 (
    echo [ERROR] Dependency check failed
    pause
    exit /b 1
)
echo.

echo Default hotkey: Right Alt (hold to talk)
echo Right-click system tray icon for settings
echo Press Ctrl+C to exit
echo ============================================
echo.

title VoiceInk
python -m voiceink.main

if errorlevel 1 (
    echo.
    echo [ERROR] VoiceInk exited with error code: %errorlevel%
    echo Check logs: %USERPROFILE%\.voiceink\logs\voiceink.log
    pause
)
