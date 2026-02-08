============================================
  VoiceInk - Windows Local Voice Input
  Speak -> Recognize -> Polish -> Type
============================================

VoiceInk is a local voice input tool for Windows.
Hold a hotkey, speak, release -- your words are automatically
recognized (ASR), polished by a local LLM, and typed into any app.

Everything runs locally on your machine. No cloud, no subscription.


1. SYSTEM REQUIREMENTS
----------------------
- Windows 10 / 11  (64-bit)
- Python 3.10 or later  (https://www.python.org/downloads/)
  * IMPORTANT: Check "Add Python to PATH" during installation
- A working microphone
- ~2 GB disk space  (for models and dependencies)


2. QUICK START
--------------
Step 1 -- Install
  Double-click  setup.bat
  It will:
    - Create a Python virtual environment (.venv)
    - Install all dependencies
    - Check if ASR / LLM models exist

  First-time install takes 5-15 minutes depending on your network.

Step 2 -- Launch
  Double-click  start.bat
  A system tray icon will appear (bottom-right of taskbar).

Step 3 -- Use
  Hold  Right Alt  and speak into your microphone.
  Release the key -- VoiceInk will:
    1) Recognize your speech  (Paraformer ASR)
    2) Correct with custom dictionary
    3) Polish with local LLM  (Qwen3 0.6B)
    4) Type the result into your active window

  That's it!


3. DEFAULT HOTKEYS
------------------
  Right Alt (hold)     Push-to-Talk  (hold to record, release to process)
  Ctrl+Shift+V         Toggle voice input on/off


4. SYSTEM TRAY MENU
--------------------
Right-click the VoiceInk tray icon to access:
  - Settings        Adjust ASR, LLM, audio, hotkey, output options
  - Dictionary      Add/edit custom correction terms
  - Exit            Quit VoiceInk


5. CONFIGURATION
----------------
Default config file:  voiceink\config.yaml
User override:        %USERPROFILE%\.voiceink\config.yaml

Key settings you may want to change:

  [Hotkey]
    app.hotkey_push_to_talk    Default: right alt
    app.hotkey_toggle          Default: ctrl+shift+v

  [ASR Engine]
    asr.backend                Default: paraformer_onnx
                               Options: paraformer_onnx, whisper_cpp, faster_whisper

  [Language]
    asr.language               Default: zh  (Chinese)
                               Change to "en" for English with Whisper backends

  [LLM]
    llm.backend                Default: llama_cpp  (local Qwen3 0.6B)
                               Change to "api" to use cloud API (e.g. GPT-4o-mini)

  [Output Method]
    output.method              Default: keyboard  (simulated keystrokes)


6. CUSTOM DICTIONARY
--------------------
Edit  voiceink\custom_dictionary.json  to add correction terms.
Example -- fix common ASR mistakes:

  {
    "terms": [
      {"wrong": "wei xin", "correct": "WeChat"},
      {"wrong": "bo ke",   "correct": "blog"}
    ]
  }

You can also manage the dictionary through the tray menu -> Dictionary.


7. MODEL DOWNLOAD
-----------------
On first launch, VoiceInk will automatically download required models:
  - Paraformer ASR model    (~200 MB, from ModelScope)
  - Qwen3 0.6B LLM model   (~700 MB, GGUF format)

Models are saved to the  models\  folder. This is a one-time download.

If the download is slow, you can manually download models and place them in:
  models\asr\       ASR model files
  models\llm\       LLM model files  (Qwen3-0.6B-Q8_0.gguf)


8. LOGS & TROUBLESHOOTING
--------------------------
Log file:  %USERPROFILE%\.voiceink\logs\voiceink.log

Common issues:

  Q: "Python not found"
  A: Install Python 3.10+ and check "Add Python to PATH" during install.
     Then rerun setup.bat.

  Q: No sound is being captured
  A: Check your microphone is set as the default recording device in
     Windows Sound settings. You can also specify a device in config.yaml
     under audio.device.

  Q: Recognition quality is poor
  A: - Make sure you are using Paraformer (default) for Chinese input
     - Speak clearly and avoid background noise
     - Add frequently misrecognized words to the custom dictionary

  Q: Text is typed too slowly or has errors
  A: Adjust output.typing_delay_ms in config.yaml (default: 5 ms).
     Increase the value if characters are being dropped.

  Q: Model download fails
  A: Check your internet connection. If behind a proxy, set the
     HTTP_PROXY / HTTPS_PROXY environment variables before running setup.bat.


9. FILE STRUCTURE
-----------------
  VoiceInk\
  |-- setup.bat              Install script (run once)
  |-- start.bat              Launch script (run every time)
  |-- requirements.txt       Python dependencies
  |-- scripts\
  |   +-- check_deps.py      Dependency checker
  +-- voiceink\
      |-- main.py            Application entry point
      |-- config.py          Configuration loader
      |-- config.yaml        Default configuration
      |-- custom_dictionary.json   Dictionary data
      |-- asr\               Speech recognition backends
      |-- core\              Audio capture, pipeline, text output
      |-- dictionary\        Dictionary correction engine
      |-- llm\               LLM polish backends
      |-- ui\                System tray, settings, status indicator
      +-- utils\             Hotkey, logger, model downloader


10. UNINSTALL
-------------
Simply delete the entire VoiceInk folder.
User data is stored in  %USERPROFILE%\.voiceink\  -- delete that too
if you want a clean removal.

============================================
