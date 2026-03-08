"""
check_deps.py - Verify and auto-install dependencies based on current config.

Reads voiceink/config.yaml, determines required packages for the configured
ASR and LLM backends, and installs any that are missing via pip.
"""

import sys
import subprocess
from pathlib import Path

import yaml


# Backend -> list of (import_name, pip_package)
ASR_DEPS = {
    "sensevoice_onnx": [("sherpa_onnx",   "sherpa-onnx")],   # pre-compiled, no VC++ needed
    "whisper_cpp":     [("pywhispercpp",  "pywhispercpp")],
    "faster_whisper":  [("faster_whisper", "faster-whisper")],
}

LLM_DEPS = {
    "llama_cpp": [("llama_cpp", "llama-cpp-python")],
    "api":       [("openai",    "openai")],
    # "disabled" needs nothing
}

CORE_DEPS = [
    # (import_name, pip_package)
    ("numpy",       "numpy"),
    ("yaml",        "pyyaml"),
    ("sounddevice", "sounddevice"),
    ("keyboard",    "keyboard"),
    ("pystray",     "pystray"),
    ("PIL",         "Pillow"),
    ("pypinyin",    "pypinyin"),
    ("pyperclip",   "pyperclip"),
]


def can_import(module_name: str) -> bool:
    """Check if a module can be imported."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def pip_install(package: str) -> bool:
    """Install a package via pip. Returns True on success."""
    print(f"  Installing {package} ...")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", package],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"  [FAIL] pip install {package} failed:")
        print(result.stderr[:500])
        return False
    print(f"  {package} installed OK")
    return True


def main() -> int:
    # Locate config file (same directory structure as voiceink.config)
    config_path = Path(__file__).resolve().parent.parent / "voiceink" / "config.yaml"
    if not config_path.exists():
        print(f"[ERROR] Config file not found: {config_path}")
        return 1

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    asr_backend = raw.get("asr", {}).get("backend", "sensevoice_onnx")
    llm_backend = raw.get("llm", {}).get("backend", "llama_cpp")

    print(f"  ASR backend : {asr_backend}")
    print(f"  LLM backend : {llm_backend}")
    print()

    ok = True

    # --- Check core dependencies ---
    print("[Core dependencies]")
    for import_name, pip_pkg in CORE_DEPS:
        if not can_import(import_name):
            print(f"  {pip_pkg} missing, installing...")
            if not pip_install(pip_pkg):
                ok = False
        else:
            print(f"  {pip_pkg} OK")
    print()

    # --- Check ASR backend dependency ---
    print("[ASR backend dependency]")
    if asr_backend in ASR_DEPS:
        for import_name, pip_pkg in ASR_DEPS[asr_backend]:
            if not can_import(import_name):
                print(f"  {pip_pkg} not found (required by {asr_backend})")
                if not pip_install(pip_pkg):
                    ok = False
            else:
                print(f"  {pip_pkg} OK")
    else:
        print(f"  Unknown ASR backend: {asr_backend}")
        ok = False
    print()

    # --- Check LLM backend dependency ---
    print("[LLM backend dependency]")
    if llm_backend == "disabled":
        print("  LLM disabled, no extra deps needed")
    elif llm_backend in LLM_DEPS:
        for import_name, pip_pkg in LLM_DEPS[llm_backend]:
            if not can_import(import_name):
                print(f"  {pip_pkg} not found (required by {llm_backend})")
                if not pip_install(pip_pkg):
                    ok = False
            else:
                print(f"  {pip_pkg} OK")
    else:
        print(f"  Unknown LLM backend: {llm_backend}")
        ok = False
    print()

    if ok:
        print("[OK] All dependencies satisfied.")
        return 0
    else:
        print("[ERROR] Some dependencies could not be installed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
