============================================
  VoiceInk - Windows 本地语音输入工具
  说话 -> 识别 -> 润色 -> 输入
============================================

VoiceInk 是一款 Windows 本地语音输入工具。
按住快捷键说话，松开后自动识别语音、LLM润色、输入到任意应用。

所有处理均在本地完成，无需联网，保护隐私。
无需安装 Visual C++ 运行时，开箱即用。


1. 系统要求
----------------------
- Windows 10 / 11 (64位)
- Python 3.10 或更高版本 (https://www.python.org/downloads/)
  * 重要：安装时勾选 "Add Python to PATH"
- 麦克风
- 约 1GB 磁盘空间 (含模型)


2. 快速开始
--------------
第一步：安装
  双击运行 setup.bat
  首次安装约需 5-10 分钟（取决于网络速度）

第二步：启动
  双击运行 start.bat
  系统托盘会出现 VoiceInk 图标

第三步：使用
  按住 右Alt 键说话，松开后：
    1) 语音识别 (SenseVoice ASR)
    2) 词典纠错
    3) LLM 润色 (Qwen3)
    4) 自动输入到当前窗口


3. 快捷键
------------------
  右Alt (按住)        按住说话，松开识别
  Ctrl+Shift+V       切换语音输入开关


4. 语言支持
------------------
SenseVoice 支持多种语言，在 voiceink\config.yaml 中配置：

  asr:
    backend: sensevoice_onnx
    language: auto    # 自动检测语言（推荐）

支持的语言代码：
  auto  - 自动检测
  zh    - 中文
  en    - 英文
  ja    - 日语
  ko    - 韩语
  yue   - 粤语


5. 配置文件
----------------
默认配置：voiceink\config.yaml
用户配置：%USERPROFILE%\.voiceink\config.yaml (优先级更高)

常用配置项：

  [快捷键]
    app.hotkey_push_to_talk    默认: right alt
    app.hotkey_toggle          默认: ctrl+shift+v

  [ASR 引擎]
    asr.backend                默认: sensevoice_onnx
                               可选: sensevoice_onnx, whisper_cpp, faster_whisper

  [语言]
    asr.language               默认: auto (自动检测)
                               可选: auto/zh/en/ja/ko/yue

  [LLM 润色]
    llm.backend                默认: llama_cpp (本地 Qwen3)
                               可选: api (云端 API), disabled (禁用)


6. 自定义词典
--------------------
编辑 voiceink\custom_dictionary.json 添加纠错词条：

  {
    "terms": [
      {"wrong": "wei xin", "correct": "微信"},
      {"wrong": "bo ke",   "correct": "博客"}
    ]
  }

也可通过托盘菜单 -> 词典 进行管理。


7. 模型下载
-----------------
首次启动会自动下载所需模型：

  SenseVoice ASR 模型   ~170 MB (从 GitHub 下载)
  Qwen3 0.6B LLM 模型   ~610 MB (从 HuggingFace 下载)

模型保存在 models\ 目录，只需下载一次。

手动下载地址：
  SenseVoice: https://github.com/k2-fsa/sherpa-onnx/releases
  Qwen3:      https://huggingface.co/Qwen/Qwen3-0.6B-GGUF


8. 常见问题
--------------------------
日志文件：%USERPROFILE%\.voiceink\logs\voiceink.log

Q: 提示 "Python not found"
A: 安装 Python 3.10+ 并勾选 "Add Python to PATH"，然后重新运行 setup.bat

Q: 没有录到声音
A: 检查麦克风是否设为 Windows 默认录音设备
   也可在 config.yaml 的 audio.device 指定设备名

Q: 识别不准确
A: - 说话清晰，避免背景噪音
   - 将常错的词添加到自定义词典
   - 尝试指定语言（如 language: zh）而非 auto

Q: 输入太慢或丢字
A: 增大 config.yaml 中的 output.typing_delay_ms (默认 5ms)

Q: 模型下载失败
A: 检查网络连接。如使用代理，设置环境变量：
   set HTTP_PROXY=http://127.0.0.1:7890
   set HTTPS_PROXY=http://127.0.0.1:7890


9. 文件结构
-----------------
  VoiceInk\
  |-- setup.bat              安装脚本 (运行一次)
  |-- start.bat              启动脚本 (每次运行)
  |-- requirements.txt       Python 依赖
  |-- readme.txt             本文件
  |-- scripts\
  |   +-- check_deps.py      依赖检查脚本
  +-- voiceink\
      |-- main.py            程序入口
      |-- config.yaml        默认配置
      |-- custom_dictionary.json   自定义词典
      |-- asr\               语音识别后端 (SenseVoice)
      |-- llm\               LLM 润色后端 (Qwen3)
      |-- core\              音频采集、处理管道
      |-- ui\                系统托盘、设置界面
      +-- utils\             工具函数


10. 卸载
-------------
删除 VoiceInk 整个文件夹即可。
用户数据在 %USERPROFILE%\.voiceink\，如需完全清除也一并删除。


============================================
  GitHub: https://github.com/wolaiye945/VoiceInk
============================================
