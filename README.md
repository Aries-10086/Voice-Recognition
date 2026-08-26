# 🎙️ 跨语种语音翻译与原声复刻系统

## Cross-Lingual Speech Translation & Voice Cloning System

基于**发音开闭感知**的语音时间轴自动对齐与语音生成技术

---

## 📋 项目概述

本项目面向跨语种视听内容翻译中存在的**情感失真、口型不同步、声音风格不一致**等问题，提出了一种基于多模态情感感知与人工智能大模型的跨语种语音翻译与原声复刻系统。

### 核心流程

```
语音输入 → ASR识别 → 说话人分离(不同人声) → 逐段情感分析 → LLM翻译优化
        → AOCP发音分析 → 时间轴对齐 → 分段起始/结束音检测 → 声纹克隆(复用prompt) → 语音输出
```

### 执行结构 (Pipeline 步骤)

`CrossLingualPipeline.run()` 按 `config/pipeline.steps` 依次执行 7 步：

| 步骤 | 模块 | 说明 |
|------|------|------|
| 1. 加载 | `utils.audio_utils` | 读入音频、去首尾静音 |
| 2. ASR + 情感 + 分离 | `asr/whisper_asr` `asr/emotion_recognition` `asr/speaker_diarization` | faster-whisper 批量转写；wav2vec2 情感；pyannote/离线聚类分人声 |
| 3. 翻译 | `translation/llm_translator` | 情感感知 LLM 翻译（本地/OpenAI兼容/Google） |
| 4. AOCP | `alignment/articulatory_analyzer` | 发音开闭感知（无权重时能量 VAD 兜底） |
| 5. 音素对齐 + 时间轴 | `alignment/phoneme_aligner` `alignment/timeline_generator` | 音素对齐与口型时间轴 |
| 6. 声纹克隆 | `synthesis/voice_cloner` `synthesis/tts_engine` | 自动选择 CosyVoice2→Qwen3-TTS→edge-tts；逐段严格按长度生成 |
| 7. 输出 | `utils.audio_utils` | 按绝对时间轴拼接、归一化、保存 |

---

## 🏗️ 项目结构

```
voiceclone/
├── config/
│   └── default.yaml              # 系统配置文件
├── src/
│   ├── asr/                      # 语音识别模块
│   │   ├── whisper_asr.py        # 多引擎ASR (Whisper/FunASR)
│   │   ├── emotion_recognition.py # 语音情感识别
│   │   └── language_id.py        # 语种识别 (MMS-LID)
│   ├── translation/              # 翻译优化模块
│   │   ├── llm_translator.py     # 情感感知LLM翻译
│   │   └── prompt_templates.py   # 提示词模板
│   ├── alignment/                # ⭐ 时间轴对齐模块 (核心创新)
│   │   ├── articulatory_analyzer.py  # AOCP-Net 发音开闭感知
│   │   ├── phoneme_aligner.py        # 音素对齐
│   │   └── timeline_generator.py     # 时间轴生成
│   ├── synthesis/                # 语音合成模块
│   │   ├── voice_cloner.py       # 声纹复刻
│   │   ├── tts_engine.py         # TTS引擎 (CosyVoice2等)
│   │   └── vocoder.py            # 神经声码器 (HiFi-GAN)
│   ├── integration/
│   │   └── pipeline.py           # 完整Pipeline集成
│   └── utils/
│       ├── audio_utils.py        # 音频工具
│       └── video_utils.py        # 视频工具
├── scripts/
│   ├── run_pipeline.py           # 主运行脚本
│   └── demo.py                   # 演示脚本
├── models/                       # 模型存放目录
├── data/                         # 数据目录
├── outputs/                      # 输出目录
└── requirements.txt              # 依赖清单
```

---

## 🔬 四大创新点 (可用于专利申请)

### 创新1: AOCP-Net 发音开闭感知网络

**纯音频驱动的口腔开合度预测**，无需视频即可分析说话状态：

- **多尺度声学特征提取**: 同时从20ms/50ms/100ms三个时间尺度提取频谱特征
- **时域-频域交叉注意力**: 融合声学特征与语言学特征，学习跨模态关联
- **连续开合度回归**: 输出0~1连续值，不仅判断开/闭，还量化开合程度
- **音素感知增强**: 结合跨语种音素映射表，利用先验知识提升预测精度

```python
from src.alignment.articulatory_analyzer import AOCPNet
aocp = AOCPNet(config)
result = aocp.predict(audio, sample_rate=16000)
# result.openness: 逐帧开合度 (0~1)
# result.timeline: 口型同步时间轴
```

### 创新2: 情感感知的跨语种翻译优化

将语音情感识别结果注入LLM翻译过程：

- 情感条件化翻译提示词，LLM同时优化语义和情感
- 发音长度约束，译文音节数与源文匹配，便于口型同步
- 多轮迭代优化，评估-修正循环确保质量

### 创新3: 多源信息融合的时间轴生成

联合AOCP-Net开闭状态和音素对齐结果：

- 自动识别说话段与静音段
- 跨语种时长自适应调整
- 自动生成口型同步关键点
- 提供同步质量评分

### 创新4: 情感保持的零样本语音克隆

- 情感-音色解耦建模
- 情感嵌入向量注入TTS韵律控制
- 从3-5秒参考音频即可克隆，保持情感

---

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt

# 可选: 安装CosyVoice2 (零样本语音克隆)
git clone https://github.com/FunAudioLLM/CosyVoice.git models/CosyVoice2-0.5B
```

### 2. 配置LLM后端

编辑 `config/default.yaml`:

```yaml
translation:
  engine: "openai_compatible"
  api_base: "http://localhost:8000/v1"  # vLLM/Ollama 地址
  model_name: "Qwen/Qwen3-8B"
```

### 3. 运行

```bash
# 快速演示 (无需外部模型)
python scripts/demo.py

# 完整Pipeline
python scripts/run_pipeline.py --input audio.wav --target-lang en

# 指定参考音频克隆
python scripts/run_pipeline.py --input speech.wav --ref-audio speaker.wav --target-lang ja
```

---

## 🔧 配置说明

`config/default.yaml` 包含所有模块的配置参数：

| 模块 | 关键参数 | 说明 |
|------|---------|------|
| `asr` | `engine`, `model_name`, `batch_size` | faster_whisper / funasr；`batch_size>0` 启用 GPU 批量转写提速 |
| `emotion` | `engine` | emotion2vec 或 wav2vec2_emotion |
| `pipeline.diarization` | `engine`, `distance_threshold` | 说话人分离（pyannote / 离线聚类） |
| `translation` | `engine`, `model_name` | LLM后端和模型（默认 `google` 免费可靠） |
| `alignment.aocp` | `multi_scale_windows` | 多尺度窗口配置 |
| `synthesis` | `engine` | `auto`=自动选最优：CosyVoice2 → Qwen3-TTS → edge-tts |
| `pipeline.segment` | `reference_min_seconds`, `min_group_segments` | 整段克隆 + 停顿控制（按说话人/语气分组） |

---

## 📊 技术栈

| 组件 | 技术选型 | 版本/年份 |
|------|---------|---------|
| ASR | faster-whisper / FunASR SenseVoice | 2024 |
| 情感识别 | emotion2vec / Wav2Vec2-Emotion | 2024 |
| LLM翻译 | Qwen3 / DeepSeek | 2024-2025 |
| AOCP-Net | 自研 BiLSTM+Attention | 2025 (本系统) |
| TTS/克隆 | CosyVoice2 / StyleTTS2 | 2024-2025 |
| 声码器 | HiFi-GAN | 2024 |
| 声纹提取 | ECAPA-TDNN | 2024 |

---

## 📝 专利要点

本系统的以下技术方案具有专利申请价值：

1. **一种基于纯音频信号的发音口型开合状态感知方法**: 通过多尺度频谱分析和交叉注意力机制，仅从音频预测口腔开合状态

2. **一种融合情感感知的跨语种语音翻译方法**: 将语音情感向量注入大语言模型翻译过程，实现情感保持的翻译

3. **一种多源信息融合的语音-视频时间轴自动对齐方法**: 联合发音开闭状态和音素信息自动生成口型同步时间轴

4. **一种情感保持的零样本跨语种音色克隆方法**: 解耦情感与音色表征，实现独立控制
