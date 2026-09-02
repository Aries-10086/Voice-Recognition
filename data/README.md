# 测试音频说明（F3 / F8）

## 命名规范

| 类型 | 文件名模式 | 说明 |
|---|---|---|
| 单人独白 | `solo_*.wav` | 1 人连续说话 10–30s |
| 双人对话 | `dialog_*.wav` | 2 人交替，间隔自然停顿 |
| 带噪/远场 | `noisy_*.wav` | 环境噪声或 1m+ 距离 |
| 清唱验收 | `singing_*.wav` | 无人声伴奏；配 `.gold.json`；用 `--singing` |
| 真实录音 | `real_dialog_*.wav` | 非 TTS 合成，用于 O8 听测 |

**标准演示样例**：`dialog_two_speakers_16k.wav`（双人中文 TTS）  
**F3 真实听测（推荐）**：`real_dialog_02.wav`（FLEURS 真人中文）  
**清唱验收（03）**：  
- `singing_ruguaiawang_16k.wav`（情歌）  
- `singing_walkman2000_16k.wav`（怀旧口播/清唱）  
均用 `--singing --no-gold --no-lyrics-hint` 测真 ASR。  
**带噪压力样例**：`real_dialog_01.wav` / `noisy_dialog_01.wav`（SLR162，英语嘈杂，勿作中文语义金标）

## 清唱双轨验收（C3 · 勿混结论）

| 轨 | 命令 | 识别来源 | 验收什么 |
|---|---|---|---|
| **A · gold 全链路** | `run_pipeline.py … --singing --assume-single-speaker` | `.gold.json` 文本 | 翻译、克隆、sync、length |
| **B · 纯 ASR（真 KPI）** | 同上 + `--no-gold --no-lyrics-hint` | Whisper，无歌词 hint | 清唱识别真能力 |
| **B′ · 演示 ASR** | `--no-gold`（无 `--no-lyrics-hint`） | Whisper + hint/纠错 | 听感演示，**非真 KPI** |

判定：

```bash
# A 轨
python run_pipeline.py --input data/singing_ruguaiawang_16k.wav \
  --singing --assume-single-speaker -t en --device cpu

# B 轨 · 真 ASR
python run_pipeline.py --input data/singing_ruguaiawang_16k.wav \
  --singing --assume-single-speaker --no-gold --no-lyrics-hint -t en --device cpu
python scripts/score_singing_asr.py \
  --gold data/singing_ruguaiawang_16k.gold.json \
  --audio data/singing_ruguaiawang_16k.wav --no-gold-asr --no-lyrics-hint --device cpu

# 引擎对比
python scripts/compare_singing_asr_no_hint.py data/singing_ruguaiawang_16k.wav
```

| 指标 | 通过线 |
|---|---|
| A: sync | ≥0.65 |
| A: length | 0.8–1.2 |
| B: 关键词命中率 | ≥50% 且 bigram≥0.35（脚本 `usable_candidate`） |

## 录制要求（真实双人对话）

1. **格式**：WAV，16kHz，mono，16-bit（可用 `ffmpeg` 转换）
2. **时长**：10–30s，每人至少 2 轮发言
3. **环境**：安静室内，避免混响；手机/笔电麦即可
4. **内容**：自然对话即可（项目进度、日常话题），无需照稿
5. **距离**：嘴距麦 20–40cm；两人音色尽量可区分

### 转换命令

```bash
ffmpeg -i input.m4a -ar 16000 -ac 1 -sample_fmt s16 data/real_dialog_02.wav
```

### 可选金标旁路

同目录放 `*.gold.json` 时，pipeline **跳过 ASR 文本**（用金标 + 时间戳），专测克隆/同步：

```json
{"language":"zh","segments":[{"start":0.1,"end":3.5,"text":"...","speaker":"SPEAKER_00"}]}
```

`real_dialog_02.gold.json` 已提供。测 ASR 时请暂时移走该文件。

## 跑验收

```bash
cd voiceclone
source .venv/bin/activate
export HF_HOME="$(pwd)/models/huggingface"
# F3 推荐
python run_pipeline.py --input data/real_dialog_02.wav --target-lang en --device cpu
# 带噪压力（英语嘈杂，译文语义勿硬比）
python run_pipeline.py --input data/real_dialog_01.wav --target-lang en --device cpu
# 清唱（03）
python run_pipeline.py --input data/singing_ruguaiawang_16k.wav --singing --assume-single-speaker -t en --device cpu
```

制备清唱样例：

```bash
python scripts/prepare_singing_sample.py --mp3 "/path/如果爱忘了..mp3" --name ruguaiawang --seconds 60
python scripts/compare_singing_asr.py data/singing_ruguaiawang_16k.wav
```

准备/降噪脚本：

```bash
python scripts/prepare_real_dialog.py --fleurs
python scripts/prepare_real_dialog.py --denoise data/real_dialog_01.wav -o data/noisy_dialog_01.wav
```

## 听测表（O8，主观 MOS 1–5）

| 样例 | 像度 | 可懂度 | 自然度 | 分人正确 | 备注 |
|---|---|---|---|---|---|
| dialog_two_speakers_16k | | | | | TTS 合成 |
| real_dialog_02 | | | | | FLEURS 真人中文 ★ |
| real_dialog_01 | | | | | SLR162 嘈杂（压力） |

≥3 人打分取平均；**像度目标 ≥3.5**（02 结题）。

## real_dialog_02 来源（推荐 F3）

| 项 | 值 |
|---|---|
| 文件 | `data/real_dialog_02.wav` |
| 来源 | [Google FLEURS](https://huggingface.co/datasets/google/fleurs) `cmn_hans_cn`（真人朗读，非 TTS） |
| 许可 | CC-BY |
| 规格 | 16 kHz · mono · PCM_16 · ~17 s · 两性别交替 4 句 |
| 金标 | `data/real_dialog_02.gold.json` |

## real_dialog_01 来源（带噪压力）

| 项 | 值 |
|---|---|
| 文件 | `data/real_dialog_01.wav` |
| 来源 | [OpenSLR SLR162](https://www.openslr.org/162/) · `audio_158.wav`（多人自然对话，含背景噪声；**英语**） |
| 许可 | MIT（需署名 Rasheed Mudasiru / SLR162） |
| 规格 | 16 kHz · mono · PCM_16 · **20 s** |
| 说明 | 强制 `language=zh` 时易幻觉中文，**不宜作中文语义验收**；降噪版见 `noisy_dialog_01.wav` |

重新下载 SLR162 并生成：

```bash
curl -L -o /tmp/slr162.zip \
  https://openslr.trmal.net/resources/162/audio-for-forensic-and-multi-speaker-separation.zip
unzip -p /tmp/slr162.zip audio-for-forensic-and-multi-speaker-separation/audio_158.wav > /tmp/audio_158.wav
# 再用项目 .venv 内 librosa 转为 16k mono
```
