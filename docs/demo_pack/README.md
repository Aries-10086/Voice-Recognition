# 演示包 (F10/F11 · A5)

他人按本文可在 **30 分钟内** 复现三条标准样例。

## 0. 环境

```bash
cd voiceclone
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export HF_HOME="$(pwd)/models/huggingface"
export NLTK_DATA="$(pwd)/models/nltk_data"
```

## 1. 对白 · 申请书主场景

```bash
python run_pipeline.py --input data/dialog_two_speakers_16k.wav -t en --device cpu
# 试听: outputs/run_*/cloned_output.wav
# 指标: summary.txt → length 0.8–1.2, sync ≥0.65
```

## 2. 真实中文 · FLEURS 金标

```bash
python run_pipeline.py --input data/real_dialog_02.wav -t en --device cpu
# 金标: data/real_dialog_02.gold.json（跳过 ASR 文本）
```

## 3. 清唱 · 03 双轨验收

**轨 A — gold 全链路**（测翻译/克隆/时间轴，**不测 ASR**）：

```bash
python run_pipeline.py --input data/singing_ruguaiawang_16k.wav \
  --singing --assume-single-speaker -t en --device cpu
```

**轨 B — 纯 ASR**（测清唱识别，**勿与轨 A 混结论**）：

```bash
python run_pipeline.py --input data/singing_ruguaiawang_16k.wav \
  --singing --assume-single-speaker --no-gold -t en --device cpu

python scripts/score_singing_asr.py \
  --gold data/singing_ruguaiawang_16k.gold.json --run outputs/run_最新
```

## 4. LLM 翻译主路径 (A1)

本地模型已缓存时：

```bash
python run_pipeline.py --input data/real_dialog_02.wav -t en \
  --translate-engine local --device cpu
# summary 应出现 translation_engine=local
```

无 GPU/无本地模型时用 OpenAI 兼容 API 或 `auto`（有缓存走 local，否则 google）。

## 5. 批量验收表 (A4)

```bash
python scripts/prepare_solo_sample.py   # 首次生成 solo 样例
python scripts/batch_acceptance.py --collect-only
# 输出: outputs/acceptance_*.md
```

## 6. 口型/同步素材 (A3 · 待补视频)

- 每 run 自动生成 `aocp_open_segments.json`、`timeline_sync.json`
- 有参考视频时：`python run_pipeline.py --video --input demo.mp4 -t en`

## 参考 run

| 样例 | 推荐 run | length | sync |
|---|---|---:|---:|
| 双人 TTS | `run_20260831_101504` | 1.03 | 0.78 |
| FLEURS | `run_20260831_153353` | 1.06 | 0.81 |
| 清唱 gold | `run_20260902_102910` | 1.11 | 0.66 |
