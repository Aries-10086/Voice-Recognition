# F9 口型验收视频素材说明

## 选用样例（推荐）

| 字段 | 内容 |
|---|---|
| 文件 | `data/talking_head_willkent_20s.mp4` |
| 时长 | 20s（自完整片首 20s 截取） |
| 画面 | 正脸说话（Will Kent / Wiki Education） |
| 音轨 | 有（英文口播，rms 正常） |
| 来源页 | https://commons.wikimedia.org/wiki/File:Will_Kent_on_Wikidata%27s_usefulness_for_university_knowledge_sharing.webm |
| 许可 | 以 Wikimedia Commons 该文件页标注为准（通常 CC BY-SA 类；结题注明出处） |
| 制备 | 下载 480p WebM → ffmpeg 转码 AAC/H.264 → `-t 20` |

## 弃用候选

| 文件 | 原因 |
|---|---|
| `talking_head_mixkit_28287.mp4`（若存在） | Mixkit 该片 **无音轨**，不能跑 ASR/克隆 |

## 验收命令

```bash
python scripts/validate_mp4.py --check-env --mode demo \
  --video data/talking_head_willkent_20s.mp4

python run_pipeline.py --video --input data/talking_head_willkent_20s.mp4 \
  -t zh --source-lang en --device cpu --assume-single-speaker

python scripts/validate_mp4.py --mode demo \
  --video data/talking_head_willkent_20s.mp4 \
  --run-dir outputs/run_XXXXXXXX_XXXXXX
```

**实测**（2026-09-02）：`outputs/run_20260902_164647` · success · sync 0.77 · F9 demo **13/13**。

**识别优化**（2026-09-03）：`--video` 默认 `video_talking`；

```bash
python scripts/score_video_asr.py \
  --gold data/talking_head_willkent_20s.gold.json \
  --video data/talking_head_willkent_20s.mp4 \
  --profile video_talking --no-gold-asr --device cpu
# → keyword 10/10, WER≈0
```
