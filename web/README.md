# VoiceClone Web 控制台

本地演示前端：上传音频/视频或一键跑仓库样例，查看识别、译文、同步分，并播放克隆音视频。

## 启动

```bash
cd voiceclone
source .venv/bin/activate   # 若有
python scripts/run_web.py
# 浏览器打开 http://127.0.0.1:8765
```

首次任务会加载 Whisper / 情感 / TTS，可能较慢；之后同进程复用。

## 能力

| 项 | 说明 |
|---|---|
| 上传 | wav / mp3 / mp4 … |
| 样例 | Will Kent 视频、对白、清唱、FLEURS |
| 参数 | 目标/源语种、profile、清唱、真 ASR、单人 |
| 结果 | ASR、译文、sync、克隆 wav / mp4 播放与下载 |

## API

- `GET /api/health`
- `GET /api/samples`
- `POST /api/jobs`（multipart）
- `GET /api/jobs/{id}`
- `GET /api/jobs/{id}/file/{wav|video|summary}`

任务产物在 `outputs/web_jobs/`。
