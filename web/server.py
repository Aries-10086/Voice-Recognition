#!/usr/bin/env python3
"""
VoiceClone Web 控制台 — FastAPI 后端

启动:
  python scripts/run_web.py
  # 或: uvicorn web.server:app --host 127.0.0.1 --port 8765
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from loguru import logger

# ---- 路径与模型缓存（对齐 run_pipeline）----
WEB_DIR = Path(__file__).resolve().parent
ROOT = WEB_DIR.parent
sys.path.insert(0, str(ROOT))

_MODEL_ROOT = ROOT / "models"
_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODEL_ROOT / "huggingface"))
os.environ.setdefault(
    "HUGGINGFACE_HUB_CACHE",
    str(Path(os.environ["HF_HOME"]) / "hub"),
)
os.environ.setdefault("HF_HUB_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
os.environ.setdefault("TORCH_HOME", str(_MODEL_ROOT / "torch"))
os.environ.setdefault("CTRANSLATE2_MODELS", str(_MODEL_ROOT / "ctranslate2"))
os.environ.setdefault("TRANSFORMERS_CACHE", os.environ["HUGGINGFACE_HUB_CACHE"])
_nltk = ROOT / "models" / "nltk_data"
if _nltk.exists():
    os.environ.setdefault("NLTK_DATA", str(_nltk))

STATIC_DIR = WEB_DIR / "static"
UPLOAD_DIR = ROOT / "outputs" / "web_uploads"
JOB_DIR = ROOT / "outputs" / "web_jobs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
JOB_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES = [
    {
        "id": "willkent_video",
        "label": "正脸口播 · Will Kent (mp4)",
        "path": "data/talking_head_willkent_20s.mp4",
        "video": True,
        "source_lang": "en",
        "target_lang": "zh",
        "profile": "video_talking",
        "assume_single": True,
        "singing": False,
    },
    {
        "id": "real_dialog_02",
        "label": "中文对白 · real_dialog_02",
        "path": "data/real_dialog_02.wav",
        "video": False,
        "source_lang": "zh",
        "target_lang": "en",
        "profile": "neutral",
        "assume_single": False,
        "singing": False,
    },
    {
        "id": "singing_ruguaiawang",
        "label": "清唱 · 如果爱忘了",
        "path": "data/singing_ruguaiawang_16k.wav",
        "video": False,
        "source_lang": "zh",
        "target_lang": "en",
        "profile": "singing",
        "assume_single": True,
        "singing": True,
        "no_gold": True,
        "no_lyrics_hint": True,
    },
    {
        "id": "solo_fleurs",
        "label": "朗读 · FLEURS 摘录",
        "path": "data/solo_fleurs_excerpt_16k.wav",
        "video": False,
        "source_lang": "zh",
        "target_lang": "en",
        "profile": "fleurs_dialog",
        "assume_single": True,
        "singing": False,
    },
]


@dataclass
class JobState:
    id: str
    status: str = "queued"  # queued|running|success|degraded|error
    message: str = ""
    created_at: float = field(default_factory=time.time)
    started_at: float = 0.0
    finished_at: float = 0.0
    params: Dict[str, Any] = field(default_factory=dict)
    result: Dict[str, Any] = field(default_factory=dict)
    error: str = ""
    work_dir: str = ""
    input_path: str = ""

    def public(self) -> Dict[str, Any]:
        d = asdict(self)
        # 不暴露绝对机内无关字段过多；保留路径给前端拼 URL
        return d


_jobs: Dict[str, JobState] = {}
_jobs_lock = threading.Lock()
_queue: List[str] = []
_worker_started = False
_pipeline = None
_pipeline_lock = threading.Lock()
_pipeline_device = "cpu"


def _load_config(device: str = "cpu") -> dict:
    import yaml

    cfg_path = ROOT / "config" / "default.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    config.setdefault("system", {})["device"] = device
    config.setdefault("asr", {})["device"] = device
    config["device"] = device
    for key in ["output_dir", "model_dir"]:
        if key in config.get("system", {}):
            path = config["system"][key]
            if not os.path.isabs(path):
                config["system"][key] = str(ROOT / path)
    return config


def _get_pipeline(device: str = "cpu"):
    global _pipeline, _pipeline_device
    with _pipeline_lock:
        if _pipeline is None or _pipeline_device != device:
            from src.integration.pipeline import CrossLingualPipeline

            logger.info(f"Loading pipeline (device={device})…")
            _pipeline = CrossLingualPipeline(_load_config(device))
            _pipeline_device = device
            logger.info("Pipeline ready")
        return _pipeline


def _ensure_worker():
    global _worker_started
    if _worker_started:
        return
    _worker_started = True
    t = threading.Thread(target=_worker_loop, name="voiceclone-web-worker", daemon=True)
    t.start()


def _worker_loop():
    while True:
        job_id = None
        with _jobs_lock:
            if _queue:
                job_id = _queue.pop(0)
        if not job_id:
            time.sleep(0.25)
            continue
        with _jobs_lock:
            job = _jobs.get(job_id)
        if not job:
            continue
        _run_job(job)


def _run_job(job: JobState):
    job.status = "running"
    job.started_at = time.time()
    job.message = "加载模型 / 处理中…"
    work = Path(job.work_dir)
    work.mkdir(parents=True, exist_ok=True)

    try:
        params = job.params
        device = params.get("device") or "cpu"
        pipeline = _get_pipeline(device)

        input_path = job.input_path
        is_video = bool(params.get("video"))
        audio_path = input_path
        temp_audio = None
        video_out_name = None

        if is_video:
            from src.utils.video_utils import VideoUtils
            from src.utils.audio_utils import AudioUtils
            import tempfile

            job.message = "从视频抽音…"
            audio_arr, audio_sr = VideoUtils.extract_audio_from_video(
                input_path, target_sr=16000
            )
            fd, temp_audio = tempfile.mkstemp(prefix="webvc_", suffix=".wav")
            os.close(fd)
            AudioUtils.save_audio(audio_arr, temp_audio, audio_sr)
            audio_path = temp_audio
            stem = Path(input_path).stem
            video_out_name = f"{stem}_cloned.mp4"

        profile = params.get("profile") or None
        if params.get("singing"):
            profile = "singing"
        elif is_video and not profile:
            profile = "video_talking"

        job.message = "ASR → 翻译 → 时间轴 → 克隆…"
        result = pipeline.run(
            audio_path=audio_path,
            target_lang=params.get("target_lang") or "en",
            output_dir=str(work),
            prompt_profile=profile,
            assume_single_speaker=bool(params.get("assume_single")) or None,
            skip_gold=bool(params.get("no_gold")),
            no_lyrics_hint=bool(params.get("no_lyrics_hint")),
            source_lang=params.get("source_lang") or None,
            sidecar_base_path=input_path if is_video else None,
        )

        # pipeline 会在 output_dir 下再建 run_*；定位最新
        run_dirs = sorted(work.glob("run_*"), key=lambda p: p.stat().st_mtime)
        run_dir = Path(result.output_dir) if result.output_dir else (run_dirs[-1] if run_dirs else work)

        cloned_wav = result.cloned_wav_path or str(run_dir / "cloned_output.wav")
        video_path = ""
        if video_out_name and result.status in ("success", "degraded"):
            job.message = "回贴视频音轨…"
            try:
                from src.utils.video_utils import VideoUtils
                import soundfile as sf

                if os.path.isfile(cloned_wav):
                    audio_data, audio_sr = sf.read(cloned_wav)
                    out_mp4 = str(run_dir / video_out_name)
                    VideoUtils.replace_audio_in_video(
                        input_path, audio_data, int(audio_sr), out_mp4
                    )
                    video_path = out_mp4
            except Exception as e:
                logger.warning(f"video remux failed: {e}")
                result.quality_meta.setdefault("degraded_reasons", []).append(
                    "video_remux_failed"
                )
                if result.status == "success":
                    result.status = "degraded"

        if temp_audio:
            try:
                os.unlink(temp_audio)
            except OSError:
                pass

        # 读摘要产物
        summary = ""
        translation = ""
        sync = None
        asr_text = ""
        if result.asr_result:
            asr_text = result.asr_result.text or ""
        if result.translation_result:
            translation = result.translation_result.translated_text or ""
        if result.timeline_result:
            sync = float(result.timeline_result.sync_score)
        sum_path = run_dir / "summary.txt"
        if sum_path.exists():
            summary = sum_path.read_text(encoding="utf-8")
        tr_path = run_dir / "translation.txt"
        if tr_path.exists() and not translation:
            translation = tr_path.read_text(encoding="utf-8")

        job.status = result.status if result.status in ("success", "degraded") else "error"
        job.finished_at = time.time()
        job.message = "完成" if job.status != "error" else (result.error_message or "失败")
        job.error = result.error_message or ""
        job.result = {
            "run_dir": str(run_dir),
            "status": job.status,
            "processing_time": round(float(result.processing_time or 0), 1),
            "source_lang": result.detected_language,
            "target_lang": params.get("target_lang"),
            "asr_text": asr_text,
            "translation": translation,
            "sync_score": sync,
            "summary": summary[:4000],
            "has_wav": os.path.isfile(cloned_wav),
            "has_video": bool(video_path and os.path.isfile(video_path)),
            "cloned_wav": cloned_wav if os.path.isfile(cloned_wav) else "",
            "cloned_video": video_path,
            "profile": (result.quality_meta or {}).get("prompt_profile"),
            "degraded_reasons": (result.quality_meta or {}).get("degraded_reasons") or [],
        }
        # 写一份 job.json 便于回看
        (work / "job.json").write_text(
            json.dumps(job.public(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        logger.exception("job failed")
        job.status = "error"
        job.error = str(e)
        job.message = f"失败: {e}"
        job.finished_at = time.time()


app = FastAPI(title="VoiceClone Web", version="1.0")


@app.on_event("startup")
def _startup():
    _ensure_worker()
    logger.info("VoiceClone web ready")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "pipeline_loaded": _pipeline is not None,
        "queued": len(_queue),
        "jobs": len(_jobs),
    }


@app.get("/api/samples")
def list_samples():
    out = []
    for s in SAMPLES:
        p = ROOT / s["path"]
        item = dict(s)
        item["available"] = p.is_file()
        out.append(item)
    return {"samples": out}


def _as_bool(v) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("1", "true", "yes", "on")


@app.post("/api/jobs")
async def create_job(
    file: Optional[UploadFile] = File(None),
    sample_id: Optional[str] = Form(None),
    target_lang: str = Form("zh"),
    source_lang: Optional[str] = Form(None),
    profile: Optional[str] = Form(None),
    singing: str = Form("false"),
    video: str = Form("false"),
    assume_single: str = Form("true"),
    no_gold: str = Form("true"),
    no_lyrics_hint: str = Form("false"),
    device: str = Form("cpu"),
):
    job_id = time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    work = JOB_DIR / job_id
    work.mkdir(parents=True, exist_ok=True)

    input_path = ""
    params: Dict[str, Any] = {
        "target_lang": target_lang,
        "source_lang": (source_lang or None) or None,
        "profile": (profile or None) or None,
        "singing": _as_bool(singing),
        "video": _as_bool(video),
        "assume_single": _as_bool(assume_single),
        "no_gold": _as_bool(no_gold),
        "no_lyrics_hint": _as_bool(no_lyrics_hint),
        "device": device or "cpu",
    }

    if sample_id:
        sample = next((s for s in SAMPLES if s["id"] == sample_id), None)
        if not sample:
            raise HTTPException(404, f"unknown sample: {sample_id}")
        src = ROOT / sample["path"]
        if not src.is_file():
            raise HTTPException(404, f"sample file missing: {sample['path']}")
        dest = work / src.name
        shutil.copy2(src, dest)
        input_path = str(dest)
        params["video"] = bool(sample.get("video")) or params["video"]
        if not params["source_lang"]:
            params["source_lang"] = sample.get("source_lang")
        if not params["profile"]:
            params["profile"] = sample.get("profile")
        params["singing"] = params["singing"] or bool(sample.get("singing"))
        if sample.get("no_gold"):
            params["no_gold"] = True
        if sample.get("no_lyrics_hint"):
            params["no_lyrics_hint"] = True
        # 前端点样例时已写入 checkbox；此处仅在用户未改时补 assume_single
        if sample.get("assume_single") and not params["assume_single"]:
            params["assume_single"] = True
    elif file is not None and file.filename:
        suffix = Path(file.filename).suffix.lower() or ".wav"
        dest = work / f"upload{suffix}"
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        input_path = str(dest)
        if suffix in (".mp4", ".mkv", ".mov", ".webm"):
            params["video"] = True
    else:
        raise HTTPException(400, "需要上传文件或选择 sample_id")

    job = JobState(
        id=job_id,
        status="queued",
        message="排队中",
        params=params,
        work_dir=str(work),
        input_path=input_path,
    )
    with _jobs_lock:
        _jobs[job_id] = job
        _queue.append(job_id)
    _ensure_worker()
    return {"job_id": job_id, "status": "queued"}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    if not job:
        # 尝试磁盘恢复
        meta = JOB_DIR / job_id / "job.json"
        if meta.is_file():
            return json.loads(meta.read_text(encoding="utf-8"))
        raise HTTPException(404, "job not found")
    return job.public()


@app.get("/api/jobs/{job_id}/file/{kind}")
def get_job_file(job_id: str, kind: str):
    with _jobs_lock:
        job = _jobs.get(job_id)
    result = (job.result if job else {}) or {}
    if not result and (JOB_DIR / job_id / "job.json").is_file():
        result = json.loads((JOB_DIR / job_id / "job.json").read_text(encoding="utf-8")).get(
            "result"
        ) or {}

    path = ""
    media = "application/octet-stream"
    if kind == "wav":
        path = result.get("cloned_wav") or ""
        media = "audio/wav"
    elif kind == "video":
        path = result.get("cloned_video") or ""
        media = "video/mp4"
    elif kind == "summary":
        run_dir = result.get("run_dir") or ""
        path = str(Path(run_dir) / "summary.txt") if run_dir else ""
        media = "text/plain; charset=utf-8"
    else:
        raise HTTPException(400, "kind must be wav|video|summary")

    if not path or not os.path.isfile(path):
        raise HTTPException(404, f"{kind} not ready")
    return FileResponse(path, media_type=media, filename=Path(path).name)


@app.get("/")
def index():
    index_path = STATIC_DIR / "index.html"
    if not index_path.is_file():
        return HTMLResponse("<h1>VoiceClone</h1><p>static/index.html missing</p>")
    return FileResponse(index_path)


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
