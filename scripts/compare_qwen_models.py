#!/usr/bin/env python3
"""
F5: Qwen3-TTS 0.6B vs 1.7B 同参考音对比

输出:
  outputs/f5_compare/
    clone_0.6B.wav / clone_1.7B.wav
    report.json / report.md

用法:
  source .venv/bin/activate
  export HF_HOME="$(pwd)/models/huggingface"
  python scripts/compare_qwen_models.py
  # 只跑已缓存模型:
  python scripts/compare_qwen_models.py --models 0.6B
"""
from __future__ import annotations

import argparse
import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

MODELS = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}


def _rss_mb() -> float:
    # macOS: ru_maxrss 已是字节; Linux: KB
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == "darwin":
        return rss / (1024 * 1024)
    return rss / 1024


def _pick_ref(wav_path: Path, start: float = 5.1, end: float = 8.7):
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    s, e = int(start * sr), int(end * sr)
    return audio[s:e], sr


def _mfcc_sim(a: np.ndarray, sr_a: int, b: np.ndarray, sr_b: int) -> float:
    """简易 MFCC 余弦相似度 (客观像度代理, 非 MOS)。"""
    import librosa
    if sr_a != 16000:
        a = librosa.resample(a.astype(np.float32), orig_sr=sr_a, target_sr=16000)
    if sr_b != 16000:
        b = librosa.resample(b.astype(np.float32), orig_sr=sr_b, target_sr=16000)
    fa = librosa.feature.mfcc(y=a, sr=16000, n_mfcc=20)
    fb = librosa.feature.mfcc(y=b, sr=16000, n_mfcc=20)
    va, vb = fa.mean(axis=1), fb.mean(axis=1)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) + 1e-8
    return float(np.dot(va, vb) / denom)


def _run_one(tag: str, model_id: str, ref_audio, ref_sr, text: str, ref_text: str, out_dir: Path):
    import gc
    import torch
    from qwen_tts import Qwen3TTSModel

    row = {"tag": tag, "model": model_id, "ok": False}
    rss0 = _rss_mb()

    if torch.cuda.is_available():
        device, dtype = "cuda:0", torch.bfloat16
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        device, dtype = "mps", torch.float32
    else:
        device, dtype = "cpu", torch.float32
    row["device"] = device

    t0 = time.time()
    print(f"\n=== Loading {tag}: {model_id} on {device} ===")
    model = Qwen3TTSModel.from_pretrained(model_id, device_map=device, dtype=dtype)
    row["load_sec"] = round(time.time() - t0, 2)
    row["rss_after_load_mb"] = round(_rss_mb(), 1)

    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        ref_path = f.name
    sf.write(ref_path, ref_audio, int(ref_sr))

    t1 = time.time()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        ref_audio=ref_path,
        ref_text=ref_text,
        max_new_tokens=512,
        repetition_penalty=1.1,
        top_p=0.95,
        top_k=50,
    )
    clone_sec = time.time() - t1
    os.unlink(ref_path)

    audio = wavs[0] if isinstance(wavs, list) else wavs
    audio = np.asarray(audio, dtype=np.float32)
    sr = int(sr)
    out_wav = out_dir / f"clone_{tag}.wav"
    sf.write(out_wav, audio, sr)

    sim = _mfcc_sim(ref_audio, ref_sr, audio, sr)
    row.update({
        "ok": True,
        "clone_sec": round(clone_sec, 2),
        "out_dur_sec": round(len(audio) / sr, 2),
        "out_wav": str(out_wav.relative_to(ROOT)),
        "mfcc_sim_vs_ref": round(sim, 4),
        "rss_peak_mb": round(_rss_mb(), 1),
        "rss_delta_mb": round(_rss_mb() - rss0, 1),
    })
    print(
        f"  load={row['load_sec']}s clone={row['clone_sec']}s "
        f"dur={row['out_dur_sec']}s mfcc_sim={row['mfcc_sim_vs_ref']} "
        f"rss={row['rss_peak_mb']}MB"
    )

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    elif device == "mps":
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    return row


def _write_report(rows: list, out_dir: Path, text: str):
    report = {
        "text": text,
        "models": rows,
        "recommendation": None,
        "notes": [
            "mfcc_sim_vs_ref 为客观代理, 非正式 MOS; 请听 clone_*.wav 填 1–5 像度",
            "默认模型决策见 report.md / config",
        ],
    }
    ok = [r for r in rows if r.get("ok")]
    if len(ok) >= 2:
        # 优先像度代理, 其次速度
        best_sim = max(ok, key=lambda r: r.get("mfcc_sim_vs_ref", -1))
        fastest = min(ok, key=lambda r: r.get("clone_sec", 1e9))
        # Mac 演示: 若 1.7B 像度明显更好且 clone < 3x 0.6B → 推荐 1.7B; 否则保 0.6B
        b06 = next((r for r in ok if r["tag"] == "0.6B"), None)
        b17 = next((r for r in ok if r["tag"] == "1.7B"), None)
        if b06 and b17:
            sim_gain = b17["mfcc_sim_vs_ref"] - b06["mfcc_sim_vs_ref"]
            time_ratio = b17["clone_sec"] / max(b06["clone_sec"], 0.01)
            if sim_gain >= 0.02 and time_ratio <= 3.0:
                rec = "1.7B"
                reason = f"像度代理 +{sim_gain:.3f}, 耗时比 {time_ratio:.1f}x ≤3"
            elif time_ratio > 3.0 and sim_gain < 0.05:
                rec = "0.6B"
                reason = f"1.7B 慢 {time_ratio:.1f}x 且像度增益有限 ({sim_gain:+.3f})"
            else:
                rec = "0.6B"
                reason = f"默认保速: sim_gain={sim_gain:+.3f}, time_ratio={time_ratio:.1f}x"
            report["recommendation"] = {"model": rec, "reason": reason}
        else:
            report["recommendation"] = {
                "model": best_sim["tag"],
                "reason": f"最高 mfcc_sim={best_sim['mfcc_sim_vs_ref']}",
            }
    elif len(ok) == 1:
        report["recommendation"] = {
            "model": ok[0]["tag"],
            "reason": "仅一侧跑通",
        }

    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# F5 Qwen3-TTS 0.6B vs 1.7B",
        "",
        f"克隆文本: `{text}`",
        "",
        "| 模型 | 设备 | 加载(s) | 克隆(s) | 输出时长(s) | MFCC相似度 | RSS峰值(MB) | 音频 |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        if not r.get("ok"):
            lines.append(
                f"| {r.get('tag')} | — | — | — | — | — | — | FAIL: {r.get('error','')} |"
            )
            continue
        lines.append(
            f"| {r['tag']} | {r['device']} | {r['load_sec']} | {r['clone_sec']} | "
            f"{r['out_dur_sec']} | {r['mfcc_sim_vs_ref']} | {r['rss_peak_mb']} | "
            f"`{r['out_wav']}` |"
        )
    rec = report.get("recommendation") or {}
    lines += [
        "",
        f"**建议默认**: `{rec.get('model', '?')}` — {rec.get('reason', '')}",
        "",
        "## 主观 MOS（请填写 1–5）",
        "",
        "| 听者 | 0.6B 像度 | 1.7B 像度 | 可懂度 | 备注 |",
        "|---|---|---|---|---|",
        "| A |  |  |  |  |",
        "| B |  |  |  |  |",
        "| C |  |  |  |  |",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="0.6B,1.7B", help="逗号分隔: 0.6B,1.7B")
    ap.add_argument(
        "--input",
        default=str(ROOT / "data" / "dialog_two_speakers_16k.wav"),
    )
    ap.add_argument(
        "--text",
        default="Today we will discuss the progress of this project.",
    )
    ap.add_argument(
        "--ref-text",
        default="目前语音克隆还分不太清不同说话人",
    )
    args = ap.parse_args()

    out_dir = ROOT / "outputs" / "f5_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    wav = Path(args.input)
    if not wav.exists():
        print(f"missing input: {wav}")
        return 1

    ref_audio, ref_sr = _pick_ref(wav)
    sf.write(out_dir / "reference.wav", ref_audio, ref_sr)

    tags = [t.strip() for t in args.models.split(",") if t.strip()]
    rows = []
    for tag in tags:
        if tag not in MODELS:
            print(f"unknown model tag: {tag}")
            return 1
        try:
            rows.append(
                _run_one(tag, MODELS[tag], ref_audio, ref_sr, args.text, args.ref_text, out_dir)
            )
        except Exception as e:
            print(f"FAIL {tag}: {e}")
            rows.append({"tag": tag, "model": MODELS[tag], "ok": False, "error": str(e)[:200]})

    report = _write_report(rows, out_dir, args.text)
    print("\n=== Done ===")
    print(json.dumps(report.get("recommendation"), ensure_ascii=False, indent=2))
    print(f"report: {out_dir / 'report.md'}")
    return 0 if any(r.get("ok") for r in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
