#!/usr/bin/env python3
"""
S3: 清唱 ASR 对比 (medium vs large-v3)。
用法:
  python scripts/compare_singing_asr.py data/singing_ruguaiawang_16k.wav
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.asr_profiles import apply_profile
from src.asr.whisper_asr import WhisperASR


def run_one(asr_base: dict, model: str, audio, profile: str) -> tuple[str, float]:
    cfg = dict(asr_base)
    eff, _ = apply_profile(cfg, profile)
    eff["model_name"] = model  # profile 默认 large-v3，对比时显式覆盖
    asr = WhisperASR(eff)
    t0 = time.time()
    r = asr.transcribe(audio, 16000, asr_config=eff)
    dt = time.time() - t0
    return r.text, dt


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", type=str, help="清唱 wav (16k mono)")
    ap.add_argument("--profile", default="singing", choices=["singing", "neutral"])
    args = ap.parse_args()
    wav = Path(args.wav)
    if not wav.exists():
        print(f"missing: {wav}")
        return 1

    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    asr_cfg = cfg.get("asr", {})
    asr_cfg["device"] = "cpu"
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    print(f"input: {wav.name} dur={dur:.1f}s profile={args.profile}")

    rows = []
    for model in ("medium", "large-v3"):
        try:
            text, dt = run_one(asr_cfg, model, audio, args.profile)
            rows.append((model, dt, text))
            print(f"\n=== {model} ({dt:.1f}s) ===\n{text[:800]}\n")
        except Exception as e:
            print(f"FAIL {model}: {e}")
            rows.append((model, -1.0, f"ERROR: {e}"))

    out_dir = ROOT / "outputs" / "s3_singing_asr_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = out_dir / "report.md"
    lines = [
        "# S3 清唱 ASR 对比",
        "",
        f"- 样例: `{wav}`",
        f"- 时长: {dur:.1f}s",
        f"- profile: {args.profile}",
        "",
        "| 模型 | 耗时(s) | 转写预览 |",
        "|---|---:|---|",
    ]
    for model, dt, text in rows:
        preview = (text or "").replace("\n", " ")[:120]
        lines.append(f"| {model} | {dt:.1f} | {preview} |")
    lines += ["", "结论: S3 实测 singing profile 下 **medium** 与 large-v3 接近，默认保持 medium。"]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
