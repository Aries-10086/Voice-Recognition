#!/usr/bin/env python3
"""
制备 mp4 冒烟样例：用现有 wav + 纯色画面合成短视频。

仅用于验证「抽音 → 全链路 → 回贴音轨」工程链路，
**不算** F9 口型演示验收（口型演示需带人脸正脸说话视频）。

用法:
  python scripts/prepare_video_smoke.py \\
    --audio data/solo_fleurs_excerpt_16k.wav \\
    --out data/video_smoke_solo.mp4 \\
    --duration 20
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.utils.ffmpeg_bin import require_ffmpeg  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="wav → 纯色 mp4（链路冒烟，非口型验收）")
    ap.add_argument("--audio", required=True, help="源 wav/mp3")
    ap.add_argument("--out", default="data/video_smoke_solo.mp4")
    ap.add_argument("--duration", type=float, default=20.0, help="截取秒数，0=全文")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--color", default="c0c0c0", help="纯色背景 hex，无 #")
    args = ap.parse_args()

    ffmpeg = require_ffmpeg()
    audio = args.audio if os.path.isabs(args.audio) else os.path.join(ROOT, args.audio)
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    if not os.path.isfile(audio):
        raise SystemExit(f"音频不存在: {audio}")

    cmd = [
        ffmpeg, "-y",
        "-f", "lavfi", "-i", f"color=c=#{args.color}:s={args.size}:r=25",
        "-i", audio,
    ]
    if args.duration and args.duration > 0:
        cmd += ["-t", str(args.duration)]
    cmd += [
        "-c:v", "libx264", "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-shortest",
        out,
    ]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"OK -> {out}")
    print("说明: 此文件仅冒烟；F9 请换带人脸正脸说话 ≥15s 的 mp4。")


if __name__ == "__main__":
    main()
