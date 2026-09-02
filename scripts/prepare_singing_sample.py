#!/usr/bin/env python3
"""
S4: 从本地 MP3 制备清唱验收样例 data/singing_*.wav + .gold.json
用法:
  python scripts/prepare_singing_sample.py \\
    --mp3 "/path/如果爱忘了..mp3" --name ruguaiawang --seconds 60
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mp3", required=True, help="源 MP3 (清唱/干声)")
    ap.add_argument("--name", default="ruguaiawang", help="输出基名 singing_<name>")
    ap.add_argument("--seconds", type=float, default=60.0, help="截取秒数")
    ap.add_argument(
        "--lyrics-file",
        type=str,
        default=None,
        help="可选整段歌词 txt，写入 gold.json",
    )
    args = ap.parse_args()

    src = Path(args.mp3)
    if not src.exists():
        print(f"missing mp3: {src}")
        return 1

    y, sr = librosa.load(str(src), sr=16000, mono=True)
    n = int(min(len(y), args.seconds * 16000))
    y = y[:n]
    out_wav = ROOT / "data" / f"singing_{args.name}_16k.wav"
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(out_wav, y.astype(np.float32), 16000, subtype="PCM_16")
    print(f"wav: {out_wav} ({n/sr:.1f}s)")

    lyrics = ""
    if args.lyrics_file:
        lyrics = Path(args.lyrics_file).read_text(encoding="utf-8").strip()
    gold_path = out_wav.with_suffix(".gold.json")
    gold = {
        "language": "zh",
        "source": f"local:{src.name}",
        "domain": "singing",
        "segments": [
            {
                "start": 0.0,
                "end": round(n / sr, 3),
                "text": lyrics or "(请填写歌词金标)",
                "speaker": "SPEAKER_00",
            }
        ],
    }
    gold_path.write_text(json.dumps(gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"gold: {gold_path}")
    print("运行: python run_pipeline.py --input", out_wav, "--singing --assume-single-speaker -t en")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
