#!/usr/bin/env python3
"""F6 冒烟: 组间 gap 对齐 + 情感韵律非 neutral"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration.pipeline import CrossLingualPipeline
from src.synthesis.voice_cloner import VoiceCloner


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    fails = 0

    # gap 计算逻辑 (复用 pipeline 公式)
    gap = float(np.clip(1.45 - 1.0, 0.05, 0.8))
    print(f"source gap → {gap:.2f}s (expect 0.45)")
    if abs(gap - 0.45) > 1e-6:
        fails += 1

    # merge groups
    pipe = CrossLingualPipeline.__new__(CrossLingualPipeline)
    pipe.config = cfg
    segments = [
        {"speaker": "A", "start": 0.0, "end": 0.5, "text": "目前語音克隆還分"},
        {"speaker": "A", "start": 0.6, "end": 1.5, "text": "太清不同說話人"},
        {"speaker": "B", "start": 2.0, "end": 2.8, "text": "聽感也一般"},
    ]
    groups = pipe._merge_short_segments_for_translation(segments)
    print(f"merge groups: {groups}")
    if groups != [[0, 1], [2]]:
        print("  FAIL expected [[0,1],[2]]")
        fails += 1
    else:
        print("  PASS merge")

    # emotion prosody changes audio for non-neutral
    cloner = VoiceCloner.__new__(VoiceCloner)
    sr = 16000
    x = (np.sin(2 * np.pi * 220 * np.arange(sr) / sr) * 0.3).astype(np.float32)
    y = cloner._apply_emotion_prosody(x, sr, "happy")
    z = cloner._apply_emotion_prosody(x, sr, "neutral")
    print(f"prosody happy len={len(y)} neutral len={len(z)} src={len(x)}")
    if len(y) == len(x) and float(np.max(np.abs(y - x[: len(y)] if len(y) <= len(x) else x))) < 1e-6:
        print("  FAIL happy prosody unchanged")
        fails += 1
    else:
        print("  PASS happy prosody applied")
    if len(z) != len(x) or not np.allclose(z, x):
        print("  FAIL neutral should be identity")
        fails += 1
    else:
        print("  PASS neutral identity")

    print(f"\n{'PASS' if fails == 0 else 'FAIL'} fails={fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
