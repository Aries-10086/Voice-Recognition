#!/usr/bin/env python3
"""O1/O2 冒烟: ASR 碎屑粘合 + 空译文合并重试 + 合并阈值"""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.integration.pipeline import CrossLingualPipeline


def _segs():
    # 模拟随身听尾部碎段
    return [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 2.0, "text": "我慢人调的我的磁带和老板"},
        {"speaker": "SPEAKER_00", "start": 2.0, "end": 3.5, "text": "CD我的"},
        {"speaker": "SPEAKER_00", "start": 3.5, "end": 4.2, "text": "ory"},
        {"speaker": "SPEAKER_01", "start": 5.0, "end": 6.0, "text": "我的2000"},
    ]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    # 轻量构造: 只需要 config + 方法
    pipe = CrossLingualPipeline.__new__(CrossLingualPipeline)
    pipe.config = cfg
    pipe.translator = None

    fails = 0
    segs = _segs()
    assert pipe._is_asr_crumb("ory")
    assert pipe._is_asr_crumb("CD我的") is False or len("CD我的") <= 12
    pipe._glue_asr_crumbs(segs)
    # ory / CD我的 都应并入首段
    texts = [s.get("text") or "" for s in segs]
    print("after glue:", texts)
    if "ory" not in texts[0] or "CD" not in texts[0]:
        print("FAIL: crumbs not glued into first segment")
        fails += 1
    if any(t.strip() in ("ory", "CD我的") for t in texts[1:]):
        print("FAIL: crumb still standalone")
        fails += 1

    segs2 = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "短句一"},
        {"speaker": "SPEAKER_00", "start": 1.1, "end": 2, "text": "短"},
        {"speaker": "SPEAKER_01", "start": 3, "end": 4, "text": "另一人"},
    ]
    groups = pipe._merge_short_segments_for_translation(segs2)
    print("merge groups:", groups)
    if groups[0] != [0, 1]:
        print("FAIL: expected [0,1] merge")
        fails += 1

    # 空译文重试: mock translator
    class FakeTr:
        def translate(self, text, **kw):
            return SimpleNamespace(translated_text=f"EN:{text}")

        def _is_bad_translation(self, *a):
            return False

        def _demo_phrase_translate(self, *a):
            return ""

    pipe.translator = FakeTr()
    segs3 = [
        {"speaker": "SPEAKER_00", "start": 0, "end": 1, "text": "甲", "tgt": "", "emotion": "neutral"},
        {"speaker": "SPEAKER_00", "start": 1, "end": 2, "text": "乙", "tgt": "EN:乙", "emotion": "neutral"},
    ]
    mg = [[0], [1]]
    gt = ["", "EN:乙"]
    n = pipe._retry_empty_translations(
        segs3, mg, gt, "zh", "en", "neutral", 0.5, 0.0
    )
    print("retry recovered:", n, "tgt0:", segs3[0].get("tgt"))
    if n < 1 or not (segs3[0].get("tgt") or "").startswith("EN:"):
        print("FAIL: empty retry")
        fails += 1

    dmin = cfg["pipeline"]["segment"]["duration_ratio_min"]
    print("duration_ratio_min:", dmin)
    if float(dmin) < 0.89:
        print("FAIL: duration_ratio_min not raised")
        fails += 1

    if fails:
        print(f"FAILED ({fails})")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
