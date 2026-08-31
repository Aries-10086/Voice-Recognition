#!/usr/bin/env python3
"""F7 冒烟: 错译黑名单 + 术语表 + sit down / Taiqing 修复"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.translation.llm_translator import LLMTranslator


CASES = [
    (
        "那我们先把说话人分离坐稳",
        "Then let's separate the speakers and sit down first.",
        ["sit down"],
        ["stabilize"],
    ),
    (
        "目前语音克隆还分太清不同说话人",
        "Currently Taiqing cannot tell speakers",
        ["taiqing"],
        ["speaker"],
    ),
    (
        "听感也一般",
        "The listening",
        [],
        ["listening quality"],
    ),
]


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    tr = LLMTranslator(cfg["translation"])
    fails = 0
    for src, bad_tgt, forbid, require in CASES:
        out = tr._refine_translation_quality(src, bad_tgt, "en", "neutral", 0.5)
        low = out.lower()
        print(f"SRC: {src}")
        print(f"  in : {bad_tgt}")
        print(f"  out: {out}")
        for f in forbid:
            if f.lower() in low:
                print(f"  FAIL still has {f!r}")
                fails += 1
        for r in require:
            if r.lower() not in low:
                print(f"  FAIL missing {r!r}")
                fails += 1
        # bad detector
        if tr._is_bad_translation(bad_tgt, src, "en") is False and "sit down" in bad_tgt.lower():
            print("  FAIL bad detector missed sit down")
            fails += 1
        print()

    # happy lexicon
    happy = tr._apply_emotion_lexicon("This is good.", "happy", 0.8)
    print(f"emotion happy: {happy}")
    if not happy.endswith("!"):
        print("  FAIL happy should end with !")
        fails += 1

    # live translate 关键句
    res = tr.translate(
        "那我们先把说话人分离坐稳", "zh", "en", emotion="neutral", refine=False
    )
    print(f"online: {res.translated_text}")
    if "sit down" in res.translated_text.lower():
        print("  FAIL online still sit down")
        fails += 1
    else:
        print("  PASS online no sit down")

    print(f"\n{'PASS' if fails == 0 else 'FAIL'} fails={fails}")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
