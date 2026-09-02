#!/usr/bin/env python3
"""G1 冒烟: ASR 幻觉检测与护栏逻辑。"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.asr_profiles import (
    apply_profile,
    detect_hallucination,
    DEFAULT_BANNED_PHRASES,
)

OLD_PROMPT = "这是一段中文对话。讨论科技、气候、河流与历史。"
GOOD_TEXT = "我是千夏，版本已经开启，参与活动领取月卡。"
BAD_TEXT = "讨论科技、气候、河流与历史。讨论科技、气候、河流与历史。"


def main() -> int:
    fails = 0
    is_hal, score, reason = detect_hallucination(BAD_TEXT, OLD_PROMPT, DEFAULT_BANNED_PHRASES)
    print(f"bad: hal={is_hal} score={score:.2f} reason={reason}")
    if not is_hal:
        print("FAIL: should detect banned loop")
        fails += 1

    is_hal, score, reason = detect_hallucination(GOOD_TEXT, OLD_PROMPT, DEFAULT_BANNED_PHRASES)
    print(f"good: hal={is_hal} score={score:.2f} reason={reason}")
    if is_hal:
        print("FAIL: should not flag normal speech")
        fails += 1

    cfg = {"prompt_profile": "singing"}
    eff, name = apply_profile(cfg, "singing")
    print(f"profile={name} model={eff.get('model_name')} max_seg={eff.get('max_segment_seconds')}")
    if name != "singing" or eff.get("max_segment_seconds", 0) < 5:
        print("FAIL: singing profile not applied")
        fails += 1
    if "清唱" not in (eff.get("initial_prompt") or ""):
        print("FAIL: singing prompt missing")
        fails += 1

    if fails:
        print(f"FAILED ({fails})")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
