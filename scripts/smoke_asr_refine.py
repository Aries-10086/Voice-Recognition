#!/usr/bin/env python3
"""快速验证 ASR 后处理（不跑完整 pipeline）。"""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.asr.singing_corrector import refine_singing_transcript

RAW = (
    "想爱过如果是爱了够久 分开月灯吧 想念你的脸加你的 "
    "发 我并不害怕 心中总有对你的牵挂 想遇个 不了的吧 我再也不敢碰它却一硬地动在那个时间 "
    "个玩笑吧 根本吸不掉回忆 你就在我心里面啦 我 必须要处理 为了生活能继续 但现在你在哪里 "
    "虽然已没家如果要 淚不剩落下那些幸福 提升大家"
)
HINT = "如果爱忘了 泪不想落下 那些幸福啊 想念你的脸你的发"

if __name__ == "__main__":
    text, _ = refine_singing_transcript(RAW, [], hint=HINT, title="如果爱忘了")
    print("BEFORE:", RAW[:120], "...")
    print("AFTER: ", text[:120], "...")
