#!/usr/bin/env python3
"""F4 冒烟: 音节 length_ratio 压到 0.8–1.2"""
from __future__ import annotations

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.translation.llm_translator import LLMTranslator


def main() -> int:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    tr = LLMTranslator(cfg["translation"])
    # 刻意超长英文译文
    long_en = (
        "For example, an automatic flooding system can really actually make "
        "water protection indeed make a very big difference quite basically"
    )
    src_syl = 12  # 模拟短中文源
    out, ratio, tgt = tr._constrain_length_ratio(long_en, src_syl, "en", [])
    print(f"long → ratio={ratio:.2f} syl={tgt} text={out!r}")
    ok_long = 0.8 <= ratio <= 1.2
    # 刻意过短
    short_en = "OK"
    out2, ratio2, tgt2 = tr._constrain_length_ratio(short_en, 10, "en", [])
    print(f"short → ratio={ratio2:.2f} syl={tgt2} text={out2!r}")
    ok_short = ratio2 >= 0.8
    # 在线翻译一条演示句
    demo_src = "目前语音克隆还分不太清不同说话人，听感也一般"
    res = tr.translate(demo_src, "zh", "en", emotion="neutral", refine=False)
    print(
        f"online → ratio={res.length_ratio:.2f} "
        f"({res.source_syllables}->{res.target_syllables}) {res.translated_text!r}"
    )
    ok_online = 0.8 <= res.length_ratio <= 1.2
    print(f"PASS compress: {ok_long}")
    print(f"PASS expand: {ok_short}")
    print(f"PASS online: {ok_online}")
    return 0 if (ok_long and ok_short and ok_online) else 1


if __name__ == "__main__":
    raise SystemExit(main())
