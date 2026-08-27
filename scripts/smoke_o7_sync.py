#!/usr/bin/env python3
"""O7 冒烟: AOCP VAD + 元音同步点 + sync_score (≥0.65)"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.alignment.articulatory_analyzer import AOCPNet
from src.alignment.phoneme_aligner import PhonemeAligner
from src.alignment.timeline_generator import TimelineGenerator


def main() -> int:
    wav = ROOT / "data" / "dialog_two_speakers_16k.wav"
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    align_cfg = cfg["alignment"]
    target = float(align_cfg.get("sync", {}).get("target_score", 0.65))

    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    duration = len(audio) / sr

    # 演示样例中文转写 (与 ASR 大致一致, 避免加载 Whisper)
    text = (
        "你好今天我们来讨论一下这个项目的进度"
        "目前语音克隆还分不太清不同说话人听感也一般"
        "那我们先把说话人分离做稳再提升克隆质量"
    )
    target_text = (
        "Hello, let's discuss the project progress today. "
        "Currently voice cloning still cannot tell speakers apart. "
        "Let's stabilize speaker separation first, then improve clone quality."
    )

    aocp = AOCPNet(align_cfg)
    aocp_result = aocp.predict(audio, sr, device="cpu")
    n_open = sum(1 for s in aocp_result.state_segments if s["state"] == "open")
    open_dur = sum(s["duration"] for s in aocp_result.state_segments if s["state"] == "open")
    print(f"AOCP: segs={len(aocp_result.state_segments)} open={n_open} "
          f"open_dur={open_dur:.2f}/{duration:.2f}s")

    aligner = PhonemeAligner(align_cfg)
    src_ph = aligner.text_to_phonemes(text, "zh")
    try:
        tgt_ph = aligner.text_to_phonemes(target_text, "en")
    except Exception as e:
        print(f"en phonemes fallback: {e}")
        tgt_ph = list(target_text.replace(" ", ""))
    ph_align = aligner.align(audio, src_ph, sr)
    vowels = aligner.extract_vowel_timeline(ph_align)
    core = [v for v in vowels if v["is_core"]]
    print(f"phonemes={len(ph_align.phonemes)} vowels={len(vowels)} core={len(core)}")

    # length_ratio 近似演示验收值
    length_ratio = 1.16
    tg = TimelineGenerator(align_cfg)
    tl = tg.generate(
        aocp_result=aocp_result,
        phoneme_alignment=ph_align,
        source_duration=duration,
        target_phonemes=tgt_ph,
        length_ratio=length_ratio,
        vowel_timeline=vowels,
    )
    n_vowel_pts = sum(1 for p in tl.sync_points if p.get("type") == "vowel")
    print(
        f"timeline: speech={len(tl.speech_segments)} sync_points={len(tl.sync_points)} "
        f"vowel_pts={n_vowel_pts} coverage={tl.coverage:.3f} sync_score={tl.sync_score:.3f}"
    )

    # before/after 简表 (无元音 vs 有元音)
    tl_before = tg.generate(
        aocp_result=aocp_result,
        phoneme_alignment=ph_align,
        source_duration=duration,
        target_phonemes=tgt_ph,
        length_ratio=length_ratio,
        vowel_timeline=None,
    )
    print(f"before(no vowel): sync={tl_before.sync_score:.3f} | "
          f"after(+vowel): sync={tl.sync_score:.3f} | target>={target}")

    ok = tl.sync_score >= target
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
