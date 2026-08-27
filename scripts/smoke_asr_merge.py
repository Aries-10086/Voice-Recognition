#!/usr/bin/env python3
"""ASR 碎句合并冒烟: 单元用例 + 可选真实 wav"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import yaml
from src.asr.whisper_asr import WhisperASR


def _fake_asr(cfg: dict) -> WhisperASR:
    """跳过模型加载, 只测合并逻辑"""
    obj = object.__new__(WhisperASR)
    obj.config = cfg
    obj.engine_name = "faster_whisper"
    obj.model = None
    obj.processor = None
    return obj


def test_unit() -> None:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))["asr"]
    asr = _fake_asr(cfg)

    # 复现 run_154157 的碎句
    segs = [
        {"start": 0.00, "end": 0.50, "text": "你好,", "words": [], "syllables": []},
        {"start": 1.10, "end": 4.14, "text": "今天我們來討論一下這個項目的進度", "words": [], "syllables": []},
        {"start": 5.05, "end": 7.04, "text": "目前語音克隆還分", "words": [], "syllables": []},
        {"start": 7.04, "end": 8.83, "text": "太清不同說話人", "words": [], "syllables": []},
        {"start": 8.91, "end": 10.29, "text": "聽感也一般", "words": [], "syllables": []},
        {"start": 11.20, "end": 12.54, "text": "那我們先把說", "words": [], "syllables": []},
        {"start": 12.54, "end": 14.10, "text": "話人分離坐穩", "words": [], "syllables": []},
        {"start": 14.15, "end": 15.81, "text": "再提升克隆質量", "words": [], "syllables": []},
    ]
    out = asr._merge_broken_phrases(segs)
    texts = [s["text"] for s in out]
    print("UNIT merged", len(segs), "->", len(out))
    for i, t in enumerate(texts, 1):
        print(f"  {i}. {t}")

    joined = "".join(texts)
    assert "還分太清" in joined or "还分太清" in joined or "目前語音克隆還分太清不同說話人" in joined, texts
    assert any("把說話人分離坐穩" in t or "把说话人分离坐稳" in t for t in texts), texts
    assert any("再提升克隆質量" in t or "再提升克隆质量" in t for t in texts), texts
    assert not any("再提升" in t and "把說" in t for t in texts), "should not merge 再提升 into prev"
    assert not any(t == "太清不同說話人" for t in texts), "太清 should be merged"
    assert not any(t == "話人分離坐穩" for t in texts), "話人 should be merged"
    assert len(out) >= 6, f"over-merged to {len(out)}: {texts}"
    # 不应把无关句并掉
    assert any("你好" in t for t in texts)
    assert any("再提升" in t for t in texts)
    print("UNIT PASS")


def test_wav() -> int:
    wav = ROOT / "data" / "dialog_two_speakers_16k.wav"
    if not wav.exists():
        print("WAV skip (no file)")
        return 0
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))["asr"]
    asr = WhisperASR(cfg)
    import soundfile as sf
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    result = asr.transcribe(audio, sr)
    print("WAV segs:", len(result.segments))
    for i, s in enumerate(result.segments, 1):
        print(f"  {i}. [{s['start']:.2f}-{s['end']:.2f}] {s['text']}")
    texts = [s["text"] for s in result.segments]
    bad = [t for t in texts if t.strip() in ("太清不同說話人", "話人分離坐穩", "太清不同说话人", "话人分离坐稳")]
    if bad:
        print("WAV WARN still fragmented:", bad)
        return 1
    # 期望合并后能看到完整关键短语
    blob = "".join(texts)
    ok_a = ("分不太清" in blob) or ("分太清" in blob)
    ok_b = ("把說話人" in blob) or ("把说话人" in blob)
    print("WAV phrase checks:", "分不清/分太清=", ok_a, "把说话人=", ok_b)
    if not (ok_a and ok_b):
        print("WAV FAIL phrase merge incomplete")
        return 1
    print("WAV PASS")
    return 0


if __name__ == "__main__":
    test_unit()
    raise SystemExit(test_wav())
