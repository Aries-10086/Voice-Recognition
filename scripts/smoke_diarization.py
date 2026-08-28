#!/usr/bin/env python3
"""F1/F2 冒烟: 说话人分离首轮人数 + label_confidence"""
from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.whisper_asr import WhisperASR
from src.asr.speaker_diarization import SpeakerDiarizer


def main() -> int:
    wav = ROOT / "data" / "dialog_two_speakers_16k.wav"
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    asr_cfg = cfg["asr"]
    diar_cfg = cfg["pipeline"]["diarization"]

    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    asr = WhisperASR(asr_cfg)
    result = asr.transcribe(audio, sr)
    print(f"ASR segments: {len(result.segments)}")

    diar = SpeakerDiarizer(diar_cfg)
    labels = diar.diarize(audio, sr, result.segments)
    meta = diar.last_meta

    print(f"backend: {meta.get('backend')}")
    print(f"speakers: {meta.get('n_speakers')} (first_round={meta.get('first_round_speakers')})")
    print(f"used_retry: {meta.get('used_retry')} degraded={meta.get('degraded')}")
    confs = meta.get("label_confidences") or []
    if confs:
        print(f"label_confidence avg={sum(confs)/len(confs):.3f} min={min(confs):.3f}")

    for i, (seg, lb) in enumerate(zip(result.segments, labels), 1):
        c = confs[i - 1] if i - 1 < len(confs) else None
        cs = f" conf={c:.2f}" if c is not None else ""
        print(f"  {i}. [{seg['start']:.2f}-{seg['end']:.2f}] {lb}{cs} | {seg.get('text','')[:32]}")

    ok_spk = meta.get("n_speakers", 0) >= 2
    ok_first = meta.get("first_round_speakers", 0) >= 2
    ok_no_retry = not meta.get("used_retry", True)
    print(f"PASS speakers>=2: {ok_spk}")
    print(f"PASS first_round>=2: {ok_first} (target F1)")
    print(f"PASS no_retry: {ok_no_retry} (stretch goal)")

    if not ok_spk:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
