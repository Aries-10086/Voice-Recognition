#!/usr/bin/env python3
"""
准备真实双人测试音频:
1) 从 Google FLEURS 拉中文真人短句拼成 dialog (需网络)
2) 或对本地 wav 做轻度谱减法降噪

用法:
  source .venv/bin/activate
  export HF_HOME="$(pwd)/models/huggingface"
  python scripts/prepare_real_dialog.py --fleurs
  python scripts/prepare_real_dialog.py --denoise data/real_dialog_01.wav -o data/noisy_dialog_01.wav
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def denoise(path: Path, out: Path) -> None:
    import librosa
    y, sr = sf.read(str(path), dtype="float32")
    if y.ndim > 1:
        y = y.mean(axis=1)
    S = librosa.stft(y.astype(np.float32))
    mag, phase = np.abs(S), np.angle(S)
    n_frames = max(1, int(0.3 * sr / 512))
    noise = np.median(mag[:, :n_frames], axis=1, keepdims=True)
    mask = mag > (noise * 2.0)
    y_c = librosa.istft(mag * mask * np.exp(1j * phase), length=len(y))
    p = float(np.max(np.abs(y_c)) or 1)
    y_c = (y_c / p * 0.9).astype(np.float32)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), y_c, sr, subtype="PCM_16")
    print(f"denoised → {out} ({len(y_c)/sr:.1f}s)")


def build_fleurs(out_wav: Path, out_txt: Path, out_gold: Path) -> None:
    import librosa
    from collections import defaultdict
    from datasets import load_dataset, Audio

    ds = load_dataset("google/fleurs", "cmn_hans_cn", split="validation", streaming=True)
    ds = ds.cast_column("audio", Audio(decode=False))
    clips = []
    for i, row in enumerate(ds):
        audio = row["audio"]
        try:
            if audio.get("bytes"):
                import io
                y, sr = sf.read(io.BytesIO(audio["bytes"]), dtype="float32")
            elif audio.get("path"):
                y, sr = sf.read(audio["path"], dtype="float32")
            else:
                continue
        except Exception:
            continue
        if y.ndim > 1:
            y = y.mean(axis=1)
        text = (row.get("transcription") or row.get("raw_transcription") or "").strip()
        gender = row.get("gender")
        if len(y) < sr * 1.0 or len(text) < 3:
            continue
        if len(y) > sr * 5:
            y = y[: int(sr * 4)]
        clips.append({"gender": gender, "text": text, "y": y.astype(np.float32), "sr": sr})
        if len(clips) >= 50:
            break

    by = defaultdict(list)
    for c in clips:
        by[str(c["gender"])].append(c)
    keys = sorted(by.keys(), key=lambda k: -len(by[k]))
    if len(keys) < 2:
        raise RuntimeError("not enough gender groups in FLEURS sample")
    a, b = by[keys[0]], by[keys[1]]
    pieces, texts = [], []
    gap = np.zeros(int(0.4 * 16000), dtype=np.float32)
    for i in range(2):
        for bag, tag in ((a, "A"), (b, "B")):
            c = bag[min(i, len(bag) - 1)]
            y = librosa.resample(c["y"], orig_sr=c["sr"], target_sr=16000)
            p = float(np.max(np.abs(y)) or 1)
            y = (y / p * 0.9).astype(np.float32)
            pieces += [y, gap.copy()]
            t = re.sub(r"(?<=[\u4e00-\u9fff])\s+(?=[\u4e00-\u9fff])", "", c["text"])
            texts.append(f"{tag}: {t}")
    wav = np.concatenate(pieces[:-1])
    if len(wav) > 16000 * 25:
        wav = wav[: 16000 * 22]
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_wav), wav, 16000, subtype="PCM_16")
    out_txt.write_text(
        "\n".join(
            [
                "source: Google FLEURS cmn_hans_cn (CC-BY)",
                "note: real speech, two gender groups alternating; not TTS",
                f"duration_s: {len(wav)/16000:.2f}",
                "transcript:",
                *texts,
            ]
        ),
        encoding="utf-8",
    )

    intervals = librosa.effects.split(wav, top_db=30, frame_length=1024, hop_length=256)
    merged = []
    for s, e in intervals:
        if not merged:
            merged.append([s, e])
            continue
        if s - merged[-1][1] < int(0.25 * 16000):
            merged[-1][1] = e
        else:
            merged.append([s, e])
    lines = [re.sub(r"^[AB]:\s*", "", t) for t in texts]
    segs = []
    for i in range(min(len(lines), len(merged))):
        s, e = merged[i]
        segs.append(
            {
                "start": round(s / 16000, 3),
                "end": round(e / 16000, 3),
                "text": lines[i],
                "speaker": "SPEAKER_00" if i % 2 == 0 else "SPEAKER_01",
            }
        )
    out_gold.write_text(
        json.dumps({"language": "zh", "segments": segs, "source": "FLEURS"}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {out_wav} ({len(wav)/16000:.1f}s)")
    print(f"wrote {out_txt}")
    print(f"wrote {out_gold}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fleurs", action="store_true", help="从 FLEURS 生成 real_dialog_02")
    ap.add_argument("--denoise", type=str, help="输入 wav 路径")
    ap.add_argument("-o", "--output", type=str, default="")
    args = ap.parse_args()
    if args.fleurs:
        build_fleurs(
            ROOT / "data" / "real_dialog_02.wav",
            ROOT / "data" / "real_dialog_02.txt",
            ROOT / "data" / "real_dialog_02.gold.json",
        )
        return 0
    if args.denoise:
        inp = Path(args.denoise)
        out = Path(args.output) if args.output else inp.with_name(inp.stem + "_clean.wav")
        denoise(inp, out)
        return 0
    ap.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
