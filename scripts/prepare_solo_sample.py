#!/usr/bin/env python3
"""从 real_dialog_02 截取单人独白片段，生成 solo_* 样例 (F8 solo 类)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def main():
    import numpy as np
    from src.utils.audio_utils import AudioUtils

    src_wav = PROJECT_ROOT / "data" / "real_dialog_02.wav"
    gold_path = PROJECT_ROOT / "data" / "real_dialog_02.gold.json"
    out_wav = PROJECT_ROOT / "data" / "solo_fleurs_excerpt_16k.wav"
    out_gold = PROJECT_ROOT / "data" / "solo_fleurs_excerpt_16k.gold.json"

    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    seg = gold["segments"][0]
    start, end = float(seg["start"]), float(seg["end"])

    audio, sr = AudioUtils.load_audio(str(src_wav))
    i0, i1 = int(start * sr), int(end * sr)
    clip = np.asarray(audio[i0:i1], dtype=np.float32).reshape(-1, 1)
    import soundfile as sf
    sf.write(str(out_wav), clip, sr)

    solo_gold = {
        "language": gold.get("language", "zh"),
        "source": "real_dialog_02 segment 0 (solo excerpt)",
        "segments": [
            {
                "start": 0.0,
                "end": round(end - start, 3),
                "text": seg["text"],
                "speaker": "SPEAKER_00",
            }
        ],
    }
    out_gold.write_text(json.dumps(solo_gold, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out_wav} ({end - start:.1f}s)")
    print(f"Wrote {out_gold}")


if __name__ == "__main__":
    main()
