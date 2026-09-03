#!/usr/bin/env python3
"""
视频口播 ASR 评分（英文词级关键词 + 简易 WER）。

用法:
  # 真 ASR（忽略 gold 旁路）
  python scripts/score_video_asr.py \
    --gold data/talking_head_willkent_20s.gold.json \
    --video data/talking_head_willkent_20s.mp4 \
    --profile video_talking --no-gold-asr --device cpu
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
from typing import Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _words(s: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+", (s or "").lower())


def keyword_hit_rate(hyp: str, keywords: List[str]) -> Tuple[float, List[str], List[str]]:
    h = (hyp or "").lower()
    hit, miss = [], []
    for kw in keywords:
        k = kw.lower()
        if k in h or all(p in h for p in k.split()):
            hit.append(kw)
        else:
            miss.append(kw)
    rate = len(hit) / max(len(keywords), 1)
    return rate, hit, miss


def wer(ref: str, hyp: str) -> float:
    r, h = _words(ref), _words(hyp)
    if not r:
        return 0.0 if not h else 1.0
    # classic DP
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[-1][-1] / len(r)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gold", required=True)
    ap.add_argument("--video", default="")
    ap.add_argument("--audio", default="")
    ap.add_argument("--profile", default="video_talking")
    ap.add_argument("--no-gold-asr", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    with open(args.gold, encoding="utf-8") as f:
        gold = json.load(f)
    keywords = gold.get("keywords") or []
    ref_text = " ".join(s.get("text") or "" for s in gold.get("segments") or [])

    import yaml
    from src.asr.whisper_asr import WhisperASR
    from src.asr.asr_profiles import apply_profile
    from src.utils.audio_utils import AudioUtils

    cfg_path = os.path.join(ROOT, "config", "default.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        full = yaml.safe_load(f)
    asr_cfg = dict(full.get("asr") or {})
    asr_cfg["device"] = args.device
    effective, _ = apply_profile(asr_cfg, args.profile)

    # 英文视频词表
    from src.asr.singing_corrector import load_lexicon_broadcast, refine_dialog_transcript

    lex = os.path.join(ROOT, "data", "lexicon_video_en.txt")
    hot, pairs = load_lexicon_broadcast(lex)
    if hot and args.profile == "video_talking":
        prompt = (effective.get("initial_prompt") or "").strip()
        effective["initial_prompt"] = (
            prompt + " Names that may appear: " + ", ".join(hot[:16]) + "."
        ).strip()
    if pairs:
        effective["corrections"] = list(effective.get("corrections") or []) + [
            [a, b] for a, b in pairs
        ]
    effective["language"] = gold.get("language") or effective.get("language") or "en"

    wav = args.audio
    tmp = None
    if args.video:
        from src.utils.video_utils import VideoUtils

        audio, sr = VideoUtils.extract_audio_from_video(args.video, target_sr=16000)
        fd, tmp = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        AudioUtils.save_audio(audio, tmp, sr)
        wav = tmp
    if not wav or not os.path.isfile(wav):
        raise SystemExit("需要 --video 或 --audio")

    asr = WhisperASR(effective)
    audio, sr = AudioUtils.load_audio(wav, target_sr=16000)
    result = asr.transcribe(audio, sr, asr_config=effective)
    text, segs = refine_dialog_transcript(
        result.text,
        result.segments,
        lexicon_path=lex,
        corrections=effective.get("corrections"),
    )
    if tmp:
        try:
            os.unlink(tmp)
        except OSError:
            pass

    kw_rate, hit, miss = keyword_hit_rate(text, keywords)
    w = wer(ref_text, text)
    report: Dict = {
        "profile": args.profile,
        "language": result.language,
        "n_segments": len(segs),
        "keyword_rate": round(kw_rate, 4),
        "keyword_hit": hit,
        "keyword_miss": miss,
        "wer": round(w, 4),
        "hyp": text,
        "ref": ref_text,
        "pass_kw_0_8": kw_rate >= 0.8,
        "pass_wer_0_25": w <= 0.25,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json_out:
        out = args.json_out if os.path.isabs(args.json_out) else os.path.join(ROOT, args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print("wrote", out)
    # 退出码：关键词未达 80% 则失败（便于 CI）
    sys.exit(0 if kw_rate >= 0.8 else 1)


if __name__ == "__main__":
    main()
