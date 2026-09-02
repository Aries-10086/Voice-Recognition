#!/usr/bin/env python3
"""
C2 · 清唱 ASR 评分：对比 run 输出或现场 Whisper 与 gold 关键词/句意。

用法:
  python scripts/score_singing_asr.py --gold data/singing_ruguaiawang_16k.gold.json \\
      --run outputs/run_20260902_101218

  python scripts/score_singing_asr.py --gold data/singing_ruguaiawang_16k.gold.json \\
      --audio data/singing_ruguaiawang_16k.wav --no-gold-asr
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def _norm_zh(s: str) -> str:
    s = re.sub(r"\s+", "", s or "")
    s = re.sub(r"[，。！？、；：""''…,.!?]", "", s)
    try:
        from opencc import OpenCC
        s = OpenCC("t2s").convert(s)
    except Exception:
        # 轻量兜底：常见繁简
        for a, b in (
            ("鋼琴", "钢琴"), ("練習", "练习"), ("歲月", "岁月"),
            ("磁帶", "磁带"), ("時代", "时代"), ("詞", "词"),
            ("憶", "忆"), ("聽", "听"), ("見", "见"), ("寫", "写"),
            ("記", "记"), ("電", "电"), ("競", "竞"), ("過", "过"),
        ):
            s = s.replace(a, b)
    return s.lower()


def _collect_keywords(gold: dict) -> list[str]:
    keys: list[str] = []
    for seg in gold.get("segments", []):
        for k in seg.get("keywords", []) or []:
            k = _norm_zh(k)
            if k and k not in keys:
                keys.append(k)
    hint = gold.get("lyrics_hint") or ""
    for chunk in re.split(r"[，,、\s]+", hint):
        c = _norm_zh(chunk)
        if len(c) >= 2 and c not in keys:
            keys.append(c)
    return keys


def _load_asr_text_from_run(run_dir: Path) -> str:
    seg_path = run_dir / "segments.json"
    if not seg_path.exists():
        raise FileNotFoundError(f"缺少 {seg_path}")
    data = json.loads(seg_path.read_text(encoding="utf-8"))
    parts = [(s.get("source") or "").strip() for s in data.get("segments", [])]
    return " ".join(p for p in parts if p)


def _run_whisper(
    audio: Path,
    device: str = "cpu",
    *,
    no_lyrics_hint: bool = False,
    generic_llm: bool = False,
) -> str:
    import os
    import yaml
    from src.asr.asr_profiles import apply_profile
    from src.asr.singing_corrector import refine_singing_transcript
    from src.asr.whisper_asr import WhisperASR
    from src.utils.audio_utils import AudioUtils

    if not os.environ.get("HF_HOME"):
        os.environ["HF_HOME"] = str(PROJECT_ROOT / "models" / "huggingface")

    cfg_path = PROJECT_ROOT / "config" / "default.yaml"
    config = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    config.setdefault("asr", {})["device"] = device
    asr_cfg, _ = apply_profile(config.get("asr", {}), "singing")
    asr_cfg["device"] = device
    if no_lyrics_hint:
        asr_cfg["initial_prompt"] = (
            asr_cfg.get("initial_prompt")
            or "这是中文清唱，请按歌词逐字转写，不要写成对话摘要。"
        )
        asr_cfg.pop("corrections", None)

    gold_path = audio.with_suffix(".gold.json")
    hint, title = "", ""
    if gold_path.exists() and not no_lyrics_hint:
        side = json.loads(gold_path.read_text(encoding="utf-8"))
        hint = side.get("lyrics_hint") or ""
        title = side.get("title") or ""

    audio_arr, sr = AudioUtils.load_audio(str(audio))
    asr = WhisperASR(asr_cfg)
    asr.reload_if_needed(asr_cfg)
    result = asr.transcribe(audio_arr, sr, asr_config=asr_cfg)
    text, _ = refine_singing_transcript(
        result.text or "",
        result.segments,
        hint=hint,
        title=title,
        corrections=asr_cfg.get("corrections"),
        use_lyrics_hint=not no_lyrics_hint,
        use_song_corrections=not no_lyrics_hint,
        generic_llm=generic_llm,
        llm_config=config.get("translation", {}),
    )
    return text


def score(gold_path: Path, asr_text: str) -> dict:
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    keywords = _collect_keywords(gold)
    asr_norm = _norm_zh(asr_text)
    gold_norm = _norm_zh(
        "".join((s.get("text") or "") for s in gold.get("segments", []))
    )

    hits = [k for k in keywords if k in asr_norm]
    misses = [k for k in keywords if k not in asr_norm]

    # 字符 bigram 重叠（粗句意）
    def bigrams(s: str):
        return {s[i : i + 2] for i in range(max(len(s) - 1, 0))}

    gb, ab = bigrams(gold_norm), bigrams(asr_norm)
    overlap = len(gb & ab) / max(len(gb), 1)

    hit_rate = len(hits) / max(len(keywords), 1)
    # 03 真 ASR 必达线：关键词 ≥70% 且 bigram≥0.35
    pass_hit = 0.70
    pass_bigram = 0.35
    usable = hit_rate >= pass_hit and overlap >= pass_bigram

    return {
        "gold": str(gold_path),
        "keywords_total": len(keywords),
        "keywords_hit": len(hits),
        "keyword_hit_rate": round(hit_rate, 3),
        "bigram_overlap": round(overlap, 3),
        "pass_line_hit_rate": pass_hit,
        "usable_candidate": usable,
        "hits": hits,
        "misses": misses,
        "asr_preview": asr_text[:240],
    }


def main():
    p = argparse.ArgumentParser(description="清唱 ASR 评分 (C2)")
    p.add_argument("--gold", required=True, help="gold.json 路径")
    p.add_argument("--run", help="已有 run 目录 (读 segments.json)")
    p.add_argument("--audio", help="音频路径 (配合 --no-gold-asr)")
    p.add_argument("--no-gold-asr", action="store_true", help="现场跑 Whisper singing profile")
    p.add_argument("--device", default="cpu")
    p.add_argument("--no-lyrics-hint", action="store_true", help="评分时不使用 lyrics_hint 后处理")
    p.add_argument("--generic-llm", action="store_true", help="Whisper 后通用 LLM 修错")
    p.add_argument("-o", "--output", help="写入 JSON 报告")
    args = p.parse_args()

    gold_path = Path(args.gold)
    if args.run:
        asr_text = _load_asr_text_from_run(Path(args.run))
    elif args.audio and args.no_gold_asr:
        asr_text = _run_whisper(
            Path(args.audio),
            args.device,
            no_lyrics_hint=args.no_lyrics_hint,
            generic_llm=args.generic_llm,
        )
    else:
        p.error("请指定 --run 或 (--audio + --no-gold-asr)")

    report = score(gold_path, asr_text)
    lines = [
        "# 清唱 ASR 评分",
        "",
        f"- 关键词命中: {report['keywords_hit']}/{report['keywords_total']} "
        f"({report['keyword_hit_rate']:.1%})",
        f"- Bigram 重叠: {report['bigram_overlap']:.3f}",
        f"- 必达线: 关键词≥{report.get('pass_line_hit_rate', 0.7):.0%} 且 bigram≥0.35",
        f"- 可用候选: {'✅ 达标' if report['usable_candidate'] else '❌ 未达 70% 必达线'}",
        "",
        "## 命中",
        ", ".join(report["hits"]) or "(无)",
        "",
        "## 未命中",
        ", ".join(report["misses"]) or "(无)",
        "",
        "## ASR 预览",
        report["asr_preview"],
    ]
    text = "\n".join(lines)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n已写入 {out}")


if __name__ == "__main__":
    main()
