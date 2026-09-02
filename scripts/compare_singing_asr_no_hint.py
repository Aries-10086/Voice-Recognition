#!/usr/bin/env python3
"""
无歌词 hint 清唱 ASR 对比：Whisper vs FunASR，可选通用 LLM 后处理。

用法:
  python scripts/compare_singing_asr_no_hint.py data/singing_ruguaiawang_16k.wav
  python scripts/compare_singing_asr_no_hint.py data/singing_ruguaiawang_16k.wav --generic-llm
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.asr.asr_profiles import apply_profile
from src.asr.singing_corrector import refine_singing_transcript
from src.asr.whisper_asr import WhisperASR

# 复用评分
from scripts.score_singing_asr import score as score_asr


def _singing_asr_cfg(device: str = "cpu") -> dict:
    cfg = yaml.safe_load((ROOT / "config" / "default.yaml").read_text(encoding="utf-8"))
    asr_cfg = cfg.get("asr", {})
    asr_cfg["device"] = device
    eff, _ = apply_profile(asr_cfg, "singing")
    eff["device"] = device
    # 真 ASR：不注入歌名/歌词/歌曲纠错
    eff["initial_prompt"] = (
        eff.get("initial_prompt") or "这是中文清唱，请按歌词逐字转写，不要写成对话摘要。"
    )
    eff["_no_lyrics_hint"] = True
    eff.pop("corrections", None)
    return eff, cfg.get("translation", {})


def run_whisper(audio, device: str) -> tuple[str, float, list]:
    eff, _ = _singing_asr_cfg(device)
    asr = WhisperASR(eff)
    t0 = time.time()
    r = asr.transcribe(audio, 16000, asr_config=eff)
    # 真 ASR：通用同音修正（不用歌词）
    text = apply_refine(r.text or "", r.segments, generic_llm=False, trans_cfg={})
    return text, time.time() - t0, r.segments


def run_funasr(audio, device: str) -> tuple[str, float, list]:
    from pathlib import Path as P

    local = P.home() / ".cache/modelscope/models/iic--SenseVoiceSmall/snapshots/master"
    eff, _ = _singing_asr_cfg(device)
    cfg = dict(eff)
    cfg["engine"] = "funasr"
    # 清唱对比：不要拉 VAD/标点大模型（慢且对清唱帮助有限）
    cfg["vad_model"] = None
    cfg["punc_model"] = None
    if local.exists():
        cfg["funasr_model"] = str(local)
    asr = WhisperASR(cfg)
    t0 = time.time()
    r = asr.transcribe(audio, 16000, language="zh")
    text = apply_refine(r.text or "", r.segments or [], generic_llm=False, trans_cfg={})
    return text, time.time() - t0, r.segments or []


def apply_refine(
    text: str,
    segments: list,
    *,
    generic_llm: bool,
    trans_cfg: dict,
) -> str:
    out, _ = refine_singing_transcript(
        text,
        segments,
        use_lyrics_hint=False,
        use_song_corrections=False,
        generic_llm=generic_llm,
        llm_config=trans_cfg,
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="无歌词 hint 清唱 ASR 引擎对比")
    ap.add_argument("wav", type=str)
    ap.add_argument("--gold", default="", help="gold.json（评分用）")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--generic-llm", action="store_true", help="对 raw 结果跑通用 LLM 修错")
    args = ap.parse_args()

    wav = Path(args.wav)
    if not wav.exists():
        print(f"missing: {wav}")
        return 1

    gold_path = Path(args.gold) if args.gold else wav.with_suffix(".gold.json")
    audio, sr = sf.read(str(wav), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    dur = len(audio) / sr
    _, trans_cfg = _singing_asr_cfg(args.device)

    print(f"样例: {wav.name}  时长: {dur:.1f}s  模式: 无歌词 hint")

    rows = []
    jobs = [
        ("whisper_medium", "whisper", False),
    ]
    if args.generic_llm:
        jobs.append(("whisper_medium+generic_llm", "whisper", True))
    jobs.append(("funasr_sensevoice", "funasr", False))
    if args.generic_llm:
        jobs.append(("funasr+generic_llm", "funasr", True))

    for label, engine, use_llm in jobs:
        try:
            if engine == "whisper":
                text, dt, segs = run_whisper(audio, args.device)
            else:
                text, dt, segs = run_funasr(audio, args.device)
            if use_llm:
                text = apply_refine(text, segs, generic_llm=True, trans_cfg=trans_cfg)
            rep = {"label": label, "seconds": round(dt, 1), "text": text}
            if gold_path.exists():
                rep.update(score_asr(gold_path, text))
            rows.append(rep)
            hit = rep.get("keyword_hit_rate", "?")
            print(f"\n=== {label} ({dt:.1f}s) 关键词率={hit} ===\n{text[:500]}\n")
        except Exception as e:
            rows.append({"label": label, "error": str(e)})
            print(f"\n=== {label} FAIL ===\n{e}\n")

    out_dir = ROOT / "outputs" / "singing_asr_no_hint_compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_json = out_dir / "report.json"
    report_md = out_dir / "report.md"
    report_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 无歌词 hint 清唱 ASR 对比",
        "",
        f"- 样例: `{wav}`",
        f"- 时长: {dur:.1f}s",
        f"- 说明: 不用歌名/歌词 prompt、不用歌曲纠错表",
        "",
        "| 方案 | 耗时(s) | 关键词命中 | 可用 | 预览 |",
        "|---|---:|---:|---|---|",
    ]
    for r in rows:
        if "error" in r:
            lines.append(f"| {r['label']} | — | — | ❌ | {r['error'][:60]} |")
            continue
        prev = (r.get("text") or "").replace("\n", " ")[:80]
        hit = r.get("keyword_hit_rate", "")
        hit_s = f"{hit:.0%}" if isinstance(hit, (int, float)) else "—"
        ok = "✅" if r.get("usable_candidate") else "❌"
        lines.append(
            f"| {r['label']} | {r.get('seconds', '—')} | {hit_s} | {ok} | {prev} |"
        )
    lines += [
        "",
        "## 结论指引",
        "",
        "- **Whisper raw**：基线，不借助歌词",
        "- **FunASR**：中文说话/朗读强，清唱需实测",
        "- **+generic_llm**：需 `--translate-engine local` 或 OpenAI 兼容 API",
        "",
        "真 ASR KPI 请用: `run_pipeline.py --singing --no-gold --no-lyrics-hint`",
    ]
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"report: {report_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
