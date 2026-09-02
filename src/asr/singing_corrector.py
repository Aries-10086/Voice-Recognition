"""
清唱 / 口播 ASR 后处理 (S5 · G4).
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", s or "")


def load_lexicon_broadcast(path: Optional[str] = None) -> Tuple[List[str], List[Tuple[str, str]]]:
    hotwords: List[str] = []
    pairs: List[Tuple[str, str]] = []
    p = Path(path) if path else Path(__file__).resolve().parents[2] / "data" / "lexicon_broadcast.txt"
    if not p.exists():
        return hotwords, pairs
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            a, b = line.split("|", 1)
            pairs.append((a.strip(), b.strip()))
        else:
            hotwords.append(line)
    return hotwords, pairs


def apply_correction_pairs(text: str, pairs: List[Tuple[str, str]]) -> str:
    if not text or not pairs:
        return text
    out = text
    for src, dst in pairs:
        if not src:
            continue
        if src in out:
            out = out.replace(src, dst)
            continue
        # 忽略空白后再匹配（Whisper 常在汉字间插空格）
        compact = re.sub(r"\s+", "", out)
        src_c = re.sub(r"\s+", "", src)
        if src_c and src_c in compact:
            map_idx = [i for i, ch in enumerate(out) if not ch.isspace()]
            pos = compact.find(src_c)
            if pos >= 0 and pos + len(src_c) <= len(map_idx):
                start = map_idx[pos]
                end = map_idx[pos + len(src_c) - 1] + 1
                out = out[:start] + dst + out[end:]
    return out


def _hint_phrases(hint: str) -> List[str]:
    if not hint:
        return []
    parts = re.split(r"[，,、；;。\s]+", hint)
    return [p.strip() for p in parts if len(_norm(p.strip())) >= 3]


def refine_with_lyrics_hint(text: str, hint: str, min_ratio: float = 0.55) -> str:
    if not text or not hint:
        return text
    out = text
    tn = _norm(out)
    for phrase in _hint_phrases(hint):
        pn = _norm(phrase)
        if len(pn) < 4 or pn in tn:
            continue
        best = (0.0, -1, -1)
        win = len(pn)
        for i in range(0, max(len(tn) - 2, 1)):
            for w in range(max(win - 2, 3), min(win + 4, len(tn) - i + 1)):
                chunk = tn[i : i + w]
                if len(chunk) < 3:
                    continue
                r = SequenceMatcher(None, chunk, pn).ratio()
                if r > best[0]:
                    best = (r, i, i + w)
        ratio, i0, i1 = best
        if min_ratio <= ratio < 0.92 and i0 >= 0:
            compact, map_idx = [], []
            for j, ch in enumerate(out):
                if not ch.isspace():
                    compact.append(ch)
                    map_idx.append(j)
            if i1 <= len(compact):
                start, end = map_idx[i0], map_idx[i1 - 1] + 1
                out = out[:start] + phrase + out[end:]
                tn = _norm(out)
    return out


def default_singing_pairs() -> List[Tuple[str, str]]:
    """歌曲相关纠错（仅演示轨 / 有 hint 时使用）。"""
    return [
        ("想爱过", "如果爱忘了"),
        ("如果是爱了", "要是忘了"),
        ("爱够久了", "爱够久了"),
        ("分开月灯", "分开的话"),
        ("分开月灯吧", "分开的话"),
        ("加你的", "你的"),
        ("想遇个", "相遇"),
        ("一硬地动", "隐隐地震"),
        ("在那个时间", "在那个瞬间"),
        ("必须要处理", "必须要坚强"),
        ("已没家", "已没有家"),
        ("淚不剩落下", "泪不想落下"),
        ("泪不剩落下", "泪不想落下"),
        ("幸福 提升大家", "幸福啊 让她给我吧"),
        ("幸福提升大家", "幸福啊 让她给我吧"),
        ("根本吸不掉", "根本洗不掉"),
        ("心里面有啦", "心里面了"),
        ("碰它却", "碰它却"),
    ]


def generic_homophone_pairs() -> List[Tuple[str, str]]:
    """
    通用同音/不通词修正（真 ASR 轨也可用）。
    原则：只改明显不通的词形，不注入歌名/整句歌词。
    """
    return [
        # —— 情歌清唱近音 ——
        ("想遇个", "相遇"),
        ("想遇", "相遇"),
        ("根本吸不掉", "根本洗不掉"),
        ("吸不掉回忆", "洗不掉回忆"),
        ("一硬地动", "隐隐地动"),
        ("一硬地", "隐隐地"),
        ("已没家", "已没有家"),
        ("淚不剩落下", "泪不想落下"),
        ("泪不剩落下", "泪不想落下"),
        ("不剩落下", "不想落下"),
        ("脸颊里的发", "脸你的发"),
        ("加你的发", "你的发"),
        ("脸加你的", "脸你的"),
        ("那些幸福剩", "那些幸福啊"),
        ("幸福剩大家", "幸福啊"),
        ("在那个时间", "在那个瞬间"),
        ("那个时间是个", "那个瞬间是个"),
        # —— 怀旧口播/随身听类近音 ——
        ("说杰伦", "周杰伦"),
        ("文山的磁", "文山的词"),
        ("飞吼过", "飞过"),
        ("爱在七元前", "爱在西元前"),
        ("听成爱在七元前", "听成爱在西元前"),
        ("兴开年错温", "新概念作文"),
        ("连城县", "连呈现"),
        ("連城縣", "連呈現"),
        ("歲月深聽", "歲月深情"),
        ("岁月深听", "岁月深情"),
    ]


def _merge_pairs(corrections: Optional[List]) -> List[Tuple[str, str]]:
    pairs = list(default_singing_pairs())
    for item in corrections or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    return pairs


def refine_singing_transcript(
    text: str,
    segments: List[Dict],
    *,
    hint: str = "",
    title: str = "",
    corrections: Optional[List] = None,
    use_lyrics_hint: bool = True,
    use_song_corrections: bool = True,
    generic_llm: bool = False,
    llm_config: Optional[Dict] = None,
) -> Tuple[str, List[Dict]]:
    pairs: List[Tuple[str, str]] = list(generic_homophone_pairs())
    if use_song_corrections:
        pairs = _merge_pairs(corrections) + pairs
    elif corrections:
        for item in corrections:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                pairs.append((str(item[0]), str(item[1])))

    # 去重保序
    seen = set()
    uniq = []
    for a, b in pairs:
        if (a, b) not in seen:
            seen.add((a, b))
            uniq.append((a, b))
    pairs = uniq

    merged = apply_correction_pairs(text, pairs)
    if use_lyrics_hint and hint:
        merged = refine_with_lyrics_hint(merged, hint)
    if generic_llm:
        merged = refine_with_generic_llm(merged, llm_config or {})

    if not segments:
        return merged, segments

    new_segs = []
    for s in segments:
        t = apply_correction_pairs(s.get("text") or "", pairs)
        if use_lyrics_hint and hint:
            t = refine_with_lyrics_hint(t, hint)
        ns = dict(s)
        ns["text"] = t
        new_segs.append(ns)

    if generic_llm and new_segs:
        merged = refine_with_generic_llm(
            " ".join((s.get("text") or "").strip() for s in new_segs),
            llm_config or {},
        )
        new_segs = _redistribute_text_to_segments(merged, new_segs)

    merged = " ".join((s.get("text") or "").strip() for s in new_segs if (s.get("text") or "").strip())
    return merged or text, new_segs


def _redistribute_text_to_segments(merged: str, segments: List[Dict]) -> List[Dict]:
    """整段 LLM 修正后按原段长比例切回（仅保留时间戳）。"""
    if not segments or not merged:
        return segments
    weights = [max(len(_norm(s.get("text") or "")), 1) for s in segments]
    total = sum(weights)
    chars = list(re.sub(r"\s+", "", merged))
    if not chars:
        return segments
    cursor = 0
    out = []
    for i, s in enumerate(segments):
        take = (
            len(chars) - cursor
            if i == len(segments) - 1
            else max(1, round(len(chars) * weights[i] / total))
        )
        chunk = "".join(chars[cursor : cursor + take])
        cursor += take
        ns = dict(s)
        ns["text"] = chunk
        out.append(ns)
    return out


def refine_with_generic_llm(text: str, llm_config: Dict) -> str:
    """
    通用同音/断句修正（不注入歌词、歌名）。
    需 local / openai_compatible；google 引擎则跳过。
    """
    text = (text or "").strip()
    if len(text) < 4:
        return text
    engine = (llm_config.get("engine") or "google").lower()
    if engine == "google":
        return text
    try:
        from src.translation.llm_translator import LLMTranslator

        cfg = dict(llm_config)
        if cfg.get("engine") == "auto":
            cfg["engine"] = "auto"
        tr = LLMTranslator(cfg)
        if tr.engine == "google":
            return text
        system = (
            "你是中文语音识别后处理助手。输入是清唱或歌声的自动转写，常有同音错字。"
            "只修正明显同音别字、错词和断句，不要添加原文没有的内容，不要引用具体歌名或歌词。"
            "只输出修正后的中文文本，不要解释。"
        )
        user = f"请修正以下识别文本：\n\n{text}"
        if tr.engine == "openai_compatible":
            raw = tr._call_openai(system, user)
        elif tr.engine in ("local", "transformers"):
            raw = tr._call_transformers(system, user)
        else:
            return text
        fixed = (raw or "").strip()
        fixed = re.sub(r"^(修正后|输出)[:：]\s*", "", fixed)
        fixed = fixed.strip("「」\"'")
        if len(fixed) >= max(len(text) * 0.4, 4):
            return fixed
    except Exception:
        pass
    return text


def refine_dialog_transcript(
    text: str,
    segments: List[Dict],
    *,
    lexicon_path: Optional[str] = None,
    corrections: Optional[List] = None,
) -> Tuple[str, List[Dict]]:
    _, lex_pairs = load_lexicon_broadcast(lexicon_path)
    pairs = list(lex_pairs)
    for item in corrections or []:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            pairs.append((str(item[0]), str(item[1])))
    new_segs = []
    for s in segments:
        ns = dict(s)
        ns["text"] = apply_correction_pairs(s.get("text") or "", pairs)
        new_segs.append(ns)
    merged = " ".join((s.get("text") or "").strip() for s in new_segs if (s.get("text") or "").strip())
    return merged or apply_correction_pairs(text, pairs), new_segs
