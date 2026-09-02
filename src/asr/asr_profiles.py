"""
ASR prompt profile 与幻觉检测 (03 · S1/G1/G2).
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# 内置 profile；config 可覆盖/追加
DEFAULT_PROFILES: Dict[str, Dict[str, Any]] = {
    "neutral": {
        "initial_prompt": "这是一段中文口语对话，请准确转写说话内容。",
        "condition_on_previous_text": False,
    },
    "singing": {
        "initial_prompt": "这是中文清唱，请按歌词逐字转写，不要写成对话摘要。",
        "condition_on_previous_text": False,
        "model_name": "medium",
        "beam_size": 5,
        "split_silence_seconds": 0.55,
        "max_segment_seconds": 6.0,
        "min_segment_seconds": 0.8,
        "merge_gap_seconds": 0.80,
        "merge_short_chars": 18,
        "min_silence_ms": 400,
        "crumb_max_chars": 6,
    },
    "fleurs_dialog": {
        "initial_prompt": "这是一段中文朗读，请准确转写说话内容。",
        "condition_on_previous_text": False,
    },
}

DEFAULT_BANNED_PHRASES = [
    "讨论科技、气候、河流与历史",
    "讨论科技、气候、河流",
    "科技、气候、河流与历史",
]


def list_profiles(asr_config: Dict) -> List[str]:
    custom = (asr_config.get("prompt_profiles") or {}).keys()
    return sorted(set(DEFAULT_PROFILES.keys()) | set(custom))


def resolve_profile_name(asr_config: Dict, override: Optional[str] = None) -> str:
    name = (override or asr_config.get("prompt_profile") or "neutral").strip().lower()
    profiles = merged_profiles(asr_config)
    if name not in profiles:
        return "neutral"
    return name


def merged_profiles(asr_config: Dict) -> Dict[str, Dict[str, Any]]:
    out = {k: dict(v) for k, v in DEFAULT_PROFILES.items()}
    for k, v in (asr_config.get("prompt_profiles") or {}).items():
        if isinstance(v, dict):
            out.setdefault(k, {})
            out[k].update(v)
    return out


def apply_profile(asr_config: Dict, profile_name: Optional[str] = None) -> Tuple[Dict, str]:
    """
    将 profile 字段合并进 asr 配置副本，返回 (effective_config, profile_name)。
    不修改 model 实例，仅返回用于 transcribe / reload 的配置。
    """
    name = resolve_profile_name(asr_config, profile_name)
    profiles = merged_profiles(asr_config)
    effective = dict(asr_config)
    patch = profiles.get(name) or {}
    for key, val in patch.items():
        if val is not None:
            effective[key] = val
    effective["prompt_profile_active"] = name
    return effective, name


def _norm(s: str) -> str:
    return re.sub(r"\s+", "", (s or "")).lower()


def prompt_overlap_score(text: str, prompt: Optional[str]) -> float:
    """转写与 prompt 的字符重叠比 (0~1)。"""
    if not text or not prompt:
        return 0.0
    t, p = _norm(text), _norm(prompt)
    if not t or not p:
        return 0.0
    # 连续子串最长公共片段占比
    best = 0
    for i in range(len(p)):
        for j in range(i + 2, len(p) + 1):
            frag = p[i:j]
            if frag in t:
                best = max(best, len(frag))
    return best / max(len(t), 1)


def detect_hallucination(
    text: str,
    prompt: Optional[str] = None,
    banned_phrases: Optional[List[str]] = None,
    overlap_threshold: float = 0.45,
) -> Tuple[bool, float, str]:
    """
    返回 (is_hallucination, score, reason).
  score 取 banned 命中与 prompt 重叠的较大值。
    """
    banned = banned_phrases if banned_phrases is not None else DEFAULT_BANNED_PHRASES
    norm = _norm(text)
    for phrase in banned:
        if phrase and _norm(phrase) in norm:
            return True, 1.0, f"banned:{phrase[:24]}"
    ov = prompt_overlap_score(text, prompt)
    if prompt and ov >= overlap_threshold:
        return True, ov, "prompt_overlap"
    # 短文本高频重复同一 prompt 片段
    if prompt and len(norm) >= 12:
        pnorm = _norm(prompt)
        for n in (8, 6, 4):
            if len(pnorm) < n:
                continue
            for i in range(0, len(pnorm) - n + 1):
                frag = pnorm[i : i + n]
                if frag and norm.count(frag) >= 2 and len(frag) / max(len(norm), 1) > 0.25:
                    return True, 0.9, f"repeat:{frag}"
    return False, ov, "ok"
