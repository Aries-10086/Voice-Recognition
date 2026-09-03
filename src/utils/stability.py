"""运行时稳定性小工具：语种启发、脚本检测、安全默认值。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional


_EN_PATH_HINTS = (
    "talking_head",
    "willkent",
    "will_kent",
    "english",
    "_en_",
    ".en.",
    "wikidata",
    "wikipedia",
)
_ZH_PATH_HINTS = (
    "singing_",
    "ruguaiawang",
    "walkman",
    "xiaoxiong",
    "qianxia",
    "real_dialog",
    "dialog_",
    "fleurs",
)


def infer_source_lang_from_path(path: str) -> Optional[str]:
    """从文件名启发源语言；不确定则返回 None。"""
    stem = Path(path).stem.lower()
    name = Path(path).name.lower()
    blob = f"{stem} {name}"
    for h in _EN_PATH_HINTS:
        if h in blob:
            return "en"
    for h in _ZH_PATH_HINTS:
        if h in blob:
            return "zh"
    return None


def latin_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    letters = sum(1 for c in text if c.isalpha())
    if letters <= 0:
        return 0.0
    latin = sum(1 for c in text if ("a" <= c.lower() <= "z"))
    return latin / letters


def cjk_char_ratio(text: str) -> float:
    if not text:
        return 0.0
    # 基本汉字区
    cjk = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    letters = sum(1 for c in text if c.isalpha() or ("\u4e00" <= c <= "\u9fff"))
    if letters <= 0:
        return 0.0
    return cjk / letters


def asr_lang_script_mismatch(detected_lang: str, text: str) -> Optional[str]:
    """
    若 ASR 语种与文本脚本明显不符，返回建议重跑语种；否则 None。
    典型：英文口播被中文 prompt 带成 lang=zh + 乱译中文。
    """
    lang = (detected_lang or "").lower().split("-")[0]
    text = (text or "").strip()
    if len(text) < 12:
        return None
    lat = latin_char_ratio(text)
    cjk = cjk_char_ratio(text)
    # 标成中文，但几乎全是拉丁字母 → 应重跑 en
    if lang in ("zh", "yue", "chinese") and lat >= 0.75 and cjk < 0.15:
        return "en"
    # 标成英文，但几乎全是汉字 → 应重跑 zh
    if lang in ("en", "eng") and cjk >= 0.55 and lat < 0.25:
        return "zh"
    return None


_SAFE_EMOTION = "neutral"


def coerce_emotion_label(raw) -> str:
    """把任意模型输出压成单个可哈希情感字符串。"""
    if raw is None:
        return _SAFE_EMOTION
    if isinstance(raw, (list, tuple)):
        if not raw:
            return _SAFE_EMOTION
        raw = raw[0]
    if isinstance(raw, dict):
        raw = raw.get("label") or raw.get("emotion") or next(iter(raw.values()), _SAFE_EMOTION)
    s = str(raw).strip()
    # emotion2vec 偶发 "happy/xxx" 或带空格
    s = re.split(r"[/|,;]", s)[0].strip()
    return s or _SAFE_EMOTION
