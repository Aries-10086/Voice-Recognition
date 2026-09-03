#!/usr/bin/env python3
"""稳定性冒烟：不跑全链路，只验近期失败点的回归防护。"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def test_emotion_list_labels():
    from src.asr.emotion_recognition import EmotionRecognizer

    # 模拟 emotion2vec 返回 labels=list
    class Fake:
        embedding_dim = 8
        EMOTION_CATEGORIES = EmotionRecognizer.EMOTION_CATEGORIES

        def _normalize_emotion(self, x):
            return EmotionRecognizer._normalize_emotion(x)

        def _compute_valence_arousal(self, x):
            return EmotionRecognizer._compute_valence_arousal(x)

    # 直接测 VA / normalize
    assert EmotionRecognizer._normalize_emotion(["happy", "sad"]) == "happy"
    assert EmotionRecognizer._compute_valence_arousal(["angry"])[1] > 0.5
    # recognize 外层兜底：构造假 recognizer
    rec = EmotionRecognizer.__new__(EmotionRecognizer)
    rec.engine = "emotion2vec"
    rec.embedding_dim = 8
    rec.EMOTION_CATEGORIES = EmotionRecognizer.EMOTION_CATEGORIES

    def boom(*a, **k):
        raise TypeError("unhashable type: 'list'")

    rec._recognize_emotion2vec = boom
    out = EmotionRecognizer.recognize(rec, __import__("numpy").zeros(1600), 16000, False)
    assert out.emotion == "neutral"


def test_lang_mismatch_and_path_hint():
    from src.utils.stability import (
        asr_lang_script_mismatch,
        infer_source_lang_from_path,
        coerce_emotion_label,
    )

    assert infer_source_lang_from_path("data/talking_head_willkent_20s.mp4") == "en"
    assert infer_source_lang_from_path("data/singing_ruguaiawang_16k.wav") == "zh"
    from src.asr.asr_profiles import apply_profile, list_profiles
    assert "video_talking" in list_profiles({})
    eff, name = apply_profile({"device": "cpu"}, "video_talking")
    assert name == "video_talking" and eff.get("language") == "en"
    assert asr_lang_script_mismatch(
        "zh", "Hi everybody this is Will Kent from Wiki Education about Wikipedia"
    ) == "en"
    assert asr_lang_script_mismatch("en", "大家好我是来自维基教育的老师今天讲维基数据") == "zh"
    assert asr_lang_script_mismatch("en", "Hello world from Wiki Education") is None
    assert coerce_emotion_label(["happy", "sad"]) == "happy"
    assert coerce_emotion_label({"label": "Angry"}) == "Angry"


def test_ffmpeg_finder():
    from src.utils.ffmpeg_bin import find_ffmpeg

    assert find_ffmpeg(), "需要系统 ffmpeg 或 imageio-ffmpeg"


def main():
    test_emotion_list_labels()
    test_lang_mismatch_and_path_hint()
    test_ffmpeg_finder()
    print("OK smoke_stability")


if __name__ == "__main__":
    main()
