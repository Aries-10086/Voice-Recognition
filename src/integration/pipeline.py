"""
跨语种语音翻译与原声复刻 - 完整Pipeline

系统流程:
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ 语音输入  │───▶│ ASR+情感识别  │───▶│ LLM翻译优化  │───▶│ 时间轴对齐    │
└──────────┘    └──────────────┘    └────────────┘    └──────────────┘
                                                              │
                                                              ▼
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌──────────────┐
│ 最终输出  │◀───│ 后处理与评估  │◀───│ 语音合成    │◀───│ 声纹复刻      │
└──────────┘    └──────────────┘    └────────────┘    └──────────────┘
"""

import os
import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ============================================================
# 将模型缓存重定向到项目本地 models/ 目录
# ============================================================
import platform
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_win_models = Path("D:/CodingPackage/models")
if platform.system() == "Windows" and _win_models.exists():
    _MODEL_ROOT = _win_models
else:
    _MODEL_ROOT = _PROJECT_ROOT / "models"
_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_MODEL_ROOT / "huggingface")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
    os.environ["HF_HUB_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
    os.environ["TORCH_HOME"] = str(_MODEL_ROOT / "torch")
    os.environ["XDG_CACHE_HOME"] = str(_MODEL_ROOT / ".cache")
    os.environ["TRANSFORMERS_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
    os.environ["CTRANSLATE2_MODELS"] = str(_MODEL_ROOT / "ctranslate2")
_nltk = _PROJECT_ROOT / "models" / "nltk_data"
if not _nltk.exists():
    _nltk = _MODEL_ROOT / "nltk_data"
os.environ["NLTK_DATA"] = str(_nltk)
try:
    import nltk
    nltk.data.path.insert(0, str(_nltk))
except ImportError:
    pass
# ============================================================

import numpy as np
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.asr.whisper_asr import WhisperASR, ASRResult
from src.asr.emotion_recognition import EmotionRecognizer, EmotionResult
from src.asr.speaker_diarization import SpeakerDiarizer
from src.translation.llm_translator import LLMTranslator, TranslationResult
from src.alignment.articulatory_analyzer import AOCPNet, AOCPResult
from src.alignment.phoneme_aligner import PhonemeAligner, PhonemeAlignment
from src.alignment.timeline_generator import TimelineGenerator, TimelineResult
from src.synthesis.voice_cloner import VoiceCloner, ClonedSpeech
from src.utils.audio_utils import AudioUtils


@dataclass
class PipelineResult:
    """Pipeline完整结果"""
    # 输入信息
    source_audio: np.ndarray
    source_sr: int
    source_duration: float

    # ASR结果
    asr_result: Optional[ASRResult] = None
    # 情感结果
    emotion_result: Optional[EmotionResult] = None
    # 语种识别
    detected_language: str = "unknown"

    # 翻译结果
    translation_result: Optional[TranslationResult] = None

    # 发音分析
    aocp_result: Optional[AOCPResult] = None
    # 音素对齐
    phoneme_alignment: Optional[PhonemeAlignment] = None
    # 时间轴
    timeline_result: Optional[TimelineResult] = None

    # 合成结果
    cloned_speech: Optional[ClonedSpeech] = None

    # 元数据
    processing_time: float = 0.0
    status: str = "pending"
    error_message: str = ""
    diarization_meta: Dict = field(default_factory=dict)
    clone_meta: Dict = field(default_factory=dict)


class CrossLingualPipeline:
    """
    跨语种语音翻译与原声复刻 Pipeline

    【核心集成模块 - 可用于专利申请的完整系统】

    本Pipeline串联了以下创新模块:
    1. 多语种ASR + 情感识别
    2. 情感感知的LLM翻译优化
    3. AOCP-Net发音开闭感知 + 时间轴生成
    4. 声纹复刻 + 情感保持的语音合成

    所有模块可独立运行,也可联合优化。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cuda")

        logger.info("=" * 60)
        logger.info("CrossLingual Voice Clone System v1.0")
        logger.info("=" * 60)

        # 初始化各模块
        self.asr = None
        self.emotion_recognizer = None
        self.language_identifier = None
        self.diarizer = None
        self.translator = None
        self.aocp_net = None
        self.phoneme_aligner = None
        self.timeline_generator = None
        self.voice_cloner = None

        self._init_modules()

    def _init_modules(self):
        """按需初始化各模块"""
        steps = self.config.get("pipeline", {}).get("steps", [])

        if "asr" in steps:
            self.asr = WhisperASR(self.config["asr"])
            # MMS-LID 不再加载 (whisper已内置语种检测, 省 3.7GB)
            self.language_identifier = None
            # 说话人分离 (离线聚类兜底)
            self.diarizer = SpeakerDiarizer(
                self.config.get("pipeline", {}).get("diarization", {})
            )

        if "emotion_recognition" in steps:
            try:
                self.emotion_recognizer = EmotionRecognizer(self.config["emotion"])
            except Exception as e:
                logger.warning(f"⚠️ 情感识别加载失败(跳过): {e}")
                self.emotion_recognizer = None

        if "translation" in steps:
            self.translator = LLMTranslator(self.config["translation"])

        if "aocp_analysis" in steps or "phoneme_alignment" in steps:
            self.aocp_net = AOCPNet(self.config["alignment"])
            self.phoneme_aligner = PhonemeAligner(self.config["alignment"])

        if "timeline_generation" in steps:
            self.timeline_generator = TimelineGenerator(self.config["alignment"])

        if "voice_synthesis" in steps:
            self.voice_cloner = VoiceCloner(self.config["synthesis"])

        logger.info("All modules initialized")

    def run(
        self,
        audio_path: str,
        target_lang: str = "en",
        reference_audio_path: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> PipelineResult:
        """
        运行完整Pipeline

        Args:
            audio_path: 源音频文件路径
            target_lang: 目标语言代码 (en/zh/ja/ko...)
            reference_audio_path: 参考音频(用于声纹克隆,默认使用源音频)
            output_dir: 输出目录

        Returns:
            PipelineResult
        """
        start_time = time.time()
        result = PipelineResult(
            source_audio=np.array([]),
            source_sr=16000,
            source_duration=0,
        )

        output_dir = output_dir or self.config.get("output_dir", "./outputs")
        # 每次运行放到带时间戳的子文件夹, 便于区分、归档与回看
        run_stamp = time.strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(output_dir, f"run_{run_stamp}")
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"输出目录: {output_dir}")

        times = {}  # 各步骤耗时
        t0 = time.time()

        try:
            t1 = time.time()
            logger.info("[1/7] Loading audio...")
            audio, sr = AudioUtils.load_audio(audio_path)
            gold = self._load_gold_transcript(audio_path)
            # 有金标时不要 trim_silence, 否则时间轴与 gold 错位; 且可能裁掉首句
            if gold is None:
                audio = AudioUtils.trim_silence(audio, sr)
            result.source_audio, result.source_sr = audio, sr
            result.source_duration = len(audio) / sr
            times["load"] = time.time() - t1
            logger.info(f"  {result.source_duration:.0f}s, {times['load']:.1f}s")

            t1 = time.time()
            logger.info("[2/7] ASR + Emotion + Diarization...")

            if self.asr is not None:
                # 可选金标旁路: 同名 .gold.json 存在则跳过 ASR 文本 (仍用时间戳)
                if gold is not None:
                    result.asr_result = gold
                    result.detected_language = gold.language or "zh"
                    logger.info(f"   lang: {result.detected_language} (gold transcript)")
                else:
                    result.asr_result = self.asr.transcribe(audio, sr)
                    result.detected_language = result.asr_result.language
                    logger.info(f"   lang: {result.detected_language}")

                # 说话人分离 (识别不同人声; pyannote 不可用时离线聚类兜底 + 明确降级提示)
                # 注意: 不要复用情感嵌入做说话人聚类 (情感空间会抹平音色差异)
                gold_has_spk = all(s.get("speaker") for s in result.asr_result.segments)
                if gold_has_spk and gold is not None:
                    result.diarization_meta = {
                        "backend": "gold",
                        "n_speakers": len({s["speaker"] for s in result.asr_result.segments}),
                        "degraded": False,
                        "first_round_speakers": len({s["speaker"] for s in result.asr_result.segments}),
                        "used_retry": False,
                    }
                    logger.info("   diarization: using gold speaker labels")
                elif self.diarizer is not None:
                    seg_speakers = self.diarizer.diarize(
                        audio, sr, result.asr_result.segments, embedding_fn=None
                    )
                    result.diarization_meta = dict(
                        getattr(self.diarizer, "last_meta", {}) or {}
                    )
                    confs = result.diarization_meta.get("label_confidences") or []
                    for i, (seg, spk) in enumerate(
                        zip(result.asr_result.segments, seg_speakers)
                    ):
                        seg["speaker"] = spk
                        if i < len(confs):
                            seg["label_confidence"] = round(float(confs[i]), 3)
                else:
                    for seg in result.asr_result.segments:
                        seg["speaker"] = "SPEAKER_00"
                    result.diarization_meta = {
                        "backend": "disabled", "n_speakers": 1,
                        "degraded": True, "warning": "diarizer 未启用",
                    }
                logger.info(f"   text: {result.asr_result.text[:150]}...")

            if self.emotion_recognizer is not None:
                result.emotion_result = self.emotion_recognizer.recognize(audio, sr)
            # F6: 段级情感提前到翻译前, 供措辞与后续分组
            if self.emotion_recognizer is not None and result.asr_result is not None:
                seg_emos = self._segment_emotions(audio, sr, result.asr_result.segments)
                default_emo = result.emotion_result.emotion if result.emotion_result else "neutral"
                for i, seg in enumerate(result.asr_result.segments):
                    seg["emotion"] = seg_emos.get(i, default_emo)
                    seg["emotion_bucket"] = self._emotion_bucket(seg["emotion"])
            times["asr+emotion"] = time.time() - t1
            n_spk = len(set(s.get("speaker","?") for s in result.asr_result.segments)) if result.asr_result else 1
            if n_spk <= 1 and result.asr_result and len(result.asr_result.segments) >= 4:
                logger.warning(
                    "⚠️ 说话人分离结果为 1 人 — 多人对白将被克隆成同一音色。"
                    f" backend={result.diarization_meta.get('backend')}"
                )
            logger.info(f"  {result.detected_language}, {len(result.asr_result.segments) if result.asr_result else 0} segs, {n_spk} spk, {times['asr+emotion']:.1f}s")

            t1 = time.time()
            logger.info("[3/7] Translation (per-segment)...")

            if self.translator is not None and result.asr_result is not None:
                src_lang = result.detected_language or result.asr_result.language
                emo = result.emotion_result.emotion if result.emotion_result else "neutral"
                emo_i = result.emotion_result.intensity if result.emotion_result else 0.5
                emo_v = result.emotion_result.valence if result.emotion_result else 0.0
                # O1: 超短 ASR 碎屑并入前一段文本, 再 F7 合并译
                self._glue_asr_crumbs(result.asr_result.segments)
                merge_groups = self._merge_short_segments_for_translation(
                    result.asr_result.segments
                )
                tgt_parts = []
                ok_n = 0
                group_tgts = []  # 与 merge_groups 对齐, 供空译文二次合并重试
                for idxs in merge_groups:
                    segs = [result.asr_result.segments[i] for i in idxs]
                    src = "".join((s.get("text") or "").strip() for s in segs).strip()
                    if not src:
                        for s in segs:
                            s["tgt"] = ""
                        group_tgts.append("")
                        continue
                    seg_emo = segs[0].get("emotion", emo)
                    tgt = self._translate_segment_text(
                        src, src_lang, target_lang, seg_emo, emo_i, emo_v
                    )
                    # 译文挂在合并组首段, 其余置空 (合成时 _join_with_pauses 会拼起来)
                    for j, s in enumerate(segs):
                        s["tgt"] = tgt if j == 0 else ""
                        s["translate_merged"] = len(idxs) > 1
                    group_tgts.append(tgt)
                    if tgt:
                        ok_n += len(idxs)
                        tgt_parts.append(tgt)

                # O2: 空译文组与下一同说话人组合并再译一次
                retry_n = self._retry_empty_translations(
                    result.asr_result.segments,
                    merge_groups,
                    group_tgts,
                    src_lang,
                    target_lang,
                    emo,
                    emo_i,
                    emo_v,
                )
                if retry_n:
                    # 按组统计: 首段有 tgt 则整组算成功
                    ok_n = 0
                    tgt_parts = []
                    for idxs in merge_groups:
                        t = (result.asr_result.segments[idxs[0]].get("tgt") or "").strip()
                        if t:
                            ok_n += len(idxs)
                            tgt_parts.append(t)
                    logger.info(f"  empty-translate retry recovered {retry_n} group(s)")

                joined = " ".join(t for t in tgt_parts if t)
                # 汇总 TranslationResult: 优先用逐段拼接统计音节 (避免整段失败导致 length_ratio 失真)
                src_syl = self.translator._count_syllables(result.asr_result.text, src_lang)
                tgt_syl = self.translator._count_syllables(joined, target_lang) if joined else 0
                length_ratio = tgt_syl / max(src_syl, 1)
                # 整篇再压一次 length_ratio (段级贴边汇总后可能略超)
                max_r = float(self.config.get("translation", {}).get("length_ratio_max", 1.2))
                min_r = float(self.config.get("translation", {}).get("length_ratio_min", 0.8))
                if joined and (length_ratio > max_r or length_ratio < min_r):
                    hist = [joined]
                    joined, length_ratio, tgt_syl = self.translator._constrain_length_ratio(
                        joined, src_syl, target_lang, hist
                    )
                    # 截断后残片修补
                    joined = self.translator._polish_after_length(
                        result.asr_result.text, joined, target_lang
                    )
                    tgt_syl = self.translator._count_syllables(joined, target_lang)
                    length_ratio = tgt_syl / max(src_syl, 1)
                    # 写回末段 tgt 不便拆分, 仅更新汇总译文供 summary/timeline
                    logger.info(
                        f"  doc-level length refine → ratio={length_ratio:.2f}"
                    )
                from src.translation.llm_translator import TranslationResult
                result.translation_result = TranslationResult(
                    source_text=result.asr_result.text,
                    translated_text=joined,
                    source_lang=src_lang,
                    target_lang=target_lang,
                    emotion=emo,
                    source_syllables=src_syl,
                    target_syllables=tgt_syl,
                    length_ratio=length_ratio,
                    confidence=0.85 if ok_n == len(result.asr_result.segments) else 0.6,
                    refinement_history=[joined],
                )
                logger.info(
                    f"  per-seg ok={ok_n}/{len(result.asr_result.segments)}, "
                    f"merge_groups={len(merge_groups)}, "
                    f"chars={len(joined)}, ratio={result.translation_result.length_ratio:.2f}"
                )
            times["translation"] = time.time() - t1
            logger.info(f"  {times['translation']:.1f}s")

            t1 = time.time()
            logger.info("[4/7] AOCP analysis...")

            if self.aocp_net is not None:
                result.aocp_result = self.aocp_net.predict(audio, sr, self.device)
            times["aocp"] = time.time() - t1
            n_open = sum(1 for s in result.aocp_result.state_segments if s["state"]=="open") if result.aocp_result else 0
            logger.info(f"  {n_open} open/{len(result.aocp_result.state_segments) if result.aocp_result else 0} segs, {times['aocp']:.1f}s")

            t1 = time.time()
            logger.info("[5/7] Phoneme alignment + timeline...")

            if self.phoneme_aligner is not None and result.asr_result is not None:
                # 源语言音素对齐
                try:
                    source_phonemes = self.phoneme_aligner.text_to_phonemes(
                        result.asr_result.text,
                        result.detected_language or result.asr_result.language,
                    )
                except Exception as e:
                    logger.warning(f"   ⚠️ 源语言音素转换失败: {e}, 使用字符拆分")
                    source_phonemes = list(result.asr_result.text)

                result.phoneme_alignment = self.phoneme_aligner.align(
                    audio, source_phonemes, sr
                )
                logger.info(f"   phonemes: {len(result.phoneme_alignment.phonemes)}")

                # 【元音检测】提取开口元音作为口型同步关键点
                src_vowels = self.phoneme_aligner.extract_vowel_timeline(
                    result.phoneme_alignment
                )
                core_vowels = [v for v in src_vowels if v["is_core"]]
                logger.info(f"   vowels: {len(src_vowels)}, core: {len(core_vowels)}")

                # 目标语言音素
                target_phonemes = []
                if result.translation_result is not None:
                    try:
                        target_phonemes = self.phoneme_aligner.text_to_phonemes(
                            result.translation_result.translated_text, target_lang
                        )
                    except Exception as e:
                        logger.warning(f"   ⚠️ 目标语言音素转换失败: {e}, 使用字符拆分")
                        target_phonemes = list(result.translation_result.translated_text)

                # 生成时间轴 (元音关键点传入后再计分, 避免覆盖后分数失效)
                if self.timeline_generator is not None and result.aocp_result is not None:
                    length_ratio = 1.0
                    if result.translation_result is not None:
                        length_ratio = result.translation_result.length_ratio

                    result.timeline_result = self.timeline_generator.generate(
                        aocp_result=result.aocp_result,
                        phoneme_alignment=result.phoneme_alignment,
                        source_duration=result.source_duration,
                        target_phonemes=target_phonemes,
                        length_ratio=length_ratio,
                        vowel_timeline=src_vowels,
                    )
                    n_vowel_pts = sum(
                        1 for p in result.timeline_result.sync_points
                        if p.get("type") == "vowel"
                    )
                    logger.info(
                        f"   timeline: {len(result.timeline_result.speech_segments)} segments, "
                        f"sync={result.timeline_result.sync_score:.2f}, "
                        f"vowel_points={n_vowel_pts}, coverage={result.timeline_result.coverage:.2f}"
                    )

            times["phoneme"] = time.time() - t1
            logger.info(f"  {len(source_phonemes)} phonemes, {times['phoneme']:.1f}s")

            t1 = time.time()
            t1_clone = time.time()
            logger.info("[6/7] Voice clone...")

            if self.voice_cloner is not None and result.asr_result is not None:
                self._synthesize_segments(
                    result, audio, sr, target_lang, output_dir,
                    reference_audio_path=reference_audio_path,
                )

            times["clone"] = time.time() - t1_clone
            result.status = "success"
            result.processing_time = time.time() - t0

            total = result.processing_time
            logger.info(f"[7/7] Done ({total:.0f}s)")
            parts = " | ".join(f"{k}={v:.0f}s" for k,v in times.items())
            logger.info(f"  Breakdown: {parts}")

        except Exception as e:
            result.status = "error"
            result.error_message = str(e)
            logger.error(f"❌ Pipeline执行失败: {e}")
            import traceback
            traceback.print_exc()

        logger.info(f"\nTotal time: {result.processing_time:.1f}s")
        logger.info(f"Status: {result.status}")

        return result

    # ==================================================================
    # 语音合成: 起始音/结束音 + 逐段情感 + 减少克隆段 + 声纹复用 + 间距控制
    # ==================================================================

    def _find_onset_offset(self, audio, sr, start, end):
        """
        在 [start, end] 附近检测该段的 起始音/结束音 (onset/offset)。
        用短时能量阈值切掉段首/段尾的静音, 得到真实发音边界 (秒)。
        """
        cfg = self.config.get("pipeline", {}).get("segment", {})
        pad = float(cfg.get("onset_pad", 0.05))
        ratio = float(cfg.get("energy_threshold_ratio", 0.25))
        total = len(audio) / sr
        s = max(0.0, float(start) - pad)
        e = min(total, float(end) + pad)
        si, ei = int(s * sr), int(e * sr)
        if ei <= si:
            return round(max(0.0, float(start)), 3), round(min(total, float(end)), 3)

        seg = audio[si:ei].astype(np.float64)
        hop = max(1, int(sr * 0.01))  # 10ms 帧
        nf = max(1, len(seg) // hop)
        rms = np.array([
            np.sqrt(np.mean(seg[i * hop:(i + 1) * hop] ** 2) + 1e-12)
            for i in range(nf)
        ])
        if rms.size == 0:
            return round(max(0.0, float(start)), 3), round(min(total, float(end)), 3)
        peak = float(rms.max())
        if peak <= 1e-6:
            return round(max(0.0, float(start)), 3), round(min(total, float(end)), 3)

        thr = peak * ratio
        idx = np.where(rms > thr)[0]
        if idx.size == 0:
            return round(max(0.0, float(start)), 3), round(min(total, float(end)), 3)
        onset = s + float(idx[0]) * hop / sr
        offset = s + float(idx[-1] + 1) * hop / sr
        return round(max(0.0, onset), 3), round(min(total, offset), 3)

    def _split_translation(self, result, target_lang):
        """将译文拆成句子 (句末标点保留, 用于停顿控制)"""
        sentences = []
        if result.translation_result is not None and result.translation_result.translated_text:
            sentences = [
                s.strip()
                for s in re.split(r'(?<=[.!?。！？])[\s]+', result.translation_result.translated_text)
                if len(s.strip()) > 1
            ]
        return sentences

    def _load_gold_transcript(self, audio_path: str):
        """
        若存在同名 .gold.json, 用金标文本(+可选时间戳)旁路 ASR。
        格式: {"language":"zh","segments":[{"start":0,"end":2,"text":"...","speaker":"SPEAKER_00"}]}
        """
        from pathlib import Path
        import json
        p = Path(audio_path)
        gold_path = p.with_suffix(".gold.json")
        if not gold_path.exists():
            # 也认 data/foo.wav → data/foo.gold.json 已覆盖; 再试 .transcript.gold.json
            alt = p.parent / f"{p.stem}.gold.json"
            gold_path = alt if alt.exists() else gold_path
        if not gold_path.exists():
            return None
        try:
            data = json.loads(gold_path.read_text(encoding="utf-8"))
            segs = data.get("segments") or []
            if not segs:
                return None
            from src.asr.whisper_asr import ASRResult
            norm = []
            for s in segs:
                text = (s.get("text") or "").strip()
                if not text:
                    continue
                item = {
                    "start": float(s.get("start", 0)),
                    "end": float(s.get("end", s.get("start", 0) + 1)),
                    "text": text,
                    "words": s.get("words") or [],
                    "syllables": s.get("syllables") or [],
                }
                if s.get("speaker"):
                    item["speaker"] = s["speaker"]
                norm.append(item)
            if not norm:
                return None
            text = " ".join(x["text"] for x in norm)
            dur = max(x["end"] for x in norm)
            logger.info(f"   gold transcript: {gold_path.name} ({len(norm)} segs)")
            return ASRResult(
                text=text,
                language=data.get("language") or "zh",
                segments=norm,
                duration=dur,
                confidence=1.0,
            )
        except Exception as e:
            logger.warning(f"gold transcript load fail: {e}")
            return None

    def _translate_segment_text(
        self, src, src_lang, target_lang, seg_emo, emo_i, emo_v
    ) -> str:
        """单段/合并组翻译; 失败或坏译时回落短语表。"""
        if not src or self.translator is None:
            return ""
        try:
            tr = self.translator.translate(
                text=src,
                source_lang=src_lang,
                target_lang=target_lang,
                emotion=seg_emo,
                emotion_intensity=emo_i,
                emotion_valence=emo_v,
                refine=False,
            )
            tgt = (tr.translated_text or "").strip()
            if self.translator._is_bad_translation(tgt, src, target_lang):
                demo = self.translator._demo_phrase_translate(src, target_lang)
                tgt = (demo or "").strip()
            return tgt
        except Exception as e:
            logger.warning(f"  seg translate fail: {e}")
            demo = self.translator._demo_phrase_translate(src, target_lang)
            return (demo or "").strip()

    @staticmethod
    def _is_asr_crumb(text: str, crumb_max: int = 4) -> bool:
        """超短残片: 纯英文尾巴 (ory) / 极少汉字碎屑。"""
        t = re.sub(r"\s+", "", (text or "")).strip(".,;:!?，。、；：！？…")
        if not t:
            return True
        if len(t) <= crumb_max:
            return True
        # 无汉字且极短英文词块
        if not re.search(r"[\u4e00-\u9fff]", t) and len(t) <= crumb_max + 2:
            return True
        return False

    def _glue_asr_crumbs(self, segments: list) -> None:
        """
        O1: 将超短 ASR 碎屑文本拼到前一同说话人「有文本」段 (时间戳不变)。
        避免 "CD我的" / "ory" 单独送翻失败; 跳过已清空的中间壳段。
        """
        if not segments:
            return
        asr_cfg = self.config.get("asr", {})
        crumb_max = int(asr_cfg.get("crumb_max_chars", 4))
        gap_th = float(asr_cfg.get("merge_gap_seconds", 0.5))
        i = 1
        while i < len(segments):
            seg = segments[i]
            text = (seg.get("text") or "").strip()
            if not self._is_asr_crumb(text, crumb_max):
                i += 1
                continue
            spk = seg.get("speaker")
            # 回溯到最近同说话人且仍有文本的段
            j = i - 1
            anchor = None
            while j >= 0:
                prev = segments[j]
                if prev.get("speaker") != spk:
                    break
                gap = float(seg.get("start", 0)) - float(prev.get("end", 0))
                if gap > gap_th + 0.3:
                    break
                if (prev.get("text") or "").strip():
                    anchor = j
                    break
                j -= 1
            if anchor is None:
                i += 1
                continue
            prev = segments[anchor]
            prev["text"] = ((prev.get("text") or "").rstrip() + text).strip()
            seg["text"] = ""
            prev["end"] = max(float(prev.get("end", 0)), float(seg.get("end", 0)))
            if "offset" in prev or "offset" in seg:
                prev["offset"] = max(
                    float(prev.get("offset", prev.get("end", 0))),
                    float(seg.get("offset", seg.get("end", 0))),
                )
            i += 1

    def _retry_empty_translations(
        self,
        segments,
        merge_groups,
        group_tgts,
        src_lang,
        target_lang,
        emo,
        emo_i,
        emo_v,
    ) -> int:
        """
        O2: 空译文组合并下一同说话人组再译; 成功则写回首段。
        返回恢复的组数。
        """
        recovered = 0
        n = len(merge_groups)
        i = 0
        while i < n:
            if (group_tgts[i] or "").strip():
                i += 1
                continue
            idxs = merge_groups[i]
            if not idxs:
                i += 1
                continue
            spk = segments[idxs[0]].get("speaker")
            # 向后找同说话人组 (可吞连续空组 + 一个非空组)
            j = i + 1
            bundle = list(idxs)
            while j < n:
                nxt = merge_groups[j]
                if not nxt:
                    j += 1
                    continue
                if segments[nxt[0]].get("speaker") != spk:
                    break
                bundle.extend(nxt)
                # 吃掉一个已有译文组后停止, 避免无限吞并
                if (group_tgts[j] or "").strip():
                    j += 1
                    break
                j += 1
            if len(bundle) <= len(idxs):
                i += 1
                continue
            src = "".join(
                (segments[k].get("text") or "").strip() for k in bundle
            ).strip()
            if not src:
                i += 1
                continue
            seg_emo = segments[bundle[0]].get("emotion", emo)
            tgt = self._translate_segment_text(
                src, src_lang, target_lang, seg_emo, emo_i, emo_v
            )
            if not tgt:
                i += 1
                continue
            # 清空 bundle 内旧 tgt, 写到首段
            for k in bundle:
                segments[k]["tgt"] = ""
                segments[k]["translate_merged"] = True
            segments[bundle[0]]["tgt"] = tgt
            for g in range(i, j):
                group_tgts[g] = tgt if g == i else ""
            recovered += 1
            i = j if j > i else i + 1
        return recovered

    def _merge_short_segments_for_translation(self, segments: list) -> list:
        """
        F7: 同说话人、短间隔、短文本的相邻段合并后再译。
        返回索引组列表, 例如 [[0], [1,2], [3]]。
        O1: 碎屑段强制可合并; 空文本段并入前组。
        """
        if not segments:
            return []
        asr_cfg = self.config.get("asr", {})
        gap_th = float(asr_cfg.get("merge_gap_seconds", 0.35))
        short_chars = int(asr_cfg.get("merge_short_chars", 8))
        crumb_max = int(asr_cfg.get("crumb_max_chars", 4))
        groups = []
        cur = [0]
        for i in range(1, len(segments)):
            prev, seg = segments[i - 1], segments[i]
            prev_spk = prev.get("speaker", "SPEAKER_00")
            spk = seg.get("speaker", "SPEAKER_00")
            gap = float(seg.get("start", 0)) - float(prev.get("end", 0))
            prev_text = (prev.get("text") or "").strip()
            seg_text = (seg.get("text") or "").strip()
            prev_len = len(re.sub(r"\s", "", prev_text))
            seg_len = len(re.sub(r"\s", "", seg_text))
            crumb = self._is_asr_crumb(seg_text, crumb_max) or self._is_asr_crumb(
                prev_text, crumb_max
            )
            empty_seg = not seg_text
            can_merge = (
                prev_spk == spk
                and gap <= gap_th
                and (
                    empty_seg
                    or crumb
                    or prev_len <= short_chars
                    or seg_len <= short_chars
                )
            )
            if can_merge:
                cur.append(i)
            else:
                groups.append(cur)
                cur = [i]
        groups.append(cur)
        return groups

    def _segment_emotions(self, audio, sr, segments):
        """逐段情感识别 (用该段的起始~结束音切片), 返回 {index: emotion}"""
        emotions = {}
        if self.emotion_recognizer is None:
            return emotions
        for i, seg in enumerate(segments):
            s = int(seg.get("onset", seg["start"]) * sr)
            e = int(seg.get("offset", seg["end"]) * sr)
            if e <= s:
                continue
            try:
                r = self.emotion_recognizer.recognize(audio[s:e], sr, return_timeline=False)
                emotions[i] = r.emotion
            except Exception:
                continue
        return emotions

    @staticmethod
    def _dominant_emotion(emotions):
        if not emotions:
            return "neutral"
        from collections import Counter
        return Counter(emotions).most_common(1)[0][0]

    # 情感粗分桶, 减少抖动导致的误切组 (B5)
    _EMOTION_BUCKET = {
        "happy": "positive", "surprised": "positive",
        "sad": "negative", "fearful": "negative",
        "angry": "negative", "disgusted": "negative",
        "neutral": "neutral",
    }

    def _emotion_bucket(self, emotion: str) -> str:
        return self._EMOTION_BUCKET.get((emotion or "neutral").lower(), "neutral")

    def _group_segments(self, segments):
        """
        按 (说话人, 语气桶) 分组 —— 换语气或换说话人才切分新组。
        同一组内的所有分段一次克隆整段话, 中间用标点控制停顿。
        """
        cfg = self.config.get("pipeline", {}).get("segment", {})
        min_group_segments = int(cfg.get("min_group_segments", 2))
        use_bucket = bool(cfg.get("emotion_bucket", True))

        groups = []
        for seg in segments:
            spk = seg.get("speaker", "SPEAKER_00")
            raw_emo = seg.get("emotion", "neutral")
            emotion = self._emotion_bucket(raw_emo) if use_bucket else raw_emo
            seg["emotion_bucket"] = emotion
            if groups and groups[-1]["spk"] == spk and groups[-1]["emotion"] == emotion:
                groups[-1]["segs"].append(seg)
            else:
                groups.append({"spk": spk, "emotion": emotion, "segs": [seg]})

        # 去噪: 过短的语气波动组 (同说话人) 并回上一组, 避免情绪抖动导致大量分段
        if min_group_segments > 1:
            merged = []
            for g in groups:
                if (merged and merged[-1]["spk"] == g["spk"]
                        and len(g["segs"]) < min_group_segments):
                    merged[-1]["segs"].extend(g["segs"])
                    continue
                merged.append(g)
            groups = merged
        return groups

    @staticmethod
    def _join_with_pauses(group, separator=" "):
        """把组内各句译文用标点衔接, 通过标点控制自然停顿"""
        parts = []
        for seg in group["segs"]:
            t = (seg.get("tgt") or "").strip()
            if not t:
                continue
            if t[-1] not in ".!?。！？":
                t = t + "."
            parts.append(t)
        return separator.join(parts)

    def _ref_quality_score(self, ref_audio: np.ndarray, sr: int) -> Dict:
        """参考音质控: 时长 / 能量 / 削波 / 静音占比"""
        dur = len(ref_audio) / max(sr, 1)
        if len(ref_audio) == 0:
            return {"ok": False, "score": -1e9, "reason": "empty", "duration": 0.0}
        y = ref_audio.astype(np.float64)
        rms = float(np.sqrt(np.mean(y ** 2) + 1e-12))
        peak = float(np.max(np.abs(y)))
        clip_ratio = float(np.mean(np.abs(y) > 0.99))
        # 粗略静音帧占比
        hop = max(1, int(sr * 0.02))
        frames = [y[i:i + hop] for i in range(0, len(y), hop) if len(y[i:i + hop]) > hop // 2]
        if frames:
            frame_rms = np.array([np.sqrt(np.mean(f ** 2) + 1e-12) for f in frames])
            silence_ratio = float(np.mean(frame_rms < (rms * 0.15 + 1e-6)))
        else:
            silence_ratio = 1.0

        score = 0.0
        reasons = []
        if dur < 1.0:
            reasons.append("too_short")
            score -= 5.0
        elif dur < 2.0:
            score -= 1.0
        else:
            score += min(dur, 6.0) * 0.3

        if rms < 0.01:
            reasons.append("low_energy")
            score -= 4.0
        else:
            score += min(rms, 0.2) * 10.0

        if clip_ratio > 0.02:
            reasons.append("clipping")
            score -= 3.0
        if silence_ratio > 0.55:
            reasons.append("too_silent")
            score -= 2.0
        if peak > 0:
            score += 0.5

        ok = (dur >= 1.5 and rms >= 0.012 and clip_ratio <= 0.05 and silence_ratio <= 0.65)
        return {
            "ok": ok,
            "score": score,
            "reason": ",".join(reasons) if reasons else "ok",
            "duration": round(dur, 3),
            "rms": round(rms, 5),
            "clip_ratio": round(clip_ratio, 4),
            "silence_ratio": round(silence_ratio, 3),
        }

    def _normalize_ref(self, ref_audio: np.ndarray) -> np.ndarray:
        """峰值归一, 降低克隆漂移"""
        ref = ref_audio.astype(np.float32)
        peak = float(np.max(np.abs(ref))) if len(ref) else 0.0
        if peak > 1e-6:
            ref = ref / peak * 0.95
        return ref

    def _pick_reference_for_speaker(
        self,
        audio,
        sr,
        all_segments,
        spk: str,
        ref_min: float,
        ref_max: float,
        prefer_group_segs=None,
    ) -> Tuple[np.ndarray, str, Dict]:
        """
        从该说话人全部片段中选最佳参考音 (A3/A4)。
        优先组内, 不合格则扩大到同说话人其他段。
        """
        def _dur(seg):
            onset = seg.get("onset", seg.get("start", 0.0))
            offset = seg.get("offset", seg.get("end", onset))
            return offset - onset

        pool = []
        if prefer_group_segs:
            pool.extend(prefer_group_segs)
        for seg in all_segments:
            if seg.get("speaker", "SPEAKER_00") == spk and seg not in pool:
                pool.append(seg)

        target = (ref_min + ref_max) / 2.0
        ranked = []
        for seg in pool:
            onset = seg.get("onset", seg.get("start", 0.0))
            offset = seg.get("offset", seg.get("end", onset))
            # 限制参考长度, 避免过长残留
            if offset - onset > ref_max:
                mid = (onset + offset) / 2.0
                onset = max(onset, mid - ref_max / 2.0)
                offset = onset + ref_max
            s = int(max(0.0, onset) * sr)
            e = int(min(len(audio) / sr, offset) * sr)
            if e <= s:
                continue
            clip = audio[s:e]
            q = self._ref_quality_score(clip, sr)
            # 时长接近目标加分
            q_score = q["score"] - abs(_dur(seg) - target) * 0.15
            # F6: 与当前组情感一致的参考音加分
            prefer_emo = None
            if prefer_group_segs:
                prefer_emo = (prefer_group_segs[0].get("emotion_bucket")
                              or prefer_group_segs[0].get("emotion"))
            seg_emo = seg.get("emotion_bucket") or seg.get("emotion")
            if prefer_emo and seg_emo and prefer_emo == seg_emo and prefer_emo != "neutral":
                q_score += 1.5
            ranked.append((q_score, q, seg, clip, onset, offset))

        ranked.sort(key=lambda x: x[0], reverse=True)
        weak = False
        if not ranked:
            # 极端兜底: 取该人最长段
            cands = [s for s in all_segments if s.get("speaker") == spk] or all_segments
            best = max(cands, key=_dur)
            onset = best.get("onset", best.get("start", 0.0))
            offset = best.get("offset", best.get("end", onset))
            s, e = int(onset * sr), int(offset * sr)
            clip = audio[s:e] if e > s else audio[: int(3 * sr)]
            q = self._ref_quality_score(clip, sr)
            weak = True
            ref_text = best.get("text", "").strip()[:200]
            meta = {
                "speaker": spk,
                "onset": round(onset, 3),
                "offset": round(offset, 3),
                "weak_clone": True,
                "quality": q,
                "source_text": ref_text,
            }
            return self._normalize_ref(clip), ref_text, meta

        # 拼接同说话人多段, 凑够 ref_min, 避免单段过短
        pieces = []
        texts = []
        total = 0.0
        used_onsets = []
        used_offsets = []
        for q_score, q, seg, clip, onset, offset in ranked:
            used_onsets.append(onset)
            used_offsets.append(offset)
            pieces.append(clip.astype(np.float32))
            t = (seg.get("text") or "").strip()
            if t:
                texts.append(t)
            total += len(clip) / sr
            if total >= ref_min and (q.get("ok") or total >= ref_min * 1.2):
                break
            if total >= ref_max:
                break

        concat = np.concatenate(pieces) if pieces else ranked[0][3]
        if len(concat) / sr > ref_max:
            concat = concat[: int(ref_max * sr)]
        q = self._ref_quality_score(concat, sr)
        weak = not q["ok"]
        if weak:
            logger.warning(
                f"   参考音质控未通过 [{spk}]: {q['reason']} "
                f"(dur={q['duration']}s rms={q['rms']}) → 弱克隆标记"
            )
        ref_text = " ".join(texts)[:200]
        meta = {
            "speaker": spk,
            "onset": round(float(min(used_onsets) if used_onsets else 0.0), 3),
            "offset": round(float(max(used_offsets) if used_offsets else 0.0), 3),
            "weak_clone": weak,
            "quality": q,
            "source_text": ref_text,
            "concat_pieces": len(pieces),
        }
        return self._normalize_ref(concat.astype(np.float32)), ref_text, meta

    def _pick_reference(self, audio, sr, group, ref_min, ref_max):
        """兼容旧接口: 仅从组内选参考"""
        ref_audio, ref_text, _meta = self._pick_reference_for_speaker(
            audio, sr, group["segs"], group["spk"], ref_min, ref_max,
            prefer_group_segs=group["segs"],
        )
        return ref_audio, ref_text

    @staticmethod
    def _align_sentences(segments, sentences):
        """把译文句无损分配到各分段 (最后一段吸收剩余, 绝不丢尾部文字)"""
        for seg in segments:
            seg["tgt"] = ""
        n_seg = len(segments)
        n_sen = len(sentences)
        if n_sen == 0:
            return
        if n_sen <= n_seg:
            for i in range(n_sen):
                segments[i]["tgt"] = sentences[i]
        else:
            for i in range(n_seg - 1):
                segments[i]["tgt"] = sentences[i]
            segments[-1]["tgt"] = " ".join(sentences[n_seg - 1:])

    @staticmethod
    def _chunk_text(text, max_chars=250, max_sentences=3):
        """把整段话按句边界切成小块, 每块更易被 TTS 可靠完整生成"""
        sentences = re.split(r'(?<=[.!?。！？])\s+', (text or "").strip())
        chunks = []
        cur = ""
        n = 0
        for s in sentences:
            s = s.strip()
            if not s:
                continue
            if s[-1] not in ".!?。！？":
                s += "."
            if cur and (len(cur) + len(s) > max_chars or n >= max_sentences):
                chunks.append(cur)
                cur = s
                n = 1
            else:
                cur = (cur + " " + s).strip() if cur else s
                n += 1
        if cur:
            chunks.append(cur)
        return chunks

    @staticmethod
    def _crossfade_concat(parts, sr, crossfade_ms=15):
        """交叉淡化拼接多个音频块, 消除连接处的咔哒/生硬感"""
        parts = [p for p in parts if p is not None and len(p) > 0]
        if not parts:
            return np.zeros(0, dtype=np.float32)
        if len(parts) == 1:
            return parts[0].astype(np.float32)
        cf = int(sr * crossfade_ms / 1000.0)
        out = parts[0].astype(np.float32)
        for p in parts[1:]:
            p = p.astype(np.float32)
            if cf > 0 and len(out) > cf and len(p) > cf:
                ramp_out = np.linspace(1.0, 0.0, cf, dtype=np.float32)
                ramp_in = np.linspace(0.0, 1.0, cf, dtype=np.float32)
                overlap = out[-cf:] * ramp_out + p[:cf] * ramp_in
                out = np.concatenate([out[:-cf], overlap, p[cf:]])
            else:
                out = np.concatenate([out, p])
        return out

    def _save_artifacts(self, output_dir, result, segments, groups, source_dur, out_dur, target_lang,
                        speaker_ref_meta=None):
        """把本次运行的翻译结果/分段结果等落盘, 便于直观回看"""
        import json
        from datetime import datetime
        try:
            trans = result.translation_result
            emotion = result.emotion_result.emotion if result.emotion_result else "neutral"
            src_lang = result.detected_language or (result.asr_result.language if result.asr_result else "?")
            n_spk = len({g["spk"] for g in groups})
            diar = result.diarization_meta or {}
            clone = result.clone_meta or {}
            speaker_ref_meta = speaker_ref_meta or {}

            # 摘要
            lines = ["=" * 60, "运行摘要", "=" * 60,
                     f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                     f"源语言: {src_lang} -> 目标语言: {target_lang}",
                     f"源音频时长: {source_dur:.1f}s",
                     f"输出音频时长: {out_dur:.1f}s",
                     f"ASR 分段数: {len(segments)}",
                     f"说话人数: {n_spk}",
                     f"分离后端: {diar.get('backend', '?')} | degraded={diar.get('degraded', '?')}",
                     f"首轮说话人数: {diar.get('first_round_speakers', '?')} | retry={diar.get('used_retry', '?')}",
                     f"克隆引擎: {clone.get('engine', '?')} | supports_clone={clone.get('supports_voice_clone', '?')}",
                     f"整体情感: {emotion}",
                     f"克隆分组数: {len(groups)}"]
            if diar.get("warning"):
                lines.append(f"分离警告: {diar['warning']}")
            if speaker_ref_meta:
                lines.append("说话人参考音:")
                for spk, meta in speaker_ref_meta.items():
                    q = meta.get("quality", {})
                    lines.append(
                        f"  - {spk}: {meta.get('onset')}s-{meta.get('offset')}s "
                        f"weak={meta.get('weak_clone')} reason={q.get('reason', '?')}"
                    )
            if trans is not None:
                lines.append(f"源音节: {trans.source_syllables} | 目标音节: {trans.target_syllables} | 长度比: {trans.length_ratio:.2f}")
            if result.timeline_result is not None:
                lines.append(f"时间轴: {len(result.timeline_result.speech_segments)} 说话段, 同步分 {result.timeline_result.sync_score:.2f}")
            with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")

            # 翻译结果
            tlines = ["=" * 60, "翻译结果", "=" * 60, "",
                      "[源文本]", result.asr_result.text if result.asr_result else "", "",
                      f"[译文 ({target_lang})]", trans.translated_text if trans else "", ""]
            if trans is not None:
                tlines.append(f"情感: {emotion} | 源音节: {trans.source_syllables} | 目标音节: {trans.target_syllables} | 长度比: {trans.length_ratio:.2f}")
            with open(os.path.join(output_dir, "translation.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(tlines) + "\n")

            # 分段结果 (人可读表格 + JSON)
            slines = [f"{'#':>3} | {'说话人':<12} | {'时间范围':<13} | {'起止音':<13} | {'时长':>6} | {'情感':<10} | 源文本 -> 译文",
                      "-" * 110]
            json_segs = []
            for i, seg in enumerate(segments):
                onset = seg.get("onset", seg.get("start", 0.0))
                offset = seg.get("offset", seg.get("end", onset))
                start = seg.get("start", onset)
                end = seg.get("end", offset)
                spk = seg.get("speaker", "SPEAKER_00")
                emo = seg.get("emotion", "neutral")
                src = (seg.get("text") or "").strip()[:40]
                tgt = (seg.get("tgt") or "").strip()[:60]
                slines.append(
                    f"{i + 1:>3} | {spk:<12} | {start:>6.2f}-{end:<6.2f} | "
                    f"{onset:>6.2f}-{offset:<6.2f} | {offset - onset:>5.2f}s | {emo:<10} | {src} -> {tgt}"
                )
                json_segs.append({
                    "index": i + 1, "speaker": spk,
                    "label_confidence": seg.get("label_confidence"),
                    "start": round(start, 3), "end": round(end, 3),
                    "onset": round(onset, 3), "offset": round(offset, 3),
                    "emotion": emo,
                    "emotion_bucket": seg.get("emotion_bucket", emo),
                    "source": (seg.get("text") or "").strip(),
                    "target": (seg.get("tgt") or "").strip(),
                })
            with open(os.path.join(output_dir, "segments.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(slines) + "\n")
            with open(os.path.join(output_dir, "segments.json"), "w", encoding="utf-8") as f:
                json.dump({
                    "diarization": diar,
                    "clone": clone,
                    "speaker_references": speaker_ref_meta,
                    "segments": json_segs,
                    "groups": [
                        {"speaker": g["spk"], "emotion": g["emotion"],
                         "segments": len(g["segs"]), "text": self._join_with_pauses(g)}
                        for g in groups
                    ],
                }, f, ensure_ascii=False, indent=2)
            logger.info(f"   结果文件已写入: {output_dir}")
        except Exception as e:
            logger.warning(f"   结果文件写入失败: {str(e)[:120]}")

    def _synthesize_segments(self, result, audio, sr, target_lang, output_dir,
                             reference_audio_path: Optional[str] = None):
        """
        整段克隆式合成:
        1) 每段计算 起始音/结束音 (onset/offset)
        2) 逐段情感识别
        3) 译文无损对齐
        4) 按 (说话人, 语气) 分组 —— 换语气或换说话人才切分
        5) 每说话人独立参考音 + prompt (禁止跨人复用); 多人场景忽略全局 --ref-audio
        6) 落盘翻译/分段/摘要结果
        """
        seg_cfg = self.config.get("pipeline", {}).get("segment", {})
        min_gap = float(seg_cfg.get("min_gap", 0.05))
        crossfade_ms = float(seg_cfg.get("crossfade_ms", 25))
        ref_min = float(seg_cfg.get("reference_min_seconds", 2.0))
        ref_max = float(seg_cfg.get("reference_max_seconds", 8.0))
        chunk_chars = int(seg_cfg.get("chunk_max_chars", 250))
        chunk_sents = int(seg_cfg.get("chunk_max_sentences", 3))

        asr_segments = result.asr_result.segments
        if not asr_segments:
            logger.warning("无 ASR 分段, 跳过合成")
            return None

        # 1) 起始音/结束音
        for seg in asr_segments:
            onset, offset = self._find_onset_offset(audio, sr, seg["start"], seg["end"])
            seg["onset"] = onset
            seg["offset"] = offset

        # 2) 逐段情感
        seg_emotions = self._segment_emotions(audio, sr, asr_segments)
        default_emotion = result.emotion_result.emotion if result.emotion_result else "neutral"
        for i, seg in enumerate(asr_segments):
            seg["emotion"] = seg_emotions.get(i, default_emotion)

        # 3) 译文: 优先用逐段已填的 tgt; 否则整段对齐兜底
        if not any((s.get("tgt") or "").strip() for s in asr_segments):
            trans_sentences = self._split_translation(result, target_lang)
            self._align_sentences(asr_segments, trans_sentences)
        else:
            logger.info(
                f"   使用逐段译文: "
                f"{sum(1 for s in asr_segments if (s.get('tgt') or '').strip())}/{len(asr_segments)}"
            )

        # 4) 分组
        groups = self._group_segments(asr_segments)
        n_speakers = len({g["spk"] for g in groups})
        logger.info(f"   分组: {len(asr_segments)} 段 -> {len(groups)} 组 ({n_speakers} 说话人)")

        # 全局 ref 仅允许单说话人场景 (A3: 禁止多人被同一 ref 覆盖)
        global_ref_audio = None
        if reference_audio_path and os.path.exists(reference_audio_path):
            if n_speakers <= 1:
                global_ref_audio, _ = AudioUtils.load_audio(reference_audio_path)
                logger.info(f"   单说话人: 使用全局参考音 {reference_audio_path}")
            else:
                logger.warning(
                    f"   检测到 {n_speakers} 个说话人, 忽略全局 --ref-audio, "
                    "改为每人独立参考音 (避免全片同声)"
                )

        clone_sr = self.voice_cloner.tts_engine.sample_rate
        engine_name = getattr(self.voice_cloner.tts_engine, "engine_name", "?")
        supports_clone = bool(
            getattr(self.voice_cloner.tts_engine, "supports_voice_clone", False)
        )
        use_batch = getattr(self.voice_cloner.tts_engine, "_qwen3_available", False)
        result.clone_meta = {
            "engine": engine_name,
            "supports_voice_clone": supports_clone,
            "n_speakers": n_speakers,
            "ignored_global_ref": bool(reference_audio_path) and n_speakers > 1,
            "batch_mode": bool(use_batch),
        }
        if use_batch:
            logger.info("   Qwen3 batch clone: ON (prompt 复用 + clone_batch)")
        if not supports_clone:
            logger.warning(
                "⚠️ 当前 TTS 无真克隆能力 — 听感可能仅限「能听」。"
                "请配置 CosyVoice2 / Qwen3-TTS"
            )

        lang_map = {"zh": "Chinese", "en": "English", "ja": "Japanese", "ko": "Korean",
                    "fr": "French", "de": "German", "es": "Spanish", "it": "Italian",
                    "pt": "Portuguese", "ru": "Russian", "ar": "Arabic"}
        lang_full = lang_map.get(target_lang, "English")

        # 5) 每说话人缓存参考音 + prompt (同人跨组复用, 跨人绝不复用)
        speaker_ref_cache: Dict[str, Tuple[np.ndarray, str, Dict]] = {}
        speaker_prompt_cache: Dict[str, object] = {}
        speaker_ref_meta: Dict[str, Dict] = {}

        group_audios = []
        for gi, group in enumerate(groups):
            tgt_text = self._join_with_pauses(group)
            if not tgt_text:
                group_audios.append(np.zeros(int(0.2 * clone_sr), dtype=np.float32))
                continue
            chunks = self._chunk_text(tgt_text, chunk_chars, chunk_sents)
            spk = group["spk"]
            emotion = group["emotion"]

            if spk not in speaker_ref_cache:
                if global_ref_audio is not None and n_speakers <= 1:
                    ref_audio = self._normalize_ref(global_ref_audio.astype(np.float32))
                    ref_text = ""
                    meta = {
                        "speaker": spk, "onset": 0.0,
                        "offset": round(len(ref_audio) / sr, 3),
                        "weak_clone": False,
                        "quality": self._ref_quality_score(ref_audio, sr),
                        "source_text": "",
                        "from_global_ref": True,
                    }
                else:
                    ref_audio, ref_text, meta = self._pick_reference_for_speaker(
                        audio, sr, asr_segments, spk, ref_min, ref_max,
                        prefer_group_segs=group["segs"],
                    )
                speaker_ref_cache[spk] = (ref_audio, ref_text, meta)
                speaker_ref_meta[spk] = meta
                logger.info(
                    f"   参考音[{spk}]: {meta.get('onset')}-{meta.get('offset')}s "
                    f"weak={meta.get('weak_clone')} q={meta.get('quality', {}).get('reason')}"
                )

            ref_audio, ref_text, _meta = speaker_ref_cache[spk]

            try:
                # 源组覆盖时长, 用于语速对齐 (F4 / B3)
                g0 = group["segs"][0]
                g1 = group["segs"][-1]
                src_dur = float(
                    g1.get("offset", g1.get("end", 0))
                    - g0.get("onset", g0.get("start", 0))
                )
                per_chunk = (src_dur / max(len(chunks), 1)) if src_dur > 0.3 else None

                if use_batch:
                    if spk not in speaker_prompt_cache:
                        speaker_prompt_cache[spk] = self.voice_cloner.prepare_speaker_prompt(
                            ref_audio, sr, ref_text
                        )
                    prompt = speaker_prompt_cache[spk]
                    wavs = self.voice_cloner.tts_engine.clone_batch(
                        chunks, lang=lang_full, voice_clone_prompt=prompt,
                        ref_audio=ref_audio, ref_text=ref_text, ref_sr=sr,
                    )
                    wavs = [w.astype(np.float32) for w in wavs if w is not None and len(w) > 0]
                else:
                    wavs = []
                    for c in chunks:
                        cl = self.voice_cloner.clone(
                            text=c, reference_audio=ref_audio, reference_sample_rate=sr,
                            reference_text=ref_text, emotion_label=emotion,
                            target_lang=target_lang,
                            target_duration=per_chunk if supports_clone else None,
                            speaker_id=spk,
                        )
                        wavs.append(cl.audio.astype(np.float32))
                # 轻微峰值归一, 降低组间响度跳变
                group_audio = self._crossfade_concat(wavs, clone_sr, crossfade_ms)
                # F6: 批处理路径也做情感韵律 hint
                group_audio = self.voice_cloner._apply_emotion_prosody(
                    group_audio, clone_sr, emotion
                )
                # F4: 批处理路径也按源组时长 time-stretch
                match_group = bool(seg_cfg.get("match_group_duration", True))
                if (
                    match_group and supports_clone and src_dur > 0.3
                    and len(group_audio) > 0
                ):
                    from src.synthesis.tts_engine import TTSEngine
                    cur_g = len(group_audio) / clone_sr
                    if abs(cur_g - src_dur) / src_dur > 0.05:
                        group_audio = TTSEngine._stretch_to_duration(
                            group_audio, clone_sr, src_dur
                        )
                peak = float(np.max(np.abs(group_audio))) if len(group_audio) else 0.0
                if peak > 1e-6:
                    group_audio = group_audio / peak * 0.95
                group_audios.append(group_audio)
                logger.info(f"   [{gi + 1}/{len(groups)}] {spk} {emotion} "
                            f"{len(group['segs'])}段 -> {len(chunks)}块整段克隆 ({len(group_audio) / clone_sr:.1f}s)")
            except Exception as e:
                logger.warning(f"   组 {gi} 合成失败: {str(e)[:120]}")
                group_audios.append(np.zeros(int(1.0 * clone_sr), dtype=np.float32))

        result.clone_meta["speaker_references"] = speaker_ref_meta

        # 6) 组间静音: F6 对齐源语段间隔 (可读 ASR onset/offset)
        if not group_audios:
            return None
        gap_min = float(seg_cfg.get("min_gap", 0.05))
        gap_max = float(seg_cfg.get("max_gap", 0.8))
        align_gap = bool(seg_cfg.get("align_source_gaps", True))
        parts = []
        for i, a in enumerate(group_audios):
            parts.append(a)
            if i >= len(group_audios) - 1:
                continue
            if align_gap and i + 1 < len(groups):
                prev_segs, next_segs = groups[i]["segs"], groups[i + 1]["segs"]
                prev_end = float(
                    prev_segs[-1].get("offset", prev_segs[-1].get("end", 0))
                )
                next_start = float(
                    next_segs[0].get("onset", next_segs[0].get("start", 0))
                )
                src_gap = max(0.0, next_start - prev_end)
                gap_sec = float(np.clip(src_gap, gap_min, gap_max))
            else:
                gap_sec = gap_min
            parts.append(np.zeros(int(gap_sec * clone_sr), dtype=np.float32))
        full = np.concatenate(parts) if parts else np.zeros(clone_sr, dtype=np.float32)

        # 归一化
        mv = float(np.abs(full).max()) if len(full) else 0.0
        if mv > 1.0:
            full = full / mv * 0.98
        # F4: 整轨输出压到源时长的 duration_ratio_min–max
        match_out = bool(seg_cfg.get("match_output_duration", True))
        src_d = float(result.source_duration or 0.0)
        if match_out and supports_clone and src_d > 0.5 and len(full) > 0:
            from src.synthesis.tts_engine import TTSEngine
            d_min = float(seg_cfg.get("duration_ratio_min", 0.8))
            d_max = float(seg_cfg.get("duration_ratio_max", 1.2))
            out_d = len(full) / clone_sr
            lo, hi = src_d * d_min, src_d * d_max
            if out_d > hi or out_d < lo:
                target_d = float(np.clip(out_d, lo, hi))
                logger.info(
                    f"   duration match: {out_d:.1f}s → {target_d:.1f}s "
                    f"(src={src_d:.1f}s, band=[{d_min:.2f},{d_max:.2f}])"
                )
                full = TTSEngine._stretch_to_duration(full, clone_sr, target_d)
        # 尾部收束
        fade_ms = float(seg_cfg.get("fade_ms", 12))
        tail = int(clone_sr * max(fade_ms, 8) / 1000.0)
        if len(full) > tail:
            full[-tail:] *= np.linspace(1.0, 0.0, tail, dtype=np.float32)
        # 头部淡入, 去咔哒
        head = min(tail, len(full) // 4)
        if head > 0:
            full[:head] *= np.linspace(0.0, 1.0, head, dtype=np.float32)

        output_path = os.path.join(output_dir, "cloned_output.wav")
        AudioUtils.save_audio(full, output_path, clone_sr)
        out_dur = len(full) / clone_sr

        # 7) 落盘结果 (翻译/分段/摘要)
        self._save_artifacts(
            output_dir, result, asr_segments, groups,
            result.source_duration, out_dur, target_lang,
            speaker_ref_meta=speaker_ref_meta,
        )

        logger.info(f"   output: {output_path} ({len(groups)} 组, {n_speakers} 说话人, {out_dur:.1f}s)")
        return output_path

    def run_segment_by_segment(
        self,
        audio_path: str,
        target_lang: str = "en",
        output_dir: Optional[str] = None,
    ) -> List[PipelineResult]:
        """
        逐段处理(适合长音频)

        先将音频按ASR分段,然后逐段执行完整Pipeline
        """
        logger.info("📋 启用逐段处理模式")

        output_dir = output_dir or self.config.get("output_dir", "./outputs")
        os.makedirs(output_dir, exist_ok=True)

        # 先做ASR获取分段
        audio, sr = AudioUtils.load_audio(audio_path)
        if self.asr is None:
            self.asr = WhisperASR(self.config["asr"])

        asr_result = self.asr.transcribe(audio, sr)

        results = []
        for i, seg in enumerate(asr_result.segments):
            if not seg["text"].strip():
                continue

            start_sample = int(seg["start"] * sr)
            end_sample = int(seg["end"] * sr)
            seg_audio = audio[start_sample:end_sample]

            # 保存临时音频
            temp_path = os.path.join(output_dir, f"temp_seg_{i}.wav")
            AudioUtils.save_audio(seg_audio, temp_path, sr)

            # 运行Pipeline
            result = self.run(
                audio_path=temp_path,
                target_lang=target_lang,
                reference_audio_path=audio_path,
                output_dir=output_dir,
            )
            results.append(result)

            # 清理临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

        return results
