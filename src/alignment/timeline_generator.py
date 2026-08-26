"""
时间轴生成器
Timeline Generator

【创新点】融合AOCP-Net开闭状态 + 音素对齐的联合时间轴生成

这是连接"发音分析"和"语音合成"的关键桥梁:
- 输入: AOCP-Net的开闭状态 + 音素对齐 + 译文字数
- 输出: 目标语言合成时的时间轴约束
"""

import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
from loguru import logger

from .articulatory_analyzer import AOCPResult
from .phoneme_aligner import PhonemeAlignment


@dataclass
class TimelineResult:
    """时间轴生成结果"""
    segments: List[Dict]            # 每个发音段的时间信息
    total_duration: float           # 总时长
    speech_segments: List[Dict]     # 说话段(开口段)
    silence_segments: List[Dict]    # 静音段(闭口段)
    sync_points: List[Dict]         # 同步关键点(用于口型对齐)
    phoneme_timeline: List[Dict]    # 音素级时间轴
    # 质量指标
    sync_score: float               # 同步质量评分
    coverage: float                 # 覆盖完整度


class TimelineGenerator:
    """
    时间轴生成器

    【核心创新】多源信息融合的时间轴生成

    融合以下信息生成最优时间轴:
    1. AOCP-Net 预测的发音开闭状态
    2. 音素对齐的时间信息
    3. 源语言和目标语言的音节长度比
    4. 翻译文本的结构信息

    生成的时间轴可用于:
    - 驱动TTS在指定时间点合成语音
    - 指导视频口型的同步调整
    """

    def __init__(self, config: Dict):
        self.config = config
        self.sample_rate = config.get("aocp", {}).get("sample_rate", 16000)
        self.min_speech_duration = 0.05   # 最小说话段时长
        self.min_silence_duration = 0.03  # 最小静音段时长

    def generate(
        self,
        aocp_result: AOCPResult,
        phoneme_alignment: PhonemeAlignment,
        source_duration: float,
        target_phonemes: List[str],
        length_ratio: float = 1.0,
    ) -> TimelineResult:
        """
        生成综合时间轴

        Args:
            aocp_result: AOCP-Net发音分析结果
            phoneme_alignment: 音素对齐结果
            source_duration: 源音频时长
            target_phonemes: 目标语言音素序列
            length_ratio: 音节长度比(目标/源)

        Returns:
            TimelineResult
        """
        # 步骤1: 从AOCP结果提取说话段和静音段
        speech_segments, silence_segments = self._extract_speech_silence(aocp_result)

        # 步骤2: 根据长度比调整目标时间轴
        target_duration = source_duration * length_ratio
        adjusted_speech = self._adjust_durations(
            speech_segments, source_duration, target_duration
        )

        # 步骤3: 融合音素对齐信息
        phoneme_timeline = self._build_phoneme_timeline(
            target_phonemes, adjusted_speech, target_duration
        )

        # 步骤4: 生成同步关键点
        sync_points = self._generate_sync_points(
            aocp_result, phoneme_alignment, adjusted_speech
        )

        # 步骤5: 质量评估
        sync_score = self._compute_sync_score(sync_points, adjusted_speech)
        coverage = self._compute_coverage(adjusted_speech, target_duration)

        # 构建完整段信息
        segments = []
        for seg in adjusted_speech:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "openness": seg.get("openness", 0.5),
                "type": "speech",
            })
        for seg in silence_segments:
            segments.append({
                "start": seg["start"],
                "end": seg["end"],
                "duration": seg["duration"],
                "type": "silence",
            })
        segments.sort(key=lambda x: x["start"])

        logger.info(
            f"📊 时间轴生成完成: "
            f"说话段={len(speech_segments)}, "
            f"同步点={len(sync_points)}, "
            f"同步质量={sync_score:.2f}"
        )

        return TimelineResult(
            segments=segments,
            total_duration=target_duration,
            speech_segments=adjusted_speech,
            silence_segments=silence_segments,
            sync_points=sync_points,
            phoneme_timeline=phoneme_timeline,
            sync_score=sync_score,
            coverage=coverage,
        )

    def _extract_speech_silence(
        self, aocp_result: AOCPResult
    ) -> tuple:
        """从AOCP结果提取说话段和静音段"""
        speech = []
        silence = []

        for seg in aocp_result.state_segments:
            if seg["state"] == "open":
                speech.append(seg)
            elif seg["state"] == "closed" and seg["duration"] >= self.min_silence_duration:
                silence.append(seg)
            # transition: 根据时长判断
            elif seg["state"] == "transition":
                if seg["duration"] >= self.min_speech_duration:
                    speech.append(seg)

        return speech, silence

    def _adjust_durations(
        self,
        speech_segments: List[Dict],
        source_duration: float,
        target_duration: float,
    ) -> List[Dict]:
        """
        【创新点】基于长度比的时长自适应调整

        当目标语言译文比源语言长或短时,
        按比例调整每个说话段的时长,
        保持相对时间结构不变。
        """
        if source_duration <= 0 or len(speech_segments) == 0:
            return speech_segments

        scale = target_duration / source_duration

        adjusted = []
        for seg in speech_segments:
            new_seg = seg.copy()
            new_seg["start"] = seg["start"] * scale
            new_seg["end"] = seg["end"] * scale
            new_seg["duration"] = seg["duration"] * scale
            adjusted.append(new_seg)

        return adjusted

    def _build_phoneme_timeline(
        self,
        target_phonemes: List[str],
        speech_segments: List[Dict],
        target_duration: float,
    ) -> List[Dict]:
        """
        【创新点】音素级时间轴构建

        将目标音素分配到时间轴上,
        结合AOCP-Net的开口度信息优化分配
        """
        from .phoneme_aligner import CROSS_LINGUAL_PHONEME_MAP

        n_phonemes = len(target_phonemes)
        if n_phonemes == 0:
            return []

        # 收集所有说话时间
        total_speech_time = sum(s["duration"] for s in speech_segments)

        timeline = []
        ph_idx = 0

        for seg in speech_segments:
            seg_n_phonemes = max(1, int(n_phonemes * seg["duration"] / max(total_speech_time, 0.01)))
            seg_n_phonemes = min(seg_n_phonemes, n_phonemes - ph_idx)

            if seg_n_phonemes <= 0:
                continue

            ph_duration = seg["duration"] / seg_n_phonemes

            for i in range(seg_n_phonemes):
                if ph_idx >= n_phonemes:
                    break

                ph = target_phonemes[ph_idx] if ph_idx < len(target_phonemes) else ""

                # 根据音素类型微调时长
                is_open = any(
                    ph in phoneme_list
                    for name, phoneme_list in CROSS_LINGUAL_PHONEME_MAP.items()
                    if "open" in name
                )
                duration_mult = 1.3 if is_open else 0.8

                timeline.append({
                    "phoneme": ph,
                    "start": seg["start"] + i * ph_duration,
                    "end": seg["start"] + (i + 1) * ph_duration,
                    "duration": ph_duration * duration_mult,
                    "openness": seg.get("openness", 0.5),
                })
                ph_idx += 1

        return timeline

    def _generate_sync_points(
        self,
        aocp_result: AOCPResult,
        phoneme_alignment: PhonemeAlignment,
        speech_segments: List[Dict],
    ) -> List[Dict]:
        """
        【创新点】生成口型同步关键点

        这些关键点是连接音频和视频的桥梁:
        - 每个开口段的起止点
        - 开口度峰值点
        - 状态转换点
        """
        sync_points = []

        # 1. 从AOCP结果中提取开口峰值点
        openness = aocp_result.openness
        hop_ms = self.config.get("aocp", {}).get("hop_size_ms", 10)
        hop_s = hop_ms / 1000.0

        for seg in speech_segments:
            start_frame = int(seg["start"] / hop_s)
            end_frame = int(seg["end"] / hop_s)
            start_frame = max(0, min(start_frame, len(openness) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(openness)))

            if start_frame < end_frame and end_frame <= len(openness):
                window = openness[start_frame:end_frame]
                peak_idx = start_frame + np.argmax(window)
                peak_time = peak_idx * hop_s

                sync_points.append({
                    "time": round(peak_time, 3),
                    "type": "openness_peak",
                    "value": round(float(window.max()), 3),
                })

            # 开口段起止点
            sync_points.append({
                "time": round(seg["start"], 3),
                "type": "speech_start",
                "value": seg.get("openness", 0.5),
            })
            sync_points.append({
                "time": round(seg["end"], 3),
                "type": "speech_end",
                "value": seg.get("openness", 0.5),
            })

        # 2. 从音素对齐中添加音素边界点
        for i, (start, end) in enumerate(
            zip(phoneme_alignment.start_times, phoneme_alignment.end_times)
        ):
            if i < len(phoneme_alignment.phonemes):
                ph = phoneme_alignment.phonemes[i]
                from .phoneme_aligner import CROSS_LINGUAL_PHONEME_MAP

                is_important = any(
                    ph in phoneme_list
                    for name, phoneme_list in CROSS_LINGUAL_PHONEME_MAP.items()
                    if name in ("plosive", "open", "near_open")
                )
                if is_important:
                    sync_points.append({
                        "time": round(start, 3),
                        "type": "phoneme_boundary",
                        "phoneme": ph,
                    })

        sync_points.sort(key=lambda x: x["time"])
        return sync_points

    @staticmethod
    def _compute_sync_score(
        sync_points: List[Dict], speech_segments: List[Dict]
    ) -> float:
        """计算同步质量评分"""
        if not speech_segments:
            return 0.0

        # 检查关键点覆盖
        covered_duration = sum(s["duration"] for s in speech_segments)
        total_duration = max(s["end"] for s in speech_segments) if speech_segments else 1.0

        # 开口峰值点的分布均匀度
        peak_points = [p for p in sync_points if p["type"] == "openness_peak"]
        if len(peak_points) >= 2:
            intervals = [
                peak_points[i+1]["time"] - peak_points[i]["time"]
                for i in range(len(peak_points) - 1)
            ]
            if intervals:
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                uniformity = 1.0 - min(std_interval / max(mean_interval, 0.01), 1.0)
            else:
                uniformity = 0.5
        else:
            uniformity = 0.5

        coverage_ratio = min(covered_duration / max(total_duration, 0.01), 1.0)
        score = 0.5 * coverage_ratio + 0.5 * uniformity
        return round(score, 3)

    @staticmethod
    def _compute_coverage(
        speech_segments: List[Dict], total_duration: float
    ) -> float:
        """计算时间覆盖完整度"""
        if total_duration <= 0:
            return 0.0
        covered = sum(s["duration"] for s in speech_segments)
        return round(min(covered / total_duration, 1.0), 3)
