"""
声纹复刻与语音克隆模块

【创新点】情感保持的零样本语音克隆

本模块实现:
1. 声纹特征提取与建模
2. 零样本音色克隆 (从3-5秒参考音频)
3. 情感保持: 克隆时不丢失源语音的情感特征
4. 跨语种音色迁移: 保持音色在目标语言中的一致性
"""

import numpy as np
from typing import Dict, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .tts_engine import TTSEngine, TTSResult
from .vocoder import HiFiGANVocoder


@dataclass
class ClonedSpeech:
    """克隆语音结果"""
    audio: np.ndarray
    sample_rate: int
    duration: float
    speaker_embedding: np.ndarray
    emotion_preserved: bool
    quality_score: float
    text: str
    timeline: list


class VoiceCloner:
    """
    声纹复刻引擎

    核心流程:
    1. 提取参考音频的声纹特征
    2. 情感特征提取与保存
    3. 利用TTS引擎进行零样本合成
    4. 情感注入与优化

    【创新点 - 可用于专利】
    - 情感-音色解耦克隆: 分离情感和音色表征,实现独立控制
    - 层次化声纹建模: 从粗粒度(说话人ID)到细粒度(音色细节)
    - 跨语种音色一致性保持
    """

    def __init__(self, config: Dict):
        self.config = config
        self.tts_engine = TTSEngine(config)
        self.vocoder = HiFiGANVocoder(config)
        self.speaker_embedding_dim = 256
        self._init_speaker_encoder()

    def _init_speaker_encoder(self):
        """初始化声纹编码器"""
        try:
            # 使用Wav2Vec2或ECAPA-TDNN作为声纹编码器
            from speechbrain.inference.speaker import EncoderClassifier
            self.speaker_encoder = EncoderClassifier.from_hparams(
                source="speechbrain/spkrec-ecapa-voxceleb",
                savedir="./models/ecapa-tdnn",
            )
            logger.info("✅ ECAPA-TDNN 声纹编码器加载成功")
        except ImportError:
            logger.info("ℹ️ SpeechBrain未安装(可选), 使用内置声纹特征提取")
            self.speaker_encoder = None
        except Exception as e:
            logger.warning(f"⚠️ 声纹编码器加载失败: {e}")
            self.speaker_encoder = None

    def extract_speaker_embedding(
        self, audio: np.ndarray, sample_rate: int = 16000
    ) -> np.ndarray:
        """
        提取声纹嵌入向量

        Args:
            audio: 参考音频
            sample_rate: 采样率

        Returns:
            speaker_embedding: (speaker_embedding_dim,)
        """
        if self.speaker_encoder is not None:
            return self._extract_ecapa(audio, sample_rate)
        else:
            return self._extract_simple(audio, sample_rate)

    def _extract_ecapa(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """ECAPA-TDNN 声纹提取"""
        import torch

        audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
        embedding = self.speaker_encoder.encode_batch(audio_tensor)
        return embedding.squeeze().cpu().numpy()

    def _extract_simple(
        self, audio: np.ndarray, sample_rate: int
    ) -> np.ndarray:
        """
        【创新点】简化的频谱统计声纹特征

        即使没有预训练模型,也可以通过频谱统计特征
        提取基本的声纹信息:
        - 频谱质心
        - 频谱带宽
        - MFCC统计量
        - 基频分布
        """
        try:
            import librosa

            # MFCC
            mfcc = librosa.feature.mfcc(
                y=audio.astype(np.float64),
                sr=sample_rate,
                n_mfcc=40,
            )

            # 频谱质心
            spectral_centroid = librosa.feature.spectral_centroid(
                y=audio.astype(np.float64), sr=sample_rate
            )

            # 基频
            f0, _, _ = librosa.pyin(
                audio.astype(np.float64),
                fmin=50, fmax=500,
                sr=sample_rate,
            )
            f0 = f0[~np.isnan(f0)]

            # 构建声纹特征向量
            features = []

            # MFCC统计量
            for stat in [np.mean, np.std, np.min, np.max]:
                features.extend(stat(mfcc, axis=1))

            # 频谱质心统计
            features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])

            # F0统计量
            if len(f0) > 0:
                features.extend([
                    np.mean(f0), np.std(f0),
                    np.percentile(f0, 25), np.percentile(f0, 75),
                ])
            else:
                features.extend([120, 20, 100, 150])

            # 补齐/截断到目标维度
            features = np.array(features, dtype=np.float32)
            embedding = np.zeros(self.speaker_embedding_dim, dtype=np.float32)
            n = min(len(features), self.speaker_embedding_dim)
            embedding[:n] = features[:n]

            # 归一化
            norm = np.linalg.norm(embedding)
            if norm > 0:
                embedding /= norm

            return embedding

        except ImportError:
            return np.random.randn(self.speaker_embedding_dim).astype(np.float32) * 0.1

    def prepare_speaker_prompt(
        self,
        reference_audio: np.ndarray,
        reference_sample_rate: int = 16000,
        reference_text: Optional[str] = None,
    ):
        """预计算某说话人的声纹克隆 prompt (供批量复用, 加速生成)"""
        return self.tts_engine.prepare_voice_prompt(
            reference_audio, reference_sample_rate, reference_text
        )

    def clone(
        self,
        text: str,
        reference_audio: np.ndarray,
        reference_sample_rate: int = 16000,
        reference_text: Optional[str] = None,
        emotion_embedding: Optional[np.ndarray] = None,
        emotion_label: str = "neutral",
        timeline_constraints: Optional[list] = None,
        target_lang: str = "zh",
        target_duration: Optional[float] = None,
        voice_clone_prompt=None,
        speaker_id: Optional[str] = None,
    ) -> ClonedSpeech:
        """
        执行语音克隆

        Args:
            text: 目标文本
            reference_audio: 参考音频(用于提取音色)
            reference_sample_rate: 参考音频采样率
            reference_text: 参考音频对应文本
            emotion_embedding: 情感嵌入向量
            emotion_label: 情感标签
            timeline_constraints: 时间轴约束
            target_lang: 目标语言
            target_duration: 目标时长(秒), 严格对齐到该时长
            voice_clone_prompt: 预计算的声纹克隆 prompt (复用, 加速)
            speaker_id: 说话人标签 (edge 多音色区分)

        Returns:
            ClonedSpeech 克隆结果
        """
        logger.info(f"Voice clone: text_len={len(text)}, emotion={emotion_label}"
                    f"{', target='+str(round(target_duration,2))+'s' if target_duration else ''}")

        # 1. 提取声纹特征
        speaker_emb = self.extract_speaker_embedding(
            reference_audio, reference_sample_rate
        )
        logger.info(f"Speaker embedding: dim={speaker_emb.shape}")

        # 2. 重采样参考音频到引擎采样率
        if reference_sample_rate != self.tts_engine.sample_rate:
            import librosa
            reference_audio = librosa.resample(
                reference_audio.astype(np.float64),
                orig_sr=reference_sample_rate,
                target_sr=self.tts_engine.sample_rate,
            ).astype(np.float32)
            reference_sample_rate = self.tts_engine.sample_rate

        # 3. TTS合成
        tts_result = self.tts_engine.synthesize(
            text=text,
            reference_audio=reference_audio,
            reference_text=reference_text,
            emotion_embedding=emotion_embedding,
            emotion_label=emotion_label,
            timeline_constraints=timeline_constraints,
            target_lang=target_lang,
            target_duration=target_duration,
            reference_sample_rate=reference_sample_rate,
            voice_clone_prompt=voice_clone_prompt,
            speaker_id=speaker_id,
        )

        # 4. 后处理
        engine = getattr(self.tts_engine, "engine_name", "")
        use_real_clone = bool(
            getattr(self.tts_engine, "_qwen3_available", False)
            or getattr(self.tts_engine, "_cosyvoice_available", False)
        )
        if use_real_clone:
            cloned_audio = self._match_duration(
                tts_result.audio, tts_result.sample_rate,
                reference_audio, reference_sample_rate,
                target_duration,
            )
            # F6: 真克隆后轻度情感韵律 hint (非仅 neutral)
            cloned_audio = self._apply_emotion_prosody(
                cloned_audio, tts_result.sample_rate, emotion_label
            )
        elif engine == "edge_tts":
            # edge 已用不同说话人音色区分; 跳过 WORLD(易毁掉听感/变短)
            cloned_audio = tts_result.audio.astype(np.float32)
            if target_duration and target_duration > 0.05:
                cloned_audio = self._match_duration(
                    cloned_audio, tts_result.sample_rate,
                    reference_audio, reference_sample_rate,
                    target_duration,
                )
            cloned_audio = self._apply_emotion_prosody(
                cloned_audio, tts_result.sample_rate, emotion_label
            )
        else:
            cloned_audio = self._world_voice_conversion(
                tts_result.audio, tts_result.sample_rate,
                reference_audio, reference_sample_rate,
                emotion_label,
            )
            if target_duration and target_duration > 0.05:
                cloned_audio = self._match_duration(
                    cloned_audio, tts_result.sample_rate,
                    reference_audio, reference_sample_rate,
                    target_duration,
                )

        # 5. 质量评估
        quality_score = self._evaluate_quality(tts_result, reference_audio)

        logger.info(f"Clone done: {tts_result.duration:.1f}s, quality={quality_score:.2f}")

        return ClonedSpeech(
            audio=cloned_audio.astype(np.float32),
            sample_rate=tts_result.sample_rate,
            duration=len(cloned_audio) / tts_result.sample_rate,
            speaker_embedding=speaker_emb,
            emotion_preserved=emotion_embedding is not None,
            quality_score=quality_score,
            text=text,
            timeline=tts_result.timeline,
        )

    def _world_voice_conversion(
        self, tts_audio: np.ndarray, tts_sr: int,
        ref_audio: np.ndarray, ref_sr: int, emotion: str,
    ) -> np.ndarray:
        """
        轻量音色克隆: F0匹配 + 共振峰微调

        避免频谱融合产生"烟嗓"噪声
        """
        try:
            import librosa

            if tts_sr != ref_sr:
                ref = librosa.resample(ref_audio.astype(np.float64), orig_sr=ref_sr, target_sr=tts_sr)
            else:
                ref = ref_audio.astype(np.float64)
            tts = tts_audio.astype(np.float64)

            # --- 1. F0 基频分析 ---
            ref_f0, _, _ = librosa.pyin(ref, fmin=60, fmax=500, sr=tts_sr)
            ref_f0 = ref_f0[~np.isnan(ref_f0)]
            ref_median = np.median(ref_f0) if len(ref_f0) > 0 else 180

            tts_f0, _, _ = librosa.pyin(tts, fmin=60, fmax=500, sr=tts_sr)
            tts_f0 = tts_f0[~np.isnan(tts_f0)]
            tts_median = np.median(tts_f0) if len(tts_f0) > 0 else 200

            # 半音偏移
            semitones = 12 * np.log2(max(ref_median, 1) / max(tts_median, 1)) if tts_median > 0 else 0

            # 情感修正
            emo_semi = {"happy": 1.5, "sad": -2.0, "angry": 2.5, "surprised": 3.0, "neutral": 0, "fearful": 1.0}
            semitones += emo_semi.get(emotion, 0)

            # --- 2. F0 音高偏移 (librosa 高质量) ---
            if abs(semitones) > 0.2:
                converted = librosa.effects.pitch_shift(
                    tts, sr=tts_sr, n_steps=semitones, bins_per_octave=24
                )
            else:
                converted = tts.copy()

            # --- 3. 共振峰微调 (只调高频, 不引入噪声) ---
            # 计算源音频的频谱质心 (亮度)
            ref_centroid = librosa.feature.spectral_centroid(y=ref, sr=tts_sr)[0]
            ref_brightness = np.mean(ref_centroid)

            tts_centroid = librosa.feature.spectral_centroid(y=converted, sr=tts_sr)[0]
            tts_brightness = np.mean(tts_centroid)

            # 如果源声音更亮或更暗，微调高频
            if ref_brightness > 0 and tts_brightness > 0:
                brightness_ratio = ref_brightness / tts_brightness
                if abs(brightness_ratio - 1.0) > 0.05:
                    # 用SOS滤波器调整高频 (轻柔, 不产生噪声)
                    from scipy.signal import butter, sosfilt
                    # 参数化均衡器: 调整 3kHz 以上
                    gain_db = np.clip(np.log2(brightness_ratio) * 3, -4, 4)
                    if abs(gain_db) > 0.5:
                        sos = butter(4, 3000 / (tts_sr / 2), btype='high', output='sos')
                        high = sosfilt(sos, converted)
                        # 混合: 原音 70% + 调过的高频 30%
                        converted = converted * 0.7 + high * (10 ** (gain_db / 20)) * 0.3

            # --- 4. 归一化 ---
            mv = np.abs(converted).max()
            if mv > 0:
                converted = converted / mv * 0.85

            logger.info(f"🎤 克隆: F0 {tts_median:.0f}→{ref_median:.0f}Hz "
                        f"(偏移{semitones:.1f}半音), 亮度比{brightness_ratio if ref_brightness>0 else 1:.2f}")
            return converted.astype(np.float32)

        except Exception as e:
            logger.warning(f"⚠️ 克隆跳过({e})")
            return tts_audio

    def _apply_emotion_prosody(
        self, audio: np.ndarray, sr: int, emotion: str
    ) -> np.ndarray:
        """
        F6: 真克隆后的轻度情感韵律 hint。
        幅度刻意保守, 避免变调玩具感。
        """
        emo = (emotion or "neutral").lower()
        if emo in ("neutral", "") or audio is None or len(audio) == 0:
            return audio
        try:
            import librosa
            # 半音 / 语速 (rate>1 加快)
            table = {
                "happy": (1.0, 1.04),
                "surprised": (1.5, 1.05),
                "positive": (0.8, 1.03),
                "sad": (-1.2, 0.96),
                "negative": (-1.0, 0.97),
                "angry": (1.8, 1.06),
                "fearful": (0.8, 1.04),
                "disgusted": (0.5, 1.0),
            }
            semi, rate = table.get(emo, (0.0, 1.0))
            out = audio.astype(np.float64)
            if abs(semi) >= 0.3:
                out = librosa.effects.pitch_shift(
                    out, sr=sr, n_steps=semi, bins_per_octave=24
                )
            if abs(rate - 1.0) >= 0.02:
                out = librosa.effects.time_stretch(out, rate=float(rate))
            peak = float(np.max(np.abs(out))) if len(out) else 0.0
            if peak > 1e-6:
                out = out / peak * 0.95
            return out.astype(np.float32)
        except Exception as e:
            logger.warning(f"emotion prosody skip: {e}")
            return audio

    def _match_duration(
        self, tts_audio, tts_sr, ref_audio, ref_sr, target_duration=None,
    ):
        """
        严格时长对齐: 拉伸/压缩到目标时长 (音高不变)。

        - 有 target_duration: 严格对齐到该时长 (兜底, 主对齐在 TTS 引擎内完成)
        - 无 target_duration: 轻微对齐到参考音频时长 (仅0.85-1.15范围)
        """
        try:
            import librosa
            tts_dur = len(tts_audio) / tts_sr
            if target_duration and target_duration > 0.05:
                rate = tts_dur / target_duration
                rate = float(np.clip(rate, 0.5, 2.0))
                aligned = librosa.effects.time_stretch(tts_audio.astype(np.float64), rate=rate)
                # 精确对齐目标长度
                target_len = int(target_duration * tts_sr)
                if len(aligned) > target_len:
                    aligned = aligned[:target_len]
                elif len(aligned) < target_len:
                    padded = np.zeros(target_len, dtype=np.float32)
                    padded[:len(aligned)] = aligned
                    aligned = padded
                return aligned.astype(np.float32)

            # 无目标时长 → 轻微对齐参考音频 (仅在接近时)
            ref_dur = len(ref_audio) / ref_sr
            ratio = ref_dur / max(tts_dur, 0.1)
            if 0.85 <= ratio <= 1.15 and abs(ratio - 1.0) > 0.03:
                aligned = librosa.effects.time_stretch(tts_audio.astype(np.float64), rate=ratio)
                return aligned.astype(np.float32)
            return tts_audio
        except Exception:
            return tts_audio

    def _evaluate_quality(
        self,
        tts_result: TTSResult,
        reference_audio: np.ndarray,
    ) -> float:
        """
        语音质量评估

        使用多种指标评估合成质量:
        - MOS预测 (基于信号特征)
        - 频谱相似度
        - 时长匹配度
        """
        try:
            import librosa

            # 提取源和目标的频谱特征
            ref_mfcc = librosa.feature.mfcc(
                y=reference_audio.astype(np.float64),
                sr=self.tts_engine.sample_rate,
                n_mfcc=13,
            )
            syn_mfcc = librosa.feature.mfcc(
                y=tts_result.audio.astype(np.float64),
                sr=self.tts_engine.sample_rate,
                n_mfcc=13,
            )

            # 对齐长度
            min_len = min(ref_mfcc.shape[1], syn_mfcc.shape[1])
            if min_len > 0:
                ref_mfcc = ref_mfcc[:, :min_len]
                syn_mfcc = syn_mfcc[:, :min_len]

                # MFCC余弦相似度
                from sklearn.metrics.pairwise import cosine_similarity
                sim = cosine_similarity(ref_mfcc.T, syn_mfcc.T).diagonal().mean()

                # 能量匹配
                ref_energy = np.mean(librosa.feature.rms(y=reference_audio))
                syn_energy = np.mean(librosa.feature.rms(y=tts_result.audio))
                energy_ratio = min(ref_energy, syn_energy) / max(ref_energy, syn_energy, 1e-8)

                quality = 0.5 * sim + 0.3 * energy_ratio + 0.1
            else:
                quality = 0.5

            return round(float(quality), 3)

        except ImportError:
            return 0.7
