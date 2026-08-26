"""
音频处理工具集
"""

import numpy as np
from typing import Dict, Optional, Tuple
from loguru import logger


class AudioUtils:
    """音频处理工具"""

    @staticmethod
    def load_audio(
        path: str,
        target_sr: int = 16000,
        mono: bool = True,
    ) -> Tuple[np.ndarray, int]:
        """
        加载音频文件

        Args:
            path: 音频文件路径
            target_sr: 目标采样率 (None=保持原样)
            mono: 是否转为单声道

        Returns:
            (audio, sample_rate)
        """
        try:
            import librosa
            audio, sr = librosa.load(path, sr=target_sr, mono=mono)
            logger.info(f"Loaded: {path}, sr={sr}, dur={len(audio)/sr:.1f}s")
            return audio, sr
        except ImportError:
            import soundfile as sf
            audio, sr = sf.read(path)
            if target_sr is not None and sr != target_sr:
                import librosa
                audio = librosa.resample(audio.astype(np.float64), orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            if mono and audio.ndim > 1:
                audio = audio.mean(axis=1)
            return audio, sr

    @staticmethod
    def save_audio(
        audio: np.ndarray,
        path: str,
        sample_rate: int = 24000,
    ) -> None:
        """保存音频文件"""
        import soundfile as sf
        sf.write(path, audio, sample_rate)
        logger.info(f"Saved: {path}, sr={sample_rate}, dur={len(audio)/sample_rate:.1f}s")

    @staticmethod
    def normalize(audio: np.ndarray, target_db: float = -3.0) -> np.ndarray:
        """音频响度归一化"""
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(sample_rate)
            loudness = meter.integrated_loudness(audio)
            return pyln.normalize.loudness(audio, loudness, target_db)
        except ImportError:
            # 简单峰值归一化
            max_val = np.abs(audio).max()
            if max_val > 0:
                return audio / max_val * 10 ** (target_db / 20)
            return audio

    @staticmethod
    def trim_silence(
        audio: np.ndarray,
        sample_rate: int = 16000,
        top_db: int = 30,
    ) -> np.ndarray:
        """去除首尾静音"""
        try:
            import librosa
            trimmed, _ = librosa.effects.trim(audio, top_db=top_db)
            return trimmed
        except ImportError:
            return audio

    @staticmethod
    def compute_duration(audio: np.ndarray, sample_rate: int) -> float:
        """计算音频时长"""
        return len(audio) / sample_rate

    @staticmethod
    def mix_audio(
        audio1: np.ndarray,
        audio2: np.ndarray,
        weight1: float = 0.5,
        weight2: float = 0.5,
    ) -> np.ndarray:
        """混音"""
        min_len = min(len(audio1), len(audio2))
        mixed = weight1 * audio1[:min_len] + weight2 * audio2[:min_len]
        max_val = np.abs(mixed).max()
        if max_val > 1.0:
            mixed /= max_val
        return mixed

    @staticmethod
    def estimate_quality(audio: np.ndarray, sample_rate: int) -> Dict:
        """
        音频质量评估
        返回多个质量指标
        """
        result = {}
        try:
            import librosa

            # 信噪比估计
            rms = np.sqrt(np.mean(audio ** 2))
            result["rms"] = round(float(rms), 4)

            # 频谱平坦度
            spec = np.abs(librosa.stft(audio.astype(np.float64)))
            geo_mean = np.exp(np.mean(np.log(spec + 1e-8)))
            arith_mean = np.mean(spec)
            result["spectral_flatness"] = round(float(geo_mean / (arith_mean + 1e-8)), 4)

            # 过零率
            zcr = librosa.feature.zero_crossing_rate(audio)
            result["zero_crossing_rate"] = round(float(np.mean(zcr)), 4)

        except ImportError:
            pass

        return result
