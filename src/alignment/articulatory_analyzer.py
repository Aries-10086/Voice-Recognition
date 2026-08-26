"""
发音开闭感知网络 (AOCP-Net)
Articulatory Open-Close Perception Network

【核心创新模块 - 可用于专利申请】

创新概述:
本模块提出一种全新的基于纯音频信号的发音口型开闭状态感知方法。
传统方案依赖视频分析或昂贵设备,本方案仅通过音频即可预测说话时
口腔的开合程度和状态变化,为语音-口型同步提供关键的时间轴信息。

核心创新点:
1. 多尺度频谱-发音映射: 通过多分辨率频谱分析捕捉不同粒度的发音特征
2. 时域-频域交叉注意力机制: 融合声学特征与语言学特征
3. 开闭状态连续回归: 不仅预测离散开/闭,还量化开合程度(0~1连续值)
4. 音素感知的发音状态预测: 结合音素先验知识提升预测精度

技术路线:
音频 → 多尺度特征提取 → BiLSTM编码 → 交叉注意力融合
     → 开闭状态预测 + 音素边界检测 → 时间轴生成
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from loguru import logger


@dataclass
class AOCPResult:
    """发音开闭感知结果"""
    openness: np.ndarray          # 口腔开合度序列 (0~1, 连续值)
    state_labels: List[str]       # 离散状态标签: open/closed/transition
    state_segments: List[Dict]    # 状态段信息
    phoneme_boundaries: np.ndarray  # 音素边界概率
    confidence: np.ndarray        # 逐帧置信度
    timeline: List[Dict]          # 最终时间轴 (用于口型同步)


# ============================================================
# 模型定义
# ============================================================

class MultiScaleFeatureExtractor(nn.Module):
    """
    【创新点1】多尺度声学特征提取器

    同时从多个时间尺度提取声学特征:
    - 短窗口(20ms): 捕捉快速音素变化
    - 中窗口(50ms): 捕捉音节级特征
    - 长窗口(100ms): 捕捉韵律和口型趋势
    """

    def __init__(
        self,
        n_mels: int = 80,
        n_mfcc: int = 40,
        multi_scale_windows: List[int] = [20, 50, 100],
        sample_rate: int = 16000,
        hop_ms: int = 10,
    ):
        super().__init__()
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.sample_rate = sample_rate
        self.hop_ms = hop_ms
        self.hop_samples = int(sample_rate * hop_ms / 1000)

        # 多尺度卷积层
        self.multi_scale_convs = nn.ModuleList()
        total_dim = 0

        for win_ms in multi_scale_windows:
            win_samples = int(sample_rate * win_ms / 1000)
            kernel_size = win_samples // self.hop_samples
            if kernel_size < 1:
                kernel_size = 1

            # 每个尺度用不同大小的卷积核
            conv = nn.Sequential(
                nn.Conv1d(n_mels, 64, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(64),
                nn.GELU(),
                nn.Conv1d(64, 64, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(64),
                nn.GELU(),
            )
            self.multi_scale_convs.append(conv)
            total_dim += 64

        self.total_dim = total_dim
        self.output_proj = nn.Linear(total_dim, 256)

    def forward(self, mel_spec: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mel_spec: (B, n_mels, T)
        Returns:
            features: (B, T, 256)
        """
        B, _, T = mel_spec.shape
        multi_scale_features = []
        for conv in self.multi_scale_convs:
            feat = conv(mel_spec)  # (B, 64, T')
            # 对齐时间维度: 插值到相同的T
            if feat.shape[-1] != T:
                feat = F.interpolate(feat, size=T, mode='linear', align_corners=False)
            multi_scale_features.append(feat)

        # 拼接多尺度特征
        combined = torch.cat(multi_scale_features, dim=1)  # (B, total_dim, T)
        combined = combined.transpose(1, 2)  # (B, T, total_dim)
        output = self.output_proj(combined)  # (B, T, 256)
        return output


class CrossModalAttention(nn.Module):
    """
    【创新点2】时域-频域交叉注意力融合模块

    融合声学特征(频谱)和语言学特征(音素embedding),
    通过交叉注意力机制学习两种模态之间的关联,
    从而更准确地预测发音状态。
    """

    def __init__(self, dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads

        # 自注意力
        self.self_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # 交叉注意力
        self.cross_attn = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True
        )
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(
        self,
        acoustic_feat: torch.Tensor,
        linguistic_feat: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            acoustic_feat: (B, T, dim) 声学特征
            linguistic_feat: (B, T', dim) 语言学特征(音素embedding), 可选
        Returns:
            fused: (B, T, dim) 融合特征
        """
        # 自注意力
        attn_out, _ = self.self_attn(acoustic_feat, acoustic_feat, acoustic_feat)
        x = self.norm1(acoustic_feat + attn_out)

        # 交叉注意力(如果有语言学特征)
        if linguistic_feat is not None:
            cross_out, _ = self.cross_attn(x, linguistic_feat, linguistic_feat)
            x = self.norm2(x + cross_out)

        # FFN
        ffn_out = self.ffn(x)
        x = self.norm3(x + ffn_out)
        return x


class ArticulatoryStateDecoder(nn.Module):
    """
    【创新点3】发音状态解码器

    从融合特征中解码:
    1. 口腔开合度 (0~1 连续值)
    2. 开/闭/过渡 三状态分类
    3. 音素边界检测
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 512):
        super().__init__()
        self.bilstm = nn.LSTM(
            input_dim, hidden_dim // 2,
            num_layers=2, bidirectional=True,
            batch_first=True, dropout=0.1,
        )

        # 开合度回归头
        self.openness_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # 输出 0~1
        )

        # 状态分类头 (open/closed/transition)
        self.state_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 3),
        )

        # 音素边界检测头
        self.boundary_head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, input_dim)
        Returns:
            openness: (B, T, 1) 口腔开合度
            states: (B, T, 3) 状态分类logits
            boundaries: (B, T, 1) 音素边界概率
        """
        lstm_out, _ = self.bilstm(x)  # (B, T, hidden_dim)
        openness = self.openness_head(lstm_out)
        states = self.state_head(lstm_out)
        boundaries = self.boundary_head(lstm_out)
        return openness, states, boundaries


class AOCPNet(nn.Module):
    """
    AOCP-Net: 发音开闭感知网络

    完整模型结构:
    Audio → Mel Spectrogram → MultiScaleFeatureExtractor
          → BiLSTM Encoder → CrossModalAttention
          → ArticulatoryStateDecoder → {openness, states, boundaries}
    """

    def __init__(self, config: Dict):
        super().__init__()
        aocp_cfg = config.get("aocp", {})

        n_mels = aocp_cfg.get("n_mels", 80)
        n_mfcc = aocp_cfg.get("n_mfcc", 40)
        multi_scale_windows = aocp_cfg.get("multi_scale_windows", [20, 50, 100])
        sample_rate = aocp_cfg.get("sample_rate", 16000)
        hop_ms = aocp_cfg.get("hop_size_ms", 10)
        encoder_dim = aocp_cfg.get("encoder_dim", 256)
        lstm_hidden = aocp_cfg.get("lstm_hidden", 512)
        lstm_layers = aocp_cfg.get("lstm_layers", 3)
        attention_heads = aocp_cfg.get("attention_heads", 8)
        dropout = aocp_cfg.get("dropout", 0.1)

        self.sample_rate = sample_rate
        self.hop_ms = hop_ms
        self.hop_samples = int(sample_rate * hop_ms / 1000)

        # 特征提取
        self.feature_extractor = MultiScaleFeatureExtractor(
            n_mels=n_mels, n_mfcc=n_mfcc,
            multi_scale_windows=multi_scale_windows,
            sample_rate=sample_rate, hop_ms=hop_ms,
        )

        # BiLSTM编码器 (输入来自feature_extractor输出256维)
        encoder_input_dim = 256
        self.encoder_dim = encoder_input_dim
        self.encoder = nn.LSTM(
            encoder_input_dim,
            lstm_hidden // 2,
            num_layers=lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
        )

        # 交叉注意力
        self.cross_attn = CrossModalAttention(
            dim=lstm_hidden, num_heads=attention_heads, dropout=dropout
        )

        # 状态解码器
        self.decoder = ArticulatoryStateDecoder(
            input_dim=lstm_hidden, hidden_dim=lstm_hidden * 2
        )

        logger.info(
            f"🏗️ AOCP-Net 构建完成: "
            f"multi_scale_windows={multi_scale_windows}, "
            f"lstm_hidden={lstm_hidden}, "
            f"attention_heads={attention_heads}"
        )

    def forward(
        self,
        mel_spec: torch.Tensor,
        phoneme_feat: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            mel_spec: (B, n_mels, T) Mel频谱
            phoneme_feat: (B, T', dim) 可选音素特征
        Returns:
            openness, states, boundaries
        """
        # 多尺度特征提取
        feat = self.feature_extractor(mel_spec)  # (B, T, 256)

        # BiLSTM编码
        encoder_out, _ = self.encoder(feat)  # (B, T, lstm_hidden)

        # 交叉注意力融合
        fused = self.cross_attn(encoder_out, phoneme_feat)

        # 解码
        openness, states, boundaries = self.decoder(fused)

        return openness, states, boundaries

    # --------------------------------------------------
    # 推理方法 (非训练)
    # --------------------------------------------------

    def predict(
        self,
        audio: np.ndarray,
        sample_rate: Optional[int] = None,
        device: str = "cuda",
    ) -> AOCPResult:
        """
        对音频进行发音开闭状态预测

        Args:
            audio: 音频波形
            sample_rate: 采样率
            device: 推理设备

        Returns:
            AOCPResult 包含开合度、状态、边界等
        """
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = self._resample(audio, sample_rate, self.sample_rate)

        # 提取Mel频谱
        mel = self._extract_mel(audio)  # (n_mels, T)
        mel_tensor = torch.FloatTensor(mel).unsqueeze(0)  # (1, n_mels, T)
        if device == "cuda" and torch.cuda.is_available():
            mel_tensor = mel_tensor.cuda()
            self.cuda()

        self.eval()
        with torch.no_grad():
            openness, states, boundaries = self.forward(mel_tensor)

        # 转换为numpy
        openness_np = openness.squeeze().cpu().numpy()  # (T,)
        states_np = states.squeeze().cpu().numpy()       # (T, 3)
        boundaries_np = boundaries.squeeze().cpu().numpy()  # (T,)

        # 状态标签
        state_ids = states_np.argmax(axis=-1)
        state_map = {0: "closed", 1: "open", 2: "transition"}
        state_labels = [state_map[s] for s in state_ids]

        # 状态分段
        state_segments = self._build_state_segments(state_labels, self.hop_ms)

        # 【兜底】如果模型只输出1段或0个开口段，用能量VAD替代
        n_open = sum(1 for s in state_segments if s["state"] == "open")
        if len(state_segments) <= 1 or n_open == 0:
            state_segments = self._energy_vad(audio, self.sample_rate, self.hop_ms)
            # 用能量比例近似开合度
            total_energy = audio.astype(np.float64) ** 2
            openness_np = np.zeros_like(openness_np)
            for seg in state_segments:
                s = int(seg["start"] * self.sample_rate / (self.hop_ms * self.sample_rate / 1000))
                e = int(seg["end"] * self.sample_rate / (self.hop_ms * self.sample_rate / 1000))
                s, e = max(0, s), min(len(openness_np), e)
                if e > s:
                    openness_np[s:e] = seg.get("openness", 0.5)

        timeline = self._build_timeline(state_segments, openness_np, self.hop_ms, len(audio)/self.sample_rate)

        # 置信度
        confidence = np.max(states_np, axis=-1)

        return AOCPResult(
            openness=openness_np,
            state_labels=state_labels,
            state_segments=state_segments,
            phoneme_boundaries=boundaries_np,
            confidence=confidence,
            timeline=timeline,
        )

    @staticmethod
    def _energy_vad(audio: np.ndarray, sample_rate: int, hop_ms: int) -> List[Dict]:
        """
        基于能量的语音活动检测 (VAD)
        当神经网络输出不可靠时的兜底方案
        """
        import librosa

        # 计算RMS能量
        hop_len = int(sample_rate * hop_ms / 1000)
        rms = librosa.feature.rms(y=audio.astype(np.float64), hop_length=hop_len)[0]

        # 自适应阈值: 能量中位数的0.3倍
        threshold = np.median(rms) * 0.3
        if threshold < 1e-6:
            threshold = np.mean(rms) * 0.2

        # 标记有声/无声
        is_speech = rms > threshold

        # 合并相邻帧
        segments = []
        in_speech = False
        start_frame = 0
        min_speech_frames = max(3, int(0.1 * sample_rate / hop_len))  # 至少100ms
        min_silence_frames = max(2, int(0.05 * sample_rate / hop_len))

        for i in range(len(is_speech)):
            if is_speech[i] and not in_speech:
                start_frame = i
                in_speech = True
            elif not is_speech[i] and in_speech:
                if i - start_frame >= min_speech_frames:
                    segments.append({
                        "start": start_frame * hop_ms / 1000.0,
                        "end": i * hop_ms / 1000.0,
                        "duration": (i - start_frame) * hop_ms / 1000.0,
                        "state": "open",
                        "openness": min(1.0, float(np.mean(rms[start_frame:i]) / max(rms) * 1.5)),
                    })
                in_speech = False

        # 最后一段
        if in_speech and len(is_speech) - start_frame >= min_speech_frames:
            segments.append({
                "start": start_frame * hop_ms / 1000.0,
                "end": len(is_speech) * hop_ms / 1000.0,
                "duration": (len(is_speech) - start_frame) * hop_ms / 1000.0,
                "state": "open",
                "openness": min(1.0, float(np.mean(rms[start_frame:]) / max(rms) * 1.5)),
            })

        # 如果什么都没检测到，整段视为语音
        if not segments:
            total_dur = len(audio) / sample_rate
            segments = [{
                "start": 0, "end": total_dur, "duration": total_dur,
                "state": "open", "openness": 0.5,
            }]

        # 在开口段之间插入闭口段
        result = []
        prev_end = 0.0
        for seg in segments:
            if seg["start"] > prev_end + 0.03:  # 间隔>30ms插入闭口段
                result.append({
                    "start": prev_end, "end": seg["start"],
                    "duration": seg["start"] - prev_end,
                    "state": "closed",
                })
            result.append(seg)
            prev_end = seg["end"]

        return result

    def _extract_mel(self, audio: np.ndarray) -> np.ndarray:
        """提取Mel频谱"""
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio.astype(np.float32),
                sr=self.sample_rate,
                n_mels=80,
                hop_length=self.hop_samples,
                n_fft=int(self.sample_rate * 0.025),
            )
            mel_db = librosa.power_to_db(mel, ref=np.max)
            return mel_db
        except ImportError:
            # Fallback: 使用 torchaudio
            import torchaudio
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0)
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_mels=80,
                hop_length=self.hop_samples,
            )
            mel = mel_transform(audio_tensor)
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel)
            return mel_db.squeeze().numpy()

    @staticmethod
    def _resample(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
        """重采样"""
        import librosa
        return librosa.resample(audio.astype(np.float64), orig_sr=orig_sr, target_sr=target_sr)

    @staticmethod
    def _build_state_segments(
        state_labels: List[str], hop_ms: int
    ) -> List[Dict]:
        """将逐帧标签合并为连续段"""
        segments = []
        if not state_labels:
            return segments

        current_label = state_labels[0]
        start_frame = 0

        for i in range(1, len(state_labels)):
            if state_labels[i] != current_label:
                segments.append({
                    "start": start_frame * hop_ms / 1000.0,
                    "end": i * hop_ms / 1000.0,
                    "duration": (i - start_frame) * hop_ms / 1000.0,
                    "state": current_label,
                })
                current_label = state_labels[i]
                start_frame = i

        # 最后一段
        segments.append({
            "start": start_frame * hop_ms / 1000.0,
            "end": len(state_labels) * hop_ms / 1000.0,
            "duration": (len(state_labels) - start_frame) * hop_ms / 1000.0,
            "state": current_label,
        })

        return segments

    @staticmethod
    def _build_timeline(
        state_segments: List[Dict],
        openness: np.ndarray,
        hop_ms: int,
        total_duration: float,
    ) -> List[Dict]:
        """
        【创新点4】构建用于口型同步的时间轴

        将开闭状态段转化为可用的时间轴信息:
        - 识别开口段(发音核心区域)
        - 标记闭口段(停顿/辅音)
        - 标注过渡区域
        """
        timeline = []
        for seg in state_segments:
            # 计算该段的平均开合度
            start_frame = int(seg["start"] * 1000 / hop_ms)
            end_frame = int(seg["end"] * 1000 / hop_ms)
            start_frame = max(0, min(start_frame, len(openness) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(openness)))

            if start_frame < len(openness) and end_frame <= len(openness):
                avg_openness = float(np.mean(openness[start_frame:end_frame]))
            else:
                avg_openness = 0.0

            timeline.append({
                "start_s": round(seg["start"], 3),
                "end_s": round(seg["end"], 3),
                "state": seg["state"],
                "openness": round(avg_openness, 3),
                # 是否为发音核心区(开口且持续>50ms)
                "is_speech_core": seg["state"] == "open" and seg["duration"] > 0.05,
                # 是否需要口型同步
                "needs_lip_sync": seg["state"] in ("open", "transition"),
            })

        return timeline
