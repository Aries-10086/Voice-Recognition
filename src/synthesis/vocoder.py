"""
声码器模块 - HiFi-GAN 神经声码器
将Mel频谱转换为高质量音频波形
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple
from loguru import logger


class HiFiGANVocoder:
    """
    HiFi-GAN 声码器封装

    HiFi-GAN (2020-2024): 目前最广泛使用的高质量神经声码器
    支持多说话人、高保真音频生成
    """

    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.sample_rate = 24000
        self._load_model()

    def _load_model(self):
        """加载HiFi-GAN模型"""
        try:
            # 尝试加载通用HiFi-GAN模型
            import json
            import os

            # 如果有本地模型,使用本地模型
            model_path = self.config.get("model_dir", "./models")
            config_path = os.path.join(model_path, "hifigan_config.json")
            ckpt_path = os.path.join(model_path, "hifigan_generator.pth")

            if os.path.exists(ckpt_path):
                self.model = self._build_hifigan(config_path)
                self.model.load_state_dict(
                    torch.load(ckpt_path, map_location=self.device)
                )
                self.model.to(self.device)
                self.model.eval()
                logger.info("✅ HiFi-GAN 本地模型加载成功")
            else:
                logger.info("ℹ️ 未找到本地HiFi-GAN(可选), 使用内置Griffin-Lim声码器")
                self.model = None

        except Exception as e:
            logger.warning(f"⚠️ HiFi-GAN 加载失败: {e}")
            self.model = None

    def _build_hifigan(self, config_path: str) -> nn.Module:
        """构建HiFi-GAN生成器"""
        import json

        with open(config_path, 'r') as f:
            h = json.load(f)

        # HiFi-GAN Generator 的标准结构
        # 这里使用简化的ResBlock-based生成器
        class ResBlock(nn.Module):
            def __init__(self, channels, kernel_size=3, dilations=(1, 3, 5)):
                super().__init__()
                self.convs = nn.ModuleList()
                for d in dilations:
                    self.convs.append(nn.Sequential(
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(channels, channels, kernel_size,
                                  dilation=d, padding=d * (kernel_size // 2)),
                        nn.LeakyReLU(0.1),
                        nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2),
                    ))

            def forward(self, x):
                for conv in self.convs:
                    x = x + conv(x)
                return x

        class Generator(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.h = h
                self.num_kernels = len(h.get("resblock_kernel_sizes", [3, 7, 11]))
                self.num_upsamples = len(h.get("upsample_rates", [8, 8, 2, 2]))

                self.conv_pre = nn.Conv1d(80, h.get("upsample_initial_channel", 512), 7, padding=3)

                self.ups = nn.ModuleList()
                self.resblocks = nn.ModuleList()

                for i, (u, k) in enumerate(zip(
                    h.get("upsample_rates", [8, 8, 2, 2]),
                    h.get("upsample_kernel_sizes", [16, 16, 4, 4])
                )):
                    self.ups.append(nn.ConvTranspose1d(
                        h.get("upsample_initial_channel", 512) // (2 ** i),
                        h.get("upsample_initial_channel", 512) // (2 ** (i + 1)),
                        k, u, padding=(k - u) // 2,
                    ))

                    for j in range(len(h.get("resblock_kernel_sizes", [3, 7, 11]))):
                        self.resblocks.append(
                            ResBlock(h.get("upsample_initial_channel", 512) // (2 ** (i + 1)))
                        )

                self.conv_post = nn.Conv1d(
                    h.get("upsample_initial_channel", 512) // (2 ** self.num_upsamples),
                    1, 7, padding=3
                )

            def forward(self, x):
                x = self.conv_pre(x)
                for i in range(self.num_upsamples):
                    x = self.ups[i](x)
                    xs = 0
                    for j in range(self.num_kernels):
                        xs += self.resblocks[i * self.num_kernels + j](x)
                    x = xs / self.num_kernels
                x = torch.tanh(self.conv_post(x))
                return x

        return Generator(h)

    def vocode(self, mel_spectrogram: np.ndarray) -> Tuple[np.ndarray, int]:
        """
        将Mel频谱转换为波形

        Args:
            mel_spectrogram: (n_mels, T) Mel频谱

        Returns:
            (音频波形, 采样率)
        """
        if self.model is not None:
            return self._vocode_hifigan(mel_spectrogram)
        else:
            return self._vocode_griffinlim(mel_spectrogram)

    def _vocode_hifigan(self, mel: np.ndarray) -> Tuple[np.ndarray, int]:
        """HiFi-GAN声码"""
        with torch.no_grad():
            mel_tensor = torch.FloatTensor(mel).unsqueeze(0).to(self.device)
            audio = self.model(mel_tensor)
            audio = audio.squeeze().cpu().numpy()
        return audio, self.sample_rate

    def _vocode_griffinlim(self, mel: np.ndarray) -> Tuple[np.ndarray, int]:
        """Griffin-Lim算法(降级方案)"""
        try:
            import librosa
            audio = librosa.feature.inverse.mel_to_audio(
                mel, sr=self.sample_rate, n_iter=60,
                n_fft=int(self.sample_rate * 0.025),
                hop_length=int(self.sample_rate * 0.01),
            )
            return audio, self.sample_rate
        except ImportError:
            # 最简单降级: 正弦波
            duration = mel.shape[1] * 0.01
            t = np.arange(int(duration * self.sample_rate)) / self.sample_rate
            audio = np.sin(2 * np.pi * 220 * t).astype(np.float32)
            return audio, self.sample_rate
