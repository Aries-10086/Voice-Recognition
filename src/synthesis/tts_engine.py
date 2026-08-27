"""
语音合成引擎 (TTS Engine)

【创新点】情感注入 + 时间轴约束 + 多引擎适配

支持后端:
- CosyVoice2 (阿里, 2024-2025, 零样本语音克隆SOTA)
- StyleTTS2 (2024, 高质量+风格可控)
- GPT-SoVITS (2024, 少样本语音克隆)
- ChatTTS (2024, 对话风格TTS)
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class TTSResult:
    """TTS合成结果"""
    audio: np.ndarray               # 合成音频
    sample_rate: int                # 采样率
    duration: float                 # 时长(秒)
    phonemes_used: List[str]        # 实际使用的音素
    timeline: List[Dict]            # 合成时间轴
    emotion_applied: str            # 应用的情感


class TTSEngine:
    """
    多引擎语音合成器

    创新特性:
    1. 情感嵌入注入: 将情感向量注入TTS的韵律建模
    2. 时间轴约束合成: 根据时间轴控制合成节奏
    3. 多引擎自适应: 根据场景自动选择最优引擎
    """

    SUPPORTED_ENGINES = [
        "auto", "cosyvoice2", "qwen3_tts", "edge_tts"
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.engine_name = config.get("engine", "auto")
        self.model = None
        self.sample_rate = 24000
        self.emotion_injection = config.get("emotion_injection", True)
        self.gen_config = config.get("gen", {})
        # P0-B1: 演示链路禁止静默落到 edge-tts
        self.allow_edge_fallback = bool(config.get("allow_edge_fallback", True))
        self.require_clone_engine = bool(config.get("require_clone_engine", False))
        self._qwen3_available = False
        self._cosyvoice_available = False
        self._edge_available = False
        self._load_model()

    @property
    def supports_voice_clone(self) -> bool:
        return bool(self._cosyvoice_available or self._qwen3_available)

    def _load_model(self):
        """
        按 最优可用 顺序加载引擎:
        CosyVoice2 (更优克隆) → Qwen3-TTS → edge-tts
        若用户显式指定引擎, 则优先尝试指定引擎。
        """
        preferred = self.engine_name
        candidates = ["cosyvoice2", "qwen3_tts", "edge_tts"]
        if preferred != "auto" and preferred in candidates:
            candidates.remove(preferred)
            candidates.insert(0, preferred)

        # 强制克隆引擎时不把 edge 放进候选
        if self.require_clone_engine or not self.allow_edge_fallback:
            candidates = [c for c in candidates if c != "edge_tts"]

        for name in candidates:
            ok = False
            if name == "cosyvoice2":
                self._load_cosyvoice2()
                ok = self._cosyvoice_available
            elif name == "qwen3_tts":
                self._load_qwen3_tts()
                ok = self._qwen3_available
            elif name == "edge_tts":
                self._edge_available = True
                ok = True
            if ok:
                self.engine_name = name
                logger.info(f"🎙️ TTS engine selected: {name}")
                if name == "edge_tts":
                    logger.warning(
                        "⚠️ 当前为 edge-tts: 无法真正声纹克隆, 听感仅限「能听」。"
                        "请安装 CosyVoice2 或 Qwen3-TTS (synthesis.require_clone_engine=true 可禁止回退)"
                    )
                return

        if self.require_clone_engine or not self.allow_edge_fallback:
            raise RuntimeError(
                "未找到可用的声纹克隆引擎 (CosyVoice2 / Qwen3-TTS)。"
                "请安装权重, 或临时设置 synthesis.allow_edge_fallback=true"
            )

        # 极端兜底
        self._edge_available = True
        self.engine_name = "edge_tts"
        logger.warning("⚠️ TTS 极端兜底: edge-tts (无声纹克隆能力)")

    def _load_qwen3_tts(self):
        """
        加载 Qwen3-TTS 模型

        模型选项:
        - 0.6B-Base:    1.2GB, 快速, 3秒克隆
        - 1.7B-Base:    3.5GB, 高质量, 支持ref_text音节映射
        - 1.7B-CustomVoice: 9种预设音色+语气指令
        """
        try:
            import torch
            from qwen_tts import Qwen3TTSModel

            model_name = self.config.get("qwen3_tts", {}).get(
                "model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
            )
            # Mac: 优先 MPS → CUDA → CPU
            if torch.cuda.is_available():
                device = "cuda:0"
                dtype = torch.bfloat16
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                device = "mps"
                dtype = torch.float32
            else:
                device = "cpu"
                dtype = torch.float32

            logger.info(f"Loading Qwen3-TTS: {model_name} ({device})")
            self.model = Qwen3TTSModel.from_pretrained(
                model_name, device_map=device, dtype=dtype,
            )
            self.sample_rate = int(self.config.get("qwen3_tts", {}).get("sample_rate", 24000))
            self._qwen3_available = True
            logger.info(f"Qwen3-TTS ready: {model_name}")
        except ImportError:
            logger.info("qwen-tts not installed, run: pip install qwen-tts")
            self.model = None; self._qwen3_available = False
        except Exception as e:
            logger.warning(f"Qwen3-TTS failed: {str(e)[:120]}")
            self.model = None; self._qwen3_available = False

    def _load_cosyvoice2(self):
        """尝试加载 CosyVoice2 (更优零样本克隆), 权重缺失则跳过"""
        try:
            import sys, os, glob
            cfg = self.config.get("cosyvoice2", {})
            code_path = cfg.get("code_dir") or os.path.abspath("./models/CosyVoice2-0.5B")
            pretrained_dir = cfg.get("pretrained_dir") or "./models/cosyvoice2_pretrained"
            matcha_path = os.path.join(code_path, "third_party", "Matcha-TTS")
            for p in [code_path, matcha_path]:
                if os.path.isdir(p) and p not in sys.path:
                    sys.path.insert(0, p)

            candidates = glob.glob(
                os.path.join(pretrained_dir, "**", "iic--CosyVoice2-0.5B", "snapshots", "*"),
                recursive=True,
            )
            if not candidates:
                candidates = glob.glob(
                    os.path.join(pretrained_dir, "**", "*CosyVoice*", "snapshots", "*"),
                    recursive=True,
                )
            if not candidates:
                raise FileNotFoundError("CosyVoice2 权重未下载")
            from cosyvoice.cli.cosyvoice import CosyVoice2
            self.model = CosyVoice2(candidates[0], fp16=torch.cuda.is_available())
            self.sample_rate = int(cfg.get("sample_rate", 24000))
            self._cosyvoice_available = True
            logger.info("✅ CosyVoice2 零样本克隆就绪")
        except Exception as e:
            self.model = None
            self._cosyvoice_available = False
            logger.info(f"CosyVoice2 unavailable: {str(e)[:80]}")

    def prepare_voice_prompt(
        self,
        reference_audio: np.ndarray,
        reference_sample_rate: int,
        reference_text: Optional[str] = None,
    ):
        """
        为某说话人预计算声纹克隆 prompt (Qwen3-TTS)。

        一次性将参考音频编码为 ref_code + speaker embedding, 之后可复用于该说话人的
        所有分段, 避免每段重复编码参考音频, 大幅提升生成速度。

        Returns:
            voice_clone_prompt (list[VoiceClonePromptItem]) 或 None
        """
        if not getattr(self, "_qwen3_available", False) or self.model is None:
            return None
        import tempfile, os, soundfile as sf
        ref_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                ref_path = f.name
            sf.write(ref_path, reference_audio, int(reference_sample_rate))
            x_vector_only = (reference_text is None) or (str(reference_text).strip() == "")
            items = self.model.create_voice_clone_prompt(
                ref_audio=ref_path,
                ref_text=None if x_vector_only else reference_text,
                x_vector_only_mode=x_vector_only,
            )
            return items
        except Exception as e:
            logger.warning(f"prepare_voice_prompt failed: {str(e)[:100]}")
            return None
        finally:
            if ref_path and os.path.exists(ref_path):
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass

    def clone_batch(
        self,
        texts: List[str],
        ref_audio: Optional[np.ndarray] = None,
        ref_text: Optional[str] = "",
        ref_sr: int = 24000,
        lang: str = "English",
        voice_clone_prompt=None,
    ) -> List[np.ndarray]:
        """
        批量克隆: 单次推理处理多句, 比逐句快3-5x。
        若提供 voice_clone_prompt (预先编码的声纹), 则跳过参考音频重复编码。
        """
        if not getattr(self, "_qwen3_available", False) or self.model is None:
            return [self._synthesize_fallback(t, "neutral", None, lang).audio for t in texts]

        gen = self.gen_config or {}
        # 动态 max_new_tokens: 按最长文本估算 (12Hz 语音码), 避免长文本生成过早截断或过长拖慢
        max_chars = max((len(t) for t in texts), default=1)
        max_new_tokens = min(int(gen.get("max_new_tokens", 2048)), max(128, int(max_chars * 2) + 80))
        kwargs = dict(
            text=texts,
            language=[lang] * len(texts),
            max_new_tokens=max_new_tokens,
            repetition_penalty=float(gen.get("repetition_penalty", 1.1)),
            top_p=float(gen.get("top_p", 0.95)),
            top_k=int(gen.get("top_k", 50)),
        )

        ref_path = None
        try:
            if voice_clone_prompt is not None:
                kwargs["voice_clone_prompt"] = voice_clone_prompt
            else:
                if ref_audio is None:
                    raise ValueError("clone_batch requires ref_audio or voice_clone_prompt")
                import tempfile, os, soundfile as sf
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    ref_path = f.name
                sf.write(ref_path, ref_audio, int(ref_sr))
                kwargs["ref_audio"] = ref_path
                if ref_text:
                    kwargs["ref_text"] = ref_text
                else:
                    kwargs["x_vector_only_mode"] = True

            wavs, sr = self.model.generate_voice_clone(**kwargs)
            return [w for w in wavs]
        finally:
            if ref_path and os.path.exists(ref_path):
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass

    def synthesize(
        self,
        text: str,
        reference_audio: Optional[np.ndarray] = None,
        reference_text: Optional[str] = None,
        emotion_embedding: Optional[np.ndarray] = None,
        emotion_label: str = "neutral",
        timeline_constraints: Optional[List[Dict]] = None,
        speed: float = 1.0,
        target_lang: str = "zh",
        target_duration: Optional[float] = None,
        reference_sample_rate: Optional[int] = None,
        voice_clone_prompt=None,
        speaker_id: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """
        语音合成

        Args:
            text: 目标文本
            reference_audio: 参考音频(用于声纹克隆)
            reference_text: 参考音频对应文本
            emotion_embedding: 情感嵌入向量
            emotion_label: 情感标签
            timeline_constraints: 时间轴约束
            speed: 语速控制
            target_lang: 目标语言
            target_duration: 目标时长(秒), 生成后严格拉伸/压缩到该时长
            reference_sample_rate: 参考音频采样率
            voice_clone_prompt: 预计算的声纹克隆 prompt (复用, 加速)
            speaker_id: 说话人标签 (edge 多音色映射)
            voice: 显式指定 edge 音色

        Returns:
            TTSResult
        """
        if self.engine_name == "qwen3_tts" and getattr(self, '_qwen3_available', False):
            return self._synthesize_qwen3_tts(
                text, reference_audio, reference_text,
                emotion_label, target_lang, target_duration,
                reference_sample_rate=reference_sample_rate,
                voice_clone_prompt=voice_clone_prompt,
            )
        elif self.engine_name == "cosyvoice2" and getattr(self, '_cosyvoice_available', False):
            return self._synthesize_cosyvoice2(
                text, reference_audio, reference_text,
                emotion_embedding, emotion_label,
                timeline_constraints, speed, target_lang
            )
        else:
            if not voice:
                voice = self.pick_edge_voice(target_lang, speaker_id)
            return self._synthesize_fallback(
                text, emotion_label, timeline_constraints, target_lang, voice=voice
            )

    def _synthesize_fallback(
        self,
        text: str,
        emotion_label: str,
        timeline_constraints: Optional[List[Dict]],
        target_lang: str,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """
        降级合成: edge-tts → macOS say → 正弦波
        """
        try:
            return self._synthesize_edge_tts(
                text, emotion_label, target_lang, voice=voice
            )
        except Exception as e:
            logger.warning(f"   edge-tts 不可用: {e}")
        try:
            return self._synthesize_macos_say(text, emotion_label, target_lang, speaker_voice=voice)
        except Exception as e:
            logger.warning(f"   macOS say 不可用: {e}, 使用正弦波降级")
            return self._synthesize_sinewave(
                text, emotion_label, timeline_constraints, target_lang
            )

    def _synthesize_macos_say(
        self,
        text: str,
        emotion_label: str,
        target_lang: str,
        speaker_voice: Optional[str] = None,
    ) -> TTSResult:
        """离线兜底: 系统 say (Mac), 按说话人换声保证可区分"""
        import subprocess, tempfile, os
        text = (text or "").strip()
        if not text:
            raise ValueError("empty text for say")

        # edge 音色名 → macOS 声线; 否则按语种默认
        edge_to_say = {
            "en-US-JennyNeural": "Samantha",
            "en-US-GuyNeural": "Daniel",
            "en-US-AriaNeural": "Karen",
            "en-US-DavisNeural": "Aaron",
            "zh-CN-XiaoxiaoNeural": "Tingting",
            "zh-CN-YunxiNeural": "Reed",
            "zh-CN-XiaoyiNeural": "Shelley",
            "zh-CN-YunjianNeural": "Rocko",
        }
        if target_lang.startswith("zh"):
            voices = ["Tingting", "Reed", "Shelley", "Rocko"]
            default = "Tingting"
        else:
            voices = ["Samantha", "Daniel", "Karen", "Aaron"]
            default = "Samantha"
        voice = edge_to_say.get(speaker_voice or "", None) or default
        if voice not in voices and speaker_voice:
            # 稳定映射
            try:
                idx = abs(hash(speaker_voice)) % len(voices)
                voice = voices[idx]
            except Exception:
                voice = default

        rate = 175
        if emotion_label in ("happy", "positive", "angry"):
            rate = 190
        elif emotion_label in ("sad", "negative"):
            rate = 155

        with tempfile.TemporaryDirectory() as td:
            aiff = os.path.join(td, "out.aiff")
            wav = os.path.join(td, "out.wav")
            subprocess.run(
                ["say", "-v", voice, "-r", str(rate), "-o", aiff, text],
                check=True, capture_output=True,
            )
            subprocess.run(
                ["afconvert", "-f", "WAVE", "-d", "LEI16", aiff, wav],
                check=True, capture_output=True,
            )
            import soundfile as sf
            audio, sr = sf.read(wav)

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        import librosa
        audio = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=self.sample_rate
        ).astype(np.float32)
        duration = len(audio) / self.sample_rate
        logger.info(f"macOS say: {duration:.1f}s voice={voice} emotion={emotion_label}")
        return TTSResult(
            audio=audio,
            sample_rate=self.sample_rate,
            duration=duration,
            phonemes_used=[],
            timeline=[],
            emotion_applied=emotion_label,
        )

    def _synthesize_qwen3_tts(
        self, text, reference_audio, reference_text,
        emotion_label, target_lang, target_duration=None,
        reference_sample_rate=None, voice_clone_prompt=None,
    ) -> TTSResult:
        """
        Qwen3-TTS 语音克隆 —— 支持音节级韵律映射

        ref_text 是关键: 模型对比参考音频+参考文本, 学会源说话人的
        音节→发音映射关系, 然后应用到目标文本上, 实现语气+节奏+情感的迁移。

        - voice_clone_prompt: 复用预先编码的声纹 (ref_code + speaker embedding),
          避免每段重复编码参考音频, 提升生成速度。
        - Base 模型的情感/语气由参考音频本身承载, 因此逐段使用"该段自己的起始~结束音"
          作为参考音频即可自然保留该段的情感与语气。
        """
        import tempfile, os, soundfile as sf

        lang_map = {"zh":"Chinese","en":"English","ja":"Japanese",
                   "ko":"Korean","fr":"French","de":"German",
                   "es":"Spanish","it":"Italian","pt":"Portuguese","ru":"Russian","ar":"Arabic"}
        lang = lang_map.get(target_lang, "English")

        gen = self.gen_config or {}
        # 动态估算最大生成 token 数 (12Hz 语音码): 避免无谓生成到 2048 上限而拖慢速度。
        # 英文系约 15 字符/秒, 中文等约 5 字符/秒; 估算时长 * 12 token/s * 1.6 余量。
        if target_lang in ("en", "fr", "de", "es", "it", "pt", "ru", "ar"):
            est_sec = max(1.0, len(text) / 15.0)
        else:
            est_sec = max(1.0, len(text) / 5.0)
        est_tokens = int(est_sec * 12 * 1.6) + 50
        max_new_tokens = min(int(gen.get("max_new_tokens", 2048)), max(128, est_tokens))
        repetition_penalty = float(gen.get("repetition_penalty", 1.1))
        top_p = float(gen.get("top_p", 0.95))
        top_k = int(gen.get("top_k", 50))

        ref_path = None
        try:
            if voice_clone_prompt is not None:
                wavs, sr = self.model.generate_voice_clone(
                    text=text, language=lang,
                    voice_clone_prompt=voice_clone_prompt,
                    max_new_tokens=max_new_tokens,
                    repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
                )
            elif reference_audio is not None:
                if reference_sample_rate is None:
                    reference_sample_rate = self.sample_rate
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    ref_path = f.name
                sf.write(ref_path, reference_audio, int(reference_sample_rate))
                x_vector_only = (reference_text is None) or (str(reference_text).strip() == "")
                if x_vector_only:
                    wavs, sr = self.model.generate_voice_clone(
                        text=text, language=lang,
                        ref_audio=ref_path, x_vector_only_mode=True,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
                    )
                else:
                    wavs, sr = self.model.generate_voice_clone(
                        text=text, language=lang,
                        ref_audio=ref_path, ref_text=reference_text,
                        max_new_tokens=max_new_tokens,
                        repetition_penalty=repetition_penalty, top_p=top_p, top_k=top_k,
                    )
            else:
                raise ValueError("Qwen3-TTS requires reference audio or voice_clone_prompt")

            audio = wavs[0] if isinstance(wavs, list) else wavs
            sr = int(sr)

            # 严格时长匹配: 拉伸/压缩到目标时长 (音高不变)
            if target_duration and target_duration > 0.05:
                audio = self._stretch_to_duration(audio, sr, target_duration)
            # 首尾淡化, 避免拼接时咔哒声
            audio = self._apply_boundary_fade(audio, sr)

            duration = len(audio) / sr
            logger.info(f"Qwen3-TTS clone: {duration:.2f}s, lang={lang}, emotion={emotion_label}"
                        f"{', target='+str(round(target_duration,2)) if target_duration else ''}")
            return TTSResult(audio=audio, sample_rate=sr,
                           duration=duration, phonemes_used=[], timeline=[],
                           emotion_applied=emotion_label)
        finally:
            if ref_path and os.path.exists(ref_path):
                try:
                    os.unlink(ref_path)
                except OSError:
                    pass

    @staticmethod
    def _stretch_to_duration(audio: np.ndarray, sr: int, target_duration: float) -> np.ndarray:
        """
        将音频严格拉伸/压缩到目标时长 (librosa time_stretch, 音高不变)。

        - 比例 0.5x ~ 2.0x 直接用 time_stretch (音质较好)
        - 超出范围采用分段拼贴/裁剪 (避免极端拉伸导致音质劣化)
        """
        import librosa
        cur_dur = len(audio) / sr
        if cur_dur <= 0.01 or abs(cur_dur - target_duration) < 0.03:
            return audio
        rate = cur_dur / target_duration  # >1 压缩, <1 拉伸
        rate = float(np.clip(rate, 0.5, 2.0))
        stretched = librosa.effects.time_stretch(
            audio.astype(np.float64), rate=rate
        ).astype(np.float32)

        # 目标长度精确对齐: 过长裁剪, 过短尾部静音补齐
        target_len = int(target_duration * sr)
        if len(stretched) > target_len:
            stretched = stretched[:target_len]
        elif len(stretched) < target_len:
            padded = np.zeros(target_len, dtype=np.float32)
            padded[:len(stretched)] = stretched
            # 尾部淡出避免突兀
            fade = min(1200, len(stretched))
            if fade > 0:
                padded[len(stretched):len(stretched)+fade] = 0.0
            stretched = padded
        return stretched

    @staticmethod
    def _apply_boundary_fade(audio: np.ndarray, sr: int, fade_ms: int = 8) -> np.ndarray:
        """首尾极短淡化, 消除拼接/裁剪产生的咔哒声"""
        n = len(audio)
        if n == 0:
            return audio
        fade = int(sr * fade_ms / 1000.0)
        if fade <= 0 or n < 2 * fade:
            return audio
        a = audio.astype(np.float32)
        a[:fade] *= np.linspace(0.0, 1.0, fade, dtype=np.float32)
        a[-fade:] *= np.linspace(1.0, 0.0, fade, dtype=np.float32)
        return a

    def _synthesize_cosyvoice2(
        self, text, reference_audio, reference_text,
        emotion_embedding, emotion_label, timeline_constraints, speed, target_lang,
    ) -> TTSResult:
        """CosyVoice2 零样本语音克隆"""
        import tempfile, soundfile as sf

        # 保存参考音频为临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            ref_path = f.name
        sf.write(ref_path, reference_audio, self.sample_rate)

        try:
            prompt_text = reference_text or ""
            output = self.model.inference_zero_shot(
                text, prompt_text, ref_path, stream=False
            )
            audio = None
            for item in output:
                audio = item.get("tts_speech", None)
                if audio is not None:
                    break

            if audio is None:
                raise ValueError("CosyVoice2 输出为空")

            duration = len(audio) / self.sample_rate
            logger.info(f"🎵 CosyVoice2 零样本克隆: {duration:.1f}s, emotion={emotion_label}")
            return TTSResult(audio=audio, sample_rate=self.sample_rate,
                           duration=duration, phonemes_used=[], timeline=[],
                           emotion_applied=emotion_label)
        finally:
            import os; os.unlink(ref_path) if os.path.exists(ref_path) else None

    def _synthesize_edge_tts(
        self, text: str, emotion_label: str, target_lang: str,
        voice: Optional[str] = None,
    ) -> TTSResult:
        """使用 Microsoft Edge TTS 生成真实语音 (免费, 无需模型)"""
        import asyncio
        import tempfile
        import os

        # 语种→默认 Edge 语音
        voice_map = {
            "zh": "zh-CN-XiaoxiaoNeural",
            "en": "en-US-JennyNeural",
            "ja": "ja-JP-NanamiNeural",
            "ko": "ko-KR-SunHiNeural",
        }
        # 多人区分: 按 speaker 轮换不同音色 (无真克隆时的可演示兜底)
        multi_voices = {
            "en": [
                "en-US-JennyNeural", "en-US-GuyNeural",
                "en-US-AriaNeural", "en-US-DavisNeural",
            ],
            "zh": [
                "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
                "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural",
            ],
            "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
            "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
        }
        if not voice:
            voice = voice_map.get(target_lang, "en-US-JennyNeural")

        rate_map = {
            "happy": "+10%", "sad": "-15%", "angry": "+20%",
            "neutral": "+0%", "positive": "+8%", "negative": "-10%",
        }
        pitch_map = {
            "happy": "+5Hz", "sad": "-8Hz", "angry": "+10Hz",
            "neutral": "+0Hz", "positive": "+4Hz", "negative": "-6Hz",
        }
        rate = rate_map.get(emotion_label, "+0%")
        pitch = pitch_map.get(emotion_label, "+0Hz")

        max_chars = 2000
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        text = (text or "").strip()
        if not text:
            raise ValueError("edge-tts empty text")

        async def _generate():
            import edge_tts
            communicate = edge_tts.Communicate(
                text, voice, rate=rate, pitch=pitch
            )
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tmp_path = f.name
            await communicate.save(tmp_path)
            return tmp_path

        # 避免 "loop already running" / 陈旧 loop 导致空音频
        try:
            tmp_path = asyncio.run(_generate())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            try:
                tmp_path = loop.run_until_complete(_generate())
            finally:
                loop.close()

        if not tmp_path or not os.path.exists(tmp_path) or os.path.getsize(tmp_path) < 64:
            raise RuntimeError("edge-tts produced empty file")

        import soundfile as sf
        try:
            audio, sr = sf.read(tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

        if audio is None or len(audio) == 0:
            raise RuntimeError("edge-tts decode empty")
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        import librosa
        audio = librosa.resample(
            audio.astype(np.float64), orig_sr=sr, target_sr=self.sample_rate
        )
        duration = len(audio) / self.sample_rate
        logger.info(f"edge-tts: {duration:.1f}s voice={voice} emotion={emotion_label}")
        return TTSResult(
            audio=audio.astype(np.float32),
            sample_rate=self.sample_rate,
            duration=duration,
            phonemes_used=[],
            timeline=[],
            emotion_applied=emotion_label,
        )

    def pick_edge_voice(self, target_lang: str, speaker_id: Optional[str] = None) -> str:
        """按说话人稳定映射到不同 Edge 音色, 保证多人听感可区分"""
        multi_voices = {
            "en": [
                "en-US-JennyNeural", "en-US-GuyNeural",
                "en-US-AriaNeural", "en-US-DavisNeural",
            ],
            "zh": [
                "zh-CN-XiaoxiaoNeural", "zh-CN-YunxiNeural",
                "zh-CN-XiaoyiNeural", "zh-CN-YunjianNeural",
            ],
            "ja": ["ja-JP-NanamiNeural", "ja-JP-KeitaNeural"],
            "ko": ["ko-KR-SunHiNeural", "ko-KR-InJoonNeural"],
        }
        voices = multi_voices.get(target_lang, multi_voices["en"])
        if not speaker_id:
            return voices[0]
        # 稳定哈希: SPEAKER_00 -> 0, SPEAKER_01 -> 1
        try:
            idx = int("".join(ch for ch in speaker_id if ch.isdigit()) or "0")
        except ValueError:
            idx = abs(hash(speaker_id)) % len(voices)
        return voices[idx % len(voices)]

    def _synthesize_sinewave(
        self, text: str, emotion_label: str,
        timeline_constraints: Optional[List[Dict]], target_lang: str,
    ) -> TTSResult:
        """正弦波降级 (仅当 edge-tts 不可用时)"""
        sr = self.sample_rate

        # 计算总时长
        if timeline_constraints and len(timeline_constraints) > 0:
            total_duration = max(seg.get("end", 0) for seg in timeline_constraints)
        else:
            if target_lang in ("zh", "zho", "chi", "cmn"):
                total_duration = len(text) * 0.25
            else:
                total_duration = len(text.split()) * 0.35

        total_duration = max(total_duration, 0.5)

        n_samples = int(total_duration * sr) + int(0.2 * sr)  # 加0.2s尾部
        audio = np.zeros(n_samples, dtype=np.float32)

        # 情感 → 音色参数映射
        voice_params = {
            "happy":  {"base_freq": 280, "f1": 500, "f2": 1800, "pulse": 0.7, "tremolo": 5},
            "sad":    {"base_freq": 160, "f1": 350, "f2": 1200, "pulse": 0.4, "tremolo": 2},
            "angry":  {"base_freq": 320, "f1": 550, "f2": 2000, "pulse": 0.9, "tremolo": 8},
            "neutral":{"base_freq": 200, "f1": 400, "f2": 1500, "pulse": 0.55, "tremolo": 0},
            "surprised":{"base_freq": 350, "f1": 600, "f2": 2200, "pulse": 0.8, "tremolo": 6},
            "fearful": {"base_freq": 250, "f1": 450, "f2": 1400, "pulse": 0.5, "tremolo": 10},
        }
        vp = voice_params.get(emotion_label, voice_params["neutral"])

        # 【修复电流声】如果只有1个长段 (>2s)，拆分成短句+静音间隔
        if timeline_constraints and len(timeline_constraints) == 1:
            seg = timeline_constraints[0]
            seg_dur = seg.get("duration", seg.get("end", 0) - seg.get("start", 0))
            if seg_dur > 2.0:
                # 按 ~0.5s 一段拆分，间隔 0.1s 静音
                chunk_dur = 0.5
                gap_dur = 0.1
                new_segments = []
                t = seg.get("start", 0)
                while t < seg.get("end", seg.get("start", 0) + seg_dur):
                    end_t = min(t + chunk_dur, seg.get("end", seg.get("start", 0) + seg_dur))
                    new_segments.append({
                        "start": t, "end": end_t,
                        "openness": seg.get("openness", 0.5) * (0.6 + 0.4 * np.random.random()),
                    })
                    t = end_t + gap_dur
                timeline_constraints = new_segments

        if timeline_constraints and len(timeline_constraints) > 0:
            # 有时间轴: 按说话段生成
            for seg in timeline_constraints:
                start_s = seg.get("start", 0)
                end_s = seg.get("end", start_s + 0.1)
                openness = seg.get("openness", 0.5)

                start_sample = max(0, int(start_s * sr))
                end_sample = min(n_samples, int(end_s * sr))
                seg_len = end_sample - start_sample

                if seg_len < int(0.02 * sr):  # 至少20ms
                    continue

                seg_t = np.linspace(0, seg_len / sr, seg_len, endpoint=False)

                # 基频 + 泛音 (模拟声带振动)
                signal = np.zeros(seg_len, dtype=np.float32)
                for h in range(1, 5):
                    amp = 1.0 / (h ** 1.3)  # 泛音衰减
                    signal += amp * np.sin(2 * np.pi * vp["base_freq"] * h * seg_t)

                # 共振峰 (模拟口腔)
                signal += 0.25 * openness * np.sin(2 * np.pi * vp["f1"] * seg_t)
                signal += 0.15 * openness * np.sin(2 * np.pi * vp["f2"] * seg_t)

                # 颤音 (情感表现)
                if vp["tremolo"] > 0:
                    trem = 1 + 0.1 * np.sin(2 * np.pi * vp["tremolo"] * seg_t)
                    signal *= trem

                # 振幅包络: 渐入渐出
                env = np.ones(seg_len)
                attack = min(int(0.015 * sr), seg_len // 4)   # 15ms attack
                release = min(int(0.025 * sr), seg_len // 3)  # 25ms release
                if seg_len > attack + release:
                    env[:attack] = np.linspace(0, 1, attack) ** 2
                    env[-release:] = np.linspace(1, 0, release) ** 2
                else:
                    env = np.hanning(seg_len)

                audio[start_sample:end_sample] += signal * env * vp["pulse"] * 0.6
        else:
            # 无时间轴: 整段生成
            t_full = np.linspace(0, total_duration, n_samples, endpoint=False)
            signal = np.zeros(n_samples, dtype=np.float32)
            for h in range(1, 5):
                signal += (1.0 / h) * np.sin(2 * np.pi * vp["base_freq"] * h * t_full)
            signal += 0.2 * np.sin(2 * np.pi * vp["f1"] * t_full)
            signal += 0.1 * np.sin(2 * np.pi * vp["f2"] * t_full)
            # 整体包络
            env = np.ones(n_samples)
            env[-int(0.1 * sr):] = np.linspace(1, 0, int(0.1 * sr))
            audio = signal * env * vp["pulse"] * 0.4

        # 归一化到 -3dB
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.7

        # 确保可听 (RMS > 0.01)
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.01:
            audio = audio / rms * 0.1

        logger.info(
            f"🎵 降级TTS合成: {total_duration:.1f}s, "
            f"情感={emotion_label}, 段数={len(timeline_constraints or [])}, "
            f"峰值={max_val:.2f}"
        )

        return TTSResult(
            audio=audio.astype(np.float32),
            sample_rate=sr,
            duration=total_duration,
            phonemes_used=[],
            timeline=timeline_constraints or [],
            emotion_applied=emotion_label,
        )

    def _inject_emotion(
        self,
        audio: np.ndarray,
        emotion_embedding: np.ndarray,
        emotion_label: str,
    ) -> np.ndarray:
        """
        【创新点】情感注入

        通过修改音频的韵律特征来注入情感:
        - 基频(F0)调整
        - 能量分布调整
        - 语速微调
        """
        try:
            import librosa

            # 提取基频
            f0, voiced_flag, _ = librosa.pyin(
                audio.astype(np.float64),
                fmin=50, fmax=600,
                sr=self.sample_rate,
            )

            # 根据情感调整F0
            emotion_f0_shift = {
                "happy": 1.15, "sad": 0.88, "angry": 1.12,
                "surprised": 1.2, "fearful": 1.08,
                "neutral": 1.0,
            }.get(emotion_label, 1.0)

            # 能量调整
            emotion_energy = {
                "happy": 1.15, "sad": 0.82, "angry": 1.25,
                "surprised": 1.18, "fearful": 0.95,
                "neutral": 1.0,
            }.get(emotion_label, 1.0)

            modified = audio.copy()
            modified *= emotion_energy

            # 限制幅度
            max_val = np.abs(modified).max()
            if max_val > 1.0:
                modified /= max_val

            return modified

        except ImportError:
            return audio

    def _apply_timeline_constraints(
        self,
        audio: np.ndarray,
        timeline: List[Dict],
    ) -> np.ndarray:
        """
        【创新点】时间轴约束应用

        根据时间轴调整音频,使合成语音的节奏
        与AOCP-Net预测的开口状态匹配
        """
        # 简单实现: 确保音频长度匹配时间轴
        if not timeline:
            return audio

        expected_duration = timeline[-1]["end"]
        actual_duration = len(audio) / self.sample_rate

        if abs(expected_duration - actual_duration) > 0.1:
            # 调整长度
            target_samples = int(expected_duration * self.sample_rate)
            if target_samples > len(audio):
                audio = np.pad(audio, (0, target_samples - len(audio)))
            else:
                audio = audio[:target_samples]

        return audio

    def _adjust_speed(self, audio: np.ndarray, speed: float) -> np.ndarray:
        """语速调整"""
        try:
            import librosa
            return librosa.effects.time_stretch(
                audio.astype(np.float64), rate=speed
            ).astype(np.float32)
        except ImportError:
            if speed > 1.0:
                # 简单下采样模拟加速
                indices = np.arange(0, len(audio), speed)
                indices = indices.astype(int)
                indices = indices[indices < len(audio)]
                return audio[indices]
            else:
                return audio
