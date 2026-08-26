"""
Whisper ASR 模块 - 多语种语音识别
支持 faster-whisper 和 FunASR 双引擎
"""

import numpy as np
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ASRResult:
    """ASR识别结果"""
    text: str
    language: str
    segments: List[Dict]  # 含时间戳的词/段信息
    duration: float       # 音频时长(秒)
    confidence: float     # 整体置信度


class WhisperASR:
    """
    多引擎语音识别器

    支持引擎:
    - faster-whisper: CTranslate2加速的Whisper,速度快4倍
    - funasr (Paraformer/SenseVoice): 阿里达摩院,中文识别优秀
    """

    SUPPORTED_ENGINES = ["faster_whisper", "funasr"]

    def __init__(self, config: Dict):
        self.config = config
        self.engine_name = config.get("engine", "faster_whisper")
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """加载ASR模型"""
        if self.engine_name == "faster_whisper":
            self._load_faster_whisper()
        elif self.engine_name == "funasr":
            self._load_funasr()
        else:
            raise ValueError(f"不支持的ASR引擎: {self.engine_name}")

    def _load_faster_whisper(self):
        """加载 faster-whisper 模型"""
        try:
            from faster_whisper import WhisperModel
            import os
            from pathlib import Path
            model_name = self.config.get("model_name", "medium")
            compute_type = self.config.get("compute_type", "int8")
            device = "cuda" if self.config.get("device", "cuda") == "cuda" else "cpu"
            num_workers = int(self.config.get("num_workers", 1) or 1)
            cpu_threads = int(self.config.get("cpu_threads", 0) or 0)

            # 跨平台模型目录
            download_root = self.config.get("download_root")
            if not download_root:
                win = Path("D:/CodingPackage/models")
                download_root = str(win) if win.exists() else str(
                    Path(__file__).resolve().parents[2] / "models" / "faster-whisper"
                )
            Path(download_root).mkdir(parents=True, exist_ok=True)

            logger.info(f"Downloading faster-whisper {model_name}...")
            kwargs = dict(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                num_workers=num_workers,
                download_root=download_root,
            )
            if cpu_threads > 0:
                kwargs["cpu_threads"] = cpu_threads
            self.model = WhisperModel(**kwargs)
            logger.info(f"faster-whisper loaded: {model_name} ({device})")
        except Exception as e:
            logger.error(f"❌ faster-whisper 加载失败: {e}")
            raise

    def _load_funasr(self):
        """加载 FunASR 模型 (Paraformer/SenseVoice)"""
        try:
            from funasr import AutoModel
            model_name = self.config.get(
                "funasr_model", "iic/SenseVoiceSmall"
            )
            vad_model = self.config.get("vad_model", None)
            punc_model = self.config.get("punc_model", None)

            self.model = AutoModel(
                model=model_name,
                vad_model=vad_model,
                punc_model=punc_model,
                device=self.config.get("device", "cuda:0"),
            )
            logger.info(f"✅ FunASR 模型加载成功: {model_name}")
        except Exception as e:
            logger.error(f"❌ FunASR 加载失败: {e}")
            raise

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        language: Optional[str] = None,
    ) -> ASRResult:
        """
        执行语音识别

        Args:
            audio: 音频波形 (numpy array, shape=(n_samples,))
            sample_rate: 采样率
            language: 指定语言 (None=自动检测)

        Returns:
            ASRResult 识别结果
        """
        if self.engine_name == "faster_whisper":
            return self._transcribe_whisper(audio, sample_rate, language)
        elif self.engine_name == "funasr":
            return self._transcribe_funasr(audio, sample_rate, language)

    def _transcribe_whisper(
        self, audio: np.ndarray, sample_rate: int, language: Optional[str]
    ) -> ASRResult:
        """faster-whisper 转写"""
        # 确保音频为 float32
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        beam_size = self.config.get("beam_size", 1)
        word_ts = bool(self.config.get("word_timestamps", True))
        cond_prev = bool(self.config.get("condition_on_previous_text", False))

        # 注意: 本版本 faster-whisper 的 transcribe() 不支持 batch_size/num_workers,
        # 并行度通过 WhisperModel 构造函数的 num_workers/cpu_threads 控制。
        lang = language or self.config.get("language") or None
        vad_filter = bool(self.config.get("vad_filter", True))
        # 更敏感的 VAD: 便于双人对话切出更多轮次
        vad_params = self.config.get("vad_parameters") or {
            "min_silence_duration_ms": int(self.config.get("min_silence_ms", 250)),
            "speech_pad_ms": int(self.config.get("speech_pad_ms", 120)),
        }
        transcribe_kwargs = dict(
            language=lang,
            beam_size=beam_size,
            vad_filter=vad_filter,
            vad_parameters=vad_params if vad_filter else None,
            word_timestamps=word_ts,       # 启用词级时间戳 → 音节对齐
            condition_on_previous_text=cond_prev,
        )
        # 去掉 None 值, 兼容旧版 faster-whisper
        transcribe_kwargs = {k: v for k, v in transcribe_kwargs.items() if v is not None}
        try:
            segments_raw, info = self.model.transcribe(audio, **transcribe_kwargs)
        except TypeError:
            # 旧版不支持 vad_parameters
            transcribe_kwargs.pop("vad_parameters", None)
            segments_raw, info = self.model.transcribe(audio, **transcribe_kwargs)

        segments = []
        full_text = []
        for seg in segments_raw:
            segments.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip(),
                "words": [
                    {"word": w.word, "start": w.start, "end": w.end, "probability": w.probability}
                    for w in (seg.words or [])
                ],
            })
            full_text.append(seg.text.strip())

        # 新版 faster-whisper 移除了 avg_log_prob, 用 language_probability 替代
        confidence = getattr(info, 'avg_log_prob', None)
        if confidence is None:
            confidence = getattr(info, 'language_probability', 0.5)

        # 生成音节级时间轴 (从词级时间戳推算)
        segments = self._add_syllable_timing(segments, info.language)

        # P0: 按静音/句号/最大时长再切段, 避免双人对话被合成大段
        n_before = len(segments)
        segments = self._refine_segments(audio, sample_rate, segments)
        if len(segments) != n_before:
            logger.info(f"ASR refine: {n_before} -> {len(segments)} segments")

        result = ASRResult(
            text=" ".join(full_text),
            language=info.language,
            segments=segments,
            duration=info.duration,
            confidence=confidence,
        )
        logger.info(f"ASR: lang={result.language}, segs={len(segments)}, text={result.text[:100]}...")
        return result

    def _refine_segments(
        self, audio: np.ndarray, sr: int, segments: List[Dict]
    ) -> List[Dict]:
        """
        把过粗的 ASR 段切细, 便于说话人分离:
        1) 词级时间戳上的静音缺口
        2) 段内能量静音
        3) 超过 max_segment_seconds 强制在最安静处切开
        """
        if not segments:
            return segments
        min_silence = float(self.config.get("split_silence_seconds", 0.28))
        min_seg = float(self.config.get("min_segment_seconds", 0.45))
        max_seg = float(self.config.get("max_segment_seconds", 2.8))

        refined: List[Dict] = []
        for seg in segments:
            parts = self._split_one_segment(audio, sr, seg, min_silence, min_seg, max_seg)
            refined.extend(parts)

        # 合并过短碎片到相邻段 (同向合并)
        if not refined:
            return segments
        merged: List[Dict] = [refined[0]]
        for seg in refined[1:]:
            prev = merged[-1]
            if (seg["end"] - seg["start"]) < min_seg * 0.6:
                # 并入上一段
                prev["end"] = seg["end"]
                prev["text"] = (prev.get("text", "") + seg.get("text", "")).strip()
                prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
                prev["syllables"] = (prev.get("syllables") or []) + (seg.get("syllables") or [])
            elif (prev["end"] - prev["start"]) < min_seg * 0.6:
                seg["start"] = prev["start"]
                seg["text"] = (prev.get("text", "") + seg.get("text", "")).strip()
                seg["words"] = (prev.get("words") or []) + (seg.get("words") or [])
                seg["syllables"] = (prev.get("syllables") or []) + (seg.get("syllables") or [])
                merged[-1] = seg
            else:
                merged.append(seg)
        return merged

    def _split_one_segment(
        self,
        audio: np.ndarray,
        sr: int,
        seg: Dict,
        min_silence: float,
        min_seg: float,
        max_seg: float,
    ) -> List[Dict]:
        words = seg.get("words") or []
        # 1) 优先用词间静音切
        if len(words) >= 2:
            cut_after = []  # 在第 i 个词之后切开 (i 从 0)
            for i in range(len(words) - 1):
                gap = float(words[i + 1]["start"] - words[i]["end"])
                if gap >= min_silence:
                    cut_after.append(i)
            if cut_after:
                return self._slice_by_word_cuts(seg, cut_after, min_seg)

        # 2) 段内能量静音切
        start, end = float(seg["start"]), float(seg["end"])
        if end - start <= max_seg:
            # 仍尝试能量切 (双人轮次间隔)
            energy_cuts = self._energy_silence_cuts(audio, sr, start, end, min_silence, min_seg)
            if energy_cuts:
                return self._slice_by_time_cuts(seg, energy_cuts, min_seg)
            return [seg]

        # 3) 过长: 在最安静处递归切到 max_seg 以下
        energy_cuts = self._energy_silence_cuts(audio, sr, start, end, min_silence * 0.7, min_seg)
        if not energy_cuts:
            # 强制等分, 并在边界附近找局部能量最低点
            n = int(np.ceil((end - start) / max_seg))
            energy_cuts = []
            for k in range(1, n):
                t = start + k * (end - start) / n
                energy_cuts.append(self._local_min_energy_time(audio, sr, t, window=0.25))
        parts = self._slice_by_time_cuts(seg, energy_cuts, min_seg)
        # 递归确保每段不超过 max_seg
        out = []
        for p in parts:
            if p["end"] - p["start"] > max_seg * 1.15:
                out.extend(self._split_one_segment(audio, sr, p, min_silence, min_seg, max_seg))
            else:
                out.append(p)
        return out or [seg]

    @staticmethod
    def _slice_by_word_cuts(seg: Dict, cut_after: List[int], min_seg: float) -> List[Dict]:
        words = seg.get("words") or []
        syllables = seg.get("syllables") or []
        bounds = [-1] + cut_after + [len(words) - 1]
        parts = []
        for bi in range(len(bounds) - 1):
            i0 = bounds[bi] + 1
            i1 = bounds[bi + 1]
            chunk_words = words[i0:i1 + 1]
            if not chunk_words:
                continue
            s = float(chunk_words[0]["start"])
            e = float(chunk_words[-1]["end"])
            if e - s < min_seg * 0.4 and parts:
                # 过短并入上一段
                parts[-1]["end"] = e
                parts[-1]["words"].extend(chunk_words)
                parts[-1]["text"] = "".join(w["word"] for w in parts[-1]["words"]).strip()
                continue
            text = "".join(w["word"] for w in chunk_words).strip()
            syl = [x for x in syllables if x.get("start", 0) >= s - 1e-3 and x.get("end", 0) <= e + 1e-3]
            parts.append({
                "start": s, "end": e, "text": text,
                "words": chunk_words, "syllables": syl,
            })
        return parts or [seg]

    @staticmethod
    def _slice_by_time_cuts(seg: Dict, cut_times: List[float], min_seg: float) -> List[Dict]:
        start, end = float(seg["start"]), float(seg["end"])
        words = seg.get("words") or []
        syllables = seg.get("syllables") or []
        cuts = sorted(t for t in cut_times if start + min_seg <= t <= end - min_seg)
        if not cuts:
            return [seg]
        edges = [start] + cuts + [end]
        parts = []
        for i in range(len(edges) - 1):
            s, e = edges[i], edges[i + 1]
            if e - s < min_seg * 0.35:
                if parts:
                    parts[-1]["end"] = e
                continue
            chunk_words = [w for w in words if w["start"] >= s - 0.05 and w["end"] <= e + 0.05]
            if chunk_words:
                text = "".join(w["word"] for w in chunk_words).strip()
            else:
                # 按时间比例切文本
                ratio0 = (s - start) / max(end - start, 1e-6)
                ratio1 = (e - start) / max(end - start, 1e-6)
                text = (seg.get("text") or "")
                i0, i1 = int(len(text) * ratio0), int(len(text) * ratio1)
                text = text[i0:i1].strip()
            syl = [x for x in syllables if x.get("start", 0) >= s - 1e-3 and x.get("end", 0) <= e + 1e-3]
            parts.append({
                "start": s, "end": e, "text": text or seg.get("text", ""),
                "words": chunk_words, "syllables": syl,
            })
        return parts or [seg]

    def _energy_silence_cuts(
        self, audio, sr, start, end, min_silence, min_seg
    ) -> List[float]:
        """在 [start,end] 内找足够长的低能量静音中点作为切点"""
        s = int(max(0, start) * sr)
        e = int(min(len(audio), end) * sr)
        if e - s < int(min_seg * 2 * sr):
            return []
        seg = audio[s:e].astype(np.float64)
        hop = max(1, int(sr * 0.01))
        rms = np.array([
            np.sqrt(np.mean(seg[i:i + hop] ** 2) + 1e-12)
            for i in range(0, len(seg) - hop + 1, hop)
        ])
        if rms.size == 0:
            return []
        thr = max(float(np.percentile(rms, 20)) * 1.2, float(rms.max()) * 0.08)
        silent = rms < thr
        cuts = []
        i = 0
        need = max(1, int(min_silence / 0.01))
        while i < len(silent):
            if not silent[i]:
                i += 1
                continue
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            if j - i >= need:
                mid = (i + j) / 2.0
                t = start + mid * hop / sr
                if start + min_seg <= t <= end - min_seg:
                    cuts.append(t)
            i = j
        return cuts

    @staticmethod
    def _local_min_energy_time(audio, sr, center, window=0.25) -> float:
        s = int(max(0, center - window) * sr)
        e = int(min(len(audio), center + window) * sr)
        if e <= s:
            return float(center)
        seg = audio[s:e].astype(np.float64)
        hop = max(1, int(sr * 0.01))
        best_i, best_e = 0, 1e9
        for i in range(0, len(seg) - hop + 1, hop):
            en = float(np.mean(seg[i:i + hop] ** 2))
            if en < best_e:
                best_e = en
                best_i = i
        return (s + best_i) / sr

    @staticmethod
    def _add_syllable_timing(segments: list, lang: str) -> list:
        """
        从词级时间戳推算音节级开闭时间轴

        中文: 每字=1音节, 韵母含a/o/e的为开口
        英文: 按元音字母估算音节数
        """
        import re
        open_vowels = set('aeo')  # 开口元音

        for seg in segments:
            syllables = []
            words = seg.get("words", [])
            if not words:
                # 无词级信息, 按字均分
                chars = list(seg["text"].replace(" ", ""))
                n = max(1, len(chars))
                dur = seg["end"] - seg["start"]
                for i, ch in enumerate(chars):
                    is_open = ch.lower() in open_vowels or (lang=="zh" and ch in "啊哦噢呃诶")
                    syllables.append({
                        "char": ch,
                        "start": seg["start"] + i*dur/n,
                        "end": seg["start"] + (i+1)*dur/n,
                        "open": is_open,
                    })
            else:
                for w in words:
                    word_text = w["word"].strip()
                    # 估算音节数
                    if lang in ("zh","cmn","chi"):
                        n_syl = len(word_text)  # 每字1音节
                    else:
                        n_syl = max(1, len(re.findall(r'[aeiouy]+', word_text.lower())))
                    w_dur = w["end"] - w["start"]
                    for si in range(n_syl):
                        # 取该音节对应的字符
                        ch_idx = si if si < len(word_text) else -1
                        ch = word_text[ch_idx] if 0 <= ch_idx < len(word_text) else word_text[-1]
                        is_open = ch.lower() in open_vowels
                        syllables.append({
                            "char": ch,
                            "start": w["start"] + si*w_dur/n_syl,
                            "end": w["start"] + (si+1)*w_dur/n_syl,
                            "open": is_open,
                        })
            seg["syllables"] = syllables
        return segments

    def _transcribe_funasr(
        self, audio: np.ndarray, sample_rate: int, language: Optional[str]
    ) -> ASRResult:
        """FunASR 转写"""
        result = self.model.generate(
            input=audio,
            cache={},
            language=language,
        )

        if result and len(result) > 0:
            r = result[0]
            text = r.get("text", "")
            # FunASR返回的时间戳可能格式不同
            segments = []
            if "timestamp" in r:
                for ts_item in r["timestamp"]:
                    segments.append({
                        "start": ts_item[0] / 1000.0 if ts_item[0] > 1 else ts_item[0],
                        "end": ts_item[1] / 1000.0 if ts_item[1] > 1 else ts_item[1],
                        "text": ts_item[2] if len(ts_item) > 2 else "",
                    })

            result = ASRResult(
                text=text,
                language=language or "zh",
                segments=segments,
                duration=len(audio) / sample_rate,
                confidence=0.9,
            )
            logger.info(f"📝 FunASR结果: text={result.text[:100]}...")
            return result
        else:
            return ASRResult(text="", language="unknown", segments=[], duration=0, confidence=0)
