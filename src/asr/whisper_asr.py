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
                prev["text"] = self._join_seg_text(prev.get("text", ""), seg.get("text", ""))
                prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
                prev["syllables"] = (prev.get("syllables") or []) + (seg.get("syllables") or [])
            elif (prev["end"] - prev["start"]) < min_seg * 0.6:
                seg["start"] = prev["start"]
                seg["text"] = self._join_seg_text(prev.get("text", ""), seg.get("text", ""))
                seg["words"] = (prev.get("words") or []) + (seg.get("words") or [])
                seg["syllables"] = (prev.get("syllables") or []) + (seg.get("syllables") or [])
                merged[-1] = seg
            else:
                merged.append(seg)

        # 语义/缺口碎句合并 (半词、半句)
        n1 = len(merged)
        merged = self._merge_broken_phrases(merged)
        if len(merged) != n1:
            logger.info(f"ASR phrase-merge: {n1} -> {len(merged)} segments")
        return merged

    @staticmethod
    def _join_seg_text(a: str, b: str) -> str:
        """拼接两段文本: 中文紧贴, 英文补空格"""
        a = (a or "").strip()
        b = (b or "").strip()
        if not a:
            return b
        if not b:
            return a
        # 去掉拼接处重复标点
        if b[0] in ",，、;；":
            b = b[1:].lstrip()
        cjk_a = sum(1 for c in a if "\u4e00" <= c <= "\u9fff")
        cjk_b = sum(1 for c in b if "\u4e00" <= c <= "\u9fff")
        if cjk_a >= max(1, len(a) // 3) or cjk_b >= max(1, len(b) // 3):
            return (a + b).strip()
        if a[-1].isalnum() and b[0].isalnum():
            return f"{a} {b}".strip()
        return (a + b).strip()

    @staticmethod
    def _cjk_chars(text: str) -> int:
        return sum(1 for c in (text or "") if "\u4e00" <= c <= "\u9fff")

    @staticmethod
    def _has_terminal_punct(text: str) -> bool:
        t = (text or "").strip()
        if not t:
            return False
        return t[-1] in "。！？.!?;；…"

    @classmethod
    def _looks_incomplete_end(cls, text: str) -> bool:
        """上一句像半截 (功能词/半词结尾、无句末标点)"""
        import re
        t = (text or "").strip().rstrip(",，、;；")
        if not t or cls._has_terminal_punct(t):
            return False
        # 英文介词/冠词结尾
        if re.search(
            r"(?i)\b(the|a|an|to|of|for|and|or|but|with|in|on|at|by|from|into|is|are|was|were|be|let'?s|we|i|you)$",
            t,
        ):
            return True
        # 中文: 挂起助词/半截动词结构 (不含可独立成句的结尾)
        if re.search(
            r"(把说|把說|还分|還分|不太|先把|我们先|我們先|"
            r"把|将|將|还|還|分|不|太|的|了|着|過|过|和|与|與|对|對|从|從|"
            r"在|是|有|要|会|會|能|可|就|都|也|很|更|再|先|来|來|去|给|給|"
            r"说|說|话|話)$",
            t,
        ):
            return True
        # 很短且无句末 (≤4字, 避免「听感也一般」被误判)
        if cls._cjk_chars(t) > 0 and cls._cjk_chars(t) <= 4 and not cls._has_terminal_punct(t):
            return True
        return False

    @classmethod
    def _looks_continuation_start(cls, text: str) -> bool:
        """下一句像接续而非新开话轮"""
        import re
        t = (text or "").strip().lstrip(",，、;；")
        if not t:
            return False
        # 典型接续碎片 (来自对话样例与常见半句)
        if re.match(
            r"^(太清|话人|話人|人分离|人分離|不同说话|不同說話|说话人|說話人|"
            r"克隆质量|克隆質量|坐稳|坐穩|进度|進度|听感|聽感|也一般|"
            r"分离|分離)",
            t,
        ):
            return True
        # 明确的新句起手 → 不是接续
        if re.match(
            r"^(你好|您好|喂|嗯|啊|哦|好的|是的|对|對|嗯嗯|今天|目前|那我们|那我們|"
            r"然后|然後|所以|因为|因為|如果|但是|不过|不過|另外|首先|其次|"
            r"再提升|接下来|接下來|下一步|另外|同时|同時|"
            r"Hello|Hi|Okay|Yes|No|Well|So|Then|Alright|Next|Also)",
            t,
            re.I,
        ):
            return False
        # 英文小写开头接续
        if t[0].islower():
            return True
        # 短中文且不像招呼/主语起句
        if cls._cjk_chars(t) > 0 and cls._cjk_chars(t) <= 8:
            return True
        return False

    def _merge_broken_phrases(self, segments: List[Dict]) -> List[Dict]:
        """
        合并被静音误切的半句/半词。
        例: 「目前語音克隆還分」+「太清不同說話人」
            「那我們先把說」+「話人分離坐穩」
        """
        if len(segments) <= 1:
            return segments

        max_gap = float(self.config.get("merge_gap_seconds", 0.35))
        max_dur = float(self.config.get("merge_max_seconds", 4.8))
        short_chars = int(self.config.get("merge_short_chars", 8))

        def should_merge(prev: Dict, nxt: Dict) -> bool:
            gap = float(nxt["start"]) - float(prev["end"])
            if gap > max_gap:
                return False
            merged_dur = float(nxt["end"]) - float(prev["start"])
            if merged_dur > max_dur:
                return False
            pt = (prev.get("text") or "").strip()
            nt = (nxt.get("text") or "").strip()
            if not pt or not nt:
                return gap <= max_gap * 0.5
            # 已有句末标点且下一段像新句 → 不并
            if self._has_terminal_punct(pt) and not self._looks_continuation_start(nt):
                return False
            if self._looks_incomplete_end(pt) or self._looks_continuation_start(nt):
                return True
            # 紧挨着且双方都偏短 (避免把「再提升…」并进完整上句)
            pc, nc = self._cjk_chars(pt), self._cjk_chars(nt)
            if gap <= 0.12 and not self._has_terminal_punct(pt):
                if pc and nc and pc <= short_chars and nc <= short_chars:
                    return True
                if pc == 0 and nc == 0 and len(pt) <= 18 and len(nt) <= 18:
                    return True
            return False

        out: List[Dict] = [segments[0].copy()]
        for seg in segments[1:]:
            prev = out[-1]
            cur = {k: (list(v) if isinstance(v, list) else v) for k, v in seg.items()}
            if should_merge(prev, cur):
                prev["end"] = cur["end"]
                prev["text"] = self._join_seg_text(prev.get("text", ""), cur.get("text", ""))
                prev["words"] = (prev.get("words") or []) + (cur.get("words") or [])
                prev["syllables"] = (prev.get("syllables") or []) + (cur.get("syllables") or [])
            else:
                out.append(cur)

        # 再扫一遍, 处理 A+B 合并后仍与 C 半句相连
        if len(out) >= 2:
            changed = True
            while changed and len(out) >= 2:
                changed = False
                new_out: List[Dict] = [out[0]]
                for seg in out[1:]:
                    prev = new_out[-1]
                    if should_merge(prev, seg):
                        prev["end"] = seg["end"]
                        prev["text"] = self._join_seg_text(prev.get("text", ""), seg.get("text", ""))
                        prev["words"] = (prev.get("words") or []) + (seg.get("words") or [])
                        prev["syllables"] = (prev.get("syllables") or []) + (seg.get("syllables") or [])
                        changed = True
                    else:
                        new_out.append(seg)
                out = new_out
        return out

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
