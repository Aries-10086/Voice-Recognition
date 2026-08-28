"""
说话人分离模块 (Speaker Diarization)
识别音频中的不同人声, 支持两种后端:
- pyannote/speaker-diarization-3.1  (需权重 + HF token, 精度最高)
- 离线回退: wav2vec2 / 频谱嵌入 + 层次聚类 (无需额外下载, 可离线运行)

P0 优化:
- 不可用时明确降级提示, 禁止静默当作单人
- 聚类阈值更敏感; 单人塌缩时自动重试拆分
- 记录 backend / 说话人数供 summary 观测
"""

import os
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple
from loguru import logger


class SpeakerDiarizer:
    """
    说话人分离器

    diarize() 返回与输入 segments 等长的说话人标签列表。
    last_meta 记录本次分离后端与结果, 供 pipeline 落盘。
    """

    def __init__(self, config: Dict):
        self.config = config
        self.engine = config.get("engine", "auto")
        self.max_speakers = int(config.get("max_speakers", 8))
        self.min_speakers = int(config.get("min_speakers", 1))
        # 对白场景可设 expected_speakers=2, 段数足够时强制至少分出这么多
        self.expected_speakers = int(
            config.get("expected_speakers", config.get("min_speakers", 1))
        )
        # 越小分得越细; 原默认 0.9 极易塌成 1 人
        self.distance_threshold = float(config.get("distance_threshold", 0.40))
        self.wav2vec2_model = config.get(
            "wav2vec2_model", "superb/wav2vec2-base-superb-er"
        )
        self.allow_single_speaker = bool(config.get("allow_single_speaker", True))
        # 段数足够却仍只有 1 人时, 强制尝试二次拆分
        self.retry_split_min_segments = int(config.get("retry_split_min_segments", 2))
        # spectral=频谱+F0(对音色更敏感); wav2vec2=情感模型骨干(易抹平说话人差异)
        self.embedding_backend = config.get("embedding", "spectral")
        # 双人对话: 静音轮次交替校正 (A-B-A-B), 纠正聚类错标
        self.turn_taking_correct = bool(config.get("turn_taking_correct", True))
        self.turn_gap_seconds = float(config.get("turn_gap_seconds", 0.35))
        self.prefer_f0 = bool(config.get("prefer_f0", True))
        self.f0_min_spread_hz = float(config.get("f0_min_spread_hz", 12.0))
        self.refine_by_embedding = bool(config.get("refine_by_embedding", True))
        self._w2v = None
        self._processor = None
        self.last_meta: Dict = {
            "backend": "none",
            "n_speakers": 0,
            "degraded": False,
            "warning": "",
            "first_round_speakers": 0,
            "used_retry": False,
            "label_confidences": [],
        }

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def diarize(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict],
        embedding_fn: Optional[Callable] = None,
    ) -> List[str]:
        """
        Args:
            audio: 原始音频 (mono float32)
            sr: 采样率
            segments: ASR 分段 [{start, end, text}, ...]
            embedding_fn: 可选 (audio, sr) -> embedding, 用于复用已加载模型
        Returns:
            与 segments 等长的说话人标签列表 ["SPEAKER_00", ...]
        """
        n = len(segments)
        if n == 0:
            self.last_meta = {
                "backend": "none", "n_speakers": 0, "degraded": False, "warning": ""
            }
            return []
        if n == 1:
            self.last_meta = {
                "backend": "single_segment",
                "n_speakers": 1,
                "degraded": False,
                "warning": "",
            }
            return ["SPEAKER_00"]

        labels: Optional[List[str]] = None
        backend = "none"

        # 1) 优先 pyannote
        if self.engine in ("auto", "pyannote"):
            if self._pyannote_cached():
                labels = self._diarize_pyannote(audio, sr, segments)
                if labels is not None:
                    backend = "pyannote"
            elif self.engine == "pyannote":
                msg = (
                    "已指定 pyannote, 但本地无缓存权重。"
                    "请设置 HF_TOKEN 并预下载 pyannote/speaker-diarization-3.1"
                )
                logger.error(f"[diarization] {msg}")
                if not bool(self.config.get("fallback_on_pyannote_fail", True)):
                    raise RuntimeError(msg)
                logger.warning("[diarization] 降级到离线聚类 (效果可能较差)")

        # 2) 离线聚类回退 — 多候选择优 (F1: 减少首轮塌缩与 retry)
        if labels is None:
            if self.engine in ("auto", "pyannote", "wav2vec2_cluster", "cluster"):
                target_spk = max(self.min_speakers, self.expected_speakers)
                labels, backend = self._pick_offline_labeling(
                    audio, sr, segments, embedding_fn, target_spk
                )
            else:
                labels = ["SPEAKER_00"] * n
                backend = "forced_single"

        first_round_n_spk = len(set(labels))
        n_spk = first_round_n_spk
        warning = ""
        degraded = backend != "pyannote"
        target_spk = max(self.min_speakers, self.expected_speakers)
        used_retry = False

        # 段数很多却仍少于期望人数 → 二次拆分 (仅当首轮不足)
        if n_spk < target_spk and n >= max(self.retry_split_min_segments, target_spk):
            warning = (
                f"分离仅得到 {n_spk} 人 (期望≥{target_spk}, 共 {n} 段), "
                "疑似塌缩; 已尝试更敏感二次拆分"
            )
            logger.warning(f"[diarization] {warning}")
            used_retry = True
            retry = self._cluster_segments(
                audio, sr, segments, embedding_fn,
                distance_threshold=max(0.22, self.distance_threshold * 0.5),
                force_min_speakers=target_spk,
            )
            if len(set(retry)) > n_spk:
                labels = retry
                n_spk = len(set(labels))
                backend = f"{backend}+retry_split"
                logger.info(f"[diarization] 二次拆分成功: {n_spk} speakers")
            else:
                # 最后手段: 强制按期望人数硬拆
                retry2 = self._cluster_segments(
                    audio, sr, segments, embedding_fn,
                    distance_threshold=0.22,
                    force_min_speakers=target_spk,
                )
                labels = retry2
                n_spk = len(set(labels))
                backend = f"{backend}+force_{target_spk}"
                degraded = True
                logger.warning(f"[diarization] 强制拆分为 {n_spk} speakers")

            if n_spk < target_spk and not self.allow_single_speaker:
                raise RuntimeError(
                    "说话人分离失败: 无法区分多人。请检查音频或启用 pyannote。"
                )

        if n_spk <= 1 and n >= self.retry_split_min_segments:
            logger.warning(
                "[diarization] ⚠️ 仍为单说话人标签 — 后续克隆可能全片同声。"
                "建议: 安装/缓存 pyannote, 或调低 distance_threshold / 提高 expected_speakers"
            )

        # O3: 双人对话轮次校正 (静音切轮 + F0/交替)
        corrected = False
        if (
            self.turn_taking_correct
            and target_spk == 2
            and n >= 2
            and backend != "pyannote"
        ):
            fixed = self._correct_turn_taking(audio, sr, segments, labels)
            if fixed is not None and fixed != labels:
                n_before = len(set(labels))
                labels = fixed
                n_spk = len(set(labels))
                backend = f"{backend}+turn_correct"
                corrected = True
                logger.info(
                    f"[diarization] 轮次校正: {n_before} -> {n_spk} speakers "
                    f"(gap>={self.turn_gap_seconds}s)"
                )

        label_confidences: List[float] = []

        # F2: 段级嵌入重标 + 置信度
        if self.refine_by_embedding and n >= 2 and n_spk >= 2:
            labels, label_confidences = self._refine_labels_by_embedding(
                audio, sr, segments, labels, target_spk=target_spk
            )
            n_spk = len(set(labels))
            if label_confidences:
                backend = f"{backend}+embed_refine"

        # 规范化标签顺序
        labels = self._renumber(labels)
        n_spk = len(set(labels))
        self.last_meta = {
            "backend": backend,
            "n_speakers": n_spk,
            "degraded": degraded and not corrected,
            "warning": warning,
            "n_segments": n,
            "distance_threshold": self.distance_threshold,
            "turn_corrected": corrected,
            "first_round_speakers": first_round_n_spk,
            "used_retry": used_retry,
            "label_confidences": label_confidences,
        }
        logger.info(
            f"Diarization({backend}): {n_spk} speakers / {n} segments"
            + (" [DEGRADED]" if degraded and not corrected else "")
        )
        return labels

    def _pick_offline_labeling(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict],
        embedding_fn: Optional[Callable],
        target_spk: int,
    ) -> Tuple[List[str], str]:
        """F1: 多离线候选 (F0 / spectral / 敏感 spectral) 择优, 减少首轮塌缩"""
        n = len(segments)
        candidates: List[Tuple[str, List[str]]] = []

        if self.prefer_f0 and target_spk == 2 and n >= 2:
            f0_labels = self._cluster_by_f0(audio, sr, segments)
            if f0_labels is not None and len(set(f0_labels)) >= 2:
                candidates.append(("f0_kmeans", f0_labels))

        spec_labels = self._cluster_segments(audio, sr, segments, embedding_fn)
        spec_name = (
            "wav2vec2_cluster"
            if self.embedding_backend == "wav2vec2"
            else "spectral_cluster"
        )
        candidates.append((spec_name, spec_labels))

        if target_spk >= 2 and len(set(spec_labels)) < target_spk:
            sens = self._cluster_segments(
                audio, sr, segments, embedding_fn,
                distance_threshold=max(0.24, self.distance_threshold * 0.65),
                force_min_speakers=target_spk,
            )
            candidates.append((f"{spec_name}_sensitive", sens))

        best_name, best_labels = max(
            candidates,
            key=lambda item: self._evaluate_labeling(
                segments, item[1], target_spk
            ),
        )
        logger.info(
            f"[diarization] 离线候选 {len(candidates)} 个, 选用 {best_name} "
            f"({len(set(best_labels))} spk)"
        )
        return best_labels, best_name

    def _evaluate_labeling(
        self,
        segments: List[Dict],
        labels: List[str],
        target_spk: int,
    ) -> float:
        """标签方案打分: 人数匹配 + 话轮交替一致性"""
        n = len(labels)
        if n == 0:
            return -1.0
        n_spk = len(set(labels))
        score = 0.0
        if n_spk == target_spk:
            score += 3.0
        elif n_spk >= 2:
            score += 1.5
        else:
            score -= 2.0
        # 惩罚过多簇
        if n_spk > target_spk + 1:
            score -= 0.5 * (n_spk - target_spk)
        # 话轮交替: 大静音后换人加分
        if target_spk == 2 and n >= 3:
            switches = 0
            same_after_gap = 0
            for i in range(1, n):
                gap = float(segments[i].get("start", 0)) - float(
                    segments[i - 1].get("end", 0)
                )
                if labels[i] != labels[i - 1]:
                    switches += 1
                elif gap >= self.turn_gap_seconds:
                    same_after_gap += 1
            score += min(1.5, switches * 0.25)
            score -= same_after_gap * 0.4
        # 标签均衡 (避免全标一人)
        from collections import Counter
        counts = Counter(labels)
        if n_spk >= 2:
            ratio = min(counts.values()) / max(counts.values())
            score += ratio * 1.0
        return score

    def _compute_segment_embeddings(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict],
        embedding_fn: Optional[Callable] = None,
    ) -> np.ndarray:
        """各 ASR 段频谱/声纹嵌入 (L2 归一化)"""
        if embedding_fn is None:
            embedding_fn = (
                self._embed
                if self.embedding_backend == "wav2vec2"
                else self._spectral
            )
        embeds = []
        for seg in segments:
            s = int(max(0.0, seg.get("start", 0.0)) * sr)
            e = int(min(len(audio), seg.get("end", s / sr + 0.1)) * sr)
            min_len = int(0.8 * sr)
            if e - s < min_len:
                pad = (min_len - (e - s)) // 2
                s = max(0, s - pad)
                e = min(len(audio), e + pad)
            seg_audio = audio[s:e] if e > s else audio[: int(0.1 * sr)]
            try:
                emb = embedding_fn(seg_audio, sr)
                if emb is None:
                    emb = self._spectral(seg_audio, sr)
            except Exception:
                emb = self._spectral(seg_audio, sr)
            v = np.asarray(emb, dtype=np.float32).reshape(-1)
            norm = np.linalg.norm(v)
            embeds.append(v / norm if norm > 1e-8 else v)
        dim = min(e.shape[0] for e in embeds)
        return np.vstack([e[:dim] for e in embeds]).astype(np.float32)

    def _refine_labels_by_embedding(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict],
        labels: List[str],
        target_spk: int = 2,
    ) -> Tuple[List[str], List[float]]:
        """
        F2: 用段嵌入 KMeans 重标, 并输出每段 label_confidence (余弦相似度)
        """
        n = len(segments)
        if n < 2:
            return labels, [1.0] * n
        try:
            X = self._compute_segment_embeddings(audio, sr, segments)
        except Exception as e:
            logger.warning(f"[diarization] 嵌入重标跳过: {e}")
            return labels, [0.5] * n

        k = min(max(target_spk, 2), n, self.max_speakers)
        try:
            from sklearn.cluster import KMeans
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            ids = km.fit_predict(X)
            centers = km.cluster_centers_.astype(np.float32)
            norms = np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8
            centers = centers / norms
            confidences = []
            for i in range(n):
                sims = centers @ X[i]
                confidences.append(float(np.max(sims)))
            new_labels = [f"SPEAKER_{i:02d}" for i in self._stable_ids(ids)]
            # 与话轮校正结果融合: 大静音边界处优先保留交替
            if self.turn_taking_correct and k == 2 and n >= 3:
                turn_fixed = self._correct_turn_taking(audio, sr, segments, new_labels)
                if turn_fixed is not None:
                    new_labels = turn_fixed
            avg_conf = float(np.mean(confidences)) if confidences else 0.0
            logger.info(
                f"[diarization] 嵌入重标: k={k}, avg_conf={avg_conf:.2f}"
            )
            return new_labels, confidences
        except Exception as e:
            logger.warning(f"[diarization] KMeans 重标失败: {e}")
            return labels, [0.5] * n

    # ------------------------------------------------------------------
    # pyannote
    # ------------------------------------------------------------------
    def _pyannote_cached(self) -> bool:
        hf_home = os.environ.get("HF_HOME", "")
        hub = os.path.join(hf_home, "hub") if hf_home else ""
        search_dirs = [hub] if hub else []
        # 也扫默认缓存
        default_hub = os.path.expanduser("~/.cache/huggingface/hub")
        if default_hub not in search_dirs:
            search_dirs.append(default_hub)
        for d in search_dirs:
            if not d or not os.path.isdir(d):
                continue
            try:
                names = os.listdir(d)
            except OSError:
                continue
            if any("pyannote--speaker-diarization" in n for n in names):
                return True
        return False

    def _diarize_pyannote(self, audio, sr, segments) -> Optional[List[str]]:
        try:
            from pyannote.audio import Pipeline
            import torch

            token = (
                os.environ.get("HF_TOKEN")
                or os.environ.get("HUGGINGFACE_TOKEN")
                or os.environ.get("HUGGING_FACE_HUB_TOKEN")
            )
            kwargs = {}
            if token:
                kwargs["use_auth_token"] = token
            try:
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", **kwargs
                )
            except TypeError:
                # 新版 huggingface_hub 用 token=
                kwargs.pop("use_auth_token", None)
                if token:
                    kwargs["token"] = token
                pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1", **kwargs
                )

            if torch.cuda.is_available():
                pipeline = pipeline.to(torch.device("cuda"))
            waveform = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            diar = pipeline({"waveform": waveform, "sample_rate": sr})
            labels = []
            for seg in segments:
                mid = (seg["start"] + seg["end"]) / 2
                lbl = "SPEAKER_00"
                best_overlap = 0.0
                for turn, _, spk in diar.itertracks(yield_label=True):
                    # 优先用中点命中; 否则用最大重叠
                    if turn.start <= mid <= turn.end:
                        lbl = spk
                        break
                    overlap = min(turn.end, seg["end"]) - max(turn.start, seg["start"])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        lbl = spk
                labels.append(lbl)
            logger.info(f"Diarization(pyannote): {len(set(labels))} speakers")
            return labels
        except Exception as e:
            logger.warning(f"pyannote 分离失败({str(e)[:120]}), 回退离线聚类")
            return None

    def _cluster_by_f0(self, audio, sr, segments) -> Optional[List[str]]:
        """用各段中位基频做 2 均值聚类 — 适合男女/高低音双人对话"""
        try:
            import librosa
        except ImportError:
            return None
        f0s = []
        for seg in segments:
            s = int(max(0.0, seg.get("start", 0.0)) * sr)
            e = int(min(len(audio), seg.get("end", s / sr + 0.1)) * sr)
            if e - s < int(0.3 * sr):
                f0s.append(np.nan)
                continue
            y = audio[s:e].astype(np.float64)
            try:
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
                f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
                f0s.append(float(np.median(f0)) if len(f0) > 3 else np.nan)
            except Exception:
                f0s.append(np.nan)
        valid = [f for f in f0s if not np.isnan(f)]
        if len(valid) < 2:
            return None
        # 两个质心: 低 F0 / 高 F0
        lo, hi = float(np.percentile(valid, 25)), float(np.percentile(valid, 75))
        if hi - lo < self.f0_min_spread_hz:  # 区分度不够
            return None
        mid = 0.5 * (lo + hi)
        ids = []
        for f in f0s:
            if np.isnan(f):
                ids.append(0)
            else:
                ids.append(0 if f < mid else 1)
        if len(set(ids)) < 2:
            return None
        labels = [f"SPEAKER_{i:02d}" for i in self._stable_ids(ids)]
        logger.info(
            f"Diarization(f0_kmeans): {len(set(labels))} speakers "
            f"(F0 mid={mid:.0f}Hz, spread={hi-lo:.0f}Hz)"
        )
        return labels

    def _segment_f0_medians(
        self, audio: np.ndarray, sr: int, segments: List[Dict]
    ) -> List[float]:
        """各段中位基频; 失败则为 nan"""
        try:
            import librosa
        except ImportError:
            return [float("nan")] * len(segments)
        out = []
        for seg in segments:
            s = int(max(0.0, seg.get("start", 0.0)) * sr)
            e = int(min(len(audio), seg.get("end", s / sr + 0.1)) * sr)
            if e - s < int(0.25 * sr):
                out.append(float("nan"))
                continue
            y = audio[s:e].astype(np.float64)
            try:
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
                f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
                out.append(float(np.median(f0)) if len(f0) > 3 else float("nan"))
            except Exception:
                out.append(float("nan"))
        return out

    def _correct_turn_taking(
        self,
        audio: np.ndarray,
        sr: int,
        segments: List[Dict],
        labels: List[str],
    ) -> Optional[List[str]]:
        """
        双人对话后验校正:
        1) 按静音间隙切成「话轮」
        2) 每轮用 F0 中位数判定高/低说话人; F0 不够则用原标签多数票
        3) 相邻话轮若同人则强制交替 (对白假设)
        """
        n = len(segments)
        if n < 2:
            return None

        # 切话轮: gap >= turn_gap_seconds 视为换人边界
        turns: List[List[int]] = [[0]]
        for i in range(1, n):
            gap = float(segments[i].get("start", 0)) - float(
                segments[i - 1].get("end", 0)
            )
            if gap >= self.turn_gap_seconds:
                turns.append([i])
            else:
                turns[-1].append(i)

        # 单轮过多段且无间隙 → 仍可按段交替 (合成样例常见)
        if len(turns) < 2 and n >= 4:
            turns = [[i] for i in range(n)]

        if len(turns) < 2:
            return None

        f0s = self._segment_f0_medians(audio, sr, segments)
        turn_f0 = []
        for idxs in turns:
            vals = [f0s[i] for i in idxs if not np.isnan(f0s[i])]
            turn_f0.append(float(np.median(vals)) if vals else float("nan"))

        valid_f0 = [f for f in turn_f0 if not np.isnan(f)]
        use_f0 = len(valid_f0) >= 2 and (
            max(valid_f0) - min(valid_f0) >= self.f0_min_spread_hz
        )
        mid = 0.5 * (min(valid_f0) + max(valid_f0)) if use_f0 else None

        # 话轮级说话人 id (0/1)
        turn_ids: List[int] = []
        for ti, idxs in enumerate(turns):
            if use_f0 and not np.isnan(turn_f0[ti]):
                tid = 0 if turn_f0[ti] >= mid else 1  # 高 F0 = 0 (女声常见)
            else:
                # 多数票
                votes = [labels[i] for i in idxs]
                from collections import Counter
                tid = 0 if Counter(votes).most_common(1)[0][0].endswith("00") else 1
            # 相邻同人 → 强制交替
            if turn_ids and tid == turn_ids[-1]:
                tid = 1 - tid
            turn_ids.append(tid)

        out = list(labels)
        for ti, idxs in enumerate(turns):
            spk = f"SPEAKER_{turn_ids[ti]:02d}"
            for i in idxs:
                out[i] = spk

        if len(set(out)) < 2:
            # 纯交替兜底
            out = [f"SPEAKER_{i % 2:02d}" for i in range(n)]
        return out

    # ------------------------------------------------------------------
    # 离线聚类回退
    # ------------------------------------------------------------------
    def _cluster_segments(
        self,
        audio,
        sr,
        segments,
        embedding_fn=None,
        distance_threshold: Optional[float] = None,
        force_min_speakers: int = 1,
    ) -> List[str]:
        if embedding_fn is None:
            if self.embedding_backend == "wav2vec2":
                embedding_fn = self._embed
            else:
                embedding_fn = self._spectral

        try:
            X = self._compute_segment_embeddings(
                audio, sr, segments, embedding_fn=embedding_fn
            )
        except Exception:
            embeds = []
            for seg in segments:
                s = int(max(0.0, seg.get("start", 0.0)) * sr)
                e = int(min(len(audio), seg.get("end", s / sr + 0.1)) * sr)
                min_len = int(0.8 * sr)
                if e - s < min_len:
                    pad = (min_len - (e - s)) // 2
                    s = max(0, s - pad)
                    e = min(len(audio), e + pad)
                seg_audio = audio[s:e] if e > s else audio[: int(0.1 * sr)]
                try:
                    emb = embedding_fn(seg_audio, sr)
                    if emb is None:
                        emb = self._spectral(seg_audio, sr)
                except Exception:
                    emb = self._spectral(seg_audio, sr)
                embeds.append(np.asarray(emb, dtype=np.float32).reshape(-1))
            dim = min(e.shape[0] for e in embeds)
            X = np.vstack([e[:dim] for e in embeds]).astype(np.float32)

        thresh = (
            self.distance_threshold
            if distance_threshold is None
            else float(distance_threshold)
        )
        cluster_ids = self._cluster(
            [X[i] for i in range(len(X))],
            distance_threshold=thresh,
            force_min_speakers=max(force_min_speakers, self.min_speakers),
        )
        labels = [f"SPEAKER_{i:02d}" for i in cluster_ids]
        return labels

    def _cluster(
        self,
        embeds: List[np.ndarray],
        distance_threshold: Optional[float] = None,
        force_min_speakers: int = 1,
    ) -> List[int]:
        n = len(embeds)
        if n <= 1:
            return [0] * n

        thresh = (
            self.distance_threshold
            if distance_threshold is None
            else float(distance_threshold)
        )

        try:
            from sklearn.preprocessing import normalize
            from sklearn.cluster import AgglomerativeClustering
        except ImportError:
            return self._greedy_cluster(embeds, thresh, force_min_speakers)

        # 对齐维度 (不同回退嵌入长度可能不同)
        dim = min(e.shape[0] for e in embeds)
        X = np.vstack([e[:dim] for e in embeds]).astype(np.float32)
        X = normalize(X, norm="l2")

        try:
            clust = AgglomerativeClustering(
                n_clusters=None,
                distance_threshold=thresh,
                metric="cosine",
                linkage="average",
            )
            ids = clust.fit_predict(X)
            n_found = len(set(ids))

            if n_found > self.max_speakers > 1:
                clust = AgglomerativeClustering(
                    n_clusters=self.max_speakers,
                    metric="cosine",
                    linkage="average",
                )
                ids = clust.fit_predict(X)
            elif n_found < force_min_speakers and force_min_speakers >= 2 and n >= force_min_speakers:
                # 强制拆到至少 min_speakers
                k = min(force_min_speakers, n, self.max_speakers)
                clust = AgglomerativeClustering(
                    n_clusters=k, metric="cosine", linkage="average"
                )
                ids = clust.fit_predict(X)
        except Exception:
            ids = self._greedy_cluster(embeds, thresh, force_min_speakers)

        return self._stable_ids(ids)

    def _greedy_cluster(
        self,
        embeds: List[np.ndarray],
        thresh: float,
        force_min_speakers: int = 1,
    ) -> List[int]:
        """无 sklearn 时的贪心余弦距离聚类"""
        n = len(embeds)
        dim = min(e.shape[0] for e in embeds)
        X = np.vstack([e[:dim] for e in embeds]).astype(np.float32)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / norms
        ids = [0] * n
        centroids = [X[0].copy()]
        for i in range(1, n):
            sims = [float(X[i] @ c) for c in centroids]
            best = int(np.argmax(sims))
            if sims[best] >= (1.0 - thresh):
                ids[i] = best
                # 在线更新质心
                centroids[best] = centroids[best] + X[i]
                centroids[best] /= np.linalg.norm(centroids[best]) + 1e-8
            else:
                if len(centroids) < self.max_speakers:
                    centroids.append(X[i].copy())
                    ids[i] = len(centroids) - 1
                else:
                    ids[i] = best

        if len(set(ids)) < force_min_speakers and force_min_speakers >= 2 and n >= 2:
            # 按与质心距离最远的点强制拆出新人
            return self._force_split_by_distance(X, force_min_speakers)
        return ids

    @staticmethod
    def _force_split_by_distance(X: np.ndarray, k: int) -> List[int]:
        """用最远点启发式强制拆成 k 簇"""
        n = X.shape[0]
        k = min(k, n)
        centers = [0]
        for _ in range(1, k):
            dists = []
            for i in range(n):
                d = min(1.0 - float(X[i] @ X[c]) for c in centers)
                dists.append(d)
            centers.append(int(np.argmax(dists)))
        ids = []
        for i in range(n):
            sims = [float(X[i] @ X[c]) for c in centers]
            ids.append(int(np.argmax(sims)))
        return ids

    @staticmethod
    def _stable_ids(ids) -> List[int]:
        seen = {}
        out = []
        for c in ids:
            c = int(c)
            if c not in seen:
                seen[c] = len(seen)
            out.append(seen[c])
        return out

    @staticmethod
    def _renumber(labels: List[str]) -> List[str]:
        order = {}
        out = []
        for lb in labels:
            if lb not in order:
                order[lb] = f"SPEAKER_{len(order):02d}"
            out.append(order[lb])
        return out

    # ------------------------------------------------------------------
    # 嵌入器
    # ------------------------------------------------------------------
    def _ensure_w2v(self):
        if self._w2v is not None:
            return self._w2v, self._processor
        try:
            from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
            import torch

            processor = AutoFeatureExtractor.from_pretrained(self.wav2vec2_model)
            cls = Wav2Vec2ForSequenceClassification.from_pretrained(self.wav2vec2_model)
            cls.eval()
            if torch.cuda.is_available():
                cls = cls.cuda()
            self._w2v = cls.wav2vec2
            self._processor = processor
            logger.info("说话人聚类嵌入器(wav2vec2)已加载")
        except Exception as e:
            logger.warning(f"wav2vec2 嵌入器加载失败({str(e)[:80]}), 使用频谱特征")
            self._w2v = False
        return self._w2v, self._processor

    def _embed(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        base, processor = self._ensure_w2v()
        if base is None or base is False:
            return None
        import torch

        # wav2vec2 常见训练采样率 16k
        if sr != 16000:
            try:
                import librosa
                audio = librosa.resample(
                    audio.astype(np.float64), orig_sr=sr, target_sr=16000
                ).astype(np.float32)
                sr = 16000
            except Exception:
                pass
        inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
        dev = next(base.parameters()).device
        inputs = {k: v.to(dev) for k, v in inputs.items() if hasattr(v, "to")}
        with torch.no_grad():
            out = base(**inputs, output_hidden_states=True)
            hidden = out.last_hidden_state  # (1, T, D)
        return hidden.mean(dim=1)[0].cpu().numpy().astype(np.float32)

    @staticmethod
    def _spectral(audio: np.ndarray, sr: int) -> np.ndarray:
        try:
            import librosa

            y = audio.astype(np.float64)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=26)
            feats = []
            for stat in (np.mean, np.std):
                feats.extend(stat(mfcc, axis=1))
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            feats.extend([np.mean(centroid), np.std(centroid)])
            feats.extend([np.mean(bandwidth), np.std(bandwidth)])
            feats.extend([np.mean(contrast), np.std(contrast)])
            # 基频统计增强说话人区分
            try:
                f0, _, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
                f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
                if len(f0) > 0:
                    feats.extend([np.mean(f0), np.std(f0), np.median(f0)])
                else:
                    feats.extend([0.0, 0.0, 0.0])
            except Exception:
                feats.extend([0.0, 0.0, 0.0])
            emb = np.array(feats, dtype=np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            return np.zeros(60, dtype=np.float32)
