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
        self._w2v = None
        self._processor = None
        self.last_meta: Dict = {
            "backend": "none",
            "n_speakers": 0,
            "degraded": False,
            "warning": "",
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

        # 2) 离线聚类回退
        if labels is None:
            if self.engine in ("auto", "pyannote", "wav2vec2_cluster", "cluster"):
                labels = self._cluster_segments(audio, sr, segments, embedding_fn)
                backend = "wav2vec2_cluster"
            else:
                labels = ["SPEAKER_00"] * n
                backend = "forced_single"

        n_spk = len(set(labels))
        warning = ""
        degraded = backend != "pyannote"
        target_spk = max(self.min_speakers, self.expected_speakers)

        # 段数很多却仍少于期望人数 → 大声警告 + 二次拆分
        if n_spk < target_spk and n >= max(self.retry_split_min_segments, target_spk):
            warning = (
                f"分离仅得到 {n_spk} 人 (期望≥{target_spk}, 共 {n} 段), "
                "疑似塌缩; 已尝试更敏感二次拆分"
            )
            logger.warning(f"[diarization] {warning}")
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

        # 规范化标签顺序
        labels = self._renumber(labels)
        n_spk = len(set(labels))
        self.last_meta = {
            "backend": backend,
            "n_speakers": n_spk,
            "degraded": degraded,
            "warning": warning,
            "n_segments": n,
            "distance_threshold": self.distance_threshold,
        }
        logger.info(
            f"Diarization({backend}): {n_spk} speakers / {n} segments"
            + (" [DEGRADED]" if degraded else "")
        )
        return labels

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

        embeds = []
        for seg in segments:
            s = int(max(0.0, seg.get("start", 0.0)) * sr)
            e = int(min(len(audio), seg.get("end", s / sr + 0.1)) * sr)
            # 过短段向两侧扩展, 提高嵌入稳定性
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

        thresh = (
            self.distance_threshold
            if distance_threshold is None
            else float(distance_threshold)
        )
        cluster_ids = self._cluster(
            embeds,
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
