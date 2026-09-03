"""
语音情感识别模块
支持 emotion2vec (2024) / Wav2Vec2-Emotion / SpeechBrain
创新点: 多尺度情感特征提取 + 情感强度量化
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class EmotionResult:
    """情感识别结果"""
    emotion: str                    # 主导情感类别
    scores: Dict[str, float]        # 各类别概率
    embedding: np.ndarray           # 情感嵌入向量
    intensity: float                # 情感强度 (0~1)
    valence: float                  # 效价 (-1~1, 负面~正面)
    arousal: float                  # 唤醒度 (0~1, 平静~激动)
    timeline: List[Dict]            # 情感时间线


class EmotionRecognizer:
    """
    多引擎语音情感识别器 【创新点】

    创新1: 多尺度情感特征提取
    - 同时分析短时(帧级)和长时(句级)情感特征
    - 引入情感强度量化模块,不仅识别类别还量化强度

    创新2: 情感嵌入向量的跨模态对齐
    - 生成与文本embedding同维度的情感向量
    - 可注入到LLM翻译和TTS合成中
    """

    EMOTION_CATEGORIES = [
        "neutral", "happy", "sad", "angry",
        "surprised", "fearful", "disgusted"
    ]

    def __init__(self, config: Dict):
        self.config = config
        self.engine = config.get("engine", "emotion2vec")
        self.model = None
        self.processor = None
        self.embedding_model = None
        self.id2label = {}
        self.embedding_dim = config.get("embedding_dim", 768)
        self._load_model()

    def _load_model(self):
        """加载情感识别模型"""
        if self.engine == "emotion2vec":
            self._load_emotion2vec()
        elif self.engine == "wav2vec2_emotion":
            self._load_wav2vec2_emotion()
        else:
            logger.warning(f"未知引擎 {self.engine}, 使用 emotion2vec")
            self._load_emotion2vec()

    def _load_emotion2vec(self):
        """
        加载 emotion2vec 模型
        emotion2vec 是2024年提出的通用语音情感表示模型
        通过自监督预训练 + 情感微调获得
        """
        try:
            from funasr import AutoModel
            model_name = self.config.get(
                "model_name", "iic/emotion2vec_base_finetuned"
            )
            self.model = AutoModel(
                model=model_name,
                device=self.config.get("device", "cuda:0"),
            )
            logger.info(f"✅ emotion2vec 加载成功: {model_name}")
        except ImportError:
            logger.warning("⚠️ emotion2vec 未安装,使用HuggingFace wav2vec2替代")
            self.engine = "wav2vec2_emotion"
            self._load_wav2vec2_emotion()
        except Exception as e:
            logger.error(f"❌ emotion2vec 加载失败: {e}")
            raise

    def _load_wav2vec2_emotion(self):
        """加载 HuggingFace wav2vec2 情感模型"""
        from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
        # 注意: emotion.model_name 指向 emotion2vec, wav2vec2 使用独立模型名
        model_name = self.config.get("wav2vec2_model", "superb/wav2vec2-base-superb-er")
        self.processor = AutoFeatureExtractor.from_pretrained(model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        import torch
        if torch.cuda.is_available():
            self.model = self.model.cuda()
        # 类别映射: 模型自带的 id2label → 本系统统一情感类别
        self.id2label = dict(self.model.config.id2label) if getattr(self.model.config, "id2label", None) else {}
        # 暴露 wav2vec2 骨干网络, 供说话人聚类复用 (避免重复加载)
        self.embedding_model = getattr(self.model, "wav2vec2", None)
        logger.info("Wav2Vec2-Emotion loaded: " + model_name)

    def recognize(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        return_timeline: bool = True,
    ) -> EmotionResult:
        """
        识别语音中的情感

        Args:
            audio: 音频波形
            sample_rate: 采样率
            return_timeline: 是否返回情感时间线

        Returns:
            EmotionResult 情感分析结果
        """
        try:
            if self.engine == "emotion2vec":
                return self._recognize_emotion2vec(audio, sample_rate, return_timeline)
            return self._recognize_wav2vec2(audio, sample_rate, return_timeline)
        except Exception as e:
            logger.warning(f"⚠️ 情感识别失败，回落 neutral: {e}")
            return EmotionResult(
                emotion="neutral",
                scores={e: 0.0 for e in self.EMOTION_CATEGORIES},
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                intensity=0.0,
                valence=0.0,
                arousal=0.5,
                timeline=[],
            )
    def _recognize_emotion2vec(
        self, audio: np.ndarray, sample_rate: int, return_timeline: bool
    ) -> EmotionResult:
        """emotion2vec 情感识别"""
        result = self.model.generate(
            input=audio,
            cache={},
            granularity="utterance" if not return_timeline else "frame",
        )

        if result and len(result) > 0:
            r = result[0]

            # 获取情感标签和分数（emotion2vec 常返回 labels=list + scores=list）
            labels = r.get("labels")
            scores_raw = r.get("scores")
            if isinstance(labels, (list, tuple)) and labels:
                if isinstance(scores_raw, (list, tuple)) and len(scores_raw) == len(labels):
                    best_i = int(np.argmax(np.asarray(scores_raw, dtype=np.float64)))
                    raw_emotion = labels[best_i]
                    scores_dict = {
                        self._normalize_emotion(str(lbl)): float(sc)
                        for lbl, sc in zip(labels, scores_raw)
                    }
                else:
                    raw_emotion = labels[0]
                    scores_dict = {
                        self._normalize_emotion(str(e)): (
                            float(scores_raw[0]) if isinstance(scores_raw, (list, tuple)) and scores_raw else 1.0
                        )
                        if i == 0 else 0.0
                        for i, e in enumerate(labels)
                    }
                emotion = self._normalize_emotion(str(raw_emotion))
            elif "labels" in r and isinstance(labels, str):
                emotion = self._normalize_emotion(labels)
                scores_dict = {e: 0.0 for e in self.EMOTION_CATEGORIES}
                scores_dict[emotion] = float(
                    scores_raw[0] if isinstance(scores_raw, (list, tuple)) and scores_raw else 1.0
                )
            else:
                emotion = self._normalize_emotion(str(r.get("label", "neutral")))
                scores_dict = {e: float(r.get(f"{e}_score", 0.0)) for e in self.EMOTION_CATEGORIES}
            for cat in self.EMOTION_CATEGORIES:
                scores_dict.setdefault(cat, 0.0)

            # 获取embedding
            if "feats" in r:
                embedding = np.array(r["feats"]).flatten()
                if len(embedding) > self.embedding_dim:
                    embedding = embedding[:self.embedding_dim]
                elif len(embedding) < self.embedding_dim:
                    embedding = np.pad(embedding, (0, self.embedding_dim - len(embedding)))
            else:
                embedding = np.random.randn(self.embedding_dim).astype(np.float32) * 0.01

            # 计算效价和唤醒度
            valence, arousal = self._compute_valence_arousal(emotion)

            # 情感强度
            max_score = max(scores_dict.values()) if scores_dict else 0.5
            intensity = min(max_score / 0.8, 1.0) if max_score > 0 else 0.5

            # 时间线
            timeline = []
            if return_timeline and "frame_labels" in r:
                frame_labels = r["frame_labels"]
                frame_duration = len(audio) / sample_rate / len(frame_labels) if frame_labels else 0
                for i, label in enumerate(frame_labels):
                    timeline.append({
                        "start": i * frame_duration,
                        "end": (i + 1) * frame_duration,
                        "emotion": self._normalize_emotion(label),
                    })

            return EmotionResult(
                emotion=emotion,
                scores=scores_dict,
                embedding=embedding,
                intensity=intensity,
                valence=valence,
                arousal=arousal,
                timeline=timeline,
            )
        else:
            return EmotionResult(
                emotion="neutral",
                scores={e: 0.0 for e in self.EMOTION_CATEGORIES},
                embedding=np.zeros(self.embedding_dim, dtype=np.float32),
                intensity=0.0,
                valence=0.0,
                arousal=0.5,
                timeline=[],
            )

    def _recognize_wav2vec2(
        self, audio: np.ndarray, sample_rate: int, return_timeline: bool
    ) -> EmotionResult:
        """Wav2Vec2 情感识别"""
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        import torch
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # 获取 hidden states 作为 embedding
        hidden = outputs.hidden_states[-1].mean(dim=1)[0].cpu().numpy()
        if len(hidden) > self.embedding_dim:
            hidden = hidden[:self.embedding_dim]

        pred_idx = int(probs.argmax())
        raw_label = self.id2label.get(pred_idx, self.EMOTION_CATEGORIES[pred_idx] if pred_idx < len(self.EMOTION_CATEGORIES) else "neutral")
        emotion = self._normalize_emotion(raw_label)
        scores = {}
        for i in range(len(probs)):
            lbl = self.id2label.get(i, None)
            if lbl is None:
                continue
            scores[self._normalize_emotion(lbl)] = float(probs[i])
        # 补齐缺失类别为 0
        for cat in self.EMOTION_CATEGORIES:
            scores.setdefault(cat, 0.0)

        valence, arousal = self._compute_valence_arousal(emotion)
        intensity = float(probs.max())

        return EmotionResult(
            emotion=emotion,
            scores=scores,
            embedding=hidden,
            intensity=min(intensity / 0.8, 1.0),
            valence=valence,
            arousal=arousal,
            timeline=[],
        )

    @staticmethod
    def _compute_valence_arousal(emotion: str) -> Tuple[float, float]:
        """
        【创新点】情感效价-唤醒度映射
        基于Russell情感环状模型,将离散情感映射到连续VA空间
        """
        from src.utils.stability import coerce_emotion_label

        key = EmotionRecognizer._normalize_emotion(coerce_emotion_label(emotion))
        mapping = {
            "neutral":    (0.0,  0.3),
            "happy":      (0.8,  0.7),
            "sad":        (-0.7, 0.2),
            "angry":      (-0.5, 0.9),
            "surprised":  (0.4,  0.8),
            "fearful":    (-0.6, 0.8),
            "disgusted":  (-0.7, 0.5),
        }
        return mapping.get(key, (0.0, 0.5))

    @staticmethod
    def _normalize_emotion(label: str) -> str:
        """将任意情感标签归一化到本系统统一类别"""
        from src.utils.stability import coerce_emotion_label

        l = coerce_emotion_label(label).lower()
        exact = {
            "happy": "happy", "happiness": "happy", "joy": "happy", "hap": "happy",
            "sad": "sad", "sadness": "sad",
            "angry": "angry", "anger": "angry", "ang": "angry",
            "fearful": "fearful", "fear": "fearful", "fea": "fearful",
            "disgusted": "disgusted", "disgust": "disgusted", "disg": "disgusted",
            "surprised": "surprised", "surprise": "surprised", "surp": "surprised",
            "neutral": "neutral", "calm": "neutral", "neu": "neutral",
        }
        if l in exact:
            return exact[l]
        for k in ("hap", "sad", "ang", "fea", "disg", "surp"):
            if k in l:
                return exact[k]
        return "neutral"

    def extract_embedding(self, audio: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
        """
        提取整段音频的固定维度声学嵌入 (用于说话人聚类/复用)。
        优先使用 wav2vec2 骨干网络, 不可用时回退到频谱统计特征。
        """
        if self.embedding_model is not None and self.processor is not None:
            try:
                import torch
                inputs = self.processor(
                    audio, sampling_rate=sample_rate, return_tensors="pt"
                )
                dev = next(self.embedding_model.parameters()).device
                inputs = {k: v.to(dev) for k, v in inputs.items()}
                with torch.no_grad():
                    out = self.embedding_model(**inputs, output_hidden_states=True)
                    hidden = out.last_hidden_state  # (1, T, D)
                emb = hidden.mean(dim=1)[0].cpu().numpy().astype(np.float32)
                return emb
            except Exception as e:
                logger.warning(f"wav2vec2 嵌入提取失败({e}), 回退统计特征")
        return self._spectral_embedding(audio, sample_rate)

    @staticmethod
    def _spectral_embedding(audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """频谱统计嵌入 (离线兜底)"""
        try:
            import librosa
            mfcc = librosa.feature.mfcc(y=audio.astype(np.float64), sr=sample_rate, n_mfcc=26)
            feats = []
            for stat in (np.mean, np.std):
                feats.extend(stat(mfcc, axis=1))
            centroid = librosa.feature.spectral_centroid(y=audio.astype(np.float64), sr=sample_rate)
            feats.extend([np.mean(centroid), np.std(centroid)])
            emb = np.array(feats, dtype=np.float32)
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception:
            return np.zeros(54, dtype=np.float32)
