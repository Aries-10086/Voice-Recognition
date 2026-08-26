"""
语种识别模块
支持 MMS-LID / Whisper 内置 / fastText
"""

import numpy as np
from typing import Dict, Optional
from loguru import logger


class LanguageIdentifier:
    """
    语种识别器

    支持:
    - MMS-LID (Meta, 4000+语种)
    - Whisper 内置语言检测
    - fastText 语言识别
    """

    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """加载语种识别模型"""
        try:
            from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
            model_name = self.config.get(
                "model_name", "facebook/mms-lid-4017"
            )
            self.processor = AutoFeatureExtractor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.model.eval()
            import torch
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            logger.info(f"✅ MMS-LID 语种识别模型加载成功: {model_name}")
        except Exception as e:
            logger.warning(f"⚠️ MMS-LID 加载失败,将使用Whisper内置检测: {e}")
            self.model = None

    def identify(self, audio: np.ndarray, sample_rate: int = 16000) -> Dict:
        """
        识别音频语种

        Returns:
            Dict with keys: language, confidence, alternatives
        """
        if self.model is not None:
            return self._identify_mms(audio, sample_rate)
        else:
            # 降级: 返回未知,由ASR模块内部检测
            return {"language": "auto", "confidence": 1.0, "alternatives": []}

    def _identify_mms(self, audio: np.ndarray, sample_rate: int) -> Dict:
        """MMS-LID 语种识别"""
        import torch

        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)[0]

        top_k = min(5, len(probs))
        top_probs, top_indices = torch.topk(probs, top_k)

        lang_map = {0: "eng", 1: "fra", 2: "deu", 3: "spa", 4: "zho", 5: "jpn", 6: "kor"}

        alternatives = []
        for i in range(top_k):
            idx = top_indices[i].item()
            lang_id = self.model.config.id2label.get(idx, f"lang_{idx}")
            alternatives.append({
                "language": lang_id,
                "confidence": round(top_probs[i].item(), 4),
            })

        return {
            "language": alternatives[0]["language"],
            "confidence": alternatives[0]["confidence"],
            "alternatives": alternatives,
        }
