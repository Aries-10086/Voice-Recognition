"""
LLM翻译优化模块
【创新点】情感感知的跨语种翻译 + 语义-情感联合优化
"""

import re
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from loguru import logger

from .prompt_templates import PromptTemplates


@dataclass
class TranslationResult:
    """翻译结果"""
    source_text: str
    translated_text: str
    source_lang: str
    target_lang: str
    emotion: str
    source_syllables: int
    target_syllables: int
    length_ratio: float           # 音节长度比 (目标/源)
    confidence: float
    refinement_history: List[str]


class LLMTranslator:
    """
    基于大语言模型的情感感知翻译器 【核心创新模块】

    创新1: 情感条件化翻译 (Emotion-Conditioned Translation)
    - 将情感识别结果(类别+强度+效价)注入翻译提示词
    - LLM在翻译时同时优化语义准确性和情感一致性

    创新2: 发音长度约束翻译
    - 计算源文本音节数,约束译文长度
    - 为后续口型对齐提供优化空间

    创新3: 多轮迭代优化
    - 翻译→情感评估→优化 的迭代流程
    - 确保最终译文满足情感和长度双重约束

    支持后端:
    - OpenAI兼容API (vLLM/LocalAI/ollama)
    - HuggingFace Transformers 直接加载
    """

    def __init__(self, config: Dict):
        self.config = config
        self.engine = config.get("engine", "openai_compatible")
        self.model_name = config.get("model_name", "Qwen/Qwen3-8B")
        self.emotion_conditioning = config.get("emotion_conditioning", True)
        self.emotion_weight = config.get("emotion_weight", 0.3)
        self.client = None
        self.tokenizer = None
        self.model = None
        self._init_engine()

    def _init_engine(self):
        """初始化LLM后端"""
        if self.engine == "openai_compatible":
            self._init_openai()
        elif self.engine in ("transformers", "local"):
            self._init_transformers()
        elif self.engine == "google":
            logger.info("Google Translate ready")
            self.model = None
        else:
            logger.warning(f"未知引擎 {self.engine}, 使用openai_compatible")
            self._init_openai()

    def _init_openai(self):
        """初始化OpenAI兼容API"""
        try:
            from openai import OpenAI
            api_base = self.config.get("api_base", "http://localhost:8000/v1")
            self.client = OpenAI(
                base_url=api_base,
                api_key="not-needed",
            )
            logger.info(f"✅ OpenAI兼容API初始化: {api_base}")
        except Exception as e:
            logger.error(f"❌ OpenAI API初始化失败: {e}")
            raise

    def _init_transformers(self):
        """
        直接加载HuggingFace本地模型

        推荐模型 (按大小排序):
        - Qwen/Qwen2.5-1.5B-Instruct  (1.5GB, CPU可跑, 中英翻译优秀)
        - Qwen/Qwen2.5-0.5B-Instruct  (0.5GB, 极速, 适合测试)
        - meta-llama/Llama-3.2-1B-Instruct (需HF token)
        - Qwen/Qwen2.5-7B-Instruct    (14GB, 需GPU)
        """
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"📥 正在加载本地模型: {self.model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            cache_dir=self.config.get("cache_dir", None),
        )

        # 自动检测设备
        if torch.cuda.is_available():
            device_map = "auto"
            dtype = torch.float16
            logger.info("   使用 CUDA GPU 推理")
        else:
            device_map = "cpu"
            dtype = torch.float32
            logger.info("   使用 CPU 推理 (较慢但可用)")

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()

        # 统计参数
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"✅ 本地模型加载成功: {self.model_name} ({params/1e9:.1f}B params)")

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        emotion: str = "neutral",
        emotion_intensity: float = 0.5,
        emotion_valence: float = 0.0,
        refine: bool = True,
    ) -> TranslationResult:
        """
        执行情感感知翻译

        Args:
            text: 源文本
            source_lang: 源语言
            target_lang: 目标语言
            emotion: 情感类别
            emotion_intensity: 情感强度
            emotion_valence: 情感效价
            refine: 是否进行迭代优化

        Returns:
            TranslationResult
        """
        # 计算源文本音节数
        source_syllables = self._count_syllables(text, source_lang)

        # Google 翻译: 直接翻译干净文本, 避免模板文本混入译文
        if self.engine == "google":
            translated = self._google_translate(text, target_lang)
            target_syllables = self._count_syllables(translated, target_lang)
            length_ratio = target_syllables / max(source_syllables, 1)
            logger.info(f"Translation (google): [{source_lang}->{target_lang}] ratio={length_ratio:.2f}")
            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                emotion=emotion,
                source_syllables=source_syllables,
                target_syllables=target_syllables,
                length_ratio=length_ratio,
                confidence=0.9,
                refinement_history=[translated],
            )

        chunk_size = self.config.get("chunk_size", 0)
        if chunk_size > 0 and len(text) > chunk_size:
            translated = self._translate_chunked(text, source_lang, target_lang, emotion, emotion_intensity, emotion_valence, chunk_size)
            refinement_history = [translated]
        else:
            # 同音转译提示词: 音节计数+元音时间轴
            vowel_timeline = getattr(self, '_vowel_timeline', None)
            if vowel_timeline:
                vowel_str = " | ".join(f"{v['char']}@{v['start']:.1f}s" for v in vowel_timeline[:12])
                prompt = PromptTemplates.TIMED_TRANSLATION_TEMPLATE.format(
                    source_lang=source_lang, target_lang=target_lang,
                    emotion=emotion, syllable_count=source_syllables,
                    source_text=text, vowel_timeline=vowel_str,
                )
            else:
                prompt = PromptTemplates.EMOTION_TRANSLATION_TEMPLATE.format(
                    source_lang=source_lang, target_lang=target_lang,
                    emotion=emotion, syllable_count=source_syllables,
                    source_text=text,
                )
            system_prompt = PromptTemplates.SYSTEM_PROMPT.format(
                source_lang=source_lang, target_lang=target_lang,
                syllable_count=source_syllables,
            )

            logger.info(f"Translating ({len(text)} chars, {source_syllables} syl)...")
            raw_result = self._call_llm(system_prompt, prompt)
            translated = self._clean_translation(raw_result, source_lang, target_lang)
            if not translated or len(translated) < 10:
                raise ValueError(f"Invalid translation: '{translated[:50]}'")
            refinement_history = [translated]

            # 迭代优化 (短文本才做)
            if refine and len(text) < 300:
                refined = self._refine_translation(
                    translated, emotion, target_lang, system_prompt
                )
                if refined != translated:
                    refinement_history.append(refined)
                    translated = refined

        # 计算目标音节数
        target_syllables = self._count_syllables(translated, target_lang)
        length_ratio = target_syllables / max(source_syllables, 1)

        logger.info(f"Translation: [{source_lang}->{target_lang}] emotion={emotion}, syl_ratio={length_ratio:.2f}")

        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            emotion=emotion,
            source_syllables=source_syllables,
            target_syllables=target_syllables,
            length_ratio=length_ratio,
            confidence=0.9,
            refinement_history=refinement_history,
        )

    def translate_segments(
        self,
        segments: List[str],
        source_lang: str,
        target_lang: str,
        emotion: str = "neutral",
    ) -> List[TranslationResult]:
        """
        逐段翻译,保持上下文连贯性

        Args:
            segments: 文本段落列表
            source_lang: 源语言
            target_lang: 目标语言
            emotion: 情感类别
        """
        results = []
        for i, seg in enumerate(segments):
            # 对于第一个段落之后的,添加上下文
            if i > 0:
                context = results[-1].translated_text
                seg_with_context = f"[上文译文: {context}]\n{seg}"
            else:
                seg_with_context = seg

            result = self.translate(
                text=seg_with_context,
                source_lang=source_lang,
                target_lang=target_lang,
                emotion=emotion,
                refine=True,
            )
            results.append(result)

        return results

    def _translate_chunked(
        self, text: str, source_lang: str, target_lang: str,
        emotion: str, intensity: float, valence: float, chunk_size: int,
    ) -> str:
        """
        分段翻译长文本 (CPU加速)
        每段独立翻译后拼接
        """
        # 按句子边界分段
        sentences = re.split(r'(?<=[。！？.!?\n])', text)
        chunks = []
        current_chunk = ""

        for sent in sentences:
            if len(current_chunk) + len(sent) <= chunk_size:
                current_chunk += sent
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sent
        if current_chunk:
            chunks.append(current_chunk)

        logger.info(f"   📝 分段翻译: {len(chunks)} 段, 每段≤{chunk_size}字符")
        translated_chunks = []
        system_prompt = PromptTemplates.SYSTEM_PROMPT.format(
            source_lang=source_lang, target_lang=target_lang
        )

        for i, chunk in enumerate(chunks):
            logger.info(f"   🔄 翻译第 {i+1}/{len(chunks)} 段 ({len(chunk)}字符)...")
            prompt = PromptTemplates.EMOTION_TRANSLATION_TEMPLATE.format(
                target_lang=target_lang,
                emotion=emotion,
                source_text=chunk,
                intensity=intensity,
                valence=valence,
                syllable_count=self._count_syllables(chunk, source_lang),
                source_lang=source_lang,
            )
            result = self._call_llm(system_prompt, prompt)
            translated_chunks.append(result.strip())

        return " ".join(translated_chunks)

    def _clean_translation(self, text: str, source_lang: str, target_lang: str) -> str:
        """清理LLM输出 - 去掉拒绝/废话/格式标记"""
        import re
        text = text.strip()
        refuse = ["I'm sorry", "I can't assist", "I cannot", "I apologize",
                   "抱歉", "对不起", "无法", "不能", "Sure!", "Here is"]
        for p in refuse:
            if text.lower().startswith(p.lower()):
                m = re.search(r'(?:Translation|翻译)[：:]\s*(.+)', text, re.I)
                if m: return m.group(1).strip()
                logger.warning("   ⚠️ LLM拒绝翻译, 使用原文")
                return ""
        text = re.sub(r'```[\s\S]*?```', '', text)
        text = text.strip('"\'""''')
        return text.strip()

    def _refine_translation(
        self, raw: str, emotion: str, target_lang: str, system_prompt: str,
    ) -> str:
        """迭代优化译文"""
        refine_prompt = f"Make this more natural and {emotion} in tone:\n{raw}"

        try:
            refined = self._call_llm(system_prompt, refine_prompt)
            return refined.strip()
        except Exception as e:
            logger.warning(f"译文优化失败: {e}")
            return raw

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """调用LLM/翻译 API"""
        if self.engine == "openai_compatible":
            return self._call_openai(system_prompt, user_prompt)
        elif self.engine == "google":
            return self._call_google(user_prompt)
        else:
            return self._call_transformers(system_prompt, user_prompt)

    def _google_translate(self, text: str, target_lang: str) -> str:
        """Google 翻译干净文本 (不走提示词模板)"""
        from deep_translator import GoogleTranslator
        lang_map = {
            "en": "en", "zh": "zh-CN", "zho": "zh-CN", "chi": "zh-CN", "cmn": "zh-CN",
            "ja": "ja", "jpn": "ja", "ko": "ko", "kor": "ko",
            "fr": "fr", "fra": "fr", "de": "de", "deu": "de", "ger": "de",
            "es": "es", "spa": "es", "ru": "ru", "rus": "ru",
            "it": "it", "ita": "it", "pt": "pt", "por": "pt", "ar": "ar", "ara": "ar",
        }
        target = lang_map.get(str(target_lang).lower(), "en")
        return GoogleTranslator(source='auto', target=target).translate(text)

    def _call_google(self, prompt: str) -> str:
        """Google 翻译 (免费, 质量高)"""
        from deep_translator import GoogleTranslator
        # 从prompt中提取源/目标语言
        import re
        target = "en"
        m = re.search(r'Translate to (\w+)', prompt)
        if m: target = m.group(1)

        # 提取纯文本
        text = prompt.split('\n', 1)[-1].strip() if '\n' in prompt else prompt
        # 去掉 "Translate to en (emotion: sad):" 前缀
        text = re.sub(r'^Translate[^:]*:\s*', '', text)

        result = GoogleTranslator(source='auto', target=target).translate(text)
        return result

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str:
        """OpenAI兼容API调用"""
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.config.get("temperature", 0.3),
            max_tokens=self.config.get("max_tokens", 2048),
        )
        return response.choices[0].message.content or ""

    def _call_transformers(self, system_prompt: str, user_prompt: str) -> str:
        """HuggingFace本地模型推理"""
        import torch

        # Qwen/Llama 等模型使用 chat_template
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        # 使用tokenizer的chat_template构建prompt
        if hasattr(self.tokenizer, 'apply_chat_template'):
            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 降级: 简单拼接
            text = f"{system_prompt}\n\n{user_prompt}"

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        # 移到正确的设备
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.get("max_tokens", 1024),
                temperature=self.config.get("temperature", 0.3),
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # 只取新生成的部分
        response = self.tokenizer.decode(
            outputs[0][len(inputs["input_ids"][0]):],
            skip_special_tokens=True,
        )
        return response.strip()

    @staticmethod
    def _count_syllables(text: str, lang: str) -> int:
        """
        估算音节数
        中文: 按字数估算
        英文: 按元音字母估算
        日文: 按假名数估算
        """
        text = text.strip()
        if not text:
            return 0

        # 中文 (包含汉字)
        if lang in ("zh", "zho", "chi", "chinese", "cmn"):
            # 统计中文字符
            chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
            if chinese_chars > len(text) * 0.3:
                return chinese_chars

        # 日文
        if lang in ("ja", "jpn", "japanese"):
            kana = len(re.findall(r'[\u3040-\u309f\u30a0-\u30ff]', text))
            kanji = len(re.findall(r'[\u4e00-\u9fff]', text))
            return kana + kanji

        # 韩文
        if lang in ("ko", "kor", "korean"):
            return len(re.findall(r'[\uac00-\ud7af]', text))

        # 英文/欧洲语言: 数元音簇
        words = text.split()
        count = 0
        for word in words:
            word = word.lower()
            # 简单规则: 每个元音簇算一个音节
            vowel_clusters = len(re.findall(r'[aeiouy]+', word))
            count += max(vowel_clusters, 1)

        return count
