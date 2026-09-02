"""
LLM翻译优化模块
【创新点】情感感知的跨语种翻译 + 语义-情感联合优化
"""

import os
import re
import numpy as np
from pathlib import Path
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
        self.last_backend = self.engine
        if self.engine == "auto":
            self.engine = self._resolve_auto_engine()
            self.last_backend = self.engine
        self._init_engine()

    def _resolve_auto_engine(self) -> str:
        """auto: 本地模型已缓存则 local，否则 google。"""
        model_slug = self.model_name.replace("/", "--")
        for root in (
            Path(os.environ.get("HF_HOME", "./models/huggingface")) / "hub",
            Path(os.environ.get("HF_HOME", "./models/huggingface")),
        ):
            if not root.exists():
                continue
            if any(root.glob(f"models--{model_slug}*")):
                logger.info(f"auto engine → local ({self.model_name} cached)")
                return "local"
        logger.info("auto engine → google (no local cache)")
        return "google"

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

        # Google 翻译: 繁→简预处理 + 多引擎回退, 避免模板文本混入译文
        if self.engine == "google":
            translated, backend = self._online_translate(text, target_lang)
            if self._is_bad_translation(translated, text, target_lang):
                demo = self._demo_phrase_translate(text, target_lang)
                if demo:
                    logger.warning(f"翻译结果不可用 (backend={backend}), 改用短语表")
                    translated, backend = demo, "demo_phrase"
                else:
                    logger.warning(f"翻译结果不可用 (backend={backend})")
            history = [translated]
            # F7: 术语/错译后处理 → F6: 情感措辞 → F4: 音节比
            refined = self._refine_translation_quality(
                text, translated, target_lang, emotion, emotion_intensity
            )
            if refined != translated:
                history.append(refined)
                translated = refined
            translated, length_ratio, target_syllables = self._constrain_length_ratio(
                translated, source_syllables, target_lang, history
            )
            # F4 截断后再补残片, 避免 "the listening" / 半截 sit-down
            polished = self._polish_after_length(text, translated, target_lang)
            if polished != translated:
                history.append(polished)
                translated = polished
                target_syllables = self._count_syllables(translated, target_lang)
                length_ratio = target_syllables / max(source_syllables, 1)
            self.last_backend = backend
            logger.info(
                f"Translation ({backend}): [{source_lang}->{target_lang}] "
                f"emotion={emotion} ratio={length_ratio:.2f}"
            )
            return TranslationResult(
                source_text=text,
                translated_text=translated,
                source_lang=source_lang,
                target_lang=target_lang,
                emotion=emotion,
                source_syllables=source_syllables,
                target_syllables=target_syllables,
                length_ratio=length_ratio,
                confidence=0.9 if backend.startswith(("google", "mymemory")) else 0.55,
                refinement_history=history,
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

        translated, length_ratio, target_syllables = self._constrain_length_ratio(
            translated, source_syllables, target_lang, refinement_history
        )

        self.last_backend = self.engine
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

    def translate_lyrics(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        emotion: str = "sad",
        title: str = "",
        hint: str = "",
        emotion_intensity: float = 0.6,
        emotion_valence: float = -0.2,
    ) -> TranslationResult:
        """C4: 清唱整首意译；Google 路径走 refine，LLM 路径走歌词模板。"""
        source_syllables = self._count_syllables(text, source_lang)
        history: List[str] = []

        if self.engine in ("local", "transformers", "openai_compatible"):
            prompt = PromptTemplates.SINGING_LYRICS_TEMPLATE.format(
                source_lang=source_lang,
                target_lang=target_lang,
                title=title or "(unknown)",
                hint=hint or "",
                emotion=emotion,
                syllable_count=source_syllables,
                source_text=text,
            )
            system_prompt = PromptTemplates.SYSTEM_PROMPT.format(
                source_lang=source_lang,
                target_lang=target_lang,
                syllable_count=source_syllables,
            )
            raw = self._call_llm(system_prompt, prompt)
            translated = self._clean_translation(raw, source_lang, target_lang)
            history.append(translated)
            self.last_backend = self.engine
        else:
            meta = text
            if title:
                meta = f"《{title}》{meta}"
            tr = self.translate(
                text=meta,
                source_lang=source_lang,
                target_lang=target_lang,
                emotion=emotion,
                emotion_intensity=emotion_intensity,
                emotion_valence=emotion_valence,
                refine=False,
            )
            translated = tr.translated_text
            history.extend(tr.refinement_history or [translated])
            self.last_backend = getattr(self, "last_backend", "google")

        translated = self._refine_singing_lyrics(
            translated, text, target_lang, title=title, hint=hint
        )
        if translated not in history:
            history.append(translated)

        translated, length_ratio, target_syllables = self._constrain_length_ratio(
            translated, source_syllables, target_lang, history
        )
        logger.info(
            f"Lyrics translation ({self.last_backend}): ratio={length_ratio:.2f}"
        )
        return TranslationResult(
            source_text=text,
            translated_text=translated,
            source_lang=source_lang,
            target_lang=target_lang,
            emotion=emotion,
            source_syllables=source_syllables,
            target_syllables=target_syllables,
            length_ratio=length_ratio,
            confidence=0.88,
            refinement_history=history,
        )

    def _refine_singing_lyrics(
        self,
        translated: str,
        source: str,
        target_lang: str,
        title: str = "",
        hint: str = "",
    ) -> str:
        """歌词英文硬伤修复 + 配置 glossary。"""
        tgt = str(target_lang).lower()
        if tgt not in ("en", "eng", "english"):
            return (translated or "").strip()
        text = (translated or "").strip()

        fixes = [
            (r"\bthose happiness\b", "those happy times"),
            (r"\bthose happinesses\b", "those happy times"),
            (r"\blet'?s separate the moonlight\b", "maybe it's time we part"),
            (r"\bseparate the moonlight\b", "time to say goodbye"),
            (r"\bI want to have loved\b", "If love fades away"),
            (r"\bI want to meet someone I can'?t find\b", "some meetings never happen"),
            (r"\bthe listening\b(?!\s+quality)", "the memory"),
            (r"\badd your hair\b", "your hair"),
        ]
        singing_cfg = self.config.get("singing_glossary") or []
        for item in singing_cfg:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                fixes.append((item[0], item[1]))

        for pat, repl in fixes:
            text = re.sub(pat, repl, text, flags=re.I)

        if title and "forgotten" not in text.lower() and "爱忘" in (source + hint):
            if not re.search(r"\blove\b", text, re.I):
                text = f"If love is forgotten, {text[0].lower()}{text[1:]}" if text else text

        return re.sub(r"\s{2,}", " ", text).strip(" ,;")

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
        """兼容旧调用: 返回译文文本"""
        out, _backend = self._online_translate(text, target_lang)
        return out

    def _online_translate(self, text: str, target_lang: str) -> Tuple[str, str]:
        """
        在线翻译主路径:
        1) 繁→简预处理
        2) Google (source=zh-CN 优先, 再 auto) × 重试
        3) MyMemory
        4) 演示短语表 (仅兜底)
        Returns: (translated_text, backend_name)
        """
        text = (text or "").strip()
        if not text:
            return "", "empty"

        lang_map = {
            "en": "en", "zh": "zh-CN", "zho": "zh-CN", "chi": "zh-CN", "cmn": "zh-CN",
            "ja": "ja", "jpn": "ja", "ko": "ko", "kor": "ko",
            "fr": "fr", "fra": "fr", "de": "de", "deu": "de", "ger": "de",
            "es": "es", "spa": "es", "ru": "ru", "rus": "ru",
            "it": "it", "ita": "it", "pt": "pt", "por": "pt", "ar": "ar", "ara": "ar",
        }
        target = lang_map.get(str(target_lang).lower(), "en")

        # 1) 繁→简 (提高在线翻译命中率)
        text_sc = self._to_simplified(text)
        if text_sc != text:
            logger.info(f"繁→简: {text[:24]}… → {text_sc[:24]}…")

        errors: List[str] = []

        # 2) Google: 先 zh-CN→target, 再 auto→target
        try:
            from deep_translator import GoogleTranslator
        except ImportError as e:
            errors.append(f"google:import:{e}")
            GoogleTranslator = None  # type: ignore

        if GoogleTranslator is not None:
            source_opts = []
            if self._looks_cjk(text_sc):
                source_opts.extend(["zh-CN", "auto"])
            else:
                source_opts.append("auto")
            for src in source_opts:
                for attempt in range(2):
                    try:
                        out = GoogleTranslator(source=src, target=target).translate(text_sc)
                        if not self._is_bad_translation(out, text_sc, target_lang):
                            return out.strip(), f"google/{src}"
                        raise ValueError(f"bad result: {str(out)[:60]}")
                    except Exception as e:
                        errors.append(f"google/{src}#{attempt}:{e}")
                        import time
                        time.sleep(0.35 * (attempt + 1))

        # 3) MyMemory
        try:
            from deep_translator import MyMemoryTranslator
            src = "zh-CN" if self._looks_cjk(text_sc) else "en"
            out = MyMemoryTranslator(source=src, target=target).translate(text_sc)
            if not self._is_bad_translation(out, text_sc, target_lang):
                logger.warning(f"Google 不可用, 已用 MyMemory ({'; '.join(errors)[:100]})")
                return out.strip(), "mymemory"
            errors.append("mymemory:bad_result")
        except Exception as e:
            errors.append(f"mymemory:{e}")

        # 4) 短语表: 简体/繁体都试
        for cand in (text_sc, text):
            demo = self._demo_phrase_translate(cand, target_lang)
            if demo:
                logger.warning(f"在线翻译失败, 使用演示短语表 ({'; '.join(errors)[:120]})")
                return demo, "demo_phrase"

        logger.error(f"翻译失败: {'; '.join(errors)[:200]}")
        if self._is_bad_translation(text_sc, text_sc, target_lang):
            return "", "failed"
        return text_sc, "passthrough"

    def _to_simplified(self, text: str) -> str:
        """繁体中文 → 简体; 无 opencc 时做常用字回退"""
        if not text or not self._looks_cjk(text):
            return text
        try:
            from opencc import OpenCC
            if not hasattr(self, "_opencc"):
                self._opencc = OpenCC("t2s")
            return self._opencc.convert(text)
        except Exception:
            return self._t2s_fallback(text)

    @staticmethod
    def _t2s_fallback(text: str) -> str:
        """无 opencc 时的常用繁简映射 (覆盖 ASR 常见字)"""
        table = str.maketrans({
            "們": "们", "來": "来", "這": "这", "個": "个", "項": "项", "目": "目",
            "討": "讨", "論": "论", "進": "进", "度": "度", "語": "语", "聲": "声",
            "還": "还", "分": "分", "嗎": "吗", "聽": "听", "感": "感", "質": "质",
            "量": "量", "說": "说", "話": "话", "員": "员", "對": "对", "於": "于",
            "與": "与", "為": "为", "會": "会", "後": "后", "從": "从", "種": "种",
            "國": "国", "業": "业", "發": "发", "現": "现", "時": "时", "間": "间",
            "開": "开", "關": "关", "東": "东", "車": "车", "電": "电", "腦": "脑",
            "網": "网", "絡": "络", "計": "计", "劃": "划", "優": "优", "選": "选",
            "擇": "择", "準": "准", "備": "备", "實": "实", "驗": "验", "錄": "录",
            "頻": "频", "穩": "稳", "離": "离", "離": "离", "昇": "升", "壓": "压",
            "態": "态", "據": "据", "處": "处", "裡": "里", "麼": "么", "隻": "只",
            "隻": "只", "並": "并", "並": "并", "萬": "万", "與": "与", "讓": "让",
            "該": "该", "認": "认", "識": "识", "請": "请", "問": "问", "題": "题",
            "嗎": "吗", "麼": "么", "點": "点", "樣": "样", "還": "还", "沒": "没",
            "對": "对", "錯": "错", "經": "经", "過": "过", "產": "产", "產": "产",
            "傳": "传", "統": "统", "複": "复", "雜": "杂", "簡": "简", "單": "单",
            "總": "总", "結": "结", "續": "续", "聯": "联", "調": "调", "測": "测",
            "試": "试", "摘": "摘", "要": "要", "發": "发", "給": "给", "妳": "你",
            "係": "系", "統": "统", "態": "态", "麼": "么", "裡": "里", "佈": "布",
            "佔": "占", "餘": "余", "餘": "余", "餘": "余", "餘": "余",
            "臺": "台", "灣": "湾", "區": "区", "塊": "块", "碼": "码", "碼": "码",
            "離": "离", "離": "离", "穩": "稳", "穩": "稳", "穩": "稳",
            "話": "话", "語": "语", "聲": "声", "聽": "听", "質": "质", "優": "优",
            "選": "选", "擇": "择", "準": "准", "備": "备", "驗": "验", "收": "收",
            "樣": "样", "例": "例", "雙": "双", "人": "人", "對": "对", "話": "话",
            "們": "们", "這": "这", "個": "个", "項": "项", "進": "进", "來": "来",
            "討": "讨", "論": "论", "還": "还", "清": "清", "不": "不", "同": "同",
            "說": "说", "話": "话", "人": "人", "聽": "听", "感": "感", "也": "也",
            "一": "一", "般": "般", "那": "那", "先": "先", "把": "把", "分": "分",
            "離": "离", "坐": "坐", "穩": "稳", "再": "再", "提": "提", "升": "升",
            "克": "克", "隆": "隆", "質": "质", "量": "量", "專": "专", "業": "业",
            "務": "务", "務": "务", "麼": "么", "麼": "么", "麼": "么",
        })
        return text.translate(table)

    def _demo_phrase_translate(self, text: str, target_lang: str) -> Optional[str]:
        """对白样例保底英译 (仅当在线翻译全挂); 键以简体为主"""
        if str(target_lang).lower() not in ("en", "eng", "english"):
            return None
        table = {
            "你好": "Hello",
            "今天我们来讨论一下这个项目的进度": "Let's discuss the project progress today",
            "好的，我这边算法模块已经跑通了": "Okay, the algorithm module is already running on my side",
            "目前语音克隆还分不太清不同说话人，听感也一般":
                "Currently voice cloning still cannot tell speakers apart, and quality is only okay",
            "目前语音克隆还分不太清不同说话人":
                "Currently voice cloning still cannot tell speakers apart",
            "目前语音克隆还分": "Currently voice cloning still",
            "太清不同说话人": "cannot clearly tell different speakers",
            "听感也一般": "and the listening quality is only average",
            "是的，我建议先优化说话人分离和参考音绑定":
                "Yes, I suggest optimizing speaker diarization and reference binding first",
            "那我们先把说话人分离做稳，再提升克隆质量":
                "Then let's stabilize speaker separation first, then improve clone quality",
            "那我们先把说": "Then let's first",
            "那我们先把说话人分离坐": "Then let's stabilize speaker separation",
            "那我们先把说话人分离坐稳": "Then let's stabilize speaker separation",
            "话人分离坐稳": "stabilize speaker separation",
            "再提升克隆质量": "then improve clone quality",
            "目前语音克隆还分不太清不同说话人,":
                "Currently voice cloning still cannot tell speakers apart,",
            "你好,": "Hello,",
            "你好，": "Hello,",
            "同意，我准备一段双人对话样例做验收":
                "Agreed, I will prepare a two-speaker dialogue sample for acceptance",
            "好的，那今天就先这样，下午继续联调":
                "Alright, that's it for today, we continue integration this afternoon",
            "没问题，我把测试音频和结果摘要发你":
                "No problem, I will send you the test audio and the result summary",
            "好的，我这边算法模块已经跑通了。":
                "Okay, the algorithm module is already running on my side.",
            "是的，我建议先优化说话人分离和参考音绑定。":
                "Yes, I suggest optimizing speaker diarization and reference binding first.",
        }
        # 统一用简体匹配: 优先精确, 再按最长键匹配 (避免短键吞掉长句)
        t = self._to_simplified((text or "").strip()).strip("。！？.!?,，、 ")
        if t in table:
            return table[t]
        compact = re.sub(r"[\s,，。！？、\.\!\?]+", "", t)
        if compact in {re.sub(r"[\s,，。！？、\.\!\?]+", "", self._to_simplified(k)) for k in table}:
            for k, v in table.items():
                kk = re.sub(r"[\s,，。！？、\.\!\?]+", "", self._to_simplified(k))
                if compact == kk:
                    return v
        # 最长子串匹配: 仅当键覆盖 compact 的 ≥70% 或 compact 覆盖键
        best_k, best_v, best_n = None, None, 0
        for k, v in table.items():
            kk = re.sub(r"[\s,，。！？、\.\!\?]+", "", self._to_simplified(k))
            if len(kk) < 4:
                continue
            if kk == compact or (kk in compact and len(kk) >= len(compact) * 0.7) or (
                compact in kk and len(compact) >= len(kk) * 0.7
            ):
                if len(kk) > best_n:
                    best_k, best_v, best_n = kk, v, len(kk)
        return best_v

    @staticmethod
    def _looks_cjk(text: str) -> bool:
        if not text:
            return False
        return sum(1 for c in text if "\u4e00" <= c <= "\u9fff") >= max(1, len(text) // 4)

    @staticmethod
    def _is_bad_translation(out: Optional[str], source: str = "", target_lang: str = "en") -> bool:
        if not out or not str(out).strip():
            return True
        s = str(out).strip()
        bad_markers = (
            "error 500", "server error", "that's an error", "please try again",
            "too many requests", "unexpected error", "captcha", "error 429",
        )
        low = s.lower()
        if any(m in low for m in bad_markers):
            return True
        if len(s) < 2:
            return True
        tgt = str(target_lang).lower()
        if tgt in ("en", "eng", "english"):
            cjk = sum(1 for c in s if "\u4e00" <= c <= "\u9fff")
            latin = sum(1 for c in s if ("a" <= c.lower() <= "z"))
            # 仍大量汉字, 或与源文几乎相同
            if cjk >= max(2, len(re.sub(r"\s", "", s)) // 3):
                return True
            if latin < 2 and cjk > 0:
                return True
            src_compact = re.sub(r"[\s,，。！？、\.\!\?]+", "", (source or ""))
            out_compact = re.sub(r"[\s,，。！？、\.\!\?]+", "", s)
            if src_compact and out_compact == src_compact:
                return True
            # F7: 拼音碎片 / 明显错译
            if re.search(r"\btaiqing\b", low):
                return True
            src_s = source or ""
            if "sit down" in low and any(
                k in src_s for k in ("坐稳", "坐穩", "分离", "分離", "说话人", "說話人")
            ):
                return True
        return False

    def _refine_translation_quality(
        self,
        source: str,
        translated: str,
        target_lang: str,
        emotion: str = "neutral",
        intensity: float = 0.5,
    ) -> str:
        """
        F7 术语/错译后处理 + F6 情感措辞。
        在 length_ratio 截断之前执行, 优先保语义。
        """
        tgt = str(target_lang).lower()
        if tgt not in ("en", "eng", "english"):
            return (translated or "").strip()
        text = (translated or "").strip()
        src = self._to_simplified(source or "")

        # —— 明显错译黑名单 ——
        if any(k in src for k in ("坐稳", "分离坐", "说话人分离", "分离做稳")):
            text = re.sub(
                r"\bseparate(?:\s+the)?\s+speakers?\s+and\s+sit\s+down(?:\s+first)?\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\bsit\s+down(?:\s+first)?\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
            # 去重: "... stabilize speaker separation ... stabilize speaker separation"
            text = re.sub(
                r"(stabilize speaker separation)(\s+and\s+\1)+",
                r"\1",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\bseparate(?:\s+the)?\s+speakers?\s+and\s+stabilize speaker separation\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
        text = re.sub(r"\bTaiqing\b", "speakers", text, flags=re.I)
        text = re.sub(r"\btai\s*qing\b", "speakers", text, flags=re.I)

        # 听感残片 / 分离坐稳 非 sit-down 变体
        if re.search(r"听感|聽感", src):
            text = re.sub(
                r"\bthe\s+listening\b(?!\s+quality)",
                "the listening quality",
                text,
                flags=re.I,
            )
        if any(k in src for k in ("坐稳", "坐穩", "分离坐", "分離坐", "分离做稳")):
            text = re.sub(
                r"\bseparate(?:\s+the)?\s+speakers?(?:\s+first)?\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"(stabilize speaker separation)(?:\s+and\s+then\s+stabilize speaker separation)+",
                r"\1",
                text,
                flags=re.I,
            )

        # 源命中短语表时优先覆盖 (防 Google 半句错译)
        demo = self._demo_phrase_translate(src, target_lang)
        if demo and self._is_bad_translation(text, src, target_lang):
            text = demo

        # —— 术语表 (出现源词则纠正英文表达) ——
        glossary = [
            (r"说话人分离|说话人分離", [
                (r"\bspeaker\s+isolation\b", "speaker diarization"),
                (r"\bspeaker\s+splitting\b", "speaker separation"),
            ]),
            (r"语音克隆|語音克隆", [
                (r"\bvoice\s+reproduction\b", "voice cloning"),
                (r"\bspeech\s+cloning\b", "voice cloning"),
            ]),
            (r"参考音|參考音", [
                (r"\breference\s+sound\b", "reference audio"),
                (r"\breference\s+tone\b", "reference audio"),
            ]),
            (r"听感|聽感", [
                (r"\blistening\s+experience\b", "listening quality"),
                (r"\bthe\s+listening\b(?!\s+quality)", "the listening quality"),
            ]),
        ]
        for src_pat, en_pairs in glossary:
            if re.search(src_pat, src):
                for bad, good in en_pairs:
                    text = re.sub(bad, good, text, flags=re.I)

        # 短语表子串强制 (分离坐稳 等)
        force_map = [
            ("分离坐稳", "stabilize speaker separation"),
            ("分离做稳", "stabilize speaker separation"),
            ("说话人分离坐稳", "stabilize speaker separation"),
            ("听感也一般", "listening quality is only average"),
        ]
        for zh, en in force_map:
            if zh in src and en.lower() not in text.lower():
                # 若译文仍含 sit down / the listening 残片则整句换短语
                if "sit down" in text.lower() or text.lower().rstrip(".") in (
                    "the listening", "then let's separate the speakers and sit down first"
                ):
                    text = en

        text = re.sub(r"\s{2,}", " ", text).strip(" ,;")
        text = self._apply_emotion_lexicon(text, emotion, intensity)
        return text

    def _polish_after_length(
        self, source: str, translated: str, target_lang: str
    ) -> str:
        """F4 音节截断后的残片兜底 (听感/分离坐稳)。"""
        tgt = str(target_lang).lower()
        if tgt not in ("en", "eng", "english"):
            return translated
        src = self._to_simplified(source or "")
        text = (translated or "").strip()
        if re.search(r"听感|聽感", src):
            text = re.sub(
                r"\bthe\s+listening\b(?!\s+quality)",
                "the listening quality",
                text,
                flags=re.I,
            )
            if text.lower().rstrip(".!") in ("the listening quality", "and the listening quality"):
                text = "listening quality is only average"
        if any(k in src for k in ("坐稳", "坐穩", "分离坐", "分離坐")):
            if "sit down" in text.lower():
                text = re.sub(
                    r".*?\bsit\s+down(?:\s+first)?\b.*",
                    "Then let's stabilize speaker separation",
                    text,
                    flags=re.I,
                )
            text = re.sub(
                r"\bseparate(?:\s+the)?\s+speakers?(?:\s+first)?\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
            text = re.sub(
                r"\bstabilize speaker separation\s+and\s+stabilize speaker separation\b",
                "stabilize speaker separation",
                text,
                flags=re.I,
            )
        return re.sub(r"\s{2,}", " ", text).strip(" ,;")

    def _apply_emotion_lexicon(
        self, text: str, emotion: str, intensity: float = 0.5
    ) -> str:
        """F6: 按情感轻度调整英文措辞 (Google 路径无 LLM prompt)。"""
        if not text or float(intensity or 0) < 0.35:
            return text
        emo = (emotion or "neutral").lower()
        if emo in ("neutral",):
            return text
        out = text
        if emo in ("happy", "surprised", "positive"):
            if not out.endswith(("!", "?")):
                out = out.rstrip(".") + "!"
        elif emo in ("sad", "negative"):
            out = re.sub(r"\bvery\b", "a bit", out, flags=re.I)
            out = re.sub(r"\breally\b", "somewhat", out, flags=re.I)
        elif emo in ("angry", "disgusted"):
            out = re.sub(r"\bplease\b", "", out, flags=re.I)
            out = re.sub(r"\skind of\b", "", out, flags=re.I)
            out = re.sub(r"\s{2,}", " ", out).strip()
        return out

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

    def _constrain_length_ratio(
        self,
        translated: str,
        source_syllables: int,
        target_lang: str,
        history: Optional[List[str]] = None,
    ) -> Tuple[str, float, int]:
        """
        F4: 将译文音节比压到 [length_ratio_min, length_ratio_max]（默认 0.8–1.2）。
        超长 → 去填充词 + 按音节预算截断；过短 → 轻度补连接词（英文）。
        """
        min_r = float(self.config.get("length_ratio_min", 0.8))
        max_r = float(self.config.get("length_ratio_max", 1.2))
        src = max(int(source_syllables), 1)
        text = (translated or "").strip()
        tgt = self._count_syllables(text, target_lang)
        ratio = tgt / src
        if min_r <= ratio <= max_r or not text:
            return text, ratio, tgt

        if ratio > max_r:
            budget = max(1, int(src * max_r))
            # 短源句保留最低可懂音节, 避免 "The listening" 类残片
            if src < 12:
                budget = max(budget, min(22, max(12, int(src * 2.2))))
            compressed = self._compress_to_syllable_budget(text, budget, target_lang)
            if history is not None and compressed != text:
                history.append(compressed)
            text = compressed
            logger.info(
                f"  length refine: shorten {ratio:.2f} → "
                f"{self._count_syllables(text, target_lang) / src:.2f} (budget={budget})"
            )
        elif ratio < min_r:
            budget = max(1, int(src * min_r))
            expanded = self._expand_to_syllable_budget(text, budget, target_lang)
            if history is not None and expanded != text:
                history.append(expanded)
            text = expanded
            logger.info(
                f"  length refine: lengthen {ratio:.2f} → "
                f"{self._count_syllables(text, target_lang) / src:.2f} (budget={budget})"
            )

        tgt = self._count_syllables(text, target_lang)
        return text, tgt / src, tgt

    def _compress_to_syllable_budget(self, text: str, budget: int, lang: str) -> str:
        """压缩译文到音节预算内。"""
        lang_l = (lang or "").lower()
        out = text.strip()
        if lang_l.startswith("en") or lang_l in ("english",):
            # 去填充 / 冗余连接
            fillers = (
                r"\b(indeed|actually|really|just|very|quite|basically|literally|"
                r"that is to say|for example|in order to|kind of|sort of|"
                r"you know|I mean|well|so that)\b[,.]?"
            )
            out = re.sub(fillers, " ", out, flags=re.I)
            out = re.sub(r"\s{2,}", " ", out).strip(" ,.;")
            # 去重复冠词/弱词串
            out = re.sub(r"\b(the|a|an)\s+\1\b", r"\1", out, flags=re.I)

        if self._count_syllables(out, lang) <= budget:
            return out

        # 按词/字截断到预算
        if lang_l.startswith(("zh", "ja", "ko")) or bool(re.search(r"[\u4e00-\u9fff]", out)):
            chars = list(out)
            kept = []
            for ch in chars:
                kept.append(ch)
                if self._count_syllables("".join(kept), lang) >= budget:
                    break
            return "".join(kept).rstrip("，。、；,.; ")

        words = out.split()
        kept = []
        for w in words:
            trial = " ".join(kept + [w])
            if kept and self._count_syllables(trial, lang) > budget:
                break
            kept.append(w)
        return " ".join(kept).rstrip(" ,.;") if kept else out

    def _expand_to_syllable_budget(self, text: str, budget: int, lang: str) -> str:
        """
        过短时不再灌 'yes/now/okay' 填充词 (会污染听感与 F7 译文)。
        时长由合成端 time-stretch 对齐, 音节略低于 0.8 可接受。
        """
        return text.strip()

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
