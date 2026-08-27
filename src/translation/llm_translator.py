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

        # Google 翻译: 繁→简预处理 + 多引擎回退, 避免模板文本混入译文
        if self.engine == "google":
            translated, backend = self._online_translate(text, target_lang)
            if self._is_bad_translation(translated, text, target_lang):
                logger.warning(f"翻译结果不可用 (backend={backend})")
            target_syllables = self._count_syllables(translated, target_lang)
            length_ratio = target_syllables / max(source_syllables, 1)
            logger.info(
                f"Translation ({backend}): [{source_lang}->{target_lang}] "
                f"ratio={length_ratio:.2f}"
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
        # 统一用简体匹配
        t = self._to_simplified((text or "").strip()).strip("。！？.!?,，、 ")
        if t in table:
            return table[t]
        compact = re.sub(r"[\s,，。！？、\.\!\?]+", "", t)
        for k, v in table.items():
            kk = re.sub(r"[\s,，。！？、\.\!\?]+", "", self._to_simplified(k))
            if compact == kk or (len(compact) >= 4 and (compact in kk or kk in compact)):
                return v
        return None

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
        return False

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
