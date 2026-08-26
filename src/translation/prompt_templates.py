"""
提示词模板 - 同音转译 (Phonetic-Aware Translation)
【创新点】翻译时考虑源语言发音特征，实现口型同步的跨语种转译
"""

class PromptTemplates:
    # 同音转译系统提示词
    SYSTEM_PROMPT = """You are a phonetic-aware translator for dubbing. Translate {source_lang} to {target_lang} while:

1. **Phonetic matching**: Choose target words that sound similar to source syllables when possible
2. **Syllable count**: Keep target syllable count close to source ({syllable_count})
3. **Mouth shapes**: Prefer words with similar open/close vowel patterns to source
4. **Natural flow**: Keep the translation conversational and natural for speaking
5. **Meaning**: Preserve the original meaning while optimizing for phonetic fit

Output ONLY the translation, no explanations."""

    EMOTION_TRANSLATION_TEMPLATE = """Source text ({source_lang}, {emotion} tone, {syllable_count} syllables):
{source_text}

Translate to {target_lang} with similar phonetic rhythm:"""

    # 带音节时间轴的翻译提示词
    TIMED_TRANSLATION_TEMPLATE = """Source ({source_lang}, {emotion}, {syllable_count} syllables):
{source_text}

Key vowels (timing in seconds):
{vowel_timeline}

Translate to {target_lang}, matching vowel rhythm and syllable count:"""

    # 翻译后优化提示词
    REFINEMENT_TEMPLATE = """【译文优化任务】
原始翻译: {raw_translation}
情感要求: {emotion}
目标语言: {target_lang}

请优化以上译文，使其：
1. 更符合{emotion}的情感表达
2. 更加口语化和自然
3. 适合语音朗读

只返回优化后的文本。"""

    # 多句段批量翻译
    BATCH_TRANSLATION_TEMPLATE = """请将以下{source_lang}文本翻译为{target_lang}。
整体情感基调: {emotion}
要求保持口语化、情感一致、长度适中。

{segments}

请逐句翻译，每句一行。"""

    @classmethod
    def get_translation_prompt(
        cls,
        source_text: str,
        source_lang: str,
        target_lang: str,
        emotion: str = "neutral",
        intensity: float = 0.5,
        valence: float = 0.0,
        syllable_count: int = 0,
    ) -> str:
        """构建带情感的翻译提示词"""
        return cls.EMOTION_TRANSLATION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            emotion=emotion,
            intensity=intensity,
            valence=valence,
            source_text=source_text,
            syllable_count=syllable_count,
        )

    @classmethod
    def get_refinement_prompt(
        cls,
        raw_translation: str,
        emotion: str,
        target_lang: str,
    ) -> str:
        """构建译文优化提示词"""
        return cls.REFINEMENT_TEMPLATE.format(
            raw_translation=raw_translation,
            emotion=emotion,
            target_lang=target_lang,
        )

    @classmethod
    def get_batch_prompt(
        cls,
        segments: str,
        source_lang: str,
        target_lang: str,
        emotion: str,
    ) -> str:
        """构建批量翻译提示词"""
        return cls.BATCH_TRANSLATION_TEMPLATE.format(
            source_lang=source_lang,
            target_lang=target_lang,
            emotion=emotion,
            segments=segments,
        )
