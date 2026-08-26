#!/usr/bin/env python3
"""
Demo脚本 - 快速演示系统核心功能

此脚本生成演示音频并运行完整Pipeline,
展示AOCP-Net发音开闭感知和时间轴生成的核心创新。
"""

import os
import sys
import time
import numpy as np
from pathlib import Path

# 屏蔽无关警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 【关键】将模型缓存目录重定向到项目本地，避免占用C盘
# ============================================================
_MODEL_ROOT = Path("D:/CodingPackage/models")
_MODEL_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(_MODEL_ROOT / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
os.environ["HF_HUB_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
os.environ["TORCH_HOME"] = str(_MODEL_ROOT / "torch")
os.environ["XDG_CACHE_HOME"] = str(_MODEL_ROOT / ".cache")
os.environ["TRANSFORMERS_CACHE"] = str(_MODEL_ROOT / "huggingface" / "hub")
os.environ["NLTK_DATA"] = str(_MODEL_ROOT / "nltk_data")
import nltk
nltk.data.path.insert(0, str(_MODEL_ROOT / "nltk_data"))
# ============================================================

import yaml
from loguru import logger


def generate_demo_audio(output_path: str, duration: float = 5.0):
    """
    生成演示用模拟语音

    模拟一段中文语音:
    "你好，今天天气真不错，我们一起去公园散步吧"

    包含不同情感色彩和自然的停顿
    """
    sr = 16000
    t = np.arange(int(duration * sr)) / sr

    # 模拟不同发音段
    audio = np.zeros_like(t, dtype=np.float32)

    # 定义发音段: (开始时间, 结束时间, 开口度, 基频, 文本)
    segments = [
        (0.2, 0.8, 0.7, 220, "你好"),           # 开口较大
        (0.9, 1.0, 0.2, 0, ""),                  # 停顿
        (1.0, 1.7, 0.6, 230, "今天天气"),         # 正常开口
        (1.8, 2.5, 0.85, 250, "真不错"),          # 开心,开口大
        (2.6, 2.8, 0.2, 0, ""),                  # 停顿
        (2.8, 3.4, 0.65, 225, "我们一起去"),      # 正常
        (3.5, 4.1, 0.7, 235, "公园散步"),         # 较大开口
        (4.2, 4.6, 0.55, 210, "吧"),             # 结尾收束
    ]

    for start_s, end_s, openness, base_freq, text in segments:
        if base_freq == 0:  # 静音
            continue

        seg_t = t[int(start_s * sr):int(end_s * sr)]
        seg_dur = end_s - start_s

        # 复杂谐波信号
        signal = np.zeros_like(seg_t, dtype=np.float32)

        # 基频 + 谐波
        for harmonic in range(1, 6):
            amplitude = 1.0 / (harmonic ** 1.5)  # 谐波衰减
            signal += amplitude * np.sin(2 * np.pi * base_freq * harmonic * seg_t)

        # 共振峰效果
        f1 = 300 + openness * 600
        f2 = 1200 + openness * 1500
        signal += 0.15 * np.sin(2 * np.pi * f1 * seg_t) * openness
        signal += 0.1 * np.sin(2 * np.pi * f2 * seg_t) * openness

        # 添加少量噪声(模拟真实语音)
        signal += 0.02 * np.random.randn(len(seg_t))

        # 包络
        env = np.ones(len(seg_t))
        attack = int(0.02 * sr)  # 20ms attack
        release = int(0.03 * sr)  # 30ms release
        if len(env) > attack + release:
            env[:attack] = np.linspace(0, 1, attack)
            env[-release:] = np.linspace(1, 0, release)
        else:
            env = np.hanning(len(env))

        signal *= env
        signal *= 0.7  # 整体音量

        start_idx = int(start_s * sr)
        end_idx = start_idx + len(signal)
        if end_idx <= len(audio):
            audio[start_idx:end_idx] += signal

    # 归一化
    max_val = np.abs(audio).max()
    if max_val > 1.0:
        audio /= max_val

    # 保存
    import soundfile as sf
    sf.write(output_path, audio, sr)
    logger.info(f"📁 演示音频已生成: {output_path} ({duration:.1f}s)")


def main():
    """运行Demo"""

    # 加载配置
    config_path = PROJECT_ROOT / "config" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_dir = str(PROJECT_ROOT / "outputs" / "demo")
    os.makedirs(output_dir, exist_ok=True)

    # 生成演示音频
    demo_audio_path = os.path.join(output_dir, "demo_input.wav")
    generate_demo_audio(demo_audio_path, duration=5.0)

    # ================================================================
    # Demo Part 1: 独立测试 AOCP-Net (核心创新模块)
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🔬 Demo Part 1: AOCP-Net 发音开闭感知测试")
    logger.info("=" * 60)

    from src.alignment.articulatory_analyzer import AOCPNet
    from src.utils.audio_utils import AudioUtils

    audio, sr = AudioUtils.load_audio(demo_audio_path)

    # 初始化AOCP-Net
    aocp = AOCPNet(config["alignment"])
    result = aocp.predict(audio, sr, device="cpu")

    logger.info(f"📊 AOCP-Net 分析结果:")
    logger.info(f"   - 检测到 {len(result.state_segments)} 个状态段")
    logger.info(f"   - 开口段数: {sum(1 for s in result.state_segments if s['state'] == 'open')}")
    logger.info(f"   - 闭口段数: {sum(1 for s in result.state_segments if s['state'] == 'closed')}")
    logger.info(f"   - 过渡段数: {sum(1 for s in result.state_segments if s['state'] == 'transition')}")
    logger.info(f"   - 平均开合度: {result.openness.mean():.3f}")

    logger.info(f"\n📋 状态段时间轴:")
    for seg in result.state_segments:
        bar = "█" * int(seg["duration"] * 20) if seg["state"] == "open" else "░" * int(seg["duration"] * 20)
        logger.info(f"   [{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['state']:12s} {bar}")

    logger.info(f"\n📋 口型同步关键点:")
    for item in result.timeline[:10]:  # 显示前10个
        if item["is_speech_core"]:
            logger.info(
                f"   ⭐ [{item['start_s']:.2f}s - {item['end_s']:.2f}s] "
                f"openness={item['openness']:.2f} (核心发音区)"
            )

    # ================================================================
    # Demo Part 2: 音素对齐与时间轴生成
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🔬 Demo Part 2: 音素对齐与时间轴生成")
    logger.info("=" * 60)

    from src.alignment.phoneme_aligner import PhonemeAligner
    from src.alignment.timeline_generator import TimelineGenerator

    # 模拟ASR文本
    demo_text = "你好今天天气真不错我们一起去公园散步吧"
    demo_phonemes = [
        "ni", "hao", "jin", "tian", "tian", "qi", "zhen", "bu",
        "cuo", "wo", "men", "yi", "qi", "qu", "gong", "yuan",
        "san", "bu", "ba"
    ]

    # 音素对齐
    aligner = PhonemeAligner(config["alignment"])
    alignment = aligner.align(audio, demo_phonemes, sr)
    logger.info(f"📊 音素对齐: {len(alignment.phonemes)} 个音素")

    # 显示每个音素的开合度分类
    logger.info(f"\n📋 音素-开合度映射:")
    for ph, start, end in zip(alignment.phonemes[:10], alignment.start_times[:10], alignment.end_times[:10]):
        open_class, openness = aligner.classify_articulatory_openness(ph)
        bar = "▓" * int(openness * 20)
        logger.info(f"   {ph:8s} [{start:.2f}s-{end:.2f}s] {open_class:8s} openness={openness:.2f} {bar}")

    # 生成时间轴
    timeline_gen = TimelineGenerator(config["alignment"])
    timeline = timeline_gen.generate(
        aocp_result=result,
        phoneme_alignment=alignment,
        source_duration=len(audio) / sr,
        target_phonemes=demo_phonemes,  # 假设翻译后音素类似
        length_ratio=1.1,  # 假设译文稍长
    )
    logger.info(f"\n📊 时间轴生成结果:")
    logger.info(f"   - 总时长: {timeline.total_duration:.2f}s")
    logger.info(f"   - 说话段数: {len(timeline.speech_segments)}")
    logger.info(f"   - 同步关键点: {len(timeline.sync_points)}")
    logger.info(f"   - 同步质量: {timeline.sync_score:.2f}")
    logger.info(f"   - 覆盖度: {timeline.coverage:.2f}")

    # ================================================================
    # Demo Part 3: 语音合成
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("🔬 Demo Part 3: 情感保持的语音克隆")
    logger.info("=" * 60)

    from src.asr.emotion_recognition import EmotionRecognizer

    emotion_recognizer = EmotionRecognizer(config["emotion"])
    emotion_result = emotion_recognizer.recognize(audio, sr)
    logger.info(f"📊 情感识别: {emotion_result.emotion} (强度={emotion_result.intensity:.2f})")

    from src.synthesis.voice_cloner import VoiceCloner

    cloner = VoiceCloner(config["synthesis"])
    cloned = cloner.clone(
        text="Hello, the weather is really nice today, let's go for a walk in the park together!",
        reference_audio=audio,
        reference_sample_rate=sr,
        emotion_embedding=emotion_result.embedding,
        emotion_label=emotion_result.emotion,
        timeline_constraints=timeline.speech_segments,
        target_lang="en",
    )

    # 保存克隆语音
    output_path = os.path.join(output_dir, "demo_cloned_output.wav")
    AudioUtils.save_audio(cloned.audio, output_path, cloned.sample_rate)
    logger.info(f"\n✅ Demo完成! 输出保存至: {output_path}")
    logger.info(f"   - 时长: {cloned.duration:.2f}s")
    logger.info(f"   - 质量评分: {cloned.quality_score:.2f}")
    logger.info(f"   - 情感保持: {'✓' if cloned.emotion_preserved else '✗'}")

    # ================================================================
    # 总结报告
    # ================================================================
    logger.info("\n" + "=" * 60)
    logger.info("📊 系统创新点总结")
    logger.info("=" * 60)
    logger.info("""
    1. AOCP-Net (发音开闭感知网络):
       - 纯音频驱动的口腔开合度预测
       - 多尺度声学特征提取
       - 时域-频域交叉注意力融合

    2. 情感感知的LLM翻译优化:
       - 情感条件化翻译提示词设计
       - 语义-情感联合优化
       - 发音长度约束的译文生成

    3. 多源信息融合的时间轴生成:
       - AOCP开闭状态 + 音素对齐联合
       - 跨语种时长自适应调整
       - 口型同步关键点自动提取

    4. 情感保持的零样本语音克隆:
       - 情感-音色解耦建模
       - 情感嵌入注入TTS韵律控制
       - 跨语种音色一致性保持
    """)


if __name__ == "__main__":
    main()
