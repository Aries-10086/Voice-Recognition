#!/usr/bin/env python3
"""
跨语种语音翻译与原声复刻系统 - 主运行脚本0

用法:
    # 基本运行
    python run_pipeline.py --input audio.wav --target-lang en

    # 指定输出目录
    python run_pipeline.py --input audio.wav --target-lang ja --output ./my_outputs

    # 使用不同的参考音频进行声纹克隆
    python run_pipeline.py --input speech.wav --ref-audio speaker_ref.wav --target-lang en

    # 逐段处理模式(适合长音频)
    python run_pipeline.py --input long_audio.wav --target-lang en --segment-mode

    # 使用配置文件
    python run_pipeline.py --config config/custom.yaml --input audio.wav
"""

import os
import sys
import argparse
import yaml
from pathlib import Path

# 屏蔽无关警告
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*Passing `gradient_checkpointing`.*")
warnings.filterwarnings("ignore", message=".*`torch_dtype` is deprecated.*")
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "0"   # 保留下载进度条
os.environ["TRANSFORMERS_VERBOSITY"] = "error"       # 只显示错误

# 添加项目根目录 (脚本可能位于项目根目录, 也可能位于 scripts/ 子目录)
def _detect_project_root() -> Path:
    here = Path(__file__).resolve().parent
    for candidate in (here, here.parent):
        if (candidate / "config" / "default.yaml").exists():
            return candidate
    return here

PROJECT_ROOT = _detect_project_root()
sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# 【关键】将模型缓存目录重定向到项目本地
# 仅在 Windows 且 D:/CodingPackage/models 存在时用该路径;
# macOS 上勿匹配项目内名为 "D:" 的文件夹
# ============================================================
import platform
_win_models = Path("D:/CodingPackage/models")
if platform.system() == "Windows" and _win_models.exists():
    _MODEL_ROOT = _win_models
else:
    _MODEL_ROOT = PROJECT_ROOT / "models"
_MODEL_ROOT.mkdir(parents=True, exist_ok=True)

# 若外部已指定 HF_HOME 则保留; 否则写入项目 models/
if not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = str(_MODEL_ROOT / "huggingface")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.environ.get(
    "HUGGINGFACE_HUB_CACHE", str(Path(os.environ["HF_HOME"]) / "hub")
)
os.environ["HF_HUB_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
os.environ["TORCH_HOME"] = str(_MODEL_ROOT / "torch")
os.environ["XDG_CACHE_HOME"] = str(_MODEL_ROOT / ".cache")
# faster-whisper 下载目录
os.environ["CTRANSLATE2_MODELS"] = str(_MODEL_ROOT / "ctranslate2")
# transformers 离线缓存
os.environ["TRANSFORMERS_CACHE"] = os.environ["HUGGINGFACE_HUB_CACHE"]
# sentencepiece / tokenizers
os.environ["SENTENCEPIECE_HOME"] = str(_MODEL_ROOT / "sentencepiece")
# NLTK 数据 (g2p-en 音素转换) — 优先项目 models/nltk_data
_nltk = PROJECT_ROOT / "models" / "nltk_data"
if not _nltk.exists():
    _nltk = _MODEL_ROOT / "nltk_data"
os.environ["NLTK_DATA"] = str(_nltk)
# HF 镜像 (如网络正常则注释掉)
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 注册 NLTK 路径
try:
    import nltk
    nltk.data.path.insert(0, str(_nltk))
except ImportError:
    pass

print(f"Model cache: {_MODEL_ROOT}")
print(f"   HF_HOME = {os.environ['HF_HOME']}")
# ============================================================

from loguru import logger


def load_config(config_path: str = None) -> dict:
    """加载配置文件"""
    if config_path is None:
        config_path = PROJECT_ROOT / "config" / "default.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 转换为绝对路径
    for key in ["output_dir", "model_dir"]:
        if key in config.get("system", {}):
            path = config["system"][key]
            if not os.path.isabs(path):
                config["system"][key] = str(PROJECT_ROOT / path)

    return config


def main():
    parser = argparse.ArgumentParser(
        description="跨语种语音翻译与原声复刻系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --input speech.wav --target-lang en
  %(prog)s --input speech.wav --target-lang ja --ref-audio speaker.wav
  %(prog)s --input long_audio.wav --target-lang en --segment-mode
        """,
    )

    parser.add_argument(
        "--input", "-i", type=str, required=True,
        help="输入音频/视频文件路径 (.wav/.mp4/.mkv等)"
    )
    parser.add_argument(
        "--target-lang", "-t", type=str, default="en",
        help="目标语言代码 (en/zh/ja/ko/fr/de/es...)"
    )
    parser.add_argument(
        "--video", action="store_true",
        help="输入为视频文件, 处理后替换音轨输出视频"
    )
    parser.add_argument(
        "--ref-audio", "-r", type=str, default=None,
        help="参考音频路径(用于声纹克隆,默认使用输入音频)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default=None,
        help="输出目录"
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="配置文件路径"
    )
    parser.add_argument(
        "--segment-mode", action="store_true",
        help="启用逐段处理模式(适合长音频)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="推理设备 (cuda/cpu)"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="详细日志输出"
    )
    parser.add_argument(
        "--prompt-profile",
        choices=["neutral", "singing", "fleurs_dialog"],
        default=None,
        help="ASR 提示 profile (清唱用 singing)",
    )
    parser.add_argument(
        "--singing", action="store_true",
        help="清唱模式 (= --prompt-profile singing)",
    )
    parser.add_argument(
        "--assume-single-speaker", action="store_true",
        help="强制单人分轨/克隆 (清唱推荐)",
    )
    parser.add_argument(
        "--multi-speaker", action="store_true",
        help="清唱也启用多人分离 (覆盖 singing 默认单人)",
    )
    parser.add_argument(
        "--no-gold", action="store_true",
        help="忽略 .gold.json 旁路，强制走 ASR（测清唱识别）",
    )
    parser.add_argument(
        "--asr-engine",
        choices=["faster_whisper", "funasr"],
        default=None,
        help="覆盖 ASR 引擎 (清唱对比可用 funasr)",
    )
    parser.add_argument(
        "--no-lyrics-hint",
        action="store_true",
        help="清唱 ASR 真能力：不用歌名/歌词 hint、不用歌曲纠错表",
    )
    parser.add_argument(
        "--generic-asr-refine",
        action="store_true",
        help="Whisper 后用通用 LLM 修同音错字（不注入歌词）",
    )
    parser.add_argument(
        "--translate-engine",
        choices=["google", "local", "openai_compatible", "auto"],
        default=None,
        help="覆盖 translation.engine (A1: local/auto 走 LLM 主路径)",
    )

    args = parser.parse_args()

    # 配置日志
    if not args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | {message}")

    # 加载配置
    config = load_config(args.config)
    if args.device:
        config.setdefault("system", {})["device"] = args.device
        config.setdefault("asr", {})["device"] = args.device
        # 展平给 pipeline 里少数直接读 config["device"] 的地方
        config["device"] = args.device

    prompt_profile = args.prompt_profile
    if args.singing:
        prompt_profile = "singing"
    if prompt_profile:
        config.setdefault("asr", {})["prompt_profile"] = prompt_profile

    if args.translate_engine:
        config.setdefault("translation", {})["engine"] = args.translate_engine

    if args.asr_engine:
        config.setdefault("asr", {})["engine"] = args.asr_engine

    assume_single = None
    if args.assume_single_speaker:
        assume_single = True
    elif args.multi_speaker:
        assume_single = False

    # 检查输入文件
    if not os.path.exists(args.input):
        logger.error(f"❌ 输入文件不存在: {args.input}")
        sys.exit(1)

    # 导入Pipeline
    from src.integration.pipeline import CrossLingualPipeline

    # 视频处理: 提取音频
    audio_input = args.input
    video_output = None
    if args.video:
        import subprocess, tempfile
        temp_audio = os.path.join(tempfile.gettempdir(), "voiceclone_temp.wav")
        logger.info(f"Extracting audio from video...")
        subprocess.run(["ffmpeg", "-i", args.input, "-ar", "16000", "-ac", "1",
                       temp_audio, "-y"], capture_output=True, check=True)
        audio_input = temp_audio
        video_output = args.output or args.input.replace('.mp4','_cloned.mp4').replace('.mkv','_cloned.mkv')

    # 初始化Pipeline
    pipeline = CrossLingualPipeline(config)

    result = pipeline.run(
        audio_path=audio_input,
        target_lang=args.target_lang,
        reference_audio_path=args.ref_audio,
        output_dir=args.output,
        prompt_profile=prompt_profile,
        assume_single_speaker=assume_single,
        skip_gold=args.no_gold,
        no_lyrics_hint=args.no_lyrics_hint,
        generic_asr_refine=args.generic_asr_refine,
    )

    # 视频: 替换音轨
    if video_output and result.status in ("success", "degraded"):
        import subprocess
        wav_path = os.path.join(args.output or "./outputs", "cloned_output.wav")
        logger.info(f"Replacing audio in video -> {video_output}")
        subprocess.run(["ffmpeg", "-i", args.input, "-i", wav_path,
                       "-c:v", "copy", "-c:a", "aac", "-map", "0:v:0", "-map", "1:a:0",
                       "-shortest", video_output, "-y"],
                      capture_output=True, check=True)
        logger.info(f"Video output: {video_output}")

    if result.status in ("success", "degraded"):
        logger.info(f"Done: {result.processing_time:.1f}s ({result.status})")
    else:
        logger.error(f"Failed: {result.error_message}")
        sys.exit(1)


if __name__ == "__main__":
    main()
