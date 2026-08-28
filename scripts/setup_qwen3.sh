#!/usr/bin/env bash
# Qwen3-TTS 环境 (Mac/Linux)
# 要求: Python >= 3.10 (系统 3.9 无法安装 qwen-tts)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ ! -d .venv ]]; then
  echo "Creating .venv (Python 3.11+)..."
  if command -v uv >/dev/null 2>&1; then
    uv venv .venv --python 3.11
    source .venv/bin/activate
    uv pip install torch torchaudio transformers accelerate
    uv pip install qwen-tts
  else
    python3.11 -m venv .venv 2>/dev/null || python3.12 -m venv .venv
    source .venv/bin/activate
    pip install -U pip
    pip install torch torchaudio transformers accelerate qwen-tts
  fi
else
  source .venv/bin/activate
fi

export HF_HOME="${ROOT}/models/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"

echo "OK: $(python -V)"
python -c "import qwen_tts; print('qwen-tts', qwen_tts.__file__)"
echo "HF_HOME=$HF_HOME"
echo ""
echo "可选预拉模型权重:"
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-0.6B-Base')\""
echo "  python -c \"from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen3-TTS-12Hz-1.7B-Base')\""
echo ""
echo "F5 对比 0.6B vs 1.7B:"
echo "  bash scripts/compare_qwen_models.sh"
echo ""
echo "Run pipeline:"
echo "  source .venv/bin/activate"
echo "  export HF_HOME=$HF_HOME"
echo "  python run_pipeline.py --input data/dialog_two_speakers_16k.wav --target-lang en --device cpu"
