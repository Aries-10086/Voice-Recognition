#!/usr/bin/env bash
# F1: 可选安装 pyannote 说话人分离 (需 HuggingFace Token)
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

export HF_HOME="${HF_HOME:-$(pwd)/models/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"

if [[ -z "${HF_TOKEN:-}" && -z "${HUGGINGFACE_TOKEN:-}" ]]; then
  echo "请先在 https://huggingface.co/settings/tokens 创建 Token"
  echo "并接受 pyannote 模型协议:"
  echo "  https://huggingface.co/pyannote/speaker-diarization-3.1"
  echo "  https://huggingface.co/pyannote/segmentation-3.0"
  echo ""
  echo "然后: export HF_TOKEN=hf_xxx"
  exit 1
fi

pip install -q "pyannote.audio>=3.1" torch torchaudio

python - <<'PY'
import os, torch
from pyannote.audio import Pipeline
token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
kwargs = {"token": token} if token else {}
print("Downloading pyannote/speaker-diarization-3.1 ...")
Pipeline.from_pretrained("pyannote/speaker-diarization-3.1", **kwargs)
print("OK: pyannote cached under", os.environ.get("HF_HOME", "~/.cache"))
PY

echo "Done. Set pipeline.diarization.engine: auto (default) to use pyannote when cached."
