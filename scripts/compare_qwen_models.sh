#!/usr/bin/env bash
# F5: 对比 Qwen3-TTS 0.6B vs 1.7B (下载缺失权重 + 跑对比)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
source .venv/bin/activate
export HF_HOME="${ROOT}/models/huggingface"
export HUGGINGFACE_HUB_CACHE="${HF_HOME}/hub"
mkdir -p "$HF_HOME"

MODELS="${1:-0.6B,1.7B}"
echo "HF_HOME=$HF_HOME"
echo "Comparing: $MODELS"

# 预拉权重 (huggingface_hub)
python - <<'PY' "$MODELS"
import sys
from huggingface_hub import snapshot_download

tags = [t.strip() for t in sys.argv[1].split(",") if t.strip()]
ids = {
    "0.6B": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "1.7B": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
}
for t in tags:
    mid = ids[t]
    print(f"Ensuring {mid} ...")
    path = snapshot_download(repo_id=mid)
    print(f"  -> {path}")
PY

python scripts/compare_qwen_models.py --models "$MODELS"
echo ""
echo "Listen:"
echo "  outputs/f5_compare/clone_0.6B.wav"
echo "  outputs/f5_compare/clone_1.7B.wav"
echo "Report: outputs/f5_compare/report.md"
