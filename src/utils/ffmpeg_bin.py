"""定位本机 ffmpeg 可执行文件。"""

from __future__ import annotations

import os
import shutil
from functools import lru_cache
from typing import Optional


_CANDIDATES = (
    "ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
    "/usr/bin/ffmpeg",
)


@lru_cache(maxsize=1)
def find_ffmpeg() -> Optional[str]:
    """返回可用 ffmpeg 路径；找不到则 None。"""
    for cand in _CANDIDATES:
        if os.path.sep in cand or cand.startswith("/"):
            if os.path.isfile(cand) and os.access(cand, os.X_OK):
                return cand
        else:
            hit = shutil.which(cand)
            if hit:
                return hit
    # 回退：imageio-ffmpeg 自带静态二进制（本机未装 brew ffmpeg 时可用）
    try:
        import imageio_ffmpeg

        path = imageio_ffmpeg.get_ffmpeg_exe()
        if path and os.path.isfile(path):
            return path
    except Exception:
        pass
    return None


def require_ffmpeg() -> str:
    path = find_ffmpeg()
    if not path:
        raise RuntimeError(
            "未找到 ffmpeg。请: brew install ffmpeg 或 pip install imageio-ffmpeg"
        )
    return path
