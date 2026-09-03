"""
视频处理工具集
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from loguru import logger


class VideoUtils:
    """视频处理工具"""

    @staticmethod
    def extract_audio_from_video(
        video_path: str,
        target_sr: int = 16000,
    ) -> Tuple[np.ndarray, int]:
        """从视频中提取音频"""
        try:
            import subprocess
            import tempfile
            import os
            from .ffmpeg_bin import require_ffmpeg

            ffmpeg = require_ffmpeg()
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            cmd = [
                ffmpeg, "-i", video_path,
                "-af", "highpass=f=80,loudnorm=I=-16:TP=-1.5:LRA=11",
                "-ar", str(target_sr),
                "-ac", "1",
                "-f", "wav",
                "-y", temp_path,
            ]
            proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                # loudnorm 在极短/静音片上可能失败，回退无滤镜抽音
                logger.warning("带增强抽音失败，回退纯抽音")
                cmd = [
                    ffmpeg, "-i", video_path,
                    "-ar", str(target_sr),
                    "-ac", "1",
                    "-f", "wav",
                    "-y", temp_path,
                ]
                proc = subprocess.run(cmd, capture_output=True, text=True)
            if proc.returncode != 0:
                err = (proc.stderr or proc.stdout or "")[-800:]
                try:
                    os.unlink(temp_path)
                except OSError:
                    pass
                raise RuntimeError(f"ffmpeg 抽音失败 (code={proc.returncode}): {err}")

            from .audio_utils import AudioUtils
            audio, sr = AudioUtils.load_audio(temp_path, target_sr=target_sr)
            os.unlink(temp_path)

            logger.info(f"🎬 从视频提取音频: {video_path}")
            return audio, sr

        except Exception as e:
            logger.error(f"❌ 提取音频失败: {e}")
            raise

    @staticmethod
    def replace_audio_in_video(
        video_path: str,
        audio: np.ndarray,
        audio_sr: int,
        output_path: str,
    ) -> str:
        """替换视频中的音频"""
        import subprocess
        import tempfile
        import os
        from .ffmpeg_bin import require_ffmpeg

        ffmpeg = require_ffmpeg()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name

        import soundfile as sf
        sf.write(temp_audio, audio, audio_sr)

        cmd = [
            ffmpeg, "-i", video_path,
            "-i", temp_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y", output_path,
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        os.unlink(temp_audio)
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout or "")[-800:]
            raise RuntimeError(f"ffmpeg 换轨失败 (code={proc.returncode}): {err}")

        logger.info(f"🎬 视频音频替换完成: {output_path}")
        return output_path

    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """获取视频信息（优先 ffprobe，其次 ffmpeg -i 解析，再次 OpenCV）"""
        import re
        import subprocess
        import json
        import os
        from .ffmpeg_bin import find_ffmpeg

        ffmpeg = find_ffmpeg()
        if ffmpeg:
            # 系统 ffmpeg 常带同目录 ffprobe；imageio 静态包通常只有 ffmpeg
            probe_dir = os.path.dirname(ffmpeg)
            ffprobe_cand = os.path.join(probe_dir, "ffprobe")
            ffprobe = ffprobe_cand if os.path.isfile(ffprobe_cand) else shutil_which("ffprobe")
            if ffprobe:
                try:
                    cmd = [
                        ffprobe, "-v", "quiet", "-print_format", "json",
                        "-show_format", "-show_streams", video_path,
                    ]
                    raw = subprocess.run(cmd, capture_output=True, check=True, text=True)
                    meta = json.loads(raw.stdout)
                    vstream = next(
                        (s for s in meta.get("streams", []) if s.get("codec_type") == "video"),
                        {},
                    )
                    fmt = meta.get("format", {})
                    return {
                        "width": int(vstream.get("width") or 0),
                        "height": int(vstream.get("height") or 0),
                        "fps": _parse_fps(vstream.get("r_frame_rate")),
                        "duration": float(fmt.get("duration") or 0),
                        "has_audio": any(
                            s.get("codec_type") == "audio" for s in meta.get("streams", [])
                        ),
                    }
                except Exception as e:
                    logger.warning(f"ffprobe 失败: {e}")

            # ffmpeg -i：信息在 stderr
            try:
                raw = subprocess.run(
                    [ffmpeg, "-hide_banner", "-i", video_path],
                    capture_output=True, text=True,
                )
                err = raw.stderr or ""
                dur_m = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", err)
                duration = 0.0
                if dur_m:
                    h, m, s = dur_m.groups()
                    duration = int(h) * 3600 + int(m) * 60 + float(s)
                wh = re.search(r"Video:.*?,\s*(\d{2,5})x(\d{2,5})", err)
                width = int(wh.group(1)) if wh else 0
                height = int(wh.group(2)) if wh else 0
                fps_m = re.search(r"([\d.]+)\s*fps", err)
                return {
                    "width": width,
                    "height": height,
                    "fps": float(fps_m.group(1)) if fps_m else 0.0,
                    "duration": duration,
                    "has_audio": "Audio:" in err,
                }
            except Exception as e:
                logger.warning(f"ffmpeg -i 解析失败: {e}")

        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": (
                    cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS)
                    if cap.get(cv2.CAP_PROP_FPS) else 0
                ),
            }
            cap.release()
            return info
        except ImportError:
            logger.warning("⚠️ OpenCV 不可用")
            return {}


def shutil_which(cmd: str) -> Optional[str]:
    import shutil
    return shutil.which(cmd)


def _parse_fps(rate: Optional[str]) -> float:
    if not rate or rate == "0/0":
        return 0.0
    if "/" in rate:
        a, b = rate.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(rate)
    except ValueError:
        return 0.0
