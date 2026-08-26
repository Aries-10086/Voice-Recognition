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
            import ffmpeg
            import subprocess
            import tempfile
            import os

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                temp_path = f.name

            cmd = [
                "ffmpeg", "-i", video_path,
                "-ar", str(target_sr),
                "-ac", "1",
                "-f", "wav",
                "-y", temp_path,
            ]
            subprocess.run(cmd, capture_output=True, check=True)

            from .audio_utils import AudioUtils
            audio, sr = AudioUtils.load_audio(temp_path, target_sr=target_sr)
            os.unlink(temp_path)

            logger.info(f"🎬 从视频提取音频: {video_path}")
            return audio, sr

        except ImportError:
            logger.warning("⚠️ ffmpeg 不可用")
            raise
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

        # 保存音频到临时文件
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            temp_audio = f.name

        import soundfile as sf
        sf.write(temp_audio, audio, audio_sr)

        # 使用ffmpeg替换音轨
        cmd = [
            "ffmpeg", "-i", video_path,
            "-i", temp_audio,
            "-c:v", "copy",
            "-c:a", "aac",
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-shortest",
            "-y", output_path,
        ]
        subprocess.run(cmd, capture_output=True, check=True)
        os.unlink(temp_audio)

        logger.info(f"🎬 视频音频替换完成: {output_path}")
        return output_path

    @staticmethod
    def get_video_info(video_path: str) -> Dict:
        """获取视频信息"""
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            info = {
                "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                "fps": cap.get(cv2.CAP_PROP_FPS),
                "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                "duration": cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
            }
            cap.release()
            return info
        except ImportError:
            logger.warning("⚠️ OpenCV 不可用")
            return {}
