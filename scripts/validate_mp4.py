#!/usr/bin/env python3
"""
mp4 / 口型相关验收检查（F9 / A3 / A6）。

两档:
  --mode smoke   工程链路：可抽音、有 cloned wav、可回贴、产物可播
  --mode demo    口型演示：在 smoke 基础上要求人脸素材元数据 + sync≥0.65 + 时间轴产物

不自动跑全链路（太慢）；对已有 run_* 目录做静态验收，或仅检查环境/素材。

用法:
  # 环境 + 素材体检
  python scripts/validate_mp4.py --check-env --video data/your_talking_head.mp4

  # 对某次 run 做 F9 产物验收
  python scripts/validate_mp4.py --mode demo --run-dir outputs/run_YYYYMMDD_HHMMSS \\
    --video data/your_talking_head.mp4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)


def _ok(msg: str) -> Tuple[bool, str]:
    return True, msg


def _fail(msg: str) -> Tuple[bool, str]:
    return False, msg


def check_env() -> List[Tuple[bool, str]]:
    from src.utils.ffmpeg_bin import find_ffmpeg

    rows = []
    ff = find_ffmpeg()
    if ff:
        rows.append(_ok(f"ffmpeg: {ff}"))
    else:
        rows.append(_fail("ffmpeg 未安装（brew install ffmpeg）"))
    return rows


def check_video_asset(path: str, require_face_hint: bool) -> List[Tuple[bool, str]]:
    rows = []
    if not path:
        rows.append(_fail("未提供 --video"))
        return rows
    if not os.path.isfile(path):
        rows.append(_fail(f"视频不存在: {path}"))
        return rows
    rows.append(_ok(f"视频存在: {path}"))

    try:
        from src.utils.video_utils import VideoUtils
        from src.utils.ffmpeg_bin import find_ffmpeg

        if not find_ffmpeg():
            rows.append(_fail("无 ffmpeg，无法读视频元数据"))
            return rows
        info = VideoUtils.get_video_info(path)
        dur = float(info.get("duration") or 0)
        w = int(info.get("width") or 0)
        h = int(info.get("height") or 0)
        rows.append(_ok(f"分辨率 {w}x{h}, 时长 {dur:.1f}s"))
        if dur < 15:
            rows.append(_fail(f"时长 {dur:.1f}s < 15s（F9 要求 ≥15s）"))
        else:
            rows.append(_ok("时长 ≥15s"))
        if require_face_hint:
            # 无法自动检人脸：标记为人工项
            rows.append(
                _ok("【人工】确认画面为人脸正脸说话（非纯色/风景）；口型可见")
            )
    except Exception as e:
        rows.append(_fail(f"读视频信息失败: {e}"))
    return rows


def check_run_dir(run_dir: str, mode: str) -> List[Tuple[bool, str]]:
    rows = []
    if not run_dir or not os.path.isdir(run_dir):
        rows.append(_fail(f"run 目录无效: {run_dir}"))
        return rows
    rows.append(_ok(f"run 目录: {run_dir}"))

    wav = os.path.join(run_dir, "cloned_output.wav")
    if os.path.isfile(wav):
        rows.append(_ok("cloned_output.wav 存在"))
    else:
        rows.append(_fail("缺少 cloned_output.wav"))

    tl_path = os.path.join(run_dir, "timeline_sync.json")
    aocp_path = os.path.join(run_dir, "aocp_open_segments.json")
    for p, name in ((tl_path, "timeline_sync.json"), (aocp_path, "aocp_open_segments.json")):
        if os.path.isfile(p):
            rows.append(_ok(f"{name} 存在"))
        else:
            rows.append(_fail(f"缺少 {name}") if mode == "demo" else _ok(f"{name} 缺失（smoke 可放宽）"))

    sync = None
    if os.path.isfile(tl_path):
        try:
            with open(tl_path, encoding="utf-8") as f:
                tl = json.load(f)
            sync = float(tl.get("sync_score", 0))
            rows.append(_ok(f"sync_score={sync:.3f}"))
            if mode == "demo":
                if sync >= 0.65:
                    rows.append(_ok("sync ≥ 0.65"))
                else:
                    rows.append(_fail(f"sync {sync:.3f} < 0.65"))
        except Exception as e:
            rows.append(_fail(f"解析 timeline_sync.json 失败: {e}"))

    # 回贴后的 mp4（任意 *_cloned.mp4）
    cloned_videos = [
        os.path.join(run_dir, n)
        for n in os.listdir(run_dir)
        if n.endswith("_cloned.mp4") or n.endswith("_cloned.mkv")
    ]
    if cloned_videos:
        rows.append(_ok(f"克隆视频: {os.path.basename(cloned_videos[0])}"))
    else:
        rows.append(
            _fail("缺少 *_cloned.mp4（需 --video 跑通回贴）")
            if mode in ("smoke", "demo")
            else _ok("无克隆视频")
        )

    summary = os.path.join(run_dir, "summary.txt")
    if os.path.isfile(summary):
        rows.append(_ok("summary.txt 存在"))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="F9/mp4 验收检查")
    ap.add_argument("--mode", choices=("smoke", "demo"), default="smoke")
    ap.add_argument("--check-env", action="store_true")
    ap.add_argument("--video", default="", help="待验收源视频")
    ap.add_argument("--run-dir", default="", help="pipeline 输出 run_* 目录")
    ap.add_argument("--json-out", default="", help="可选：写入验收 JSON")
    args = ap.parse_args()

    checks: List[Tuple[bool, str]] = []
    if args.check_env or True:
        checks.extend(check_env())
    if args.video:
        checks.extend(check_video_asset(args.video, require_face_hint=(args.mode == "demo")))
    if args.run_dir:
        checks.extend(check_run_dir(args.run_dir, args.mode))

    if not args.video and not args.run_dir and args.check_env:
        pass
    elif not args.video and not args.run_dir:
        print("提示: 至少传 --check-env / --video / --run-dir 之一")

    passed = sum(1 for ok, _ in checks if ok)
    failed = sum(1 for ok, _ in checks if not ok)
    print(f"=== mp4 验收 · mode={args.mode} ===")
    for ok, msg in checks:
        print(f"{'✅' if ok else '❌'} {msg}")
    print(f"--- {passed} passed / {failed} failed ---")

    report: Dict[str, Any] = {
        "mode": args.mode,
        "passed": passed,
        "failed": failed,
        "checks": [{"ok": ok, "msg": msg} for ok, msg in checks],
    }
    if args.json_out:
        out = args.json_out if os.path.isabs(args.json_out) else os.path.join(ROOT, args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"wrote {out}")

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
