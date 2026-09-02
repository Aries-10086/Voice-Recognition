#!/usr/bin/env python3
"""
A4 · F8 四类样例批量验收：解析已有 run 或触发 pipeline，汇总 length/sync/译段比。

用法:
  python scripts/batch_acceptance.py --collect-only
  python scripts/batch_acceptance.py --run-missing --device cpu
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "data" / "acceptance_manifest.yaml"


def _load_manifest() -> list[dict]:
    import yaml

    if not MANIFEST.exists():
        raise FileNotFoundError(f"缺少 {MANIFEST}")
    data = yaml.safe_load(MANIFEST.read_text(encoding="utf-8"))
    return list(data.get("samples", []))


def _parse_summary(summary_path: Path) -> dict:
    text = summary_path.read_text(encoding="utf-8")
    out = {"run_dir": str(summary_path.parent.name)}

    def grab(pattern, cast=str):
        m = re.search(pattern, text)
        if not m:
            return None
        v = m.group(1)
        return cast(v) if cast != str else v

    out["length_ratio"] = grab(r"长度比:\s*([\d.]+)", float)
    out["sync_score"] = grab(r"同步分\s*([\d.]+)", float)
    out["status"] = grab(r"运行状态:\s*(\w+)")
    out["domain"] = grab(r"场景域:\s*(\w+)")
    m = re.search(r"per_seg_translate_ratio=(\d+\.?\d*)", text)
    if not m:
        m = re.search(r"克隆分组数:\s*(\d+)", text)
    return out


def _latest_run_for_input(stem: str) -> Path | None:
    out_root = PROJECT_ROOT / "outputs"
    if not out_root.exists():
        return None
    candidates = sorted(out_root.glob("run_*"), reverse=True)
    for run in candidates:
        summ = run / "summary.txt"
        if not summ.exists():
            continue
        seg = run / "segments.json"
        if seg.exists():
            try:
                data = json.loads(seg.read_text(encoding="utf-8"))
                # 粗略匹配：segments 非空即可
                if data.get("segments"):
                    return run
            except json.JSONDecodeError:
                continue
    return candidates[0] if candidates else None


def _run_pipeline(sample: dict, device: str) -> Path:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "run_pipeline.py"),
        "--input",
        str(PROJECT_ROOT / sample["file"]),
        "--target-lang",
        sample.get("target_lang", "en"),
        "--device",
        device,
    ]
    for flag in sample.get("args", []) or []:
        cmd.append(flag)
    print("→", " ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    stem = Path(sample["file"]).stem
    run = _latest_run_for_input(stem)
    if run is None:
        raise RuntimeError("pipeline 完成但未找到 outputs/run_*")
    return run


def main():
    p = argparse.ArgumentParser(description="F8 批量验收 (A4)")
    p.add_argument("--collect-only", action="store_true", help="只汇总已有 run")
    p.add_argument("--run-missing", action="store_true", help="缺 run 时触发 pipeline")
    p.add_argument("--device", default="cpu")
    p.add_argument("-o", "--output", default=str(PROJECT_ROOT / "outputs" / "acceptance_report"))
    args = p.parse_args()

    samples = _load_manifest()
    rows = []
    for s in samples:
        fpath = PROJECT_ROOT / s["file"]
        row = {
            "id": s["id"],
            "category": s["category"],
            "file": s["file"],
            "exists": fpath.exists(),
            "length_ratio": "",
            "sync_score": "",
            "status": "missing_file" if not fpath.exists() else "no_run",
            "run_dir": "",
        }
        if not fpath.exists():
            rows.append(row)
            continue

        run_dir = s.get("run_dir")
        if run_dir:
            run_path = PROJECT_ROOT / run_dir
        elif args.run_missing:
            run_path = None
        else:
            run_path = None  # 无固定 run 时不猜测最新 run，避免串样例

        if (run_path is None or not (run_path / "summary.txt").exists()) and args.run_missing:
            run_path = _run_pipeline(s, args.device)

        if run_path and (run_path / "summary.txt").exists():
            meta = _parse_summary(run_path / "summary.txt")
            row.update(
                length_ratio=meta.get("length_ratio", ""),
                sync_score=meta.get("sync_score", ""),
                status=meta.get("status", ""),
                run_dir=run_path.name,
            )
        rows.append(row)

    out_base = Path(args.output)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_base.parent / f"acceptance_{ts}.csv"
    md_path = out_base.parent / f"acceptance_{ts}.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "category",
                "file",
                "exists",
                "length_ratio",
                "sync_score",
                "status",
                "run_dir",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    md_lines = [
        "# F8 验收指标表",
        "",
        f"生成: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "| id | 类别 | 文件 | length | sync | 状态 | run |",
        "|---|---|---|---:|---:|---|---|",
    ]
    for r in rows:
        md_lines.append(
            f"| {r['id']} | {r['category']} | `{r['file']}` | "
            f"{r['length_ratio']} | {r['sync_score']} | {r['status']} | {r['run_dir']} |"
        )
    md_lines += [
        "",
        "## MOS（主观，≥3 人填写）",
        "",
        "| 样例 | 像度 | 可懂度 | 自然度 | 备注 |",
        "|---|---:|---:|---:|---|",
        "| dialog_two_speakers_16k | | | | |",
        "| real_dialog_02 | | | | |",
        "| singing_ruguaiawang_16k | | | | 清唱 gold |",
        "",
    ]
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"\nCSV: {csv_path}\nMD:  {md_path}")


if __name__ == "__main__":
    main()
