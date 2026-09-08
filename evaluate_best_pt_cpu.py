#!/usr/bin/env python3
"""
evaluate_best_pt_cpu.py

Evaluate a YOLOv8 detection checkpoint (best.pt) on CPU over the train /
valid / test splits defined in data.yaml, using the same COCO/Ultralytics
conventions as the Q6A NPU evaluation (yolov8_inference_dragon_q6a):

    conf=0.001, NMS IoU=0.7, max_det=300, 101-point interpolated AP,
    square 640x640 letterbox, mAP@0.5, mAP@0.5:0.95,
    Precision / Recall / F1 at the max-F1 operating point.

The report is printed to the console and saved to a text file
(default: eval_report_best_pt_cpu.txt next to this script).

Usage:
    python evaluate_best_pt_cpu.py
    python evaluate_best_pt_cpu.py --splits valid,test
    python evaluate_best_pt_cpu.py --output result/eval_report.txt
"""

from __future__ import annotations

import argparse
import contextlib
import io
import platform
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate best.pt on CPU (COCO-style mAP report, Q6A-NPU report format).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=Path, default=SCRIPT_DIR / "best.pt", help="path to the .pt checkpoint")
    p.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.yaml", help="path to the dataset yaml")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "eval_report_best_pt_cpu.txt",
                   help="text file the report is saved to")
    p.add_argument("--splits", default="train,valid,test",
                   help="comma-separated list of dataset splits to evaluate")
    p.add_argument("--device", default="cpu", help="torch device (cpu)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size (square letterbox)")
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold (eval + NMS)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    p.add_argument("--batch", type=int, default=16, help="validation batch size")
    p.add_argument("--runs-dir", type=Path, default=SCRIPT_DIR / "runs" / "eval_cpu",
                   help="directory for ultralytics' internal val outputs")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt(v: float, nd: int = 4) -> str:
    """Format a float, tolerant of nan/inf."""
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return f"{0.0:.{nd}f}"
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return f"{0.0:.{nd}f}"


def best_conf_index(f1_curve: np.ndarray) -> int | None:
    """Index of the max-F1 operating point (max of smoothed mean F1 curve),
    same convention as ultralytics ap_per_class()."""
    if f1_curve.size == 0:
        return None
    from ultralytics.utils.metrics import smooth  # ultralytics-specific

    return int(smooth(f1_curve.mean(0), 0.1).argmax())


def get_cpu_name() -> str:
    try:
        from ultralytics.utils.torch_utils import get_cpu_info

        return get_cpu_info()
    except Exception:
        return platform.machine() or "cpu"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_split(model, split: str, args: argparse.Namespace) -> dict:
    """Run model.val() on one split and collect the report numbers."""
    stdout_buf, stderr_buf = io.StringIO(), io.StringIO()

    t0 = time.perf_counter()
    try:
        with contextlib.redirect_stdout(stdout_buf), contextlib.redirect_stderr(stderr_buf):
            metrics = model.val(
                data=str(args.data),
                split=split,
                device=args.device,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                max_det=args.max_det,
                batch=args.batch,
                rect=False,               # square letterbox, matches the NPU pipeline
                augment=False,
                plots=False,
                save_json=False,
                save_txt=False,
                save_hybrid=False,
                verbose=False,
                project=str(args.runs_dir),
                name="best_pt",
                exist_ok=True,
            )
    except Exception:
        # surface ultralytics logs if validation crashed
        print(stdout_buf.getvalue())
        print(stderr_buf.getvalue())
        raise
    elapsed = time.perf_counter() - t0

    box = metrics.box

    # per-image tp/fp/fn (IoU>=0.5 matching) -> image / GT / prediction counts
    image_metrics = getattr(box, "image_metrics", None) or {}
    n_images = len(image_metrics)
    n_gt = int(sum(v.get("tp", 0) + v.get("fn", 0) for v in image_metrics.values()))
    n_pred = int(sum(v.get("tp", 0) + v.get("fp", 0) for v in image_metrics.values()))

    # scalar metrics at the max-F1 operating point (same convention as the C++ eval)
    p_arr = np.asarray(getattr(box, "p", []), dtype=float).reshape(-1)
    r_arr = np.asarray(getattr(box, "r", []), dtype=float).reshape(-1)
    f1_arr = np.asarray(getattr(box, "f1", []), dtype=float).reshape(-1)

    f1_curve = np.asarray(getattr(box, "f1_curve", []), dtype=float)
    px = np.asarray(getattr(box, "px", []), dtype=float)
    best_conf = None
    if f1_curve.size and px.size:
        idx = best_conf_index(f1_curve)
        if idx is not None:
            best_conf = float(px[idx])

    all_ap = np.asarray(getattr(box, "all_ap", []), dtype=float)
    ap_per_iou = all_ap.mean(0) if all_ap.size else np.zeros(10)

    # per-class rows
    ap_class_index = [int(c) for c in getattr(box, "ap_class_index", [])]
    nt_per_class = getattr(metrics, "nt_per_class", None)
    nt_per_image = getattr(metrics, "nt_per_image", None)
    class_rows = []
    for j, c in enumerate(ap_class_index):
        class_rows.append({
            "name": args.names.get(c, f"class {c}"),
            "img": int(nt_per_image[c]) if nt_per_image is not None and c < len(nt_per_image) else 0,
            "gt": int(nt_per_class[c]) if nt_per_class is not None and c < len(nt_per_class) else 0,
            "ap50": float(all_ap[j, 0]) if all_ap.size else 0.0,
            "ap5095": float(all_ap[j].mean()) if all_ap.size else 0.0,
            "p": float(p_arr[j]) if j < p_arr.size else 0.0,
            "r": float(r_arr[j]) if j < r_arr.size else 0.0,
            "f1": float(f1_arr[j]) if j < f1_arr.size else 0.0,
        })

    return {
        "split": split,
        "images": n_images,
        "gt_boxes": n_gt,
        "predictions": n_pred,
        "map50": float(getattr(box, "map50", 0.0)),
        "map5095": float(getattr(box, "map", 0.0)),
        "precision": float(p_arr.mean()) if p_arr.size else 0.0,
        "recall": float(r_arr.mean()) if r_arr.size else 0.0,
        "f1": float(f1_arr.mean()) if f1_arr.size else 0.0,
        "best_conf": best_conf,
        "ap_per_iou": ap_per_iou,
        "class_rows": class_rows,
        "speed": dict(getattr(metrics, "speed", {}) or {}),
        "elapsed": elapsed,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

SEP = "=" * 60
SEP_SMALL = "-" * 60


def render_split_section(r: dict, args: argparse.Namespace) -> str:
    """One per-split section in the style of the Q6A eval_report_<dataset>.txt."""
    lines = []
    lines.append(SEP_SMALL)
    lines.append(f"  Dataset:         data/{r['split']}")
    lines.append(f"  Images:          {r['images']}")
    lines.append(f"  GT boxes:        {r['gt_boxes']}")
    lines.append(f"  Predictions:     {r['predictions']}")
    lines.append(SEP_SMALL)
    lines.append(f"  mAP@0.5          {fmt(r['map50'])}")
    lines.append(f"  mAP@0.5:0.95     {fmt(r['map5095'])}")
    lines.append(f"  Precision        {fmt(r['precision'])}")
    lines.append(f"  Recall           {fmt(r['recall'])}")
    lines.append(f"  F1-score         {fmt(r['f1'])}")
    best_conf = f"{r['best_conf']:.3f}" if r["best_conf"] is not None else "n/a"
    lines.append(f"  Best conf        {best_conf}   (operating point of P/R/F1)")
    lines.append(SEP_SMALL)
    lines.append("  AP per IoU threshold:")
    lines.append("    IoU      AP")
    ious = np.linspace(0.5, 0.95, 10)
    for iou, ap in zip(ious, r["ap_per_iou"]):
        lines.append(f"    {iou:.2f}    {fmt(ap)}")
    lines.append(SEP_SMALL)
    lines.append("  Per class:")
    lines.append("    Class              Img    GT    AP50    AP50-95   Prec    Rec     F1")
    for c in r["class_rows"]:
        lines.append(
            f"    {c['name']:<18} {c['img']:>4} {c['gt']:>6} {fmt(c['ap50']):>8} "
            f"{fmt(c['ap5095']):>9} {fmt(c['p']):>7} {fmt(c['r']):>6} {fmt(c['f1']):>7}"
        )
    lines.append(SEP_SMALL)
    lines.append("  Timing:")
    speed = r["speed"]
    for key, label in (("preprocess", "Preprocess"), ("inference", "Inference"), ("postprocess", "Postprocess")):
        ms = float(speed.get(key, 0.0) or 0.0)
        lines.append(f"    {label}:{ms:>14.1f} ms/image")
    ips = r["images"] / r["elapsed"] if r["elapsed"] > 0 else 0.0
    lines.append(f"    Total:  {r['elapsed']:>10.1f} s   ({ips:.1f} images/s)")
    return "\n".join(lines)


def render_summary(results: list[dict]) -> str:
    """Summary table across splits (text + markdown)."""
    text_lines = [
        SEP,
        "  Summary — all splits (float32 on CPU)",
        SEP,
        "  Split    Images    GT boxes    mAP@0.5    mAP@0.5:0.95    Precision    Recall    F1",
    ]
    md_lines = [
        SEP_SMALL,
        "  Markdown table:",
        SEP_SMALL,
        "| Split | Images | GT boxes | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in results:
        text_lines.append(
            f"  {r['split']:<6} {r['images']:>8} {r['gt_boxes']:>11} {fmt(r['map50']):>10} "
            f"{fmt(r['map5095']):>14} {fmt(r['precision']):>11} {fmt(r['recall']):>8} {fmt(r['f1']):>8}"
        )
        md_lines.append(
            f"| {r['split']} | {r['images']} | {r['gt_boxes']} | {fmt(r['map50'])} | "
            f"{fmt(r['map5095'])} | {fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} |"
        )
    return "\n".join(text_lines) + "\n" + "\n".join(md_lines)


# NPU int8 reference values from the README of
# github.com/ImRIzo/yolov8_inference_dragon_q6a (for comparison only).
NPU_REFERENCE = """\
{sep}
  Reference (for comparison) — same model, int8 on Radxa Dragon Q6A NPU
  Source: github.com/ImRIzo/yolov8_inference_dragon_q6a README
{sep}
  Split    Images    GT boxes    mAP@0.5    mAP@0.5:0.95    Precision    Recall    F1
  train       987       1215      0.9305         0.5732       0.8271    0.8856  0.8553
  valid       104        126      0.6980         0.3488       0.5652    0.7222  0.6341
  test         42         50      0.7004         0.4050       0.6364    0.7000  0.6667
""".format(sep=SEP)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    from ultralytics import YOLO
    import torch
    import ultralytics

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split name")

    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    args.names = dict(model.names)
    if getattr(model, "task", None) != "detect":
        raise SystemExit(f"Expected a detection model, got task={model.task!r}")

    # sanity-check the requested splits exist in the dataset yaml.
    # yaml keys use ultralytics naming ('val'), the report uses folder names ('valid').
    import yaml

    with data_path.open() as f:
        data_cfg = yaml.safe_load(f)
    aliases = {"valid": "val", "validation": "val", "val": "val"}
    for split in splits:
        key = aliases.get(split, split)
        if key not in data_cfg:
            raise SystemExit(f"Split {split!r} not found in {data_path} (keys: {sorted(data_cfg.keys())})")

    results = []
    for i, split in enumerate(splits, 1):
        print(f"[{i}/{len(splits)}] Evaluating split '{split}' on {args.device} ...", flush=True)
        r = evaluate_split(model, aliases.get(split, split), args)
        r["split"] = split  # keep the user-facing split name in the report
        results.append(r)
        print(
            f"      done in {r['elapsed']:.1f} s — images={r['images']} gt={r['gt_boxes']} "
            f"mAP50={r['map50']:.4f} mAP50-95={r['map5095']:.4f} P={r['precision']:.4f} "
            f"R={r['recall']:.4f} F1={r['f1']:.4f}"
        )

    header = "\n".join([
        SEP,
        "  YOLOv8 Evaluation Report (COCO-style mAP)",
        f"  {model_path.name} on CPU (PyTorch, float32)",
        SEP,
        f"  Date:            {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Model:           {model_path}",
        f"  Device:          {args.device} ({get_cpu_name()})",
        f"  Ultralytics:     {ultralytics.__version__}   PyTorch: {torch.__version__}",
        f"  Data:            {data_path}  (splits: {', '.join(splits)})",
        f"  Classes:         {len(args.names)} ({', '.join(args.names.values())})",
        f"  Letterbox:       square {args.imgsz}x{args.imgsz} (matches NPU pipeline)",
        f"  Conf: {args.conf:g}   NMS IoU: {args.iou:g}   Classes: {len(args.names)}   Max det: {args.max_det}",
    ])

    report = "\n\n".join([header] + [render_split_section(r, args) for r in results]
                          + [render_summary(results), NPU_REFERENCE]) + "\n"

    out_path = args.output.resolve()
    out_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
