#!/usr/bin/env python3
"""
evaluate_best_pt_int8_cpu.py

Int8 evaluation of a YOLOv8 detection checkpoint on CPU:

    best.pt  ->  ONNX (opset 13, raw BCN output)
              ->  static QDQ int8 quantization (onnxruntime, MinMax,
                  per-channel weights, calibration images from the train
                  split, output quantization excluded for Concat)
              ->  COCO/Ultralytics-style eval on train/valid/test with the
                  same conventions as the Q6A NPU evaluation:
                  conf=0.001, NMS IoU=0.7, max_det=300, 101-point
                  interpolated AP, square 640x640 letterbox,
                  Precision/Recall/F1 at the max-F1 operating point.

A pipeline check first runs the *float* ONNX on the test split and compares
mAP@0.5 against the float32 PyTorch report (eval_report_best_pt_cpu.txt) to
verify export/decode/NMS/matching before quantizing.

The report is printed and saved (default: eval_report_best_pt_int8_cpu.txt).

Usage:
    python evaluate_best_pt_int8_cpu.py
    python evaluate_best_pt_int8_cpu.py --splits valid,test --calib-samples 100
"""

from __future__ import annotations

import argparse
import contextlib
import glob
import io
import os
import platform
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate best.pt on CPU with int8 quantization (COCO-style mAP report).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", type=Path, default=SCRIPT_DIR / "best.pt", help="float32 .pt checkpoint")
    p.add_argument("--data", type=Path, default=SCRIPT_DIR / "data.yaml", help="dataset yaml (train/val/test paths)")
    p.add_argument("--output", type=Path, default=SCRIPT_DIR / "eval_report_best_pt_int8_cpu.txt",
                   help="text file the report is saved to")
    p.add_argument("--onnx", type=Path, default=SCRIPT_DIR / "best.onnx", help="intermediate float ONNX path")
    p.add_argument("--int8-onnx", type=Path, default=SCRIPT_DIR / "best_int8.onnx", help="quantized int8 ONNX path")
    p.add_argument("--float-report", type=Path, default=SCRIPT_DIR / "eval_report_best_pt_cpu.txt",
                   help="float32 CPU report used for the pipeline check and comparison table")
    p.add_argument("--splits", default="train,valid,test",
                   help="comma-separated splits to evaluate (report names; valid->val)")
    p.add_argument("--imgsz", type=int, default=640, help="inference image size (square letterbox)")
    p.add_argument("--conf", type=float, default=0.001, help="confidence threshold (eval + NMS)")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    p.add_argument("--max-det", type=int, default=300, help="maximum detections per image")
    p.add_argument("--calib-samples", type=int, default=200, help="calibration images sampled from the train split")
    p.add_argument("--calibrate-method", choices=["minmax", "percentile"], default="minmax",
                   help="activation calibration method")
    p.add_argument("--exclude-output-optypes", default="Concat",
                   help="op types whose outputs are kept float (comma-separated); 'Concat' keeps the final "
                        "model output float so class scores survive quantization")
    p.add_argument("--nodes-to-exclude", default="",
                   help="comma-separated substrings: nodes with matching names are kept float (optional)")
    p.add_argument("--tolerance", type=float, default=0.02,
                   help="max allowed mAP@0.5 gap between float ONNX and the float32 report in the pipeline check")
    p.add_argument("--skip-check", action="store_true", help="skip the float-ONNX pipeline check")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def resolve_data_splits(data_path: Path) -> dict[str, Path]:
    """Split name (ultralytics key) -> images directory (absolute)."""
    import yaml

    with data_path.open() as f:
        cfg = yaml.safe_load(f)
    base = data_path.resolve().parent
    out = {}
    for key in ("train", "val", "test"):
        p = cfg.get(key)
        if isinstance(p, (list, tuple)):
            p = p[0]
        if not p:
            continue
        pp = Path(str(p))
        out[key] = (pp if pp.is_absolute() else base / pp).resolve()
    return out


def label_dir_for(images_dir: Path) -> Path:
    s = str(images_dir)
    if "/images" in s:
        return Path(s.rsplit("/images", 1)[0]) / "labels"
    return images_dir.parent / "labels"


def list_images(d: Path) -> list[Path]:
    return sorted([*d.glob("*.jpg"), *d.glob("*.jpeg"), *d.glob("*.png")])


def load_gt(label_path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """YOLO label file -> (class ids, xyxy boxes in pixels)."""
    cls_list, box_list = [], []
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = map(float, parts[:5])
            cls_list.append(int(c))
            box_list.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.asarray(cls_list, dtype=int), np.asarray(box_list, dtype=np.float32).reshape(-1, 4)


# ---------------------------------------------------------------------------
# Metrics (same conventions as ultralytics 8.4.115 / the Q6A eval)
# ---------------------------------------------------------------------------

def match_predictions(pred_cls: torch.Tensor, true_cls: torch.Tensor, iou: torch.Tensor) -> torch.Tensor:
    """Replica of DetectionValidator.match_predictions(): greedy one-to-one
    matching per IoU threshold, classes must agree. Returns (N, 10) bool."""
    iouv = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((pred_cls.shape[0], iouv.shape[0]), dtype=bool)
    iou_np = (iou * (true_cls[:, None] == pred_cls)).cpu().numpy()
    for i, threshold in enumerate(iouv.tolist()):
        matches = np.array(np.nonzero(iou_np >= threshold)).T
        if matches.shape[0]:
            if matches.shape[0] > 1:
                matches = matches[iou_np[matches[:, 0], matches[:, 1]].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return torch.from_numpy(correct)


def evaluate_session(sess, input_name: str, images: list[Path], labels_dir: Path, args) -> dict:
    """Run one ORT session over a split and compute COCO-style metrics."""
    from ultralytics.data.augment import LetterBox
    from ultralytics.utils.nms import box_iou, non_max_suppression
    from ultralytics.utils.ops import scale_boxes
    from ultralytics.utils.metrics import ap_per_class

    letterbox = LetterBox(new_shape=(args.imgsz, args.imgsz), auto=False, stride=32)

    tp_all, conf_all, pcls_all, tcls_all = [], [], [], []
    imgs_per_class = Counter()
    pre_ms, inf_ms = [], []
    n_pred = 0

    t0 = time.perf_counter()
    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  warning: cannot read {img_path}, skipped")
            continue
        h0, w0 = img.shape[:2]

        s = time.perf_counter()
        im = letterbox(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pre_ms.append((time.perf_counter() - s) * 1000)
        im_t = im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0

        s = time.perf_counter()
        out = sess.run(None, {input_name: im_t})[0]  # (1, 4+nc, anchors) xywh
        inf_ms.append((time.perf_counter() - s) * 1000)

        dets = non_max_suppression(
            torch.from_numpy(np.asarray(out)), conf_thres=args.conf, iou_thres=args.iou, max_det=args.max_det
        )[0]

        tcls, tbox = load_gt(labels_dir / f"{img_path.stem}.txt", w0, h0)
        imgs_per_class.update(set(tcls.tolist()))
        if dets is not None and len(dets):
            dets[:, :4] = scale_boxes((args.imgsz, args.imgsz), dets[:, :4], (h0, w0))
            correct = match_predictions(dets[:, 5].long(), torch.from_numpy(tcls),
                                        box_iou(torch.from_numpy(tbox), dets[:, :4]))
            tp_all.append(correct.numpy())
            conf_all.append(dets[:, 4].numpy())
            pcls_all.append(dets[:, 5].numpy().astype(int))
            n_pred += len(dets)
        tcls_all.append(tcls)
    wall = time.perf_counter() - t0

    n_images = len(images)
    tp = np.concatenate(tp_all) if tp_all else np.zeros((0, 10))
    conf = np.concatenate(conf_all) if conf_all else np.zeros(0)
    pcls = np.concatenate(pcls_all) if pcls_all else np.zeros(0, int)
    tcls = np.concatenate(tcls_all) if tcls_all else np.zeros(0, int)

    # ap_per_class -> tp, fp, p, r, f1 (at max-F1 point), ap (nc, 10),
    # unique_classes, p_curve, r_curve, f1_curve, x, prec_values
    (_, _, p, r, f1, ap, unique_classes,
     p_curve, r_curve, f1_curve, x, _) = ap_per_class(tp, conf, pcls, tcls)

    p = np.asarray(p, dtype=float).reshape(-1)
    r = np.asarray(r, dtype=float).reshape(-1)
    f1 = np.asarray(f1, dtype=float).reshape(-1)
    ap = np.asarray(ap, dtype=float)
    unique_classes = np.asarray(unique_classes, dtype=int)
    nt_per_class = np.bincount(tcls, minlength=args.nc)

    best_conf = None
    if f1_curve.size and x.size:
        from ultralytics.utils.metrics import smooth

        best_conf = float(np.asarray(x)[int(smooth(f1_curve.mean(0), 0.1).argmax())])

    class_rows = []
    for j, c in enumerate(unique_classes.tolist()):
        class_rows.append({
            "name": args.names.get(c, f"class {c}"),
            "img": imgs_per_class.get(c, 0),
            "gt": int(nt_per_class[c]),
            "ap50": float(ap[j, 0]) if ap.size else 0.0,
            "ap5095": float(ap[j].mean()) if ap.size else 0.0,
            "p": float(p[j]) if j < p.size else 0.0,
            "r": float(r[j]) if j < r.size else 0.0,
            "f1": float(f1[j]) if j < f1.size else 0.0,
        })

    return {
        "images": n_images,
        "gt_boxes": int(tcls.size),
        "predictions": n_pred,
        "map50": float(ap[:, 0].mean()) if ap.size else 0.0,
        "map5095": float(ap.mean()) if ap.size else 0.0,
        "precision": float(p.mean()) if p.size else 0.0,
        "recall": float(r.mean()) if r.size else 0.0,
        "f1": float(f1.mean()) if f1.size else 0.0,
        "best_conf": best_conf,
        "ap_per_iou": ap.mean(0) if ap.size else np.zeros(10),
        "class_rows": class_rows,
        "preprocess_ms": float(np.mean(pre_ms)) if pre_ms else 0.0,
        "inference_ms": float(np.mean(inf_ms)) if inf_ms else 0.0,
        "elapsed": wall,
    }


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

from onnxruntime.quantization.calibrate import CalibrationDataReader


class CalibrationReader(CalibrationDataReader):
    """Feeds letterboxed images to onnxruntime.quantization."""

    def __init__(self, paths: list[Path], imgsz: int):
        from ultralytics.data.augment import LetterBox

        self.paths = list(paths)
        self.letterbox = LetterBox(new_shape=(imgsz, imgsz), auto=False, stride=32)
        self._it = iter(self._gen())

    def _gen(self):
        for p in self.paths:
            img = cv2.imread(str(p))
            if img is None:
                continue
            im = self.letterbox(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            yield {"images": im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0}

    def get_next(self):
        return next(self._it, None)

    def rewind(self):
        self._it = iter(self._gen())

    def __len__(self) -> int:
        return len(self.paths)

    def set_range(self, start_index: int, end_index: int):
        self._it = iter(self._gen_range(start_index, end_index))

    def _gen_range(self, start_index: int, end_index: int):
        for p in self.paths[start_index:end_index]:
            img = cv2.imread(str(p))
            if img is None:
                continue
            im = self.letterbox(image=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            yield {"images": im.transpose(2, 0, 1)[None].astype(np.float32) / 255.0}


def quantize_model(onnx_path: Path, int8_path: Path, calib_images: list[Path], args) -> dict:
    """Static QDQ int8 quantization; returns a config summary dict."""
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static

    method = CalibrationMethod.MinMax if args.calibrate_method == "minmax" else CalibrationMethod.Percentile
    extra_options = {"OpTypesToExcludeOutputQuantization":
                     [s.strip() for s in args.exclude_output_optypes.split(",") if s.strip()]}
    if method == CalibrationMethod.Percentile:
        extra_options["CalibPercentile"] = 99.99

    nodes_to_exclude = []
    if args.nodes_to_exclude:
        import onnx as onnx_pkg

        patterns = [s.strip() for s in args.nodes_to_exclude.split(",") if s.strip()]
        m = onnx_pkg.load(str(onnx_path))
        nodes_to_exclude = [n.name for n in m.graph.node if any(pat in n.name for pat in patterns)]

    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        quantize_static(
            str(onnx_path),
            str(int8_path),
            calibration_data_reader=CalibrationReader(calib_images, args.imgsz),
            quant_format=QuantFormat.QDQ,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            per_channel=True,
            calibrate_method=method,
            nodes_to_exclude=nodes_to_exclude,
            extra_options=extra_options,
        )
    return {
        "method": args.calibrate_method,
        "calib_images": len(calib_images),
        "excluded_optypes": extra_options["OpTypesToExcludeOutputQuantization"],
        "excluded_nodes": nodes_to_exclude,
        "seconds": time.perf_counter() - t0,
    }


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

SEP = "=" * 60
SEP_SMALL = "-" * 60


def fmt(v: float, nd: int = 4) -> str:
    try:
        f = float(v)
        if np.isnan(f) or np.isinf(f):
            return f"{0.0:.{nd}f}"
        return f"{f:.{nd}f}"
    except (TypeError, ValueError):
        return f"{0.0:.{nd}f}"


def render_split_section(r: dict, tag: str) -> str:
    lines = [
        SEP_SMALL,
        f"  Dataset:         data/{r['split']}   ({tag})",
        f"  Images:          {r['images']}",
        f"  GT boxes:        {r['gt_boxes']}",
        f"  Predictions:     {r['predictions']}",
        SEP_SMALL,
        f"  mAP@0.5          {fmt(r['map50'])}",
        f"  mAP@0.5:0.95     {fmt(r['map5095'])}",
        f"  Precision        {fmt(r['precision'])}",
        f"  Recall           {fmt(r['recall'])}",
        f"  F1-score         {fmt(r['f1'])}",
        f"  Best conf        {r['best_conf']:.3f}   (operating point of P/R/F1)"
        if r["best_conf"] is not None else "  Best conf        n/a    (operating point of P/R/F1)",
        SEP_SMALL,
        "  AP per IoU threshold:",
        "    IoU      AP",
    ]
    for iou, ap in zip(np.linspace(0.5, 0.95, 10), r["ap_per_iou"]):
        lines.append(f"    {iou:.2f}    {fmt(ap)}")
    lines += [
        SEP_SMALL,
        "  Per class:",
        "    Class              Img    GT    AP50    AP50-95   Prec    Rec     F1",
    ]
    for c in r["class_rows"]:
        lines.append(
            f"    {c['name']:<18} {c['img']:>4} {c['gt']:>6} {fmt(c['ap50']):>8} "
            f"{fmt(c['ap5095']):>9} {fmt(c['p']):>7} {fmt(c['r']):>6} {fmt(c['f1']):>7}"
        )
    lines += [
        SEP_SMALL,
        "  Timing:",
        f"    Preprocess:      {r['preprocess_ms']:>5.1f} ms/image",
        f"    Inference:       {r['inference_ms']:>5.1f} ms/image",
    ]
    ips = r["images"] / r["elapsed"] if r["elapsed"] > 0 else 0.0
    lines.append(f"    Total:  {r['elapsed']:>10.1f} s   ({ips:.1f} images/s)")
    return "\n".join(lines)


def render_summary(results: list[dict], title: str) -> str:
    text_lines = [
        SEP,
        f"  {title}",
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


def parse_float_report(path: Path) -> dict[str, dict]:
    """Read the float32 report's markdown table: split -> metric dict."""
    rows = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) == 8 and cells[0] in {"train", "valid", "test"}:
            try:
                rows[cells[0]] = {
                    "images": int(cells[1]), "gt_boxes": int(cells[2]),
                    "map50": float(cells[3]), "map5095": float(cells[4]),
                    "precision": float(cells[5]), "recall": float(cells[6]), "f1": float(cells[7]),
                }
            except ValueError:
                continue
    return rows


def render_comparison(int8_results: list[dict], float_rows: dict[str, dict]) -> str:
    lines = [
        SEP,
        "  Comparison — float32 vs int8 on the same CPU",
        SEP,
        "  Split    mAP@0.5 fp32   mAP@0.5 int8   ΔmAP@0.5   mAP@0.5:0.95 fp32   mAP@0.5:0.95 int8   Δ",
    ]
    for r in int8_results:
        fp = float_rows.get(r["split"])
        if fp is None:
            lines.append(f"  {r['split']:<6} {fmt(r['map50']):>13} {'—':>13} {'—':>10} {fmt(r['map5095']):>20} {'—':>18} {'—'}")
            continue
        d50 = r["map50"] - fp["map50"]
        d95 = r["map5095"] - fp["map5095"]
        lines.append(
            f"  {r['split']:<6} {fmt(fp['map50']):>13} {fmt(r['map50']):>13} {d50:>+10.4f} "
            f"{fmt(fp['map5095']):>20} {fmt(r['map5095']):>18} {d95:>+10.4f}"
        )
    lines.append("")
    lines.append("  (Δ = int8 − float32; the gap is the quantization cost)")
    return "\n".join(lines)


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

    import onnxruntime as ort
    import ultralytics
    from ultralytics import YOLO

    model_path = args.model.resolve()
    data_path = args.data.resolve()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    if not splits:
        raise SystemExit("--splits must contain at least one split name")

    # dataset layout
    split_dirs = resolve_data_splits(data_path)
    aliases = {"valid": "val", "validation": "val", "val": "val"}
    for s in splits:
        key = aliases.get(s, s)
        if key not in split_dirs:
            raise SystemExit(f"Split {s!r} not found in {data_path} (keys: {sorted(split_dirs)})")

    float_rows = parse_float_report(args.float_report)

    # 1) load model + export float ONNX --------------------------------------
    print(f"Loading model: {model_path}")
    model = YOLO(str(model_path))
    args.names = dict(model.names)
    args.nc = len(args.names)
    if getattr(model, "task", None) != "detect":
        raise SystemExit(f"Expected a detection model, got task={model.task!r}")

    onnx_path = args.onnx.resolve()
    print(f"Exporting float32 ONNX (opset 13, raw output, no NMS) -> {onnx_path}")
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        model.export(format="onnx", imgsz=args.imgsz, opset=13, nms=False,
                     simplify=False, dynamic=False, device="cpu")

    # 2) pipeline check: float ONNX on the test split vs float32 report -------
    if not args.skip_check:
        key = aliases.get("test", "test")
        check_images = list_images(split_dirs[key])
        print(f"Pipeline check: float ONNX on '{key}' ({len(check_images)} images) ...")
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        r = evaluate_session(sess, sess.get_inputs()[0].name, check_images,
                             label_dir_for(split_dirs[key]), args)
        expected = float_rows.get("test", {}).get("map50")
        print(f"  float ONNX: mAP@0.5={r['map50']:.4f}  mAP@0.5:0.95={r['map5095']:.4f}")
        if expected is not None:
            diff = abs(r["map50"] - expected)
            print(f"  float32 PyTorch report: mAP@0.5={expected:.4f}  (|Δ| = {diff:.4f})")
            if diff > args.tolerance:
                raise SystemExit(
                    f"Pipeline check FAILED: float ONNX mAP@0.5 differs from the float32 report "
                    f"by {diff:.4f} (> {args.tolerance}). Export/decode/NMS/matching needs fixing."
                )
            print(f"  pipeline OK (|Δ| <= {args.tolerance})")

    # 3) quantize ------------------------------------------------------------
    calib_key = "train" if "train" in split_dirs else aliases.get(splits[0], splits[0])
    all_calib = list_images(split_dirs[calib_key])
    if len(all_calib) > args.calib_samples:
        calib = [all_calib[i] for i in np.linspace(0, len(all_calib) - 1, args.calib_samples).astype(int)]
    else:
        calib = all_calib
    print(f"Quantizing ({args.calibrate_method}, per-channel, {len(calib)} calibration images "
          f"from '{calib_key}', output quantization excluded for: {args.exclude_output_optypes}) ...")
    qinfo = quantize_model(onnx_path, args.int8_onnx, calib, args)
    print(f"  done in {qinfo['seconds']:.1f}s -> {args.int8_onnx}")

    # 4) int8 evaluation ------------------------------------------------------
    sess = ort.InferenceSession(str(args.int8_onnx), providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    results = []
    for i, s in enumerate(splits, 1):
        key = aliases.get(s, s)
        images = list_images(split_dirs[key])
        print(f"[{i}/{len(splits)}] Int8 inference on '{s}' ({len(images)} images) ...", flush=True)
        r = evaluate_session(sess, in_name, images, label_dir_for(split_dirs[key]), args)
        r["split"] = s
        results.append(r)
        print(
            f"      done in {r['elapsed']:.1f}s — images={r['images']} gt={r['gt_boxes']} "
            f"mAP50={r['map50']:.4f} mAP50-95={r['map5095']:.4f} P={r['precision']:.4f} "
            f"R={r['recall']:.4f} F1={r['f1']:.4f}"
        )

    # 5) report ---------------------------------------------------------------
    try:
        cpu_name = ultralytics.utils.torch_utils.get_cpu_info()
    except Exception:
        cpu_name = platform.machine() or "cpu"

    header = "\n".join([
        SEP,
        "  YOLOv8 Evaluation Report (COCO-style mAP)",
        f"  {model_path.name} on CPU (PyTorch -> ONNX -> int8 static QDQ)",
        SEP,
        f"  Date:            {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"  Model:           {model_path}",
        f"  Int8 model:      {args.int8_onnx}",
        f"  Device:          CPU ({cpu_name})   onnxruntime {ort.__version__}",
        f"  Ultralytics:     {ultralytics.__version__}   PyTorch: {torch.__version__}",
        f"  Data:            {data_path}  (splits: {', '.join(splits)})",
        f"  Classes:         {args.nc} ({', '.join(args.names.values())})",
        f"  Letterbox:       square {args.imgsz}x{args.imgsz} (matches NPU pipeline)",
        f"  Conf: {args.conf:g}   NMS IoU: {args.iou:g}   Classes: {args.nc}   Max det: {args.max_det}",
        SEP_SMALL,
        "  Quantization:",
        f"    Scheme:          static QDQ, weights int8 per-channel, activations uint8",
        f"    Calibration:     {qinfo['method']} over {qinfo['calib_images']} train images",
        f"    Kept float:      outputs of op types {qinfo['excluded_optypes']}"
        + (f" + {len(qinfo['excluded_nodes'])} excluded node(s)" if qinfo["excluded_nodes"] else ""),
    ])

    sections = [render_split_section(r, "int8") for r in results]
    report = "\n\n".join(
        [header] + sections + [
            render_summary(results, "Summary — int8 (static QDQ) on CPU"),
            render_comparison(results, float_rows),
            NPU_REFERENCE,
        ]
    ) + "\n"

    out_path = args.output.resolve()
    out_path.write_text(report, encoding="utf-8")
    print()
    print(report)
    print(f"Report saved to: {out_path}")


if __name__ == "__main__":
    main()
