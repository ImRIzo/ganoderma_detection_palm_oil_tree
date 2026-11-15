"""
Bootstrap 95% CI for mAP@0.5 and mAP@0.5:0.95 on a YOLO-format test set.

Matching and AP follow Ultralytics 8.x (greedy unique IoU matching per
threshold, 101-point interpolated AP, IoU thresholds 0.50:0.05:0.95), so the
point estimate should closely reproduce `yolo val`.
Images are resampled with replacement (default 2000 resamples, seed 0).

Two modes
  A) Run the model here (PyTorch .pt or ONNX):
     python bootstrap_ci.py --data_dir v2_external_test_s031 --model models/best.pt

  B) Use predictions produced elsewhere (e.g. INT8 ORT / QNN on the Dragon Q6A).
     One YOLO txt per image, same stem as the image, normalised xywh + confidence:
         <class> <cx> <cy> <w> <h> <conf>
     python bootstrap_ci.py --data_dir v2_external_test_s031 --pred_dir /path/to/qnn_preds

  C) predictions.csv from yolov8_inference_dragon_q6a (`yolov8_video --eval` for
     QNN, `yolov8_cpu.py --eval` for ONNX Runtime): image,class_id,confidence,x1,y1,x2,y2
     python bootstrap_ci.py --data_dir v2_external_test_s031 --pred_csv results/v2_qnn/predictions.csv

Options: --conf 0.001 --iou 0.70 --imgsz 640 --n_boot 2000 --seed 0 --save_preds DIR
"""
import argparse, json
from pathlib import Path
import numpy as np

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}
IOUV = np.linspace(0.5, 0.95, 10)


def xywhn_to_xyxy(a, W, H):
    a = np.asarray(a, dtype=float).reshape(-1, 4)
    cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def box_iou(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(br - tl, 0, None).prod(2)
    area_a = (a[:, 2:] - a[:, :2]).prod(1)
    area_b = (b[:, 2:] - b[:, :2]).prod(1)
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def match(gt_xyxy, pred_xyxy):
    """Ultralytics match_predictions (single class). Returns bool [n_pred, 10]."""
    tp = np.zeros((len(pred_xyxy), len(IOUV)), dtype=bool)
    if len(gt_xyxy) == 0 or len(pred_xyxy) == 0:
        return tp
    iou = box_iou(gt_xyxy, pred_xyxy)  # [n_gt, n_pred]
    for i, t in enumerate(IOUV):
        m = np.array(np.nonzero(iou >= t)).T  # (gt, pred)
        if m.shape[0]:
            if m.shape[0] > 1:
                m = m[iou[m[:, 0], m[:, 1]].argsort()[::-1]]
                m = m[np.unique(m[:, 1], return_index=True)[1]]
                m = m[np.unique(m[:, 0], return_index=True)[1]]
            tp[m[:, 1], i] = True
    return tp


def compute_ap(recall, precision):
    # Ultralytics 8.4: precision drops to 0 right after the final recall
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    return trapz(np.interp(x, mrec, mpre), x)


def map_from(tp, conf, n_gt):
    if n_gt == 0:
        return float("nan"), float("nan")
    if len(conf) == 0:
        return 0.0, 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    fpc = (1 - tp).cumsum(0)
    tpc = tp.cumsum(0)
    recall = tpc / (n_gt + 1e-16)
    precision = tpc / (tpc + fpc)
    ap = np.array([compute_ap(recall[:, j], precision[:, j]) for j in range(tp.shape[1])])
    return float(ap[0]), float(ap.mean())


def load_gt(lbl_path, W, H):
    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
        return np.zeros((0, 4))
    rows = [l.split() for l in lbl_path.read_text().splitlines() if l.strip()]
    return xywhn_to_xyxy([list(map(float, r[1:5])) for r in rows], W, H)


def load_pred_txt(p, W, H):
    if not p.exists() or p.stat().st_size == 0:
        return np.zeros((0, 4)), np.zeros(0)
    rows = [list(map(float, l.split())) for l in p.read_text().splitlines() if l.strip()]
    a = np.array(rows)
    return xywhn_to_xyxy(a[:, 1:5], W, H), a[:, 5]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=Path, help="folder with images/ and labels/")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--model", type=str)
    g.add_argument("--pred_dir", type=Path)
    g.add_argument("--pred_csv", type=Path)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--iou", type=float, default=0.70)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_preds", type=Path, default=None, help="write YOLO txt predictions here")
    ap.add_argument("--out_json", type=Path, default=None)
    a = ap.parse_args()

    from PIL import Image
    imgs = sorted(p for p in (a.data_dir / "images").iterdir() if p.suffix.lower() in IMG_EXT)
    model = None
    if a.model:
        from ultralytics import YOLO
        model = YOLO(a.model, task="detect")
        if a.save_preds:
            a.save_preds.mkdir(parents=True, exist_ok=True)

    csv_preds = {}
    if a.pred_csv:
        import csv
        with open(a.pred_csv) as f:
            for r in csv.DictReader(f):
                key = Path(r["image"]).stem
                csv_preds.setdefault(key, []).append(
                    [float(r["x1"]), float(r["y1"]), float(r["x2"]), float(r["y2"]), float(r["confidence"])])
        print(f"predictions.csv: {sum(map(len, csv_preds.values()))} boxes on {len(csv_preds)} images")

    per_img = []  # (tp [n,10], conf [n], n_gt)
    for p in imgs:
        W, H = Image.open(p).size
        gt = load_gt(a.data_dir / "labels" / f"{p.stem}.txt", W, H)
        if model is not None:
            r = model.predict(str(p), imgsz=a.imgsz, conf=a.conf, iou=a.iou, max_det=300,
                              device="cpu", verbose=False)[0]
            pb = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
            pc = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
            if a.save_preds:
                xywhn = r.boxes.xywhn.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
                (a.save_preds / f"{p.stem}.txt").write_text("\n".join(
                    f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f} {c:.6f}" for (x, y, w, h), c in zip(xywhn, pc)))
        elif a.pred_csv:
            arr = np.array(csv_preds.get(p.stem, []), dtype=float).reshape(-1, 5)
            pb, pc = arr[:, :4], arr[:, 4]
        else:
            pb, pc = load_pred_txt(a.pred_dir / f"{p.stem}.txt", W, H)
        per_img.append((match(gt, pb), pc, len(gt)))

    def score(idx):
        tp = np.concatenate([per_img[i][0] for i in idx]) if idx.size else np.zeros((0, 10), bool)
        cf = np.concatenate([per_img[i][1] for i in idx]) if idx.size else np.zeros(0)
        ng = sum(per_img[i][2] for i in idx)
        return map_from(tp.astype(float), cf, ng)

    n = len(per_img)
    m50, m5095 = score(np.arange(n))
    rng = np.random.default_rng(a.seed)
    boots = np.array([score(rng.integers(0, n, n)) for _ in range(a.n_boot)])
    boots = boots[~np.isnan(boots).any(1)]
    lo50, hi50 = np.percentile(boots[:, 0], [2.5, 97.5])
    lo95, hi95 = np.percentile(boots[:, 1], [2.5, 97.5])

    res = {"data_dir": str(a.data_dir), "source": a.model or str(a.pred_csv or a.pred_dir),
           "images": n, "gt_boxes": int(sum(x[2] for x in per_img)),
           "mAP50": m50, "mAP50_CI95": [lo50, hi50],
           "mAP50_95": m5095, "mAP50_95_CI95": [lo95, hi95],
           "n_boot": int(len(boots)), "seed": a.seed, "conf": a.conf, "iou": a.iou}
    print(f"\nImages {n} | GT boxes {res['gt_boxes']} | source {res['source']}")
    print(f"mAP@0.5      = {m50:.4f}  (95% CI {lo50:.3f}-{hi50:.3f})")
    print(f"mAP@0.5:0.95 = {m5095:.4f}  (95% CI {lo95:.3f}-{hi95:.3f})")
    out = a.out_json or (a.data_dir / f"bootstrap_{Path(res['source']).name.replace('.', '_')}.json")
    out.write_text(json.dumps(res, indent=2))
    print("saved:", out)


if __name__ == "__main__":
    main()
