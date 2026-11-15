"""
Prepare the Mendeley V2 test split (DOI 10.17632/s23jvbpnr3.2) as an external
test set for the final YOLOv8n Ganoderma model.

Steps
  1. Overlap audit: SHA-256 + flip-aware perceptual hash between V2 test tiles
     and every image in the original Roboflow V3 export (train/valid/test).
  2. Convert V2 COCO test annotations to YOLO format, keeping ONLY
     `ganoderma_affected_palm` (-> class 0). `survey_negative_normal_palm`
     boxes are dropped, so detections on healthy palms count as false positives.
  Optional --scale S (<1) downsamples each tile by S and pastes it centred on a
     640x640 canvas (grey 114), scaling boxes accordingly. Use it to match the
     object scale of the training data (V2 crowns ~187 px vs ~58 px in training,
     S ~= 0.31). Input stays 640x640, so all four runtimes work unchanged.
  3. Write data_v2test.yaml for Ultralytics val (PyTorch / ONNX paths).
     Point your existing QNN evaluation harness at the same images/labels.

Usage
  pip install imagehash pillow
  python prepare_v2_test.py \
      --v2_images  /path/to/v2/test/images \
      --v2_coco    /path/to/v2/test_annotations.json \
      --rf_root    /path/to/roboflow_v3_export \
      --out        ./v2_external_test \
      --phash_thr  6 \
      --scale      1.0      # native; run again with --scale 0.31 --out ./v2_external_test_s031
"""
import argparse, hashlib, json, shutil
from pathlib import Path

import imagehash
from PIL import Image, ImageOps

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
POS_CLASS = "ganoderma_affected_palm"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def phash_variants(p: Path):
    """pHash of the image and its horizontal/vertical/both flips."""
    im = Image.open(p).convert("RGB")
    return [imagehash.phash(v) for v in (
        im, ImageOps.mirror(im), ImageOps.flip(im), ImageOps.flip(ImageOps.mirror(im)))]


def list_images(root: Path):
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXT)


def audit(v2_imgs, rf_imgs, thr):
    print(f"Hashing {len(rf_imgs)} Roboflow images ...")
    rf_sha = {sha256(p): p for p in rf_imgs}
    rf_ph = [(imagehash.phash(Image.open(p).convert("RGB")), p) for p in rf_imgs]

    rows, flagged = [], set()
    for p in v2_imgs:
        s = sha256(p)
        exact = rf_sha.get(s)
        best_d, best_p = 99, None
        for v in phash_variants(p):
            for h, rp in rf_ph:
                d = v - h
                if d < best_d:
                    best_d, best_p = d, rp
        hit = exact is not None or best_d <= thr
        if hit:
            flagged.add(p.name)
        rows.append({"v2_image": p.name, "sha_match": str(exact) if exact else "",
                     "min_phash_dist": best_d, "closest_rf": str(best_p),
                     "flagged": hit})
    return rows, flagged


def coco_to_yolo(coco_path: Path, img_dir: Path, out_img: Path, out_lbl: Path, exclude, scale=1.0):
    coco = json.loads(coco_path.read_text())
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    pos_ids = {i for i, n in cats.items() if n == POS_CLASS}
    if not pos_ids:
        raise SystemExit(f"Category '{POS_CLASS}' not found. Categories: {list(cats.values())}")

    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)

    out_img.mkdir(parents=True, exist_ok=True)
    out_lbl.mkdir(parents=True, exist_ok=True)
    n_img = n_pos_box = n_neg_box = n_pos_img = 0
    for im in coco["images"]:
        name = Path(im["file_name"]).name
        if name in exclude:
            continue
        src = img_dir / name
        if not src.exists():
            print(f"  missing image: {name}")
            continue
        W, H = im["width"], im["height"]
        if scale == 1.0:
            ox = oy = 0.0
            CW, CH = W, H
            shutil.copy2(src, out_img / name)
        else:
            CW, CH = W, H                      # keep 640x640 canvas
            nw, nh = round(W * scale), round(H * scale)
            ox, oy = (CW - nw) // 2, (CH - nh) // 2
            canvas = Image.new("RGB", (CW, CH), (114, 114, 114))
            canvas.paste(Image.open(src).convert("RGB").resize((nw, nh), Image.BILINEAR), (ox, oy))
            canvas.save(out_img / name)
        lines = []
        for a in anns.get(im["id"], []):
            if a["category_id"] in pos_ids:
                x, y, w, h = a["bbox"]
                x, y, w, h = x * scale + ox, y * scale + oy, w * scale, h * scale
                lines.append(f"0 {(x + w / 2) / CW:.6f} {(y + h / 2) / CH:.6f} {w / CW:.6f} {h / CH:.6f}")
            else:
                n_neg_box += 1
        (out_lbl / (Path(name).stem + ".txt")).write_text("\n".join(lines))
        n_img += 1
        n_pos_box += len(lines)
        n_pos_img += bool(lines)
    return n_img, n_pos_img, n_pos_box, n_neg_box


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2_images", required=True, type=Path)
    ap.add_argument("--v2_coco", required=True, type=Path)
    ap.add_argument("--rf_root", required=True, type=Path)
    ap.add_argument("--out", default=Path("v2_external_test"), type=Path)
    ap.add_argument("--phash_thr", default=6, type=int)
    ap.add_argument("--scale", default=1.0, type=float,
                    help="downsample factor; 0.31 matches the training box scale")
    a = ap.parse_args()

    v2_imgs, rf_imgs = list_images(a.v2_images), list_images(a.rf_root)
    print(f"V2 test images: {len(v2_imgs)} | Roboflow images: {len(rf_imgs)}")

    rows, flagged = audit(v2_imgs, rf_imgs, a.phash_thr)
    a.out.mkdir(parents=True, exist_ok=True)
    with open(a.out / "overlap_audit.csv", "w") as f:
        f.write("v2_image,sha_match,min_phash_dist,closest_rf,flagged\n")
        for r in rows:
            f.write(f'{r["v2_image"]},{r["sha_match"]},{r["min_phash_dist"]},{r["closest_rf"]},{r["flagged"]}\n')
    print(f"Overlap flagged: {len(flagged)} -> excluded (see overlap_audit.csv; inspect pairs visually)")

    n_img, n_pos_img, n_pos, n_neg = coco_to_yolo(
        a.v2_coco, a.v2_images, a.out / "images", a.out / "labels", flagged, a.scale)
    (a.out / "data_v2test.yaml").write_text(
        f"path: {a.out.resolve()}\ntrain: images\nval: images\ntest: images\nnames:\n  0: ganoderma\n")

    print("\n=== External test set summary ===")
    print(f"images kept:              {n_img}")
    print(f"images with positives:    {n_pos_img}")
    print(f"Ganoderma boxes (GT):     {n_pos}")
    print(f"healthy boxes dropped:    {n_neg}")
    print(f"scale:                    {a.scale}")
    print(f"\nPyTorch / ONNX eval:\n  yolo val model=best.pt data={a.out}/data_v2test.yaml "
          f"imgsz=640 conf=0.001 iou=0.70 rect=False split=test")


if __name__ == "__main__":
    main()
