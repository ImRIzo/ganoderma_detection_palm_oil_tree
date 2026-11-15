# Ganoderma Detection in Oil Palm — Training & Evaluation

Training and evaluation code for the manuscript
**"Edge Deployment of Lightweight Ganoderma Detection in Oil Palm Using RGB Images"**
(M. R. Sarkar, C. S. Der — Universiti Tenaga Nasional; under review at IEEE Access).

A COCO-pretrained **YOLOv8n** is trained with **source-grouped 5-fold cross-validation**, tested once on the
untouched 42-image test split, evaluated on an **independent external test set**, and exported to ONNX for
deployment on the Qualcomm QCS6490 (Radxa Dragon Q6A) NPU.

![Example detections](example.png?raw=true "Example detections")

## Data sources

| Dataset | Use | Link |
|---|---|---|
| BINUS PTPN4 Ramadya, *Ganoderma Detection (Oil Palm)*, v3 "V3 Jakarta" (1,133 images; CC BY 4.0) | Training, CV, 42-image test | [Roboflow](https://universe.roboflow.com/binus-ptpn4-ramadya/ganoderma-detection-oil-palm/dataset/3) · [Mendeley v1](https://doi.org/10.17632/s23jvbpnr3.1) |
| Mimboro et al., *UAV RGB Image Dataset for Object Detection of Ganoderma-Affected Oil Palm Trees*, v2 (CC BY 4.0) | External test (118-tile test split) | [Mendeley v2](https://doi.org/10.17632/s23jvbpnr3.2) |

Datasets are **not** included in this repository. Download them and place them as shown below.

## Project layout

```
.
├── data/                          # Roboflow v3 export (YOLOv8 format) — not tracked
│   ├── train/ valid/ test/        #   each with images/ and labels/
│   └── data.yaml
├── Ganoderma_UAV_RGB_Dataset_v2/  # Mendeley v2 download — not tracked
│   ├── images/test/
│   └── annotations/instances_test.json
├── models/                        # final weights (best.pt, best.onnx) + Lite-SE configs
├── notebooks/                     # experiments (Google Colab)
├── external_test/                 # external-test preparation and bootstrap CI scripts
├── data.yaml
└── requirements.txt
```

## Setup

```bash
git clone https://github.com/ImRIzo/ganoderma_detection_palm_oil_tree.git
cd ganoderma_detection_palm_oil_tree
pip install -r requirements.txt          # Ultralytics 8.4.173
```

Download the Roboflow v3 export (**YOLOv8** format) and extract it into `data/`.
For the external test, download the Mendeley v2 release into `Ganoderma_UAV_RGB_Dataset_v2/`.

## Reproducing the experiments

Run the notebooks in order (Google Colab, A100; they expect the dataset under `MyDrive/ganoderma/`):

| Notebook | Purpose | Paper |
|---|---|---|
| `01_cv_phash_grouping_yolov8n.ipynb` | 5-fold CV with pHash-only grouping (shows leakage inflation) | Sec. III-B |
| `02_cv_source_grouping_yolov8n.ipynb` | Leakage audit + source-grouped 5-fold CV (recipe A) | Table 3 |
| `03_ablation_recipes_BCD.ipynb` | Paired ablation B/C/D (incl. YOLOv8s) | Tables 4–5 |
| `04_final_model_recipeA.ipynb` | Final model, recipe A (reported for transparency) | Sec. IV-C |
| `05_final_model_recipeB_deployed.ipynb` | **Final deployed model** (recipe B, seed 1041), test + ONNX export | Tables 6–7 |
| `exploratory/yolov8s_lite_se_cv.ipynb` | Exploratory Lite-SE detector | Sec. IV-B |

Key settings: 640 × 640, batch 32, AdamW (lr 0.002), recipe B = `flipud=0.5`, 100 epochs, patience 50;
evaluation at fixed 640 × 640 (`rect=False`), conf 0.001, IoU 0.70.

## External test set

```bash
# 1) Overlap audit vs. the Roboflow export + COCO -> YOLO conversion (scale 0.31 matches training object size)
python external_test/prepare_v2_test.py \
  --v2_images Ganoderma_UAV_RGB_Dataset_v2/images/test \
  --v2_coco   Ganoderma_UAV_RGB_Dataset_v2/annotations/instances_test.json \
  --rf_root   data --out v2_external_test_s031 --scale 0.31

# 2) Evaluate (PyTorch or ONNX), saving predictions
yolo val model=models/best.pt data=v2_external_test_s031/data_v2test.yaml imgsz=640 batch=1 \
  conf=0.001 iou=0.70 rect=False split=test save_txt=True save_conf=True \
  project=v2_results name=s031_pt exist_ok=True

# 3) 95% bootstrap CI (2,000 image-level resamples; Ultralytics 8.4 AP)
python external_test/bootstrap_ci.py --data_dir v2_external_test_s031 \
  --pred_dir runs/detect/v2_results/s031_pt/labels
```

Use `--scale 1.0` for the native-scale result, or other values (0.25–0.40) for the sensitivity analysis.
On-device predictions (`predictions.csv`) can be scored with `--pred_csv`.

## Related repositories

| Stage | Repository |
|---|---|
| INT8 QNN conversion | [onnx2qnn](https://github.com/ImRIzo/onnx2qnn) |
| Edge inference & power/accuracy evaluation (Dragon Q6A) | [yolov8_inference_dragon_q6a](https://github.com/ImRIzo/yolov8_inference_dragon_q6a) |
| UAV-like RTSP simulator | [metal-drone-view-simulator](https://github.com/ImRIzo/metal-drone-view-simulator) |

## License

Code: see `LICENSE`. Datasets: CC BY 4.0 — please cite the original dataset authors.
