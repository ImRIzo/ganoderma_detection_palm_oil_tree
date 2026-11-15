cd /Users/mac/Downloads/ganoderma_detection_palm_oil_tree
pip install imagehash pillow

# ১) native স্কেল
python external_test/prepare_v2_test.py \
  --v2_images Ganoderma_UAV_RGB_Dataset_v2/images/test \
  --v2_coco   Ganoderma_UAV_RGB_Dataset_v2/annotations/instances_test.json \
  --rf_root   data \
  --out       v2_external_test

# ২) scale-matched (0.31)
python external_test/prepare_v2_test.py \
  --v2_images Ganoderma_UAV_RGB_Dataset_v2/images/test \
  --v2_coco   Ganoderma_UAV_RGB_Dataset_v2/annotations/instances_test.json \
  --rf_root   data \
  --out       v2_external_test_s031 \
  --scale     0.31