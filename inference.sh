#!/bin/bash

SECONDS=0
# reset output folder and crop folder
/opt/conda/bin/python /workspace/Future_technology/utils/reset.py

echo "Reset Done"

# create json file for test images
/opt/conda/bin/python /workspace/Future_technology/utils/create_json.py

echo "Json Creation Done"

# Test images Object Detection
cd /workspace/Future_technology/detection/SKU110K-DenseDet
/opt/conda/bin/python tools/test.py configs/SKU_fusion_bfp_x101_32x4d.py /workspace/item_box_competition/model/DenseDet.pth --json_out=out

echo "Object Detection Done"

# Crop images
cd /workspace/Future_technology
/opt/conda/bin/python utils/out_bbox_crop.py

echo "Crop Done"
# Classification with cropped dataset 

/opt/conda/bin/python classification/test.py

echo "Classification Done"
# Merge bboxes which is adjacent to each other
/opt/conda/bin/python utils/bbox_merge_by_distance.py

echo "Merge Done"

# Draw result image and prediction info txt
/opt/conda/bin/python utils/test_output.py
echo "Drawing result image and prediction info txt are done"

# Checking Times Spent during full pipeline
echo $SECONDS