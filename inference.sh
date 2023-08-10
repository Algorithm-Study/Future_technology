#!/bin/bash

start=`date +%s.%N`

# Test images Object Detection
cd /workspace/Future_technology/detection/SKU110K-DenseDet
/opt/conda/bin/python tools/test.py configs/SKU_fusion_bfp_x101_32x4d.py /workspace/item_box_competition/model/DenseDet.pth --json_out=out

echo "Object Detection Done"

# Crop images
cd /workspace/Future_technology
/opt/conda/bin/python utils/out_bbox_crop.py

# Classification with cropped dataset 


/opt/conda/bin/python classification/test.py

echo "Classification Done"
# Merge bboxes which is adjacent to each other
/opt/conda/bin/python utils/bbox_merge_IOU.py

# Checking Times Spent during full pipeline
finish=`date +%s.%N`
diff=$( echo "$finish - $start" | bc -l )
echo 'diff:' $diff