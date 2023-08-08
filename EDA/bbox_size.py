import json
import os

bbox_dir = "/workspace/item_box_competition/data/train.json"
# bbox_dir = "/workspace/Future_technology/out.bbox.json"

with open(bbox_dir, 'r', encoding='utf-8') as json_file:
    bbox_data = json.load(json_file)["annotations"]

area = 0
height = 0
width = 0
m=float('inf')
for idx in range(len(bbox_data)):
    area+=bbox_data[idx]["bbox"][2]*bbox_data[idx]["bbox"][3]
    height+=bbox_data[idx]["bbox"][3]
    width+=bbox_data[idx]["bbox"][2]
    m=min(m,bbox_data[idx]["bbox"][3])
area/=len(bbox_data)
width/=len(bbox_data)
height/=len(bbox_data)
print(area)
print(height)
print(width)
print(m)