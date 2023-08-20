import json
import os
import cv2
from PIL import Image

json_path="/workspace/item_box_competition/data/test.json"
model_json_path="/workspace/item_box_competition/data/model_105.json"


model_json_data = json.load(open(model_json_path, 'r',encoding='utf-8'))
categories = model_json_data['categories']
images = []
for idx, image in enumerate(os.listdir('/workspace/item_box_competition/data/test')):
    
    picture = Image.open("/workspace/item_box_competition/data/test/"+image)
    width, height = picture.size
    images.append(
    {
      "id": idx + 1,
      "width": width,
      "height": height,
      "file_name": image,
      "license": 0,
      "flickr_url": "",
      "coco_url": "",
      "date_captured": 0
    })
model_json_data['images'] = images
json.dump(model_json_data, open(json_path, 'w',encoding='utf-8'), indent=4, ensure_ascii=False)