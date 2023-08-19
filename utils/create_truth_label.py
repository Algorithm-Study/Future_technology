import json

VALIDATION_JSON="/workspace/item_box_competition/data/validation.json"

# c x y w h
val=json.load(open(VALIDATION_JSON, 'r', encoding='utf-8'))
for image in val["images"]:
    with open(f"/workspace/item_box_competition/data/test_label/{image['file_name'].split('.')[0]}.txt", "w") as f:
        for bbox in val["annotations"]:
            if bbox["image_id"] == image["id"]:
                f.write(f"{bbox['category_id']} {bbox['bbox'][0]} {bbox['bbox'][1]} {bbox['bbox'][2]} {bbox['bbox'][3]}\n")