import json
# TODO: 물품명과 IoU를 txt값을 기준으로 처리

# TODO: 원본 사이즈 이미지 출력해서 저장

RESULT_JSON_PATH ="/workspace/item_box_competition/model/result_merged_bbox.json"
def calculate_IOU(label,d):
    result_json = open(RESULT_JSON_PATH, "r",encoding="utf-8")
    result_json = json.load(result_json)
    for x,y,w,h,c,i in label:
        highest_iou=0
        for pred in result_json:
            if i==pred["image_id"] and model_label_to_real_label(pred["category_id"],d) == c:
                # calculate iou
                real_box_area=w*h
                pred_box_area=pred["bbox"][2]*pred["bbox"][3]
                intersection_x1=max(x,pred["bbox"][0])
                intersection_y1=max(y,pred["bbox"][1])
                intersection_x2=min(x+w,pred["bbox"][0]+pred["bbox"][2])
                intersection_y2=min(y+h,pred["bbox"][1]+pred["bbox"][3])
                
                intersection_area=max(intersection_x2-intersection_x1,0)*max(intersection_y2-intersection_y1,0)
                
                iou=intersection_area/(real_box_area+pred_box_area-intersection_area)
                highest_iou=max(highest_iou,iou)
        
    

def get_real_label():
    x,y,w,h,c,i =0,0,0,0,0,0
    # SOME CODE
    return [[x,y,w,h,c,i]]

def model_label_to_real_label(num, d):
    return d[num][0]

# dict[model_num] = (real_num, real_name)
def make_label_dict():
    with open('/workspace/item_box_competition/data/labeling.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    idx_name_dict = {}
    for i, line in enumerate(lines):
        if i >= 1:
            print(line)
            line.strip().split("\u0009")
            idx_name_dict[i] = (line[0], line[1])

    return idx_name_dict

def main():
    label=get_real_label()
    d=make_label_dict()
    calculate_IOU(label,d)

print(make_label_dict())