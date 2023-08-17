import json
import pandas as pd
# TODO: 물품명과 IoU를 txt값을 기준으로 처리

# TODO: 원본 사이즈 이미지 출력해서 저장

RESULT_JSON_PATH ="/workspace/item_box_competition/model/result_merged_bbox.json"
TEST_JSON_PATH="/workspace/item_box_competition/data/test.json"
def calculate_IOU(label,d):
    result_json = json.load(open(RESULT_JSON_PATH, "r",encoding="utf-8"))
    test_json=json.load(open(TEST_JSON_PATH, "r",encoding="utf-8"))
    for image in test_json["images"]:
        id=image["id"]
        df=pd.DataFrame(columns=["상품명","IoU"])
        for x,y,w,h,c,i in label:
            if id!=i:
                continue
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
            df = df.append({"상품명":d[c][1],"IoU":highest_iou},ignore_index=True)
        df.to_csv(f"/workspace/item_box_competition/model/result_{id}.csv",index=False,encoding="utf-8")
        
        
    

def get_real_label():
    x,y,w,h,c,i =312.1711120605469,1221.7757568359375,120.94418334960938,157.3841552734375,5,1
    # SOME CODE
    return [[x,y,w,h,c,i]]

def model_label_to_real_label(num, d):
    return d[num][0]

def model_label_to_real_label(num, d):
    return d[num][0]

# dict[model_num] = (real_num, real_name)
def make_label_dict():
    data = pd.read_csv('/workspace/item_box_competition/data/labeling_copy.csv', encoding= 'UTF-8')
        
    idx_name_dict = {}
    for i in range(len(data)):
        idx_name_dict[i+1] = (data['사진 번호'][i], data['상품명'][i])
            
    # with open('/workspace/label.txt', 'w', encoding = 'UTF-8') as f:
    #     for i in range(len(data)):
    #         f.write(str(data['사진 번호'][i]) + data['상품명'][i] + '\n')
    return idx_name_dict

def main():
    label=get_real_label()
    d=make_label_dict()
    calculate_IOU(label,d)

main()