import json
import pandas as pd
import time
from tqdm import tqdm

RESULT_JSON_PATH ="/workspace/item_box_competition/model/result_merged_bbox.json"
TEST_JSON_PATH="/workspace/item_box_competition/data/test.json"
def calculate_IOU(label,d):
    result_json = json.load(open(RESULT_JSON_PATH, "r",encoding="utf-8"))
    test_json=json.load(open(TEST_JSON_PATH, "r",encoding="utf-8"))
    answer_cnt=0
    for image in tqdm(test_json["images"]):
    # for image in tqdm(range(1,46)):
        # id=image
        id=image["id"]
        df=pd.DataFrame(columns=["상품명","IoU"])
        
        for x,y,w,h,c,i in label:
            if id!=i:
                continue
            highest_iou=0
            for pred in result_json:
                if i==pred["image_id"] and model_label_to_real_label(pred["category_id"],d) == c:
                # if i==pred["image_id"] and pred["category_id"] == c:
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
            if highest_iou>=0.5:
                answer_cnt+=1
            df = df.append({"상품명":d[c][1],"IoU":highest_iou},ignore_index=True)
        df.to_csv(f"/workspace/item_box_competition/output/result_{id}.csv",index=False,encoding="utf-8")
    print(f"Right bbox: {answer_cnt} / Total bbox:  {len(label)}")
    print(f"Accuracy: {answer_cnt/len(label):.2f}")
    

    

def get_real_label():
    x,y,w,h,c,i =0,0,0,0,0,0
    bboxes=[]
    # test code
    # result_json = json.load(open(RESULT_JSON_PATH, "r",encoding="utf-8"))
    # for bbox in result_json:
    #     bboxes.append((bbox["bbox"][0],bbox["bbox"][1],bbox["bbox"][2],bbox["bbox"][3],bbox["category_id"],bbox["image_id"]))

    for image in json.load(open(TEST_JSON_PATH,"r",encoding="utf-8"))['images']:
        i=image["id"]
        file_name=image["file_name"].split(".")[0]
        with open(f"/workspace/item_box_competition/data/test_label/{file_name}.txt", "r",encoding="utf-8") as f:
            for line in f.readlines():
                c,x,y,w,h = line.split(" ")
                c=int(c)
                x=float(x)
                y=float(y)
                w=float(w)
                h=float(h)
                bboxes.append((x,y,w,h,c,i))
    
    return bboxes

def model_label_to_real_label(num, d):
    return d[num][0]

def model_label_to_real_name(num, d):
    return d[num][1]

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
    t=time.time()
    label=get_real_label()
    d=make_label_dict()
    calculate_IOU(label,d)
    print(f"{time.time()-t:.2f} seconds")

main()