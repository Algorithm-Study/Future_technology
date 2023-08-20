from itertools import combinations
import json
from PIL import Image
from tqdm import tqdm

def find(x,parent):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        parent[x] = find(parent[x],parent)
    return parent[x]

# 두 원소가 속한 집합을 합치기
def union(a, b,parent) -> None:
    a = find(a,parent)
    b = find(b,parent)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b
# 인접한 거리에 있는 bbox를 동일한 객체로 판별
def is_near(a : tuple, b: tuple) -> bool:
    a_x1=a[1]
    a_x2=a[1]+a[3]
    a_y1=a[2]
    a_y2=a[2]+a[4]
    b_x1=b[1]
    b_x2=b[1]+b[3]
    b_y1=b[2]
    b_y2=b[2]+b[4]
    if  (abs(a_x1-b_x1)<=SAME_GROUP_DISTANCE and (a_y1<=b_y1<=a_y2 or a_y1<=b_y2<=a_y2 or b_y1<=a_y1<=b_y2 or b_y1<=a_y2<=b_y2)) or \
        (abs(a_x1-b_x2)<=SAME_GROUP_DISTANCE and (a_y1<=b_y1<=a_y2 or a_y1<=b_y2<=a_y2 or b_y1<=a_y1<=b_y2 or b_y1<=a_y2<=b_y2)) or \
        (abs(a_x2-b_x1)<=SAME_GROUP_DISTANCE and (a_y1<=b_y1<=a_y2 or a_y1<=b_y2<=a_y2 or b_y1<=a_y1<=b_y2 or b_y1<=a_y2<=b_y2)) or \
        (abs(a_x2-b_x2)<=SAME_GROUP_DISTANCE and (a_y1<=b_y1<=a_y2 or a_y1<=b_y2<=a_y2 or b_y1<=a_y1<=b_y2 or b_y1<=a_y2<=b_y2)):
        return True
    if  (abs(a_y1-b_y1)<=SAME_GROUP_DISTANCE and (a_x1<=b_x1<=a_x2 or a_x1<=b_x2<=a_x2 or b_x1<=a_x1<=b_x2 or b_x1<=a_x2<=b_x2)) or \
        (abs(a_y1-b_y2)<=SAME_GROUP_DISTANCE and (a_x1<=b_x1<=a_x2 or a_x1<=b_x2<=a_x2 or b_x1<=a_x1<=b_x2 or b_x1<=a_x2<=b_x2)) or \
        (abs(a_y2-b_y1)<=SAME_GROUP_DISTANCE and (a_x1<=b_x1<=a_x2 or a_x1<=b_x2<=a_x2 or b_x1<=a_x1<=b_x2 or b_x1<=a_x2<=b_x2)) or \
        (abs(a_y2-b_y2)<=SAME_GROUP_DISTANCE and (a_x1<=b_x1<=a_x2 or a_x1<=b_x2<=a_x2 or b_x1<=a_x1<=b_x2 or b_x1<=a_x2<=b_x2)):
        return True
    return False

def merge_bbox(same_group: list):
    if MODE ==0:
        l=len(same_group)
        # sort by score decreasing
        same_group.sort(key=lambda x: x[0],reverse=True)
        same_group=same_group[:min(MAX_COMBINATION,l)]
        for k in range(1,l+1): # max 6?
            for comb in combinations(same_group,k):
                min_x=float('inf')
                min_y=float('inf')
                max_x=0
                max_y=0
                for e in comb:
                    min_x=min(min_x,e[1])
                    max_x=max(max_x,e[1]+e[3])
                    min_y=min(min_y,e[2])
                    max_y=max(max_y,e[2]+e[4])
                w=max_x-min_x
                h=max_y-min_y
                result_file.append({"image_id": id,
                                    "bbox": [min_x, min_y, w,h],
                                    "category_id": c})
    else:
        # Merge everything
        l=len(same_group)
        min_x=float('inf')
        min_y=float('inf')
        max_x=0
        max_y=0
        for e in same_group:
            #print(e)
            min_x = min(min_x,e[1])
            max_x = max(max_x,e[1]+e[3])
            min_y = min(min_y,e[2])
            max_y = max(max_y,e[2]+e[4])
        w=max_x-min_x
        h=max_y-min_y
        result_file.append({"image_id": id,
                            "bbox": [min_x, min_y, w,h],
                            "category_id": c})
    

SAME_GROUP_DISTANCE=50
CLASS_NUM=100
MAX_COMBINATION=10
# 0 means combination, 1 means merge everything
MODE=0
# Json import
test_json_path="/workspace/item_box_competition/data/test.json"
bbox_json_path="/workspace/item_box_competition/model/result_bbox.json"
result_json_path="/workspace/item_box_competition/model/result_merged_bbox.json"
data_dir= "/workspace/item_box_competition/data/test/"

result_file=[]
imgs=json.load(open(test_json_path,"r",encoding="utf-8"))

for image in tqdm(imgs['images']):
    id=image['id']
    picture = Image.open(data_dir+image['file_name'])
    for c in range(1,CLASS_NUM+1):
        all_bbox=[]
        for i, bbox in enumerate(json.load(open(bbox_json_path,"r",encoding="utf-8"))):
            if bbox['image_id']==id and bbox['category_id'] == c:
                all_bbox.append((bbox['score'],bbox['bbox'][0],bbox['bbox'][1],bbox['bbox'][2],bbox['bbox'][3]))
                
        l=len(all_bbox)
        parent=list(range(l))
        for a in range(l-1):
            for b in range(a+1,l):
                if is_near(all_bbox[a],all_bbox[b]):
                    union(a,b,parent)
        for i in range(l):
            if parent[i]!=i:
                continue
            same_category=[]
            for j in range(l):
                if i==parent[j]:
                    same_category.append(all_bbox[j])
            merge_bbox(same_category)
json.dump(result_file,open(result_json_path, 'w'), indent=4)