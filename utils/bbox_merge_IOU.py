from itertools import combinations
from pycocotools.coco import COCO
import json
from collections import defaultdict
from PIL import Image
import heapq
from math import ceil
from tqdm import tqdm

def find(x,parent):
    # 루트 노드가 아니라면, 루트 노드를 찾을 때까지 재귀적으로 호출
    if parent[x] != x:
        return find(parent[x])
    return x

# 두 원소가 속한 집합을 합치기
def union(a, b,parent):
    a = find(a,parent)
    b = find(b,parent)
    if a < b:
        parent[b] = a
    else:
        parent[a] = b

MAX_BOUNDING_BOXES = 5
X_GRID = 2
Y_GRID = 5
SAME_GROUP_DISTANCE=5
CLASS_NUM=100
# 0 means combination, 1 means merge everything
MODE=1
# Json import
test_json_path="/workspace/item_box_competition/data/validation.json"
bbox_json_path="/workspace/item_box_competition/model/result_bbox.json"
result_json_path="/workspace/item_box_competition/model/result_merged_bbox.json"

# Categories 한 이미지 내 식별된 객체 순서, 분류 score 포함
result_file=[]
imgs=json.load(open(test_json_path,"r",encoding="utf-8"))

for image in tqdm(imgs['images']):
    id=image['id']

    # TODO: 그리드 방식으로 각 이미지 별 속한 그리드 설정
    picture = Image.open("/workspace/item_box_competition/data/validation/"+image['file_name'])
    width,height = picture.size
    CELL_X = ceil(width//X_GRID)
    CELL_Y = ceil(height//Y_GRID)
    cell_state = [[[[] for _ in range(CLASS_NUM+1)] for _ in range(Y_GRID + 1)] for _ in range(X_GRID + 1)]
    # 객체를 각 grid에 넣어줌(x,y,w,h)
    # x_cord, y_cord -> 그리드 내 위치c
    for i, bbox in enumerate(json.load(open(bbox_json_path,"r",encoding="utf-8"))):
        if bbox['image_id']==id:
            x_cord=int((bbox['bbox'][0]+bbox['bbox'][2]//2)//CELL_X)
            y_cord=int((bbox['bbox'][1]+bbox['bbox'][3]//2)//CELL_Y)
            #if x_cord==8 and y_cord==8:
            #    print("Yes")
            category=bbox['category_id']
            cell_state[x_cord][y_cord][category].append((bbox['score'],bbox['bbox'][0],bbox['bbox'][1],bbox['bbox'][2],bbox['bbox'][3]))
               
    # # TODO: 일정 거리내에 존재하면 같은 그룹으로 처리(레이블:1번~100번으로 존재)
    # for c in range(1,CLASS_NUM+1):
    #     nodes=[]
    #     for x in range(CELL_X):
    #         for y in range(CELL_Y):
    #             for k in range(len(cell_state[x][y][c])):
    #                 nodes.append((x,y,k))
    #     parent=list(range(len(nodes)))
    if MODE ==0:
        # Combination
        for c in range(1,CLASS_NUM+1): #100
            for x in range(X_GRID + 1): #CELL_X+1
                for y in range(Y_GRID + 1): #CELl_Y+1
                    l=len(cell_state[x][y][c])
                    if l==0:
                        continue
                    for k in range(1,l+1): # max 6?
                        for comb in combinations(cell_state[x][y][c],k):
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
        for c in range(1,CLASS_NUM+1):
            for x in range(X_GRID + 1):
                for y in range(Y_GRID + 1):
                    l=len(cell_state[x][y][c])
                    if l==0:
                        continue
                    else:
                        min_x=float('inf')
                        min_y=float('inf')
                        max_x=0
                        max_y=0
                        for e in cell_state[x][y][c]:
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
                        
json.dump(result_file,open(result_json_path, 'w'), indent=4)