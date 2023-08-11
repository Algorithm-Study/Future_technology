from itertools import combinations
from pycocotools.coco import COCO
import json
from collections import defaultdict
from PIL import Image
import heapq
MAX_BOUNDING_BOXES = 5
X_GRID = 10
Y_GRID = 19
# Json import
test_json_path="/workspace/item_box_competition/data/validation.json"
bbox_json_path="/workspace/item_box_competition/model/result_bbox.json"

# Categories 한 이미지 내 식별된 객체 순서, 분류 score 포함
imgs=json(test_json_path)
for image in imgs['images']:
    id=image['id']
    bboxes=[]
    # key가 존재하지 않는 경우 빈 리스트 생성
    categories = {}
    for i,bbox in enumerate(bbox_json_path):
        if bbox['image_id']==id:
            bboxes.append(i)
            if bbox['category_id'] not in categories:
                categories[bbox['category_id']] = [(i, bbox['score'])]
            else:
                categories[bbox['category_id']].append((i, bbox['score'],))
                categories[bbox['category_id']]=sorted(categories[bbox['category_id']], key=lambda x: x[1], reverse=True)
                if len(categories[bbox['category_id']]) > MAX_BOUNDING_BOXES:
                    categories[bbox['category_id']].pop()
    # TODO: 그리드 방식으로 각 이미지 별 속한 그리드 설정
    picture = Image.open("image")
    width,height = picture.size
    cell_state = [[[] for _ in range(width//Y_GRID)] for _ in range(height//X_GRID)]
    for x in range(0,height,height//X_GRID):
        for y in range(0, width, width//Y_GRID):
            
            
    # TODO: 일정 거리내에 존재하면 같은 그룹으로 처리
    
    
    # TODO: 인접한 거리 내에 존재한 애들끼리 묶어서 병합
    for cat in categories.keys():
        2^5-1
            