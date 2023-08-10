import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import random
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from dataset import TestDataset
from torch.utils.data import DataLoader
import multiprocessing
import json
from tqdm import tqdm
def seed_config(seed):
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

SEED=777
BATCH_SIZE=16
MODEL_PATH='/workspace/item_box_competition/model/resnet50_32.pth'
seed_config(SEED)


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

model=models.resnet50(pretrained=True).to(device)
model.fc=nn.Linear(2048, 100).to(device)

# load model's state_dict
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

transform_test=A.Compose([
    A.Resize(224, 224, interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2(),
])

test_set= TestDataset(transform_test)
test_loader = DataLoader(
    test_set,
    batch_size=BATCH_SIZE,
    num_workers=multiprocessing.cpu_count() // 2,
    shuffle=False,
    pin_memory=True,
    drop_last=False,
)
preds = []
with torch.no_grad():
    for idx, images in enumerate(tqdm(test_loader)):
        images = images.to(device)
        pred = model(images)
        pred = pred.argmax(dim=-1)
        preds.extend(pred.cpu().numpy())
        
        
# evalutation


# json_path = '/workspace/Future_technology/detection/SKU110K-DenseDet/out.bbox.json'
# save_path= "/workspace/item_box_competition/model/result_bbox.json"
# anns=json.load(open(json_path, 'r'))
# new_json=[]
# # TODO: 이미지 순서, confidence 순서
# for i, ann in enumerate(test_set.image_paths):
#     ann_id=int(ann.split('/')[-1][:4])
#     e=anns[ann_id]
#     e["category_id"]=int(preds[i])+1
#     new_json.append(e)

# json.dump(new_json, open(save_path, 'w'), indent=4)



json_path = '/workspace/item_box_competition/data/validation.json'
save_path= "/workspace/item_box_competition/model/result_bbox.json"
anns=json.load(open(json_path, 'r',encoding='utf-8'))["annotations"]
new_json=[]
# TODO: 이미지 순서, confidence 순서
for i, ann in enumerate(test_set.image_paths):
    ann_id=int(ann.split('/')[-1][:4])
    e=anns[ann_id]
    e["category_id"]=int(preds[i])+1
    new_json.append(e)

json.dump(new_json, open(save_path, 'w'), indent=4)
