import multiprocessing
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms
import numpy as np
import albumentations as A
import random
import cv2
from tqdm import tqdm
import pandas as pd
from albumentations.pytorch import ToTensorV2
from dataset import TrainDataset, TestDataset

def seed_config(seed):
    # https://hoya012.github.io/blog/reproducible_pytorch/
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

#  Hyperparameters
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LEARNING_RATE=0.001
BATCH_SIZE = 16
EPOCHS = 100
VAL_INTERVAL=1
SEED=777
seed_config(SEED)
####################

# height : 120, width : 81
transform_train=A.Compose([
    # A.RandomSizedCrop(min_max_height = [700,2000],height = 224, width = 224, w2h_ratio = 0.66,interpolation=cv2.INTER_CUBIC),
    A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Normalize(),
    ToTensorV2(),
])

transform_val=A.Compose([
    A.Resize(224,224,interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2(),
])
####################
# Loaders
df=pd.read_csv('/workspace/item_box_competition/data/train_cropped.csv')
trainset = TrainDataset(df, transform_train)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = multiprocessing.cpu_count() // 2) 
valset = TrainDataset(df, transform_val)
valLoader = torch.utils.data.DataLoader(valset, batch_size = BATCH_SIZE, shuffle = False, num_workers = multiprocessing.cpu_count() // 2)



model=models.resnet50(pretrained=True).to(device)
model.fc=nn.Linear(2048, 100).to(device)


criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train_loss = 0
    train_acc = 0

    model = model.train()
    for train_batch in tqdm(trainLoader):
        
        inputs,labels = train_batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        # print(f'label : {labels}')
        
        optimizer.zero_grad()

        outs = model(inputs)
        preds = torch.argmax(outs, dim=-1)
        # print(f'predict: {preds}')
        loss = criterion(outs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_acc += (preds == labels).sum().item()
    print('Epoch: %d | Loss: %.4f | Train Accuracy: %.2f' %(epoch, train_loss, train_acc/len(trainset)*100)) 
        
    if epoch%VAL_INTERVAL==0:
        with torch.no_grad():
            val_loss=0
            val_acc=0
            model.eval()
            for val_batch in tqdm(valLoader):
                inputs,labels = val_batch
                inputs = inputs.to(device)
                labels = labels.to(device)
                outs = model(inputs)
                preds = torch.argmax(outs, dim=-1)
                val_acc+=(labels==preds).sum().item()
                val_loss+= criterion(outs, labels).item()
            print('Val_Loss: %.4f | Validation Accuracy: %.2f' %( val_loss, val_acc/len(valset)*100)) 
