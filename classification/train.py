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
from dataset import TrainDataset, TestDataset,ValDataset
from imbalanced_sampler import ImbalancedDatasetSampler
from model import model_import

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
MODEL = 'ResNet50'
LEARNING_RATE=0.00001 #1e-4~1e-5
BATCH_SIZE = 32
EPOCHS = 500
VAL_INTERVAL=1
SEED=777
CLASS_NUM=100
IMAGE_SIZE=224
seed_config(SEED)
####################

# height : 120, width : 81
transform_train=A.Compose([
    A.RandomResizedCrop(height = IMAGE_SIZE, width = IMAGE_SIZE, scale=(0.1, 1.0), ratio=(0.66, 1.33),interpolation=cv2.INTER_CUBIC,always_apply=True),
    # A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
    A.HorizontalFlip(),
    A.VerticalFlip(),
    A.Normalize(),
    ToTensorV2(),
])

transform_val=A.Compose([
    A.Resize(IMAGE_SIZE, IMAGE_SIZE, interpolation=cv2.INTER_CUBIC),
    A.Normalize(),
    ToTensorV2(),
])
####################
# Loaders
df_1=pd.read_csv('/workspace/item_box_competition/data/train_cropped.csv')
trainset = TrainDataset(df_1, transform_train)
trainLoader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE,sampler=ImbalancedDatasetSampler(trainset), num_workers = multiprocessing.cpu_count() // 2,pin_memory=True) 
df_2=pd.read_csv('/workspace/item_box_competition/data/val_cropped.csv')
valset = ValDataset(df_2, transform_val)
valLoader = torch.utils.data.DataLoader(valset, batch_size = BATCH_SIZE, shuffle = False, num_workers = multiprocessing.cpu_count() // 2,pin_memory=True)

model= model_import(MODEL).to(device)

criterion=nn.CrossEntropyLoss()
optimizer=torch.optim.Adam(model.parameters(),lr=LEARNING_RATE)

best_val_acc=0
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
        
    if epoch % VAL_INTERVAL==0:
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
            
            if best_val_acc < val_acc/len(valset)*100:
                best_val_acc = val_acc/len(valset)*100
                torch.save(model.state_dict(),'/workspace/item_box_competition/model/resnet50.pth')
                # save the best_val_acc to txt file
                with open('/workspace/item_box_competition/model/best_val_acc.txt', 'w') as f:
                    f.write(str(epoch) + "_" + str(best_val_acc))
                print(f'Best Model saved at epoch {epoch}')
            
