import cv2
from torch.utils.data import Dataset
import pandas as pd
import os

class TrainDataset(Dataset):


    def __init__(self,df, transform):
        self.transform=transform
        self.image_paths=[os.path.join("/workspace/item_box_competition/data/train_cropped_105", img_name) for img_name in df['file_name']]
        self.classes=df['category']

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.classes[index]

    def __len__(self):
        return len(self.image_paths)
    
class ValDataset(Dataset):


    def __init__(self,df, transform):
        self.transform=transform
        self.image_paths=[os.path.join("/workspace/item_box_competition/data/validation_crawled", img_name) for img_name in df['file_name']]
        self.classes=df['category']

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image, self.classes[index]

    def __len__(self):
        return len(self.image_paths)


class TestDataset(Dataset):


    def __init__(self,transform,image_folder="/workspace/item_box_competition/data/test_cropped"):
        self.transform=transform
        self.image_paths=sorted([os.path.join(image_folder,name) for name in os.listdir(image_folder)])

    def __getitem__(self, index):
        image = cv2.imread(self.image_paths[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            image = self.transform(image=image)["image"]
        return image

    def __len__(self):
        return len(self.image_paths)