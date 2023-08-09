import os
from tqdm import tqdm
import pandas as pd
import cv2

VAL_IMG_PATH="/workspace/item_box_competition/data/validation_crawled"

# save the image name to csv by using pandas
df=pd.DataFrame(columns=['file_name',"category"])
for img in tqdm(os.listdir(VAL_IMG_PATH)):
    df=df.append({'file_name':img,'category':int(img.split("_")[0])-1},ignore_index=True)
    # convert image into .jpg
    img_path=os.path.join(VAL_IMG_PATH,img)
    if img[-4:] == '.png':
        img_jpg_path=os.path.join(VAL_IMG_PATH,img.split(".")[0]+".jpg")

        img=cv2.imread(img_path)
        cv2.imwrite(img_jpg_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
        # delete file in img_path
        os.remove(img_path)
    
    
df.to_csv('/workspace/item_box_competition/data/val_cropped.csv',index=False)
