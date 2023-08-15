import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
annotation_file="/workspace/item_box_competition/data/validation.json"
# bbox_file="/workspace/item_box_competition/model/result_bbox.json"
bbox_file="/workspace/item_box_competition/model/result_merged_bbox.json"
image_prefix="/workspace/item_box_competition/data/validation/"

LABEL_SHOW = True
# show bbox on an image by image id from annotation file and bbox_file
ann=json.load(open(annotation_file,encoding='utf-8'))
bbox=json.load(open(bbox_file,encoding='utf-8'))

for i in tqdm(range(1,len(ann['images'])+1)):
    IMAGE_ID=i
    im = Image.open(image_prefix+ann["images"][IMAGE_ID-1]["file_name"])
    width, height = im.size
    plt.rc('font', size=5)  
    fig, ax = plt.subplots(figsize = (ann["images"][IMAGE_ID-1]["width"]/300, ann["images"][IMAGE_ID-1]["height"]/300), dpi= 300)
    # fig.set_size_inches(18.5, 10.5)

    ax.imshow(im)
    for e in bbox:
        if e['image_id']==IMAGE_ID:
            ax.add_patch(plt.Rectangle((e['bbox'][0],e['bbox'][1]),e['bbox'][2],e['bbox'][3],fill=False,edgecolor='red',linewidth=0.5))
            # 숫자 
            if LABEL_SHOW:
                plt.text(e['bbox'][0], e['bbox'][1]-10, e["category_id"], bbox=dict(fill=True, edgecolor='red', linewidth=0.7,boxstyle="Square, pad=0.1"))
    plt.savefig(f'/workspace/eval/savefig_default_{i}.png')
    plt.close()
# rec 얇게
# 숫자 onoff