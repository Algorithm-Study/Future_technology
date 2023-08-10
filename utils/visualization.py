import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
annotation_file="/workspace/item_box_competition/data/validation.json"
bbox_file="/workspace/item_box_competition/model/result_bbox.json"
image_prefix="/workspace/item_box_competition/data/validation/"


# show bbox on an image by image id from annotation file and bbox_file


ann=json.load(open(annotation_file,encoding='utf-8'))
bbox=json.load(open(bbox_file,encoding='utf-8'))
for i in tqdm(range(1,len(ann['images'])+1)):
    IMAGE_ID=i
    im = Image.open(image_prefix+ann["images"][IMAGE_ID-1]["file_name"])
    fig, ax = plt.subplots()
    fig.set_size_inches(18.5, 10.5)
    ax.imshow(im)
    for e in bbox:
        if e['image_id']==IMAGE_ID:
            
            ax.add_patch(plt.Rectangle((e['bbox'][0],e['bbox'][1]),e['bbox'][2],e['bbox'][3],fill=False,edgecolor='red',linewidth=1))
            plt.text(e['bbox'][0], e['bbox'][1]-10, e["category_id"], bbox=dict(fill=True, edgecolor='red', linewidth=2))
    plt.savefig(f'/workspace/eval/savefig_default_{i}.png')
    plt.close()
