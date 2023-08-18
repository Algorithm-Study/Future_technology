import json
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
annotation_file="/workspace/item_box_competition/data/test.json"
# bbox_file="/workspace/item_box_competition/model/result_bbox.json"
bbox_file="/workspace/item_box_competition/model/result_merged_bbox.json"
image_prefix="/workspace/item_box_competition/data/test/"

LABEL_SHOW = False
# show bbox on an image by image id from annotation file and bbox_file
ann=json.load(open(annotation_file,encoding='utf-8'))
bbox=json.load(open(bbox_file,encoding='utf-8'))
for i in tqdm(range(1,len(ann['images'])+1)):
    IMAGE_ID=i
    im = Image.open(image_prefix+ann["images"][IMAGE_ID-1]["file_name"])
    width, height = im.size
    fig=plt.figure(figsize=(width/200, height/200), dpi= 200)
    ax= fig.add_axes([0, 0, 1, 1])
    plt.axis('off')
    plt.subplots_adjust(wspace=0, hspace=0)

    ax.imshow(im)
    for e in bbox:
        if e['image_id']==IMAGE_ID:
            ax.add_patch(plt.Rectangle((e['bbox'][0],e['bbox'][1]),e['bbox'][2],e['bbox'][3],fill=False,edgecolor='red',linewidth=0.5))
            # 숫자 ON/OFF
            if LABEL_SHOW:
                plt.text(e['bbox'][0], e['bbox'][1]-10, e["category_id"], bbox=dict(fill=True, edgecolor='red', linewidth=0.5,), fontsize=1)
    file_name=ann["images"][i-1]["file_name"].split(".")[0]
    plt.savefig(f'/workspace/item_box_competition/output/{file_name}_bbox.png', bbox_inches='tight',pad_inches=0,dpi=200)
    plt.close()