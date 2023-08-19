import json
import numpy as np
import matplotlib.pyplot as plt
bbox_json="/workspace/item_box_competition/model/result_bbox.json"

# print score distribution of bbox_json file by 0.1 interval
def score_distribution(bbox_json):
    with open(bbox_json, 'r') as f:
        data = json.load(f)
    scores = []
    for i in range(len(data)):
        scores.append(data[i]['score'])
    scores = np.array(scores)
    scores = scores[np.where(scores >= 0.15)]
    scores = scores[np.where(scores < 1)]
    plt.hist(scores, bins=90, range=(0.15, 1))
    # save image
    plt.savefig('/workspace/score_distribution.png')
    
# print class bar graph of bbox_json file for given image id and there are class from 1 to 100 and put the count on the top of the bar
def class_distribution(bbox_json, image_id):
    with open(bbox_json, 'r') as f:
        data = json.load(f)
    classes = []
    for i in range(len(data)):
        if data[i]['image_id'] == image_id:
            classes.append(data[i]['category_id'])
    classes = np.array(classes)
    plt.hist(classes, bins=100, range=(1, 101))
    plt.savefig('/workspace/class_distribution.png')
    
class_distribution(bbox_json, 3)