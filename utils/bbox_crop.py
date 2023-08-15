# cropping for validation image

import cv2
import json
import os
import numpy as np
import pandas as pd

def crop_and_save_images(json_path, image_dir, output_dir, csv_filename):
    # Load COCO format JSON
    with open(json_path, 'r', encoding='utf-8') as json_file:
        coco_data = json.load(json_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    classes = []
    filenames = []
    # Loop through each annotation
    for idx, annotation in enumerate(coco_data['annotations']):
        image_id = annotation['image_id']
        image_filename = coco_data['images'][image_id-1]['file_name']
        bbox = annotation['bbox']

        # Load the image
        image_path = os.path.join(image_dir, image_filename)
        image = cv2.imread(image_path)
        # Extract bbox coordinates
        x, y, w, h = map(int, bbox)

        # Crop the image using bbox coordinates
        cropped_image = image[y:y + h, x:x + w]

        # Save the cropped image
        output_filename = f"{idx:0>4d}.jpg"
        print(output_filename)
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, cropped_image)
        filenames.append(output_filename)
        classes.append(annotation['category_id']-1)
        
    data = {'file_name': filenames, 'category': classes}
    df = pd.DataFrame(data)
    df.to_csv('/workspace/item_box_competition/data/'+csv_filename, index = False)

if __name__ == "__main__":
    json_path = "/workspace/item_box_competition/data/validation.json"
    image_dir = "/workspace/item_box_competition/data/validation"
    output_dir = "/workspace/item_box_competition/data/test_cropped"
    csv_filename = 'test_cropped_real.csv'
    
    for f in os.listdir(output_dir):
        os.remove(os.path.join(output_dir,f))
    crop_and_save_images(json_path, image_dir, output_dir, csv_filename)