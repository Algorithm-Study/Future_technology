# cropping model output bbox
import cv2
import json
import os
import pandas as pd
from tqdm import tqdm


def crop_and_save_images(test_output_json_path, test_label_json_path, image_dir, output_dir, score_threshold):
    # Load COCO format JSON
    with open(test_output_json_path, 'r', encoding='utf-8') as json_file:
        test_ouput_data = json.load(json_file)
        
    with open(test_label_json_path, 'r', encoding='utf-8') as json_file:
        test_label_data = json.load(json_file)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    cnt = 0
    # Loop through each annotation
    for idx, annotation in enumerate(tqdm(test_ouput_data)):
        if annotation['score'] >= score_threshold:
            image_id = annotation['image_id']
            image_filename = test_label_data["images"][image_id-1]['file_name']
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
            cnt += 1
            # print(output_filename)
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, cropped_image)
    print(cnt)

if __name__ == "__main__":
    # 테스트 데이터 모델 결과
    test_output_json_path = "/workspace/Future_technology/detection/SKU110K-DenseDet/out.bbox.json"
    # 정답이 있는 테스트 데이터
    test_label_json_path = "/workspace/item_box_competition/data/test.json"
    image_dir = "/workspace/item_box_competition/data/test"
    output_dir = "/workspace/item_box_competition/data/test_cropped"
    SCORE_THRESHOLD = 0.15
    
    crop_and_save_images(test_output_json_path, test_label_json_path, image_dir, output_dir, SCORE_THRESHOLD)
    