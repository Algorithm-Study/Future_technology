import os

folder_paths_to_delete=["/workspace/item_box_competition/data/test_cropped", "/workspace/item_box_competition/output"]
for folder_path in folder_paths_to_delete:
    for file in os.listdir(folder_path):
        os.remove(os.path.join(folder_path, file))