import cv2
import numpy as np
import albumentations as A
from albumentations.augmentations.bbox_utils import denormalize_bbox, normalize_bbox, union_of_bboxes


def cutmix(image, target, num_bbox_to_cut=1, beta=1.0):
    """
    CutMix augmentation function to apply CutMix on image and its corresponding target (e.g., bounding boxes).

    Parameters:
        image (numpy array): The input image as a numpy array.
        target (dict or tuple): The target information corresponding to the image.
                                It could be a dictionary containing various target information (e.g., bounding boxes),
                                or a tuple of multiple targets.
        num_bbox_to_cut (int): The number of bounding boxes to cut and paste.
        beta (float): Beta parameter for CutMix. Controls the strength of cut and paste. Set to 1.0 for CutMix, 
                      and less than 1.0 for CutMixup (a variation of CutMix).

    Returns:
        augmented_image (numpy array): The augmented image with CutMix.
        augmented_target (dict or tuple): The augmented target information corresponding to the image.
                                          It has the same structure as the input target.
    """
    if not isinstance(target, dict) and not isinstance(target, tuple):
        raise ValueError("The target should be either a dictionary or a tuple of multiple targets.")
    
    if isinstance(target, dict):
        bbox_targets = [target]
    else:
        bbox_targets = target
    
    image_height, image_width = image.shape[:2]
    
    for _ in range(num_bbox_to_cut):
        # Randomly choose another image to cut and paste a part from it
        index = np.random.choice(len(bbox_targets))
        cut_image, cut_target = image.copy(), bbox_targets[index]
        cut_h, cut_w = np.random.randint(1, image_height), np.random.randint(1, image_width)

        # Randomly choose a position to paste the cut image
        x1, y1 = np.random.randint(0, image_width - cut_w), np.random.randint(0, image_height - cut_h)
        x2, y2 = x1 + cut_w, y1 + cut_h

        # Cut and paste the selected region of the cut image to the original image
        image[y1:y2, x1:x2] = beta * image[y1:y2, x1:x2] + (1 - beta) * cut_image[y1:y2, x1:x2]

        # Adjust the bounding box coordinates for the cut region
        for bbox_target in bbox_targets:
            for bbox in bbox_target['bboxes']:
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox
                bbox_x1, bbox_y1, bbox_x2, bbox_y2 = normalize_bbox(
                    bbox_x1, bbox_y1, bbox_x2, bbox_y2, image_width, image_height
                )
                cut_x1, cut_y1, cut_x2, cut_y2 = normalize_bbox(x1, y1, x2, y2, image_width, image_height)
                union_x1, union_y1, union_x2, union_y2 = union_of_bboxes(
                    (bbox_x1, bbox_y1, bbox_x2, bbox_y2), (cut_x1, cut_y1, cut_x2, cut_y2)
                )
                bbox[0], bbox[1], bbox[2], bbox[3] = denormalize_bbox(
                    union_x1, union_y1, union_x2, union_y2, image_width, image_height
                )

    augmented_image = image
    augmented_target = target if isinstance(target, dict) else tuple(bbox_targets)

    return augmented_image, augmented_target

class CustomcocoDataset(CocoDataset):
    def __init__(self, *args, **kwargs):
        super(CustomcocoDataset, self).__init__(*args, **kwargs)
        self.train_pipeline = [
            A.RandomScale(scale_limit=(0.8, 1.2), interpolation=cv2.INTER_LINEAR, p=1.0),
            A.RandomFlip(p=0.5),
            A.Normalize(),
            A.PadIfNeeded(min_height=300, min_width=300, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Lambda(image=lambda image, **kwargs: cutmix(image, kwargs['target'], num_bbox_to_cut=1, beta=1.0),
                     bbox_params=A.BboxParams(format='coco', label_fields=['gt_labels']), p=0.5),
            A.ToTensorV2(),
        ]

    def pre_pipeline(self, results):
        results = super().pre_pipeline(results)
        results['img'], results['gt_bboxes'], results['gt_labels'] = self.train_pipeline(
            image=results['img'],
            bboxes=results['gt_bboxes'],
            labels=results['gt_labels']
        )
        return results

# dataset settings
dataset_type = 'CustomcocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(300, 300), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    # Custom cutmix augmentation
    dict(
        type='Lambda',
        lambd=lambda data: cutmix(data['img'], data['gt_bboxes'], num_bbox_to_cut=1, beta=1.0),
        bbox_params=A.BboxParams(format='coco', label_fields=['gt_labels']),
        p=0.5
    ),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(3000, 3000),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
     samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')