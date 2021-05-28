import cv2
import os
import copy
import time
from argparse import ArgumentParser

import albumentations as A
import pytorch_lightning as pl
import numpy as np


from pipeline.transforms import ResizeAndPadImage
from modules.lightning_module import LightningKeypointsEstimator

coco_sceleton = [(2, 3), (3, 4), (2, 5), (2, 8),
                 (5, 6), (5, 11), (6, 7), (8, 9), (8, 11),
                 (9, 10), (11, 12), (12, 13)]

time_accumulator = 0
counter = 0

verbose_speed = False


def process_image(image_initial,
                  model: pl.LightningModule,
                  transforms):
    global time_accumulator, counter
    image_rgb = copy.deepcopy(cv2.cvtColor(image_initial, cv2.COLOR_BGR2RGB))

    image_dict = dict(image=image_rgb)
    batch_info = transforms(**image_dict)

    time_now = time.time()
    prediction = model(batch_info['image'].cuda().unsqueeze(0))
    time_accumulator += time.time() - time_now
    counter += 1
    keypoints = model.loss_head.get_keypoints(predicted=prediction,
                                              batch_info=batch_info)
    time_per_frame = time_accumulator / counter
    if verbose_speed:
        print('Model speed = ', time_per_frame, ' ,fps=', 1 / time_per_frame)
    return keypoints


def run_test_on_image(filepath: str):
    image_initial = cv2.imread(filepath)
    cv2.imshow("initial", image_initial)
    keypoints = process_image(image_initial=image_initial,
                              model=model,
                              transforms=transforms)
    x_kps = keypoints[0][..., 0]
    y_kps = keypoints[0][..., 1]
    visabilities = keypoints[0][..., 2]
    for (x, y, v) in zip(x_kps, y_kps, visabilities):
        if v == 2:
            image_initial = cv2.circle(image_initial, (int(x), int(y)), 4, (0, 0, 255), 2)

    for (first_idx, second_idx) in coco_sceleton:
        if visabilities[first_idx] == 2 and visabilities[second_idx] == 2:
            image_initial = cv2.line(image_initial,
                                           (int(x_kps[first_idx]), int(y_kps[first_idx])),
                                           (int(x_kps[second_idx]), int(y_kps[second_idx])),
                                           (0, 0, 255), 2)
    cv2.imshow('debug', image_initial)
    cv2.waitKey(0)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('path_to_dir', type=str)
    parser.add_argument('checkpoint_path', type=str)
    parser.add_argument('--train_dir_order', default=1, required=False, type=int)
    parser.add_argument('--big_resolution', default=True, required=False, type=int)
    args = parser.parse_args()
    if args.big_resolution:
        width = 256
        height = 256
        expand_channels = True
        number_of_channels = 120
    else:
        width = 128
        height = 128
        expand_channels = False
        number_of_channels = 30

    num_classes = 17
    stride = 4
    thrs_conf = 0.1

    transforms = A.Compose([
        ResizeAndPadImage(height=height, width=width),
        A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
        A.ToTensorV2()
    ])

    backbone_cfg = dict(
        type='LiteHRNet',
        in_channels=3,
        extra=dict(
            stem=dict(stem_channels=32, out_channels=32, expand_ratio=1),
            num_stages=3,
            stages_spec=dict(
                num_modules=(2, 4, 2),
                num_branches=(2, 3, 4),
                num_blocks=(2, 2, 2),
                module_type=('NAIVE', 'NAIVE', 'NAIVE'),
                with_fuse=(True, True, True),
                reduce_ratios=(1, 1, 1),
                num_channels=(
                    (30, 60),
                    (30, 60, 120),
                    (30, 60, 120, 240),
                )),
            with_head=True,
        ))

    loss_head_cfg = dict(
        type='KeypointsExtractorHead',
        input_feature_depth=30,
        output_feature_depth=num_classes,
        channels_number=number_of_channels,
        output_stride=stride,
        thrs_conf=thrs_conf,
        use_offsets=True,
        expand_channels=expand_channels,
        test=True
    )
    model = LightningKeypointsEstimator.load_from_checkpoint(checkpoint_path=args.checkpoint_path,
                                                             backbone_cfg=backbone_cfg,
                                                             loss_head_cfg=loss_head_cfg)
    model.eval()
    model.cuda()

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 800, 600)

    if args.train_dir_order:
        for person_image_item in os.listdir(args.path_to_dir):
            person_image_folder_path = os.path.join(args.path_to_dir, person_image_item)
            for person_image in os.listdir(person_image_folder_path):
                if person_image.split('.')[-1] == 'pickle':
                    continue
                filepath = os.path.join(person_image_folder_path, person_image)
                run_test_on_image(filepath=filepath)
    else:
        for person_image in os.listdir(args.path_to_dir):
            person_image_path = os.path.join(args.path_to_dir, person_image)
            run_test_on_image(filepath=person_image_path)
