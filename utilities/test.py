import cv2
import os
import copy

import albumentations as A
import pytorch_lightning as pl
import numpy as np

from pipeline.transforms import ResizeAndPadImage
from modules.lightning_module import LightningKeypointsEstimator


def process_image(image_initial,
                  model: pl.LightningModule,
                  transforms):
    image_rgb = copy.deepcopy(cv2.cvtColor(image_initial, cv2.COLOR_BGR2RGB))

    image_dict = dict(image=image_rgb)
    batch_info = transforms(**image_dict)

    prediction = model(batch_info['image'].cuda().unsqueeze(0))
    keypoints = model.loss_head.get_keypoints(predicted=prediction,
                                              batch_info=batch_info)

    return keypoints


if __name__ == '__main__':
    path_to_dir = '/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2-single/val'
    checkpoint_path = '/home/ivan/MLTasks/home_projects/SinglePersonKpsEstimation/results/' \
                      'mhv2_epoch=27_cocoaps=0.3151.ckpt'
    width = 128
    height = 128
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
        output_stride=stride,
        thrs_conf=thrs_conf,
        use_offsets=True,
        test=True
    )
    model = LightningKeypointsEstimator.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                             load_from_checkpoint=checkpoint_path,
                                                             backbone_cfg=backbone_cfg,
                                                             loss_head_cfg=loss_head_cfg)
    model.eval()
    model.cuda()

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 800, 600)

    for person_image_folder in os.listdir(path_to_dir):
        person_image_folder_path = os.path.join(path_to_dir, person_image_folder)
        for person_image in os.listdir(person_image_folder_path):
            if person_image.split('.')[-1] == 'pickle':
                continue

            image_initial = cv2.imread(os.path.join(person_image_folder_path, person_image))
            cv2.imshow("initial", image_initial)
            keypoints = process_image(image_initial=image_initial,
                                      model=model,
                                      transforms=transforms)
            x = keypoints[0][..., 0]
            y = keypoints[0][..., 1]
            visabilities = keypoints[0][..., 2]
            for (x, y, v) in zip(x, y, visabilities):
                if v == 2:
                    image_initial = cv2.circle(image_initial, (int(x), int(y)), 4, (0, 0, 255), 2)

            cv2.imshow('debug', image_initial)
            cv2.waitKey(0)
