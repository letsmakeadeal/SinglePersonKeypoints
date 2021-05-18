import torch
from torch import nn
from typing import Tuple

from pipeline.losses import L1Loss, SigmoidFocalLoss
from pipeline.utils import draw_heatmap


class ConvHead(nn.Module):
    def __init__(self,
                 input_feature_depth: int,
                 output_stride: int = 1,
                 thrs_conf: float = 0.3,
                 test=False):
        super(ConvHead, self).__init__()
        self._test = test
        self._score_threshold = thrs_conf
        self._loss = SigmoidFocalLoss()
        self._output_stride = output_stride
        self._first_conv_block = nn.Sequential(
            nn.Conv2d(input_feature_depth, input_feature_depth, 3, 1, 1),
            nn.BatchNorm2d(input_feature_depth),
            nn.ReLU(inplace=False)
        )

        self._second_conv_block = nn.Sequential(
            nn.Conv2d(input_feature_depth, input_feature_depth, 3, 1, 1),
            nn.BatchNorm2d(input_feature_depth),
            nn.ReLU(inplace=False)
        )

        self._last_conv_module = nn.Conv2d(input_feature_depth, input_feature_depth, 3, 1, 1)

    def forward(self, x: torch.tensor):
        x = self._first_conv_block(x)
        x = self._second_conv_block(x)
        x = self._last_conv_module(x)

        return x

    def loss(self, predictions: torch.tensor, keypoints_gt: torch.tensor):
        heatmaps_for_persons = torch.stack([draw_heatmap(gt_keypoints=keypoints_gt[person_kps],
                                                         feat_shape=predictions.shape[1:],
                                                         stride=1,
                                                         radius=0.1) for person_kps in range(len(keypoints_gt))])

        total_loss = self._loss(predicted=predictions, gt=heatmaps_for_persons)

        return dict(heatmaploss=total_loss)

    def get_keypoints(self, predicted: torch.tensor, batch_info: torch.tensor):
        predicted = predicted.sigmoid()
        feature_shape = predicted.shape
        predicted_flattened = predicted.view(feature_shape[0], feature_shape[1],
                                             feature_shape[2] * feature_shape[3])
        maxs, argmaxes = torch.max(predicted_flattened, dim=-1)
        argmaxes_to_keypoints = torch.zeros((argmaxes.shape[0], argmaxes.shape[1], 2))
        argmaxes_to_keypoints[..., 0] = argmaxes % feature_shape[3]
        argmaxes_to_keypoints[..., 0][argmaxes_to_keypoints[..., 0] < 0] = 0
        argmaxes_to_keypoints[..., 1] = argmaxes // feature_shape[2]

        argmaxes_to_keypoints *= self._output_stride
        visability = torch.zeros_like(argmaxes_to_keypoints[..., 1]).unsqueeze(-1)
        validated_maxs_idxs = maxs > self._score_threshold
        visability[validated_maxs_idxs] = 2
        argmaxes_to_keypoints = torch.cat([argmaxes_to_keypoints, visability], dim=-1)

        output_keypoints_as_array = []
        if self._test:
            if False:
                import cv2
                import numpy as np
                canvas = np.zeros((predicted.shape[-2], predicted.shape[-1], 1))
                for kps in argmaxes_to_keypoints[0]:
                    canvas = cv2.circle(canvas, (int(kps[0]), int(kps[1])), 4, (255, 255, 255), 2)
                cv2.imshow('canvas', canvas)
                cv2.waitKey(0)

            for image_idx in range(argmaxes_to_keypoints.shape[0]):
                keypoints = argmaxes_to_keypoints[image_idx]
                pad_x = batch_info['pad_x']
                pad_y = batch_info['pad_y']
                scale = batch_info['scale']
                keypoints[..., 0] -= pad_x
                keypoints[..., 1] -= pad_y
                keypoints[..., :2] /= scale

                output_keypoints_as_array.append(keypoints)
        else:
            output_keypoints_as_array = argmaxes_to_keypoints.cpu().numpy()

        return output_keypoints_as_array
