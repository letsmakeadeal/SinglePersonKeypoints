import torch
from torch import nn
from typing import Tuple, List, Union


from pipeline.losses import L1Loss, SigmoidFocalLoss, MSELoss
from pipeline.utils import draw_heatmap


class KeypointsExtractorHead(nn.Module):
    def __init__(self,
                 input_feature_depth: int,
                 output_feature_depth: int,
                 use_offsets=False,
                 offsets_factor=1,
                 output_stride: int = 1,
                 thrs_conf: float = 0.3,
                 test=False):
        super(KeypointsExtractorHead, self).__init__()
        self._input_feature_depth = input_feature_depth
        self._output_feature_depth = output_feature_depth
        self._use_offsets = use_offsets
        self._offsets_factor = offsets_factor
        self._test = test
        self._score_threshold = thrs_conf
        self._output_stride = output_stride
        self._hm_loss = SigmoidFocalLoss()

        if self._use_offsets:
            self._offsets_loss = MSELoss()

        self._init_layers()

    def _init_layers(self):
        self._first_hm_conv_block = nn.Sequential(
            nn.Conv2d(self._input_feature_depth, self._input_feature_depth, 3, 1, 1),
            nn.BatchNorm2d(self._input_feature_depth),
            nn.ReLU(inplace=False)
        )

        self._second_hm_conv_block = nn.Sequential(
            nn.Conv2d(self._input_feature_depth, self._input_feature_depth, 3, 1, 1),
            nn.BatchNorm2d(self._input_feature_depth),
            nn.ReLU(inplace=False)
        )

        self._last_hm_conv_module = nn.Conv2d(self._input_feature_depth, self._output_feature_depth, 3, 1, 1)
        self._hm_layers = nn.Sequential(self._first_hm_conv_block,
                                        self._second_hm_conv_block,
                                        self._last_hm_conv_module)
        if self._use_offsets:
            self._first_offset_conv_block = nn.Sequential(
                nn.Conv2d(self._input_feature_depth, self._input_feature_depth, 3, 1, 1),
                nn.BatchNorm2d(self._input_feature_depth),
                nn.ReLU(inplace=False)
            )

            self._second_offset_conv_block = nn.Sequential(
                nn.Conv2d(self._input_feature_depth, self._input_feature_depth, 3, 1, 1),
                nn.BatchNorm2d(self._input_feature_depth),
                nn.ReLU(inplace=False)
            )

            self._last_offset_conv_module = nn.Conv2d(self._input_feature_depth,
                                                      self._output_feature_depth * 2, 3, 1, 1)
            self._offset_layers = nn.Sequential(self._first_offset_conv_block,
                                                self._second_offset_conv_block,
                                                self._last_offset_conv_module)

    def forward(self, x: Union[torch.tensor, List]):
        if isinstance(x, List):
            x = x[0]

        hm_out = self._hm_layers(x)
        if self._use_offsets:
            offsets_out = self._offset_layers(x)

        return hm_out, offsets_out if self._use_offsets else hm_out

    def loss(self, predictions: Union[torch.tensor, Tuple],
             keypoints_gt: torch.tensor):
        if self._use_offsets:
            hm_pred = predictions[0]
            offset_pred = predictions[1]
        else:
            hm_pred = predictions

        heatmaps_for_persons = torch.stack([draw_heatmap(gt_keypoints=keypoints_gt[person_idx],
                                                         feat_shape=hm_pred.shape[1:],
                                                         stride=self._output_stride,
                                                         radius=0.1) for person_idx in range(len(keypoints_gt))])

        hm_loss = self._hm_loss(predicted=hm_pred, gt=heatmaps_for_persons)
        total_loss = dict(heatmaploss=hm_loss, total_loss=hm_loss)
        if self._use_offsets:
            encoded_offsets_gt = torch.stack(
                [self._generate_offsets(person_heatmaps_gt=heatmaps_for_persons[person_idx],
                                        gt_keypoints=keypoints_gt[person_idx])
                 for person_idx in range(len(keypoints_gt))])

            offsets_loss = self._offsets_loss(predicted=offset_pred, gt=encoded_offsets_gt)
            total_loss = dict(heatmaploss=hm_loss,
                              offsets_loss=offsets_loss,
                              total_loss=hm_loss + self._offsets_factor * offsets_loss)

        return total_loss

    def _generate_offsets(self, person_heatmaps_gt: torch.tensor,
                          gt_keypoints: torch.tensor):
        gt_keypoints_int = gt_keypoints.int()
        hm_shape = person_heatmaps_gt.shape
        offsets = torch.zeros((hm_shape[0] * 2, hm_shape[1], hm_shape[2]),
                              device=person_heatmaps_gt.device, dtype=torch.int64)
        heatmaps_flattened = person_heatmaps_gt.view(hm_shape[0], hm_shape[1] * hm_shape[2])
        non_zero_idxs = heatmaps_flattened.nonzero()
        for idx, keypoint in enumerate(gt_keypoints_int):
            if keypoint[2] != 2:
                continue
            current_gaussian_idxs = non_zero_idxs[non_zero_idxs[..., 0] == idx][..., 1]
            xs = current_gaussian_idxs % hm_shape[2]
            ys = current_gaussian_idxs // hm_shape[1]
            encoded_offsets_x = -(xs * self._output_stride - keypoint[0])
            encoded_offsets_y = -(ys * self._output_stride - keypoint[1])

            offsets[idx * 2, ys.long(), xs.long()] = encoded_offsets_x
            offsets[idx * 2 + 1, ys.long(), xs.long()] = encoded_offsets_y

        return offsets

    def get_keypoints(self, predicted: Union[torch.tensor, Tuple], batch_info: torch.tensor):
        if self._use_offsets:
            hm_predicted = predicted[0]
            offsets = predicted[1]
        else:
            hm_predicted = predicted

        hm_predicted = hm_predicted.sigmoid()
        feature_shape = hm_predicted.shape
        predicted_flattened = hm_predicted.view(feature_shape[0], feature_shape[1],
                                                feature_shape[2] * feature_shape[3])
        maxs, argmaxes = torch.max(predicted_flattened, dim=-1)

        argmaxes_to_keypoints = torch.zeros((argmaxes.shape[0], argmaxes.shape[1], 2))
        argmaxes_to_keypoints[..., 0] = argmaxes % feature_shape[3]
        argmaxes_to_keypoints[..., 0][argmaxes_to_keypoints[..., 0] < 0] = 0
        argmaxes_to_keypoints[..., 1] = argmaxes // feature_shape[2]

        if self._use_offsets:
            x_offsets = offsets[:, 0::2, ...]
            y_offsets = offsets[:, 1::2, ...]
            x_offsets_flattened = x_offsets.view(x_offsets.shape[0], x_offsets.shape[1], x_offsets.shape[2] *
                                                 x_offsets.shape[3])
            y_offsets_flattened = y_offsets.view(y_offsets.shape[0], y_offsets.shape[1], y_offsets.shape[2] *
                                                 y_offsets.shape[3])

            x_offsets_from_argmaxes = torch.gather(x_offsets_flattened, 2, argmaxes.unsqueeze(-1))
            y_offsets_from_argmaxes = torch.gather(y_offsets_flattened, 2, argmaxes.unsqueeze(-1))

            argmaxes_to_keypoints *= self._output_stride
            argmaxes_to_keypoints[..., 0] += x_offsets_from_argmaxes.squeeze(-1).to(device=argmaxes_to_keypoints.device)
            argmaxes_to_keypoints[..., 1] += y_offsets_from_argmaxes.squeeze(-1).to(device=argmaxes_to_keypoints.device)
        else:
            argmaxes_to_keypoints *= self._output_stride

        visability = torch.zeros_like(argmaxes_to_keypoints[..., 1]).unsqueeze(-1)

        validated_maxs_idxs = maxs >= self._score_threshold
        unvalidated_idxs = maxs < self._score_threshold
        visability[validated_maxs_idxs] = 2
        argmaxes_to_keypoints[unvalidated_idxs] = 0

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
