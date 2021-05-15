import cv2
import numpy as np
import random
import torch
import torch.nn as nn

from typing import Tuple


def bbox_from_keypoints(keypoints):
    x0, y0 = keypoints[..., :2].min(axis=0)
    x1, y1 = keypoints[..., :2].max(axis=0)
    return x0, y0, x1, y1


def random_interpolation():
    return random.choice([
        cv2.INTER_NEAREST,
        cv2.INTER_LINEAR,
        cv2.INTER_AREA,
        cv2.INTER_CUBIC,
        cv2.INTER_LANCZOS4,
    ])


def gaussian_2d(shape, sigma_x=1, sigma_y=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x / (2 * sigma_x * sigma_x) + y * y / (2 * sigma_y * sigma_y)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_truncate_gaussian(heatmap, center, h_radius, w_radius, k=1):
    h, w = 2 * h_radius + 1, 2 * w_radius + 1
    sigma_x = w / 6
    sigma_y = h / 6
    gaussian = gaussian_2d((h, w), sigma_x=sigma_x, sigma_y=sigma_y)
    gaussian = heatmap.new_tensor(gaussian)

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape[0:2]

    left, right = min(x, w_radius), min(width - x, w_radius + 1)
    top, bottom = min(y, h_radius), min(height - y, h_radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[h_radius - top:h_radius + bottom, w_radius - left:w_radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
        torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    return heatmap


def draw_heatmap(gt_keypoints, feat_shape: Tuple[int, int], stride: int, radius: float):
    output_h, output_w = feat_shape[-2:]
    num_keypoints, dimensions = gt_keypoints.shape

    assert dimensions in (2, 3)

    heatmap = gt_keypoints.new_zeros((num_keypoints, output_h, output_w))

    keys = gt_keypoints / stride
    keys_ints = keys[..., :2].to(torch.int)
    visibilities = keys[..., 2].to(torch.int)

    h_radius = int(radius * output_h)
    w_radius = int(radius * output_w)

    canvas = gt_keypoints.new_zeros((output_h, output_w))

    for idx in range(num_keypoints):
        canvas = canvas.zero_()
        if visibilities[idx] == 2:
            draw_truncate_gaussian(canvas, keys_ints[idx], h_radius, w_radius)
            heatmap[idx] = torch.max(heatmap[idx], canvas)

    return heatmap

