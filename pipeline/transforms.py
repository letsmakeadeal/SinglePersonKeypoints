import random
import cv2
import albumentations as AT
import numpy as np

from .utils import bbox_from_keypoints, random_interpolation


__all__ = ['Flip', 'Rotate', 'ResizeAndPadImage']


class ResizeAndPadImage(object):

    def __init__(self,
                 width: int,
                 height: int,
                 stride: int = 1,
                 pad: int = 0):
        self.size = height, width
        self.stride = stride
        self.pad = pad

    def __call__(self,  force_apply: bool, **data):
        image = data['image']
        canvas = np.ones((*self.size, 3), dtype=np.uint8) * self.pad

        h, w = image.shape[:2]
        vertically_orientation = h > w
        scale = min(self.size[0] / float(h), self.size[1] / float(w))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=random_interpolation())

        if vertically_orientation:
            x1 = int(self.size[1] / 2 - w * scale / 2)
            x2 = int(x1) + image.shape[1]
            canvas[:image.shape[0], x1: x2] = image
        else:
            y1 = int(self.size[0] / 2 - h * scale / 2)
            y2 = int(y1) + image.shape[0]
            canvas[y1:y2, :image.shape[1]] = image

        data['scale'] = scale
        data['image'] = canvas

        if 'keypoints' in data:
            keypoints = data['keypoints']
            visible_kps_idx = keypoints[:, 2] > 0
            keypoints[visible_kps_idx] *= scale
            keypoints[visible_kps_idx, 2] = 2
            if vertically_orientation:
                pad_x = self.size[1] / 2 - w * scale / 2
                keypoints[visible_kps_idx, 0] += pad_x
                data['pad_x'] = pad_x
                data['pad_y'] = 0
            else:
                pad_y = self.size[0] / 2 - h * scale / 2
                keypoints[visible_kps_idx, 1] += pad_y
                data['pad_x'] = 0
                data['pad_y'] = pad_y

            data['keypoints'] = keypoints

        return data


class Flip(object):
    def __call__(self, force_apply: bool, **data):
        image = data['image']
        data['image'] = image[:, ::-1]

        if 'keypoints' in data:
            keypoints = data['keypoints']

            flipped = keypoints.copy()
            flipped[..., 0] = image.shape[1] - keypoints[..., 0] - 1
            flipped_pairs = flipped.copy()
            for a, b in data['flip_correspondence']:
                flipped_pairs[a, :] = flipped[b, :]
                flipped_pairs[b, :] = flipped[a, :]

            verified_kps = (flipped_pairs[..., 0] >= 0) * (flipped_pairs[..., 1] >= 0) *  \
                           (flipped_pairs[..., 0] <= data['image'].shape[1]) * \
                           (flipped_pairs[..., 1] <= data['image'].shape[0]) * \
                           (flipped_pairs[..., 2] > 0)
            incorrect_kps = np.array([i for i in range(len(keypoints)) if i not in verified_kps])
            flipped_pairs[incorrect_kps] = -1

            data['keypoints'] = flipped_pairs

        return data


class Rotate(object):

    def __init__(self,
                 angle: int = 45,
                 pad: int = 0):
        self.angle = angle
        self.pad = pad

    def __call__(self,  force_apply: bool, **data):
        keypoints = data['keypoints']
        visible_kps_idx = keypoints[:, 2] >= 0
        keypoints_visible = keypoints[visible_kps_idx]
        x0, y0, x1, y1 = map(int, bbox_from_keypoints(keypoints_visible))

        image = data['image']
        h, w = image.shape[:2]

        for _ in range(3):
            keypoints_rotated = keypoints_visible.copy()
            angle = random.randint(-self.angle, self.angle)
            transform = cv2.getRotationMatrix2D((x1 - x0, y1 - y0), angle, 1)
            image_rotated = cv2.warpAffine(image, transform, (w, h), flags=random_interpolation(), borderValue=self.pad)
            keypoints_rotated[..., :2] = cv2.transform(keypoints_visible[..., :2].reshape(-1, 1, 2), transform).reshape(-1, 2)

            if np.all(keypoints_rotated[..., :2] > 0) and \
                    np.all(keypoints_rotated[..., 0] < w) and \
                    np.all(keypoints_rotated[..., 1] < h):
                data['image'] = image_rotated
                keypoints[visible_kps_idx] = keypoints_rotated
                data['keypoints'] = keypoints

                return data

        return data


class SquareCrop(object):

    def __init__(self,
                 expand: float = 0.5,
                 random: bool = True,
                 pad: int = 0):
        self.min_expand = 0.1
        assert expand >= self.min_expand

        self.expand = expand
        self.random = random
        self.pad = pad

    def __call__(self, data):
        keypoints = data['keypoints']
        visible_kps_idx = keypoints[:, 2] > 0
        keypoints_visible = keypoints[visible_kps_idx]

        x0, y0, x1, y1 = map(int, bbox_from_keypoints(keypoints_visible))
        expand = self.expand
        if self.random:
            expand = float(np.random.uniform(self.min_expand, expand))
        size = int(max(x1 - x0, y1 - y0) * (1 + expand))

        image = data['image']
        canvas = np.ones((size, size, 3), dtype=np.uint8) * self.pad

        shift_x0 = (size - (x1 - x0)) // 2
        shift_y0 = (size - (y1 - y0)) // 2
        if self.random:
            shift_x0 = random.randint(0, size - (x1 - x0) + 1)
            shift_y0 = random.randint(0, size - (y1 - y0) + 1)
        crop_x0 = max(1, x0 - shift_x0)
        crop_y0 = max(1, y0 - shift_y0)

        keypoints_visible[..., :2] = keypoints_visible[..., :2] - np.array([crop_x0, crop_y0], dtype=np.float32)
        cropped = image[crop_y0: crop_y0 + size, crop_x0: crop_x0 + size]
        canvas[:cropped.shape[0], :cropped.shape[1]] = cropped

        data['image'] = canvas

        keypoints[visible_kps_idx] = keypoints_visible
        data['keypoints'] = keypoints

        return data

