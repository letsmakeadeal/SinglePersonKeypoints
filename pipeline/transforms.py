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
        self._size = height, width
        self._stride = stride
        self._pad = pad

    def __call__(self, force_apply: bool, **data):
        image = data['image']
        canvas = np.ones((*self._size, 3), dtype=np.uint8) * self._pad

        h, w = image.shape[:2]
        vertically_orientation = h > w
        scale = min(self._size[0] / float(h), self._size[1] / float(w))
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=random_interpolation())
        if vertically_orientation:
            pad_x = self._size[1] / 2 - w * scale / 2
            pad_y = 0
        else:
            pad_y = self._size[0] / 2 - h * scale / 2
            pad_x = 0

        data['pad_x'] = pad_x
        data['pad_y'] = pad_y

        if vertically_orientation:
            x1 = int(pad_x)
            x2 = int(x1) + image.shape[1]
            canvas[:image.shape[0], x1: x2] = image
        else:
            y1 = int(pad_y)
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
                keypoints[visible_kps_idx, 0] += pad_x
            else:
                keypoints[visible_kps_idx, 1] += pad_y
            data['keypoints'] = keypoints

        return data


class Flip(object):
    def __init__(self, p: float = 0.5):
        self._p = p

    def __call__(self, force_apply: bool, p=0.5, **data):
        if random.random() < self._p:
            return data
        image = data['image']
        h, w = data['image'].shape[:2]
        data['image'] = image[:, ::-1]

        if 'keypoints' in data:
            keypoints = data['keypoints']
            flipped = keypoints.copy()
            flipped[..., 0] = image.shape[1] - keypoints[..., 0] - 1
            flipped_pairs = flipped.copy()
            for a, b in data['flip_correspondence']:
                flipped_pairs[a, :] = flipped[b, :]
                flipped_pairs[b, :] = flipped[a, :]

            verified_kps = np.where((flipped_pairs[..., 0] >= 0) *
                                    (flipped_pairs[..., 1] >= 0) *
                                    (flipped_pairs[..., 0] <= w) *
                                    (flipped_pairs[..., 1] <= h) *
                                    (flipped_pairs[..., 2] > 0))[0]
            incorrect_kps = np.array([i for i in range(len(keypoints)) if i not in verified_kps])
            flipped_pairs[incorrect_kps, :2] = -1
            flipped_pairs[incorrect_kps, 2] = 0
            data['keypoints'] = flipped_pairs

        return data


class Rotate(object):

    def __init__(self,
                 p: float = 0.5,
                 angle: int = 45,
                 pad: int = 0):
        self._angle = angle
        self._pad = pad
        self._p = p

    def __call__(self, force_apply: bool, **data):
        if random.random() < self._p:
            return data

        keypoints = data['keypoints']
        visible_kps_idx = keypoints[:, 2] >= 0
        keypoints_visible = keypoints[visible_kps_idx]
        x0, y0, x1, y1 = map(int, bbox_from_keypoints(keypoints_visible))

        image = data['image']
        h, w = image.shape[:2]

        prev_incorrect_number = len(data['keypoints'])
        for _ in range(3):
            keypoints_rotated = keypoints_visible.copy()
            angle = random.randint(-self._angle, self._angle)

            transform = cv2.getRotationMatrix2D((x1 - x0, y1 - y0), angle, 1)
            image_rotated = cv2.warpAffine(image, transform, (w, h), flags=random_interpolation(),
                                           borderValue=self._pad)
            keypoints_rotated[..., :2] = cv2.transform(keypoints_visible[..., :2].reshape(-1, 1, 2), transform).reshape(
                -1, 2)

            verified_kps = np.where((keypoints_rotated[..., 0] >= 0) *
                                    (keypoints_rotated[..., 1] >= 0) *
                                    (keypoints_rotated[..., 0] <= w) *
                                    (keypoints_rotated[..., 1] <= h) *
                                    (keypoints_rotated[..., 2] > 0))[0]

            incorrect_kps = np.array([i for i in range(len(keypoints)) if i not in verified_kps])
            now_incorrect_number = len(incorrect_kps)

            if now_incorrect_number < prev_incorrect_number:
                keypoints_rotated[incorrect_kps, :2] = -1
                keypoints_rotated[incorrect_kps, 2] = 0
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
