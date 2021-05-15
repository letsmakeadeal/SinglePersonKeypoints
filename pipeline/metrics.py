from collections import defaultdict
from typing import Dict, List
import numpy as np
import json

import torch

from pytorch_lightning.metrics import Metric
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

__all__ = ['CocoAPs']


class CocoAPs(Metric):
    def __init__(self):
        super().__init__(compute_on_step=False)
        self.add_state('_keypoints_gt_accumulated', [])
        self.add_state('_keypoints_dt_accumulated', [])
        self.reset_state()

    def reset_state(self):
        self._keypoints_gt = {'images': [],
                              'annotations': [],
                              'categories': [{"supercategory": "person",
                                              "id": 1,
                                              "name": "person",
                                              "keypoints": ["unused" for _ in range(17)],
                                              "skeleton": []}]
                              }
        self._keypoints_dt = []
        self._keypoints_gt_accumulated = []
        self._keypoints_dt_accumulated = []
        self._image_counter = 0
        self.annotation_id_gt = 1

    def append_to_results(self, data, gt: bool):
        if gt:
            for image_id in range(data.shape[0]):
                keypoints_gt = data[image_id]
                keypoints_to_anns = keypoints_gt.reshape(-1).cpu().numpy().tolist()
                gt_annotation = {
                    'id': self.annotation_id_gt,
                    'image_id': self._image_counter + image_id,
                    'category_id': 1,
                    'keypoints': keypoints_to_anns,
                    'num_keypoints': 17,
                    'iscrowd': 0
                }
                x = keypoints_to_anns[0::3]
                y = keypoints_to_anns[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                gt_annotation['area'] = (x1 - x0) * (y1 - y0)
                gt_annotation['bbox'] = [x0, y0, x1 - x0, y1 - y0]

                self._keypoints_gt['images'].append({'id': self._image_counter + image_id})
                self._keypoints_gt['annotations'].append(gt_annotation)
                self.annotation_id_gt += 1
        else:
            for image_id in range(data.shape[0]):
                keypoints_to_anns = data[image_id].reshape(-1).tolist()
                self._keypoints_dt.append({
                    'image_id': self._image_counter + image_id,
                    'category_id': 1,
                    'keypoints': keypoints_to_anns,
                    'score': 1})

    def update(self, predicted_keypoints: List, gt_info: torch.tensor):
        self._keypoints_gt_accumulated.append(gt_info)
        self._keypoints_dt_accumulated.append(predicted_keypoints)

    def compute(self):
        for valid_batch_num in range(len(self._keypoints_gt_accumulated)):
            self.append_to_results(data=self._keypoints_dt_accumulated[valid_batch_num], gt=False)
            self.append_to_results(data=self._keypoints_gt_accumulated[valid_batch_num], gt=True)
            self._image_counter += len(self._keypoints_gt_accumulated[valid_batch_num])

        coco_gt = COCO()
        coco_gt.dataset = self._keypoints_gt
        coco_gt.createIndex()

        json_dt_res_file = 'temp.json'

        with open(json_dt_res_file, "w+") as f:
            json.dump(self._keypoints_dt, f)
        coco_dt = coco_gt.loadRes(resFile=json_dt_res_file)

        coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        stats = coco_eval.stats
        self.reset_state()

        return stats[0]
