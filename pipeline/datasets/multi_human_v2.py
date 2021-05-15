import os
import numpy as np
import pickle
import cv2

from torch.utils.data import Dataset

flip_correspondece_coco_mhv2 = [(2, 5), (3, 6), (4, 7), (8, 11), (9, 12), (10, 13), (16, 17), (14, 15)]


class MHV2(Dataset):
    def __init__(self,
                 is_train: bool,
                 dataset_dir: str,
                 debug: bool,
                 transforms=None):
        super(MHV2, self).__init__()
        self._is_train = is_train
        self._debug = debug
        self._transforms = transforms
        self._images_paths = []
        self._annos_path = []
        subdir_name = 'train' if self._is_train else 'val'
        stage_dir = os.path.join(dataset_dir, subdir_name)
        for image_folder in os.listdir(stage_dir):
            full_image_folder_path = os.path.join(stage_dir, image_folder)
            annotations_in_folder = [item.split('.')[0] for item in os.listdir(full_image_folder_path)
                                     if item.split('.')[1] == 'pickle']
            for person_ann in annotations_in_folder:
                person_image_name = os.path.join(full_image_folder_path, person_ann + '.jpg')
                corresponding_anno_name = os.path.join(full_image_folder_path, person_ann + '.pickle')
                if os.path.exists(person_image_name):
                    self._images_paths.append(person_image_name)
                    self._annos_path.append(corresponding_anno_name)
        print('{} set consists of {} pairs img/ann'.format('Train' if self._is_train else 'Validation',
                                                           len(self._images_paths)))

    def __getitem__(self, idx):
        image_filename = self._images_paths[idx]
        anno_filename = self._annos_path[idx]
        image = cv2.imread(image_filename)
        with open(anno_filename, 'rb') as f:
            annos_array = np.array(pickle.load(f))

        image_anno_dict = dict(image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                               keypoints=annos_array,
                               flip_correspondece=flip_correspondece_coco_mhv2)
        if self._transforms:
            image_anno_dict = self._transforms(**image_anno_dict)

        if self._debug:
            image_copy_debug = image_anno_dict['image'].detach().cpu().numpy()
            image_copy_debug = np.transpose(image_copy_debug, (1, 2, 0))
            for kps in image_anno_dict['keypoints']:
                if kps[2] > 0:
                    image_copy_debug = cv2.circle(image_copy_debug, (int(kps[0]), int(kps[1])), 4, (0, 0, 255), 2)
            cv2.imshow('debug', image_copy_debug)
            cv2.waitKey(0)

        return image_anno_dict

    def __len__(self):
        return len(self._annos_path)

