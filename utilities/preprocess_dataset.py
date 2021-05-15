import os
import scipy.io
import cv2
import numpy as np
import pickle

from tqdm import tqdm

classes_mh2v = ['Right-ankle', 'Right-knee', 'Right-hip', 'Left-hip',
                'Left-knee', 'Left-ankle', 'Pelvis', 'Thorax',
                'Upper-neck', 'Head-top', 'Right-wrist', 'Right-elbow',
                'Right-shoulder', 'Left-shoulder', 'Left-elbow', 'Left-wrist']

mhv2_to_coco_correspondence = [(0, 10), (1, 9), (2, 8), (3, 11),
                               (4, 12), (5, 13), (9, 0), (10, 4), (11, 3),
                               (12, 2), (13, 5), (14, 6), (15, 7)]

if __name__ == '__main__':
    padding_size_percent = 0
    debug = False
    path_to_dataset = '/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2' \
                      '/LV-MHP-v2'
    output_single_person_data_path = '/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2-single'

    if not os.path.exists(output_single_person_data_path):
        os.mkdir(output_single_person_data_path)

    to_process_folders = ['train', 'val']
    for stage_folder in os.listdir(path_to_dataset):
        if stage_folder not in to_process_folders:
            continue
        created_new_stage_folder_name = os.path.join(output_single_person_data_path, stage_folder)
        if not os.path.exists(created_new_stage_folder_name):
            os.mkdir(created_new_stage_folder_name)
        full_path_to_stage_folder = os.path.join(path_to_dataset, stage_folder)

        path_to_stage_anns = os.path.join(full_path_to_stage_folder, 'pose_annos')
        path_to_stage_images = os.path.join(full_path_to_stage_folder, 'images')
        stage_counter = 0
        for annotation_name in tqdm(os.listdir(path_to_stage_anns)):
            corresponding_image = annotation_name.split('.')[0]
            created_image_folder_name = os.path.join(created_new_stage_folder_name, corresponding_image)
            if not os.path.exists(created_image_folder_name):
                os.mkdir(created_image_folder_name)
            stage_counter += 1
            readed_annos = scipy.io.loadmat(os.path.join(path_to_stage_anns, annotation_name))
            image = cv2.imread(str(os.path.join(path_to_stage_images, corresponding_image)) + '.jpg')
            h_image, w_image, c_image = image.shape
            persons_keys = [key for key in readed_annos.keys() if 'person' in key]
            incorrect_persons_counter = 0
            for person in persons_keys:
                single_person_ann = np.array(readed_annos[person])

                top_left = single_person_ann[18]
                bottom_right = single_person_ann[19]
                keypoints_to_coco_format = np.ones((17, 2)) * -1
                for i in range(len(mhv2_to_coco_correspondence)):
                    coco_idx = mhv2_to_coco_correspondence[i][1]
                    mvhp_idx = mhv2_to_coco_correspondence[i][0]
                    keypoints_to_coco_format[coco_idx] = single_person_ann[mvhp_idx][:2]

                xs = keypoints_to_coco_format[..., 0]
                ys = keypoints_to_coco_format[..., 1]
                xs_transformed = xs
                ys_transformed = ys

                visability_column = np.ones(ys.shape) * 2

                verified_kps = np.where((xs >= 0) * (ys >= 0) * (xs <= w_image) * (ys <= h_image))[0]
                incorrect_kps = np.array([i for i in range(len(xs)) if i not in verified_kps])

                if len(verified_kps) < 2:
                    incorrect_persons_counter += 1
                    continue

                person_width, person_height = [bottom_right[0] - top_left[0],
                                               bottom_right[1] - top_left[1]]
                padding_x, padding_y = [person_width * padding_size_percent,
                                        person_height * padding_size_percent]

                if top_left[0] - padding_x <= 0:
                    padding_x = 0

                if top_left[1] - padding_y <= 0:
                    padding_y = 0

                xs_transformed[verified_kps] = xs[verified_kps] - top_left[0] + padding_x
                ys_transformed[verified_kps] = ys[verified_kps] - top_left[1] + padding_y

                if len(incorrect_kps) > 0:
                    xs_transformed[incorrect_kps] = -1
                    ys_transformed[incorrect_kps] = -1
                    visability_column[incorrect_kps] = 0

                single_person_ann_transformed = np.concatenate([xs_transformed[..., np.newaxis],
                                                                ys_transformed[..., np.newaxis],
                                                                visability_column[..., np.newaxis]],
                                                               axis=1)

                single_person_image = image[int(top_left[1] - padding_y): int(bottom_right[1] + padding_y),
                                      int(top_left[0] - padding_x): int(bottom_right[0] + padding_x)]
                if debug:
                    for (x, y) in zip(xs_transformed[verified_kps], ys_transformed[verified_kps]):
                        single_person_image = cv2.circle(single_person_image, (int(x), int(y)), 4, (0, 0, 255), 2)

                    cv2.namedWindow("full", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('full', 900, 900)

                    cv2.namedWindow("single", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow('single', 900, 900)

                    cv2.imshow('full', image)
                    cv2.imshow('single', single_person_image)
                    cv2.waitKey(0)

                full_image_path = str(os.path.join(created_image_folder_name, person)) + '.jpg'

                try:
                    if not os.path.exists(full_image_path):
                        cv2.imwrite(full_image_path, single_person_image)
                except cv2.error as e:
                    continue

                ann_path_filename = str(os.path.join(created_image_folder_name, person)) + '.pickle'
                if not os.path.exists(ann_path_filename):
                    with open(ann_path_filename, 'wb') as f:
                        pickle.dump(single_person_ann_transformed, f)

            if incorrect_persons_counter == len(persons_keys):
                os.rmdir(created_image_folder_name)

            # if stage_counter > 3:
            #     break
