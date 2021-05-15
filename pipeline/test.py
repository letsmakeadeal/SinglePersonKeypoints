import cv2
from
if __name__ == '__main__':
    path_to_dir = '/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2-single/val'
    checkpoint_path = '/home/ivan/MLTasks/home_projects/SinglePersonKpsEstimation/results/'
    width = 128
    height = 128

    transforms = A.Compose([
        ResizeWithKeepAspectRatio(height=height, width=width, divider=divider),
        A.Normalize(mean=(0., 0., 0.), std=(1., 1., 1.)),
        A.ToTensorV2()
    ])

    model = LightningEquipmentDetNet.load_from_checkpoint(checkpoint_path=checkpoint_path,
                                                          load_from_checkpoint=None,
                                                          backbone_cfg=backbone_cfg,
                                                          loss_head_cfg=loss_head_cfg)
    model.eval()
    model.cuda()

    thrs_conf = 0.2
    run_on_videos = False

    cv2.namedWindow("debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("debug", 800, 600)

    if run_on_videos:
        for video in os.listdir(path_to_dir):
            print(f'Playing video {video}')
            filename = os.path.join(path_to_dir, video)
            vidcap = cv2.VideoCapture(filename)
            success, image_initial = vidcap.read()
            success = True
            while success:
                success, image_initial = vidcap.read()
                if not success or image_initial is None:
                    continue
                key = cv2.waitKey(4)
                if key == 27:
                    exit(0)
                if key == ord('n'):
                    break
                if key == ord('f'):
                    continue

                bboxes, image_to_debug = process_image(image_initial=image_initial,
                                                       model=model,
                                                       transforms=transforms)

                show_bboxes(bboxes=bboxes,
                            debug_image=image_to_debug,
                            idententy_per_class=idententy_per_class)
    else:
        for filename in os.listdir(path_to_dir):
            if filename.split('.')[-1] == 'txt':
                continue
            image_initial = cv2.imread(os.path.join(path_to_dir, filename))
            bboxes, image_to_debug = process_image(image_initial=image_initial,
                                                   model=model,
                                                   transforms=transforms)

            show_bboxes(bboxes=bboxes,
                        debug_image=image_to_debug,
                        idententy_per_class=idententy_per_class)
