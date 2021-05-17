seed = 42
gpus = [0]
batch_size = 16
epochs = 32
num_workers = 4

train_dataset_len = 40184 // batch_size
height = 128
width = 128

# NOTE(i.rodin) Number of classes as in coco converted
num_classes = 17

background_color = (0, 0, 0)
stride = 1

trainer_cfg = dict(
    gpus=gpus,
    max_epochs=epochs,
    callbacks=[
        dict(type='LearningRateMonitor', logging_interval='step'),
        dict(type='ModelCheckpoint', save_top_k=2, verbose=True, mode='min',
             monitor='cocoaps', dirpath='../results/',
             filename='mhv2_{epoch:02d}_{cocoaps:.4f}')
    ],
    benchmark=True,
    deterministic=True,
    terminate_on_nan=False,
    distributed_backend='ddp',
    precision=32,
    sync_batchnorm=True
)
wandb_cfg = dict(
    name=f'{__file__.split("/")[-1].replace(".py", "")}_{height}_{width}_{batch_size}_ep{epochs}',
    project='Single person keypoints',
    entity='letsmakeadeal'
)

backbone_cfg = dict(
    type='Unet',
    encoder_name="mobilenet_v2",
    encoder_weights="imagenet",
    in_channels=3,
    classes=num_classes,
)

loss_head_cfg = dict(
    type='ConvHead',
    input_feature_depth=num_classes,
    output_stride=stride
)

metric_cfgs = [
    dict(type='CocoAPs')
]

train_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='ResizeAndPadImage', height=height, width=width),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

val_transforms_cfg = dict(
    type='Compose', transforms=[
        dict(type='ResizeAndPadImage', height=height, width=width),
        dict(type='Normalize', mean=(0., 0., 0.), std=(1., 1., 1.)),
        dict(type='ToTensorV2')
    ])

train_dataset_cfg = dict(
    type='MHV2',
    is_train=True,
    dataset_dir='/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2-single',
    debug=False

)

val_dataset_cfg = dict(
    type='MHV2',
    is_train=False,
    dataset_dir='/home/ivan/MLTasks/Datasets/PosesDatasets/LV-MHP-v2-single',
    debug=False
)

train_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=True
)

val_dataloader_cfg = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
)

optimizer_cfg = dict(
    type='SGD',
    lr=0.001,
    momentum=0.9
)
scheduler_cfg = dict(
    type='CyclicLR',
    base_lr=1e-6 * len(gpus),
    max_lr=2e-3 * len(gpus),
    step_size_up=int(train_dataset_len * epochs // (2 * len(gpus))),
    cycle_momentum=False,
)
scheduler_update_params = dict(
    interval='step',
    frequency=1
)

module_cfg = dict(
    type='LightningKeypointsEstimator',
    checkpoint_path=None,
    backbone_cfg=backbone_cfg,
    loss_head_cfg=loss_head_cfg,
    metric_cfgs=metric_cfgs,
    train_transforms_cfg=train_transforms_cfg,
    val_transforms_cfg=val_transforms_cfg,
    train_dataset_cfg=train_dataset_cfg,
    val_dataset_cfg=val_dataset_cfg,
    train_dataloader_cfg=train_dataloader_cfg,
    val_dataloader_cfg=val_dataloader_cfg,
    optimizer_cfg=optimizer_cfg,
    scheduler_cfg=scheduler_cfg,
    scheduler_update_params=scheduler_update_params
)
