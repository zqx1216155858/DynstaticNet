exp_name = 'all-in-one_DynStaticNet_psnr'

# model settings
model = dict(
    type='BasicVSR',
    generator=dict(
        type='DynStaticNet',
        in_chans=3,
        dim=48,
        num_heads=[1,2,4],
        depth=[4, 6, 8]),
    pixel_loss=dict(type='PSNRLoss', loss_weight=1.0))

# model training and testing settings
train_cfg = dict(fix_iter=5000)
test_cfg = dict(metrics=['PSNR', 'SSIM'], crop_border=0)

# dataset settings
train_dataset_type = 'SRFolderMultipleGTDataset'
val_dataset_type = 'SRFolderMultipleGTDataset'

train_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.jpg'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='PairedRandomCrop', gt_patch_size=128),
    dict(
        type='Flip', keys=['lq', 'gt'], flip_ratio=0.5,
        direction='horizontal'),
    dict(type='Flip', keys=['lq', 'gt'], flip_ratio=0.5, direction='vertical'),
    dict(type='RandomTransposeHW', keys=['lq', 'gt'], transpose_ratio=0.5),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(type='Collect', keys=['lq', 'gt'], meta_keys=['lq_path', 'gt_path'])
]

test_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.jpg'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='gt',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq', 'gt']),
    dict(type='FramesToTensor', keys=['lq', 'gt']),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key'])
]

demo_pipeline = [
    dict(
        type='GenerateSegmentIndices',
        interval_list=[1],
        filename_tmpl='{:04d}.jpg'),
    dict(
        type='LoadImageFromFileList',
        io_backend='disk',
        key='lq',
        channel_order='rgb'),
    dict(type='RescaleToZeroOne', keys=['lq']),
    dict(type='FramesToTensor', keys=['lq']),
    dict(type='Collect', keys=['lq'], meta_keys=['lq_path', 'key'])
]

data = dict(
    workers_per_gpu=1,
    train_dataloader=dict(
        samples_per_gpu=1, drop_last=True, persistent_workers=False),  # 8 gpus
    val_dataloader=dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader=dict(
        samples_per_gpu=1, workers_per_gpu=1, persistent_workers=False),

    # train
    train=dict(
        type='RepeatDataset',
        times=1000,
        dataset=dict(
            type=train_dataset_type,
            lq_folder="/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/input",
            gt_folder="/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/gt",
            num_input_frames=10,
            pipeline=train_pipeline,
            scale=1,
            ann_file="/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/test.txt",
            test_mode=False)),
    # val
    val=dict(
        type=val_dataset_type,
        lq_folder='/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/input',
        gt_folder='/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/gt',
        pipeline=test_pipeline,
        scale=1,
        ann_file='/storage/public/home/2022124023/DynStaticNet/data/all-in-one/test/test.txt',
        test_mode=True),
    # test
    test=dict(
        type=val_dataset_type,
        lq_folder='/storage/public/home/2022124023/DynStaticNet/data/all-in-one/val/input',
        gt_folder='/storage/public/home/2022124023/DynStaticNet/data/all-in-one/val/gt',
        pipeline=test_pipeline,
        scale=1,
        test_mode=True),
)

# optimizer
optimizers = dict(
    generator=dict(
        type='Adam',
        lr=1e-4,
        betas=(0.9, 0.99)))

# learning policy
total_iters = 500000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[500000],
    restart_weights=[1],
    min_lr=1e-7)

checkpoint_config = dict(interval=10000, save_optimizer=True, by_epoch=False)
# remove gpu_collect=True in non distributed training
evaluation = dict(interval=500000, save_image=False, gpu_collect=True)
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ])
visual_config = None

# runtime settings
dist_params = dict(backend='nccl')
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = None
resume_from = '/storage/public/home/2022124023/DynStaticNet/experiments/all-in-one_DynStaticNet_psnr/iter_100000.pth'
workflow = [('train', 1)]
find_unused_parameters = True
