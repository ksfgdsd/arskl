in_channel = 17
num_classes = 26
train_times = 1
batch_size = 16
num_workers = 4
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
             [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
             [1, 3], [2, 4], [11, 12]]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
thw = (48, 64, 64)
# sigma = 0.6 * (thw[1] / 64)
sigma = 0.6
train_pipeline = [
    dict(type='UniformSampleFrames', clip_len=thw[0], num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=(-1, thw[1])),
    dict(type='RandomResizedCrop', area_range=(0.56, 1.0)),
    dict(type='Resize', scale=thw[1:], keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5, left_kp=left_kp, right_kp=right_kp),

    # dict(type='JointOcclusion', p=0.5),
    # dict(type='JointJitter', jitter_amount=0.02, p=0.3),

    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:]),
    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTV'),
    dict(type='Collect', keys=['vectors', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['vectors', 'label']),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=thw[0], num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=thw[1:], keep_ratio=False),

    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:]),
    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTV'),

    dict(type='Collect', keys=['vectors', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['vectors']),
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=thw[0], num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=thw[1:], keep_ratio=False),

    # dict(type='JointOcclusion', p=0.1),

    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], left_kp=left_kp, right_kp=right_kp),
    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTV'),

    dict(type='Collect', keys=['vectors', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['vectors']),
]
dataset = dict(
    type='PoseDataLoaderX',
    train_dataset_cfg=dict(
        dataset=dict(
            type='PoseDataset',
            ann_file='data/ntu60_hrnet.pkl',
            pipeline=train_pipeline,
            split='xsub_train',
        ),
        times=train_times,
    ),
    valid_dataset_cfg=dict(
        ann_file='data/ntu60_hrnet.pkl',
        pipeline=val_pipeline,
        split='xsub_val',
    ),
    test_dataset_cfg=dict(
        ann_file='data/ntu60_hrnet.pkl',
        pipeline=test_pipeline,
        split='xsub_val',
    ),
    train_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=True,
        drop_last=True,
    ),
    valid_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=True,
    ),
    test_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=True,
        persistent_workers=False,
    ),
)
