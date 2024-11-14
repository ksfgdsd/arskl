in_channel = 17
num_classes = 99
train_times = 1
batch_size = 16
num_workers = 4
left_kp = [1, 3, 5, 7, 9, 11, 13, 15]
right_kp = [2, 4, 6, 8, 10, 12, 14, 16]
# skeletons = [[0, 5], [0, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11],
#              [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [0, 1], [0, 2],
#              [1, 3], [2, 4], [11, 12]]
# skeletons=[(5, 7, 9), (6, 8, 10), (11, 13, 15), (12, 14, 16), (0, 1, 3), (0, 2, 4), (0, 1, 2), (5, 7, 11), (5, 11, 13), (6, 8, 12), (6, 12, 14), (11, 12, 13), (11, 12, 14), (0, 1, 5), (0, 1, 6), (0, 2, 5), (0, 2, 6)]
# random
# skeletons=[(0, 1, 6, 12), (0, 6, 8, 10), (5, 11, 12, 13), (5, 7, 11, 13), (5, 7, 11, 12), (0, 1, 2, 5), (5, 11, 13, 15), (0, 5, 6, 12), (0, 1, 5, 11), (0, 5, 6, 7), (0, 1, 5, 6), (0, 2, 4, 5), (0, 1, 3, 6), (0, 2, 5, 6), (11, 12, 13, 14), (0, 5, 6, 8), (0, 2, 6, 8)]
# larger
# skeletons=[(0, 5, 6, 11), (0, 5, 6, 12), (0, 5, 11, 12), (0, 6, 11, 12), (0, 1, 5, 6), (0, 1, 5, 11), (0, 1, 6, 12), (0, 2, 5, 6), (0, 2, 5, 11), (0, 2, 6, 12), (0, 5, 6, 7), (0, 5, 6, 8), (0, 5, 7, 11), (0, 5, 11, 13), (0, 6, 8, 12), (0, 6, 12, 14), (5, 6, 11, 12)]
# random
# skeletons=[(0, 1, 2, 3, 5), (0, 1, 3, 5, 7), (0, 1, 2, 6, 12), (0, 2, 6, 11, 12), (0, 5, 7, 9, 11), (11, 12, 13, 14, 15), (0, 5, 11, 12, 14), (0, 1, 2, 6, 8), (0, 2, 5, 6, 7), (5, 11, 12, 14, 16), (0, 5, 6, 7, 11), (6, 8, 10, 12, 14), (5, 6, 8, 11, 12), (0, 5, 6, 12, 14), (0, 2, 5, 7, 11), (0, 6, 11, 12, 13), (5, 6, 11, 12, 14)]
# larger
skeletons=[(0, 5, 6, 11, 12), (0, 1, 5, 6, 11), (0, 1, 5, 6, 12), (0, 1, 5, 11, 12), (0, 1, 6, 11, 12), (0, 2, 5, 6, 11), (0, 2, 5, 6, 12), (0, 2, 5, 11, 12), (0, 2, 6, 11, 12), (0, 5, 6, 7, 11), (0, 5, 6, 7, 12), (0, 5, 6, 8, 11), (0, 5, 6, 8, 12), (0, 5, 6, 11, 13), (0, 5, 6, 12, 14), (0, 5, 7, 11, 12), (0, 5, 11, 12, 13)]
# skeletons=[(0, 1, 2, 3), (0, 1, 2, 4), (5, 7, 9, 11), (5, 11, 13, 15), (6, 8, 10, 12), (6, 12, 14, 16), (11, 12, 13, 15), (11, 12, 14, 16), (0, 1, 3, 5), (0, 1, 3, 6), (0, 2, 4, 5), (0, 2, 4, 6), (0, 5, 7, 9), (0, 6, 8, 10), (5, 7, 11, 13), (6, 8, 12, 14), (11, 12, 13, 14)]
# skeletons=[(0, 1, 2, 3, 4), (5, 7, 9, 11, 13), (5, 7, 11, 13, 15), (6, 8, 10, 12, 14), (6, 8, 12, 14, 16), (11, 12, 13, 14, 15), (11, 12, 13, 14, 16), (0, 1, 2, 3, 5), (0, 1, 2, 3, 6), (0, 1, 2, 4, 5), (0, 1, 2, 4, 6), (0, 1, 3, 5, 7), (0, 1, 3, 6, 8), (0, 1, 5, 7, 9), (0, 1, 6, 8, 10), (0, 2, 4, 5, 7), (0, 2, 4, 6, 8)]
# skeletons=[(5, 7, 9, 11, 13, 15), (6, 8, 10, 12, 14, 16), (11, 12, 13, 14, 15, 16), (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 6), (0, 1, 3, 5, 7, 9), (0, 1, 3, 6, 8, 10), (0, 2, 4, 5, 7, 9), (0, 2, 4, 6, 8, 10), (0, 1, 2, 3, 5, 7), (0, 1, 2, 3, 6, 8), (0, 1, 2, 4, 5, 7), (0, 1, 2, 4, 6, 8), (0, 1, 2, 5, 7, 9), (0, 1, 2, 6, 8, 10), (5, 7, 9, 11, 12, 13), (5, 7, 9, 11, 12, 14)]
left_limb = [0, 2, 3, 6, 7, 8, 12, 14]
right_limb = [1, 4, 5, 9, 10, 11, 13, 15]
g = 4
thw = (12 * g, 16 * g, 16 * g)
sigma = 0.6 * (thw[1] / 64)
# sigma = 0.6
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

    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:]),
    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    # dict(type='GeneratePoseVector', sigma=sigma,is_compression=True, hw=thw[1:], with_kp=False, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTV'),
    dict(type='Collect', keys=['vectors', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['vectors', 'label']),
]
val_pipeline = [
    dict(type='UniformSampleFrames', clip_len=thw[0], num_clips=1),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=thw[1:], keep_ratio=False),

    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:]),
    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    # dict(type='GeneratePoseVector', sigma=sigma, is_compression=True, hw=thw[1:], with_kp=False, skeletons=skeletons),
    dict(type='FormatShape', input_format='NCTV'),

    dict(type='Collect', keys=['vectors', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['vectors']),
]

test_pipeline = [
    dict(type='UniformSampleFrames', clip_len=thw[0], num_clips=10),
    dict(type='PoseDecode'),
    dict(type='PoseCompact', hw_ratio=1., allow_imgpad=True),
    dict(type='Resize', scale=thw[1:], keep_ratio=False),

    # dict(type='JointOcclusion', p=0.1),
    # dict(type='JointJitter', jitter_amount=0.02, p=0.5),

    # dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], left_kp=left_kp, right_kp=right_kp),
    dict(type='GeneratePoseVector', sigma=sigma, hw=thw[1:], with_kp=False, skeletons=skeletons),
    # dict(type='GeneratePoseVector', sigma=sigma, is_compression=True, hw=thw[1:], with_kp=False, skeletons=skeletons),
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
            split='train',
        ),
        times=train_times,
    ),
    valid_dataset_cfg=dict(
        ann_file='',
        pipeline=val_pipeline,
        split='val',
    ),
    test_dataset_cfg=dict(
        ann_file='',
        pipeline=test_pipeline,
        split='val',
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