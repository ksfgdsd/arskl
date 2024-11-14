_base_ = ['./ntu120.py']
data_root = 'data/ntu_inter_120.pkl'
pth_root = 'best_pth/ntu_inter_120'
log_root = 'tf-logs/ntu_inter_120'
test = dict(
    is_test=False,
    ckpt_path='',
)
reload = dict(
    is_reload=False,
    ckpt_path='',
)
precision = '16-mixed'
devices = [0]
num_workers = 32
batch_size = 32
accumulate_grad_batches = 1
train_times = 1
var = dict(
    verbose=False,
    epoch=200,
    optim_type='AdamW',
    lr=(5e-4) / 10,
    weight_decay=3e-4,
    gradient_clip_val=40,
    compile=False,
    label_smoothing=0.1,
    distillation_type='hard',
    enable_checkpointing=True,
    num_classes={{_base_.num_classes}},
    check_val_every_n_epoch=1
)

model = dict(
    type='PoseConv2D',
    num_classes={{_base_.num_classes}},
    thw={{_base_.thw}},
    is_test=test['is_test'],
)
scheduler = dict(
    type='LinearCosineScheduler',
    initial_lr=var['lr'],
    max_lr=var['lr'] * 10,
    final_lr=0,
    warmup_epochs=5,
    total_epochs=var['epoch'],
)

dataset = dict(
    train_dataset_cfg=dict(
        dataset=dict(
            ann_file=data_root,
        ),
        times=train_times,
    ),
    valid_dataset_cfg=dict(
        ann_file=data_root,
    ),
    test_dataset_cfg=dict(
        ann_file=data_root,
    ),
    train_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
    valid_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=num_workers,
    ),
    test_dataloader_cfg=dict(
        batch_size=batch_size,
        num_workers=32
    ),
)
optim = dict(
    type=var['optim_type'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
    momentum=0.9
) if var['optim_type'] == 'SGD' else dict(
    type=var['optim_type'],
    lr=var['lr'],
    weight_decay=var['weight_decay'],
)
learner = dict(
    type='Learner_Skl',
    model_cfg=model,
    optim_cfg=optim,
    scheduler_cfg=scheduler,
    hyper_cfg=var,
)
ckpt = dict(
    dirpath=pth_root,
    filename='{epoch}-{vacc:.4f}',
    save_last=False,
    monitor='vacc',
    save_top_k=1,
    mode='max',
    every_n_epochs=var['check_val_every_n_epoch'],
    verbose=True,
    save_weights_only=True,
)
trainer = dict(
    type='Trainer',
    ckpt_cfg=ckpt,
    max_epochs=var['epoch'],
    precision=precision,
    devices=devices,
    default_root_dir=log_root,
    accumulate_grad_batches=accumulate_grad_batches,
    enable_checkpointing=var['enable_checkpointing'],
    # strategy='ddp_find_unused_parameters_true',
    strategy='ddp',
    gradient_clip_val=var['gradient_clip_val'],
    benchmark=True,
    sync_batchnorm=True,
    deterministic=True,
    check_val_every_n_epoch=var['check_val_every_n_epoch'],
)
