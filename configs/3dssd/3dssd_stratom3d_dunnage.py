_base_ = [
    "../_base_/models/3dssd.py",
    "../_base_/datasets/stratom3d_dunnage.py",
    "../_base_/default_runtime.py",
]

# model settings
model = dict(
    bbox_head=dict(
        num_classes=1,
        bbox_coder=dict(type="AnchorFreeBBoxCoder", num_dir_bins=12, with_rot=True),
    )
)

# optimizer
lr = 0.002  # max learning rate
optim_wrapper = dict(
    type="OptimWrapper",
    optimizer=dict(type="AdamW", lr=lr, weight_decay=0.0),
    clip_grad=dict(max_norm=35, norm_type=2),
)

# training schedule for 1x
train_cfg = dict(type="EpochBasedTrainLoop", max_epochs=80, val_interval=2)
val_cfg = dict(type="ValLoop")
test_cfg = dict(type="TestLoop")

# learning rate
param_scheduler = [
    dict(
        type="MultiStepLR",
        begin=0,
        end=80,
        by_epoch=True,
        milestones=[45, 60],
        gamma=0.1,
    )
]
