_base_ = ['tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py']

# model settings
model = dict(cls_head=dict(dropout_ratio=0.5, init_std=0.001, num_classes=13))

# dataset settings
dataset_type = 'VideoDataset'
data_root = ''
data_root_val = ''
ann_file_train = r'D:\openmmlab\mmaction2\data\ShuttleSet\annotation_train.txt'
ann_file_val = r'D:\openmmlab\mmaction2\data\ShuttleSet\annotation_val.txt'

file_client_args = dict(io_backend='disk')

train_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(type='DenseSampleFrames', clip_len=1, frame_interval=1, num_clips=5),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(224, 224), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
val_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=5,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]
test_pipeline = [
    dict(type='DecordInit', **file_client_args),
    dict(
        type='DenseSampleFrames',
        clip_len=1,
        frame_interval=1,
        num_clips=25,
        test_mode=True),
    dict(type='DecordDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=224),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_train,
        data_prefix=dict(video=data_root),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=val_pipeline,
        test_mode=True))
test_dataloader = dict(
    batch_size=16,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file_val,
        data_prefix=dict(video=data_root_val),
        pipeline=test_pipeline,
        test_mode=True))

optim_wrapper = dict(
    optimizer=dict(type='SGD', lr=0.06, momentum=0.9, weight_decay=0.0001))

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=5, val_interval=1)

load_from = r"D:\openmmlab\mmaction2\work_dirs\tsn_imagenet-pretrained-r50_8xb32-dense-1x1x5-100e_kinetics400-rgb\best_acc_top1_epoch_5.pth"

"""vis_backends = [
    dict(type='LocalVisBackend')]  # 可视化后端的列表
visualizer = dict(
    type='ActionVisualizer',  # 可视化器的名称
    vis_backends=vis_backends)"""
