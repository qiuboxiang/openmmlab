_base_ = './mask-rcnn_r101_fpn_gn-ws-all_2x_coco.py'
# learning policy
max_epochs = 24
train_cfg = dict(max_epochs=max_epochs)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=500),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[20, 23],
        gamma=0.1)
]
