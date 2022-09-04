# ---------------------------------------------------------------
# Copyright (c) 2022 BIT-DA. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# Adapted from: https://github.com/lhoyer/DAFormer

# model settings
norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoderProjector',
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        type='ResNetV1c',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 2, 4),
        strides=(1, 2, 1, 1),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='DLV2Head',
        in_channels=2048,
        in_index=3,
        dilations=(6, 12, 18, 24),
        num_classes=19,
        align_corners=False,
        init_cfg=dict(
            type='Normal', std=0.01, override=dict(name='aspp_modules')),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='ProjHead',
        in_channels=2048,
        in_index=3,   # int or list, depending on value of input_transform
        input_transform=None,  # optional(None, 'resize_concat', 'multiple_select')
        channels=512,
        num_convs=2,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='ContrastiveLoss', use_dist=False, use_bank=False, use_reg=False,
            use_avg_pool=True, scale_min_ratio=0.75, num_classes=19,
            contrast_temp=100., loss_weight=1.0, reg_relative_weight=0.01)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole'))
