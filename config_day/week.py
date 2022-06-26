dataset_type = 'CocoDataset'
data_root = 'data/'
img_norm_cfg = dict(
    mean=[105.5, 105.5, 105.5], std=[53.9, 53.9, 53.9], to_rgb=True)
albu_train_transforms = [
    dict(
        type='RandomBrightnessContrast',
        brightness_limit=[0.1, 0.3],
        contrast_limit=[0.1, 0.3],
        p=0.2),
    dict(type='JpegCompression', quality_lower=75, quality_upper=95, p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(type='Blur', blur_limit=3, p=1.0),
            dict(type='MedianBlur', blur_limit=3, p=1.0)
        ],
        p=0.2),
    dict(type='GaussNoise', p=0.2),
    dict(
        type='OneOf',
        transforms=[
            dict(
                type='CLAHE',
                clip_limit=4.0,
                tile_grid_size=(8, 8),
                always_apply=False,
                p=1.0),
            dict(type='FancyPCA', alpha=0.1, always_apply=False, p=1.0),
            dict(
                type='Sharpen',
                alpha=(0.2, 0.5),
                lightness=(0.5, 1.0),
                always_apply=False,
                p=1.0)
        ],
        p=0.2)
]
file_client_args = dict(backend='disk')
image_size = (640, 640)
load_pipeline = [
    dict(type='LoadImageFromFile', file_client_args=dict(backend='disk')),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(
        type='Resize',
        img_scale=(640, 640),
        ratio_range=(0.5, 2.0),
        multiscale_mode='range',
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(640, 640),
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(
        type='Albu',
        transforms=[
            dict(
                type='RandomBrightnessContrast',
                brightness_limit=[0.1, 0.3],
                contrast_limit=[0.1, 0.3],
                p=0.2),
            dict(
                type='JpegCompression',
                quality_lower=75,
                quality_upper=95,
                p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(type='Blur', blur_limit=3, p=1.0),
                    dict(type='MedianBlur', blur_limit=3, p=1.0)
                ],
                p=0.2),
            dict(type='GaussNoise', p=0.2),
            dict(
                type='OneOf',
                transforms=[
                    dict(
                        type='CLAHE',
                        clip_limit=4.0,
                        tile_grid_size=(8, 8),
                        always_apply=False,
                        p=1.0),
                    dict(
                        type='FancyPCA', alpha=0.1, always_apply=False, p=1.0),
                    dict(
                        type='Sharpen',
                        alpha=(0.2, 0.5),
                        lightness=(0.5, 1.0),
                        always_apply=False,
                        p=1.0)
                ],
                p=0.2)
        ],
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_labels'],
            min_visibility=0.0,
            filter_lost_elements=True),
        keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
        update_pad_shape=False,
        skip_img_without_anno=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=640)
]
train_pipeline = [
    dict(type='CopyPaste', max_num_pasted=10),
    dict(
        type='MixUp',
        img_scale=(640, 640),
        ratio_range=(0.5, 2.0),
        pad_val=114.0),
    dict(
        type='Normalize',
        mean=[105.5, 105.5, 105.5],
        std=[53.9, 53.9, 53.9],
        to_rgb=True),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[105.5, 105.5, 105.5],
                std=[53.9, 53.9, 53.9],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ('human', 'bicycle', 'motorcycle', 'vehicle')
train_dataset = dict(
    type='MultiImageMixDataset',
    dataset=dict(
        type='CocoDataset',
        classes=('human', 'bicycle', 'motorcycle', 'vehicle'),
        ann_file=
        data_root+'train_week_small.json',
        img_prefix=
        data_root+'images/',
        pipeline=[
            dict(
                type='LoadImageFromFile',
                file_client_args=dict(backend='disk')),
            dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
            dict(
                type='Resize',
                img_scale=(640, 640),
                ratio_range=(0.5, 2.0),
                multiscale_mode='range',
                keep_ratio=True),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(640, 640),
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(
                type='Albu',
                transforms=[
                    dict(
                        type='RandomBrightnessContrast',
                        brightness_limit=[0.1, 0.3],
                        contrast_limit=[0.1, 0.3],
                        p=0.2),
                    dict(
                        type='JpegCompression',
                        quality_lower=75,
                        quality_upper=95,
                        p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(type='Blur', blur_limit=3, p=1.0),
                            dict(type='MedianBlur', blur_limit=3, p=1.0)
                        ],
                        p=0.2),
                    dict(type='GaussNoise', p=0.2),
                    dict(
                        type='OneOf',
                        transforms=[
                            dict(
                                type='CLAHE',
                                clip_limit=4.0,
                                tile_grid_size=(8, 8),
                                always_apply=False,
                                p=1.0),
                            dict(
                                type='FancyPCA',
                                alpha=0.1,
                                always_apply=False,
                                p=1.0),
                            dict(
                                type='Sharpen',
                                alpha=(0.2, 0.5),
                                lightness=(0.5, 1.0),
                                always_apply=False,
                                p=1.0)
                        ],
                        p=0.2)
                ],
                bbox_params=dict(
                    type='BboxParams',
                    format='pascal_voc',
                    label_fields=['gt_labels'],
                    min_visibility=0.0,
                    filter_lost_elements=True),
                keymap=dict(img='image', gt_masks='masks', gt_bboxes='bboxes'),
                update_pad_shape=False,
                skip_img_without_anno=True),
            dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
            dict(type='RandomFlip', flip_ratio=0.5),
            dict(type='Pad', size_divisor=640)
        ],
        filter_empty_gt=False),
    pipeline=[
        dict(type='CopyPaste', max_num_pasted=10),
        dict(
            type='MixUp',
            img_scale=(640, 640),
            ratio_range=(0.5, 2.0),
            pad_val=114.0),
        dict(
            type='Normalize',
            mean=[105.5, 105.5, 105.5],
            std=[53.9, 53.9, 53.9],
            to_rgb=True),
        dict(type='DefaultFormatBundle'),
        dict(
            type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
    ])
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset',
            classes=('human', 'bicycle', 'motorcycle', 'vehicle'),
            ann_file=
            data_root+'train.json',
            img_prefix=
            data_root+'images/',
            pipeline=[
                dict(
                    type='LoadImageFromFile',
                    file_client_args=dict(backend='disk')),
                dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
                dict(
                    type='Resize',
                    img_scale=(640, 640),
                    ratio_range=(0.5, 2.0),
                    multiscale_mode='range',
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(640, 640),
                    recompute_bbox=True,
                    allow_negative_crop=True),
                dict(
                    type='Albu',
                    transforms=[
                        dict(
                            type='RandomBrightnessContrast',
                            brightness_limit=[0.1, 0.3],
                            contrast_limit=[0.1, 0.3],
                            p=0.2),
                        dict(
                            type='JpegCompression',
                            quality_lower=75,
                            quality_upper=95,
                            p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(type='Blur', blur_limit=3, p=1.0),
                                dict(type='MedianBlur', blur_limit=3, p=1.0)
                            ],
                            p=0.2),
                        dict(type='GaussNoise', p=0.2),
                        dict(
                            type='OneOf',
                            transforms=[
                                dict(
                                    type='CLAHE',
                                    clip_limit=4.0,
                                    tile_grid_size=(8, 8),
                                    always_apply=False,
                                    p=1.0),
                                dict(
                                    type='FancyPCA',
                                    alpha=0.1,
                                    always_apply=False,
                                    p=1.0),
                                dict(
                                    type='Sharpen',
                                    alpha=(0.2, 0.5),
                                    lightness=(0.5, 1.0),
                                    always_apply=False,
                                    p=1.0)
                            ],
                            p=0.2)
                    ],
                    bbox_params=dict(
                        type='BboxParams',
                        format='pascal_voc',
                        label_fields=['gt_labels'],
                        min_visibility=0.0,
                        filter_lost_elements=True),
                    keymap=dict(
                        img='image', gt_masks='masks', gt_bboxes='bboxes'),
                    update_pad_shape=False,
                    skip_img_without_anno=True),
                dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(type='Pad', size_divisor=640)
            ],
            filter_empty_gt=False),
        pipeline=[
            dict(type='CopyPaste', max_num_pasted=10),
            dict(
                type='MixUp',
                img_scale=(640, 640),
                ratio_range=(0.5, 2.0),
                pad_val=114.0),
            dict(
                type='Normalize',
                mean=[105.5, 105.5, 105.5],
                std=[53.9, 53.9, 53.9],
                to_rgb=True),
            dict(type='DefaultFormatBundle'),
            dict(
                type='Collect',
                keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks'])
        ]),
    val=dict(
        type='CocoDataset',
        classes=('human', 'bicycle', 'motorcycle', 'vehicle'),
        ann_file=
        data_root + 'train.json',
        img_prefix=
        data_root + 'images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(640, 640),
                flip=False,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[105.5, 105.5, 105.5],
                        std=[53.9, 53.9, 53.9],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CocoDataset',
        classes=('human', 'bicycle', 'motorcycle', 'vehicle'),
        ann_file=
        data_root+'test_submit.json',
        img_prefix=data_root+'test_images/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(960, 960),
                flip=True,
                transforms=[
                    dict(type='Resize', keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[105.5, 105.5, 105.5],
                        std=[53.9, 53.9, 53.9],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='ImageToTensor', keys=['img']),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=12, metric='bbox')
optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
checkpoint_config = dict(interval=12)
log_config = dict(
    interval=1,
    hooks=[dict(type='TextLoggerHook'),
           dict(type='TensorboardLoggerHook')])
custom_hooks = [dict(type='NumClassCheckHook')]
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'weights/cascade_mask_rcnn_cbv2_swin_small_patch4_window7_mstrain_400-1400_adamw_3x_coco.pth'
resume_from = None
workflow = [('train', 1)]
opencv_num_threads = 0
mp_start_method = 'fork'
model = dict(
    type='CascadeRCNN',
    pretrained=None,
    backbone=dict(
        type='CBSwinTransformer',
        embed_dim=96,
        depths=[2, 2, 18, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        ape=False,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='CBFPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0)),
            dict(
                type='ConvFCBBoxHead',
                num_shared_convs=4,
                num_shared_fcs=1,
                in_channels=256,
                conv_out_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=4,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0.0, 0.0, 0.0, 0.0],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                norm_cfg=dict(type='SyncBN', requires_grad=True),
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='GIoULoss', loss_weight=10.0))
        ]),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_across_levels=False,
            nms_pre=2000,
            nms_post=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                mask_size=28,
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_across_levels=False,
            nms_pre=1000,
            nms_post=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='soft_nms', iou_threshold=0.5),
            max_per_img=100)))
work_dir = './work_dirs/week'
auto_resume = False
gpu_ids = range(0, 8)
