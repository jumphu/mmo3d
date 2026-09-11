# Copyright (c) OpenMMLab. All rights reserved.

# 1. Base components
dataset_type = 'DSECDataset'
data_root = 'data/DSEC-3DOD'
class_names = ('vehicle', 'pedestrian')

# 2. Model Structure
model = dict(
    type='EventStereoDetector',
    data_preprocessor=dict(
        type='EventStereoDataPreprocessor',
    ),
    backbone=dict(
        type='EventStereoBackbone',
        in_channels=15, 
        out_channels=32,
        num_blocks=3
    ),
    volume_extractor=dict(
        type='GeometricPlaneSweepVolume',
        num_depth_bins=64,
        min_depth=2.0,
        depth_interval=0.5
    ),
    dual_filter=dict(
        type='DualSemanticGeometricFilter',
        in_channels=32,
        num_depth_bins=64
    ),
    voxel_proj=dict(
        type='StereoToVoxelProj',
        voxel_size=[0.2, 0.2, 0.2],
        pc_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
        num_depth_bins=64
    ),
    bbox_head=dict(
        type='mmdet3d.models.dense_heads.CenterHead',
        in_channels=128,
        tasks=[dict(num_class=2, class_names=['vehicle', 'pedestrian'])],
        common_heads=dict(reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=1,
            voxel_size=[0.2, 0.2],
            code_size=9),
        separate_head=dict(type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True,
        train_cfg=dict(
            point_cloud_range=[0, -25.6, -3.0, 51.2, 25.6, 2.0],
            grid_size=[256, 256, 25],
            voxel_size=[0.2, 0.2, 0.2],
            out_size_factor=1,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]),
        test_cfg=dict(
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=1,
            nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            nms_thr=0.2)
    ),
    # Temporarily disabled to allow training launch
    roi_head=None
)

# 3. Data Pipeline
train_pipeline = [
    dict(type='LoadEventStereoVoxelGrid', num_bins=15),
    dict(type='CustomPack3DDetInputs', 
         keys=['event_l', 'event_r', 'cam2ego_l', 'cam2ego_r', 'cam_intrinsic', 'gt_bboxes_3d', 'gt_labels_3d'],
         meta_keys=['sample_idx', 'token', 'timestamp', 'pose', 'box_type_3d', 'box_mode_3d'])
]

test_pipeline = train_pipeline

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dsec_infos_train.pkl',
        pipeline=train_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=False,
    ))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='dsec_infos_val.pkl',
        pipeline=test_pipeline,
        metainfo=dict(classes=class_names),
        test_mode=True,
    ))

test_dataloader = val_dataloader

val_evaluator = dict(
    type='mmdet3d.NuScenesMetric',
    data_root=data_root,
    ann_file=data_root + '/dsec_infos_val.pkl',
    metric='bbox')

test_evaluator = val_evaluator

# 4. Runtime
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=20, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

# Optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.001, weight_decay=0.01),
    clip_grad=dict(max_norm=35, norm_type=2)
)

# Default Runtime
default_scope = 'mmood3d'
custom_imports = dict(imports=['mmood3d.models', 'mmood3d.datasets.transforms'], allow_failed_imports=False)
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', interval=1),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='mmdet3d.Det3DLocalVisualizer', vis_backends=vis_backends, name='visualizer')
