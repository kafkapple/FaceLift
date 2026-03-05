"""MMPose config: HRNet-w48 for mouse 22-keypoint pose estimation.

AP-10K pretrained backbone, fine-tuned on DANNCE mouse data.
Compatible with MMPose v1.x (OpenMMLab 2.0).

Usage:
    # Training (2 GPU)
    CUDA_VISIBLE_DEVICES=4,5 python -m torch.distributed.launch \
        --nproc_per_node=2 tools/train.py \
        mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
        --work-dir work_dirs/hrnet_w48_mouse_22kp \
        --launcher pytorch

    # Training (single GPU)
    CUDA_VISIBLE_DEVICES=4 python tools/train.py \
        mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
        --work-dir work_dirs/hrnet_w48_mouse_22kp

    # Evaluation
    CUDA_VISIBLE_DEVICES=4 python tools/test.py \
        mouse_extensions/configs/mmpose/hrnet_w48_mouse_22kp.py \
        work_dirs/hrnet_w48_mouse_22kp/best_coco_AP.pth
"""

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
default_scope = 'mmpose'

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=5,
        save_best='coco/AP',
        rule='greater',
        max_keep_ckpts=3,
    ),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)

custom_hooks = [
    dict(type='EMAHook', momentum=0.0002, priority='ABOVE_NORMAL'),
]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='PoseLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer',
)
log_processor = dict(by_epoch=True, window_size=50, num_digits=6)
log_level = 'INFO'
load_from = None
resume = False

# ---------------------------------------------------------------------------
# Codec (keypoint encoding/decoding)
# ---------------------------------------------------------------------------
codec = dict(
    type='MSRAHeatmap',
    input_size=(256, 192),
    heatmap_size=(64, 48),
    sigma=2,
)

# ---------------------------------------------------------------------------
# Dataset metainfo (mouse 22 keypoints)
# ---------------------------------------------------------------------------
# Flip indices: L_ear<->R_ear, L_paw<->R_paw, etc.
_flip_indices = [
    1, 0, 2, 3, 4, 5, 6, 7,
    12, 13, 14, 15, 8, 9, 10, 11,
    19, 20, 21, 16, 17, 18,
]

dataset_info = dict(
    dataset_name='mouse_22kp',
    paper_info=dict(
        author='FaceLift',
        title='Mouse 22-Keypoint Pose Estimation',
        year='2026',
    ),
    keypoint_info={
        0:  dict(name='L_ear',       id=0,  color=[255, 0, 0],   type='upper', swap='R_ear'),
        1:  dict(name='R_ear',       id=1,  color=[0, 255, 0],   type='upper', swap='L_ear'),
        2:  dict(name='nose',        id=2,  color=[0, 0, 255],   type='upper', swap=''),
        3:  dict(name='neck',        id=3,  color=[255, 255, 0], type='upper', swap=''),
        4:  dict(name='body_middle', id=4,  color=[255, 0, 255], type='upper', swap=''),
        5:  dict(name='tail_root',   id=5,  color=[0, 255, 255], type='lower', swap=''),
        6:  dict(name='tail_middle', id=6,  color=[128, 0, 0],   type='lower', swap=''),
        7:  dict(name='tail_end',    id=7,  color=[0, 128, 0],   type='lower', swap=''),
        8:  dict(name='L_paw',       id=8,  color=[0, 0, 128],   type='upper', swap='R_paw'),
        9:  dict(name='L_paw_end',   id=9,  color=[128, 128, 0], type='upper', swap='R_paw_end'),
        10: dict(name='L_elbow',     id=10, color=[128, 0, 128], type='upper', swap='R_elbow'),
        11: dict(name='L_shoulder',  id=11, color=[0, 128, 128], type='upper', swap='R_shoulder'),
        12: dict(name='R_paw',       id=12, color=[200, 100, 0], type='upper', swap='L_paw'),
        13: dict(name='R_paw_end',   id=13, color=[100, 200, 0], type='upper', swap='L_paw_end'),
        14: dict(name='R_elbow',     id=14, color=[0, 100, 200], type='upper', swap='L_elbow'),
        15: dict(name='R_shoulder',  id=15, color=[200, 0, 100], type='upper', swap='L_shoulder'),
        16: dict(name='L_foot',      id=16, color=[64, 0, 0],    type='lower', swap='R_foot'),
        17: dict(name='L_knee',      id=17, color=[0, 64, 0],    type='lower', swap='R_knee'),
        18: dict(name='L_hip',       id=18, color=[0, 0, 64],    type='lower', swap='R_hip'),
        19: dict(name='R_foot',      id=19, color=[64, 64, 0],   type='lower', swap='L_foot'),
        20: dict(name='R_knee',      id=20, color=[64, 0, 64],   type='lower', swap='L_knee'),
        21: dict(name='R_hip',       id=21, color=[0, 64, 64],   type='lower', swap='L_hip'),
    },
    skeleton_info={
        0:  dict(link=('nose', 'L_ear'), id=0, color=[255, 128, 0]),
        1:  dict(link=('nose', 'R_ear'), id=1, color=[255, 128, 0]),
        2:  dict(link=('nose', 'neck'), id=2, color=[255, 255, 0]),
        3:  dict(link=('neck', 'body_middle'), id=3, color=[200, 200, 0]),
        4:  dict(link=('body_middle', 'tail_root'), id=4, color=[150, 150, 0]),
        5:  dict(link=('tail_root', 'tail_middle'), id=5, color=[100, 100, 0]),
        6:  dict(link=('tail_middle', 'tail_end'), id=6, color=[50, 50, 0]),
        7:  dict(link=('neck', 'L_shoulder'), id=7, color=[0, 0, 255]),
        8:  dict(link=('L_shoulder', 'L_elbow'), id=8, color=[0, 0, 200]),
        9:  dict(link=('L_elbow', 'L_paw'), id=9, color=[0, 0, 150]),
        10: dict(link=('L_paw', 'L_paw_end'), id=10, color=[0, 0, 100]),
        11: dict(link=('neck', 'R_shoulder'), id=11, color=[255, 0, 0]),
        12: dict(link=('R_shoulder', 'R_elbow'), id=12, color=[200, 0, 0]),
        13: dict(link=('R_elbow', 'R_paw'), id=13, color=[150, 0, 0]),
        14: dict(link=('R_paw', 'R_paw_end'), id=14, color=[100, 0, 0]),
        15: dict(link=('tail_root', 'L_hip'), id=15, color=[0, 255, 0]),
        16: dict(link=('L_hip', 'L_knee'), id=16, color=[0, 200, 0]),
        17: dict(link=('L_knee', 'L_foot'), id=17, color=[0, 150, 0]),
        18: dict(link=('tail_root', 'R_hip'), id=18, color=[0, 128, 0]),
        19: dict(link=('R_hip', 'R_knee'), id=19, color=[0, 100, 0]),
        20: dict(link=('R_knee', 'R_foot'), id=20, color=[0, 64, 0]),
    },
    joint_weights=[
        1.0, 1.0, 1.2, 1.2, 1.0,
        1.0, 0.8, 0.5,
        1.0, 0.8, 1.0, 1.2,
        1.0, 0.8, 1.0, 1.2,
        1.0, 1.0, 1.2,
        1.0, 1.0, 1.2,
    ],
    sigmas=[
        0.035, 0.035, 0.026, 0.035, 0.035,
        0.035, 0.050, 0.070,
        0.035, 0.050, 0.035, 0.035,
        0.035, 0.050, 0.035, 0.035,
        0.035, 0.035, 0.035,
        0.035, 0.035, 0.035,
    ],
)

# ---------------------------------------------------------------------------
# Model: HRNet-w48 with AP-10K pretrained backbone
# ---------------------------------------------------------------------------
model = dict(
    type='TopdownPoseEstimator',
    data_preprocessor=dict(
        type='PoseDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
    ),
    backbone=dict(
        type='HRNet',
        in_channels=3,
        extra=dict(
            stage1=dict(
                num_modules=1,
                num_branches=1,
                block='BOTTLENECK',
                num_blocks=(4,),
                num_channels=(64,),
            ),
            stage2=dict(
                num_modules=1,
                num_branches=2,
                block='BASIC',
                num_blocks=(4, 4),
                num_channels=(48, 96),
            ),
            stage3=dict(
                num_modules=4,
                num_branches=3,
                block='BASIC',
                num_blocks=(4, 4, 4),
                num_channels=(48, 96, 192),
            ),
            stage4=dict(
                num_modules=3,
                num_branches=4,
                block='BASIC',
                num_blocks=(4, 4, 4, 4),
                num_channels=(48, 96, 192, 384),
            ),
        ),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='https://download.openmmlab.com/mmpose/animal/hrnet/'
                       'hrnet_w48_ap10k_256x256-d95ab412_20211029.pth',
        ),
    ),
    head=dict(
        type='HeatmapHead',
        in_channels=48,
        out_channels=22,
        deconv_out_channels=None,
        loss=dict(type='KeypointMSELoss', use_target_weight=True),
        decoder=codec,
    ),
    test_cfg=dict(
        flip_test=True,
        flip_mode='heatmap',
        shift_heatmap=True,
    ),
)

# ---------------------------------------------------------------------------
# Data root
# ---------------------------------------------------------------------------
data_root = '/home/joon/data/processed/mmpose_mouse/'

# ---------------------------------------------------------------------------
# Data pipelines
# ---------------------------------------------------------------------------
train_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='RandomFlip', direction='horizontal'),
    dict(
        type='RandomHalfBody',
        min_total_keypoints=6,
        min_upper_keypoints=3,
        padding=1.5,
        prob=0.3,
    ),
    dict(
        type='RandomBBoxTransform',
        rotate_factor=30,
        scale_factor=[0.75, 1.25],
        shift_factor=0.0,
    ),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='GenerateTarget', encoder=codec),
    dict(type='PackPoseInputs'),
]

val_pipeline = [
    dict(type='LoadImage'),
    dict(type='GetBBoxCenterScale'),
    dict(type='TopdownAffine', input_size=codec['input_size']),
    dict(type='PackPoseInputs'),
]

# ---------------------------------------------------------------------------
# Dataloaders
# ---------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=32,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/train.json',
        data_prefix=dict(img='images/'),
        metainfo=dataset_info,
        pipeline=train_pipeline,
    ),
)

val_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/val.json',
        data_prefix=dict(img='images/'),
        metainfo=dataset_info,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

test_dataloader = dict(
    batch_size=64,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        data_mode='topdown',
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/'),
        metainfo=dataset_info,
        pipeline=val_pipeline,
        test_mode=True,
    ),
)

# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/val.json',
    nms_mode='none',
)

test_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/test.json',
    nms_mode='none',
)

# ---------------------------------------------------------------------------
# Training schedule
# ---------------------------------------------------------------------------
train_cfg = dict(max_epochs=100, val_interval=5)
val_cfg = dict()
test_cfg = dict()

# Differential LR: backbone 10x lower than head
optim_wrapper = dict(
    optimizer=dict(type='Adam', lr=5e-4),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        },
    ),
)

param_scheduler = [
    dict(
        type='LinearLR',
        begin=0,
        end=5,
        start_factor=0.01,
        by_epoch=True,
    ),
    dict(
        type='CosineAnnealingLR',
        begin=5,
        end=100,
        eta_min=1e-6,
        by_epoch=True,
    ),
]

# Auto-scale LR based on batch size (base: 32 * 2 GPUs = 64)
auto_scale_lr = dict(base_batch_size=64)
