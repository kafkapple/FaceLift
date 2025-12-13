# Mouse-FaceLift 실험 설정 가이드

## 실험 명명 규칙

### 체크포인트 경로 구조
```
checkpoints/
├── mvdiffusion/
│   ├── pipeckpts/                    # FaceLift pretrained (human)
│   └── mouse/
│       ├── facelift_cam/             # Option A: FaceLift 카메라 설정
│       │   └── checkpoint-{step}/
│       └── mouse_cam/                # Option B: Mouse 카메라 설정
│           └── checkpoint-{step}/
│
└── gslrm/
    ├── ckpt_*.pt                     # FaceLift pretrained (human)
    └── mouse/
        ├── facelift_cam/             # Option A: FaceLift 카메라 설정
        │   └── ckpt_*.pt
        └── mouse_cam/                # Option B: Mouse 카메라 설정
            └── ckpt_*.pt
```

### 데이터 경로 구조
```
data_mouse/
├── facelift_cam/                     # Option A: FaceLift 카메라로 변환된 데이터
│   ├── sample_000000/
│   │   ├── images/
│   │   └── opencv_cameras.json       # FaceLift 카메라 파라미터
│   └── ...
│
└── mouse_cam/                        # Option B: 원본 Mouse 카메라 데이터
    ├── sample_000000/
    │   ├── images/
    │   └── opencv_cameras.json       # Mouse 카메라 파라미터
    └── ...
```

## 실험 조합

### Option A: FaceLift 카메라 통일
| 구성요소 | 카메라 | 설명 |
|----------|--------|------|
| 데이터 | FaceLift | 수평 6방향 (elevation 0°) |
| MVDiffusion | FaceLift prompt_embeds | 기존 FaceLift 것 사용 |
| GS-LRM | FaceLift 카메라 | 새로 학습 필요 |

### Option B: Mouse 카메라 통일
| 구성요소 | 카메라 | 설명 |
|----------|--------|------|
| 데이터 | Mouse | 경사 6방향 (elevation ~20°) |
| MVDiffusion | Mouse prompt_embeds | 새로 생성 필요 |
| GS-LRM | Mouse 카메라 | 기존 mouse_finetune 사용 가능 |

## Config 파일 명명
```
configs/
├── mouse_mvdiffusion_facelift_cam.yaml   # Option A
├── mouse_mvdiffusion_mouse_cam.yaml      # Option B
├── mouse_gslrm_facelift_cam.yaml         # Option A
└── mouse_gslrm_mouse_cam.yaml            # Option B
```
