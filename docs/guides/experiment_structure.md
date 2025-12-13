# 실험 폴더 구조 가이드

## 현재 구조

```
FaceLift/
├── checkpoints/
│   ├── mvdiffusion/
│   │   ├── pipeckpts/                    # pretrained
│   │   └── mouse/
│   │       ├── original_facelift_embeds/ # FaceLift embeds로 학습
│   │       ├── mouse_cam/                # Mouse embeds로 학습 (예정)
│   │       └── facelift_cam/             # FaceLift 카메라 데이터 (예정)
│   │
│   └── gslrm/
│       ├── ckpt_*.pt                     # pretrained (인간)
│       ├── mouse_finetune/               # v1
│       ├── mouse_v2/                     # v2 (불안정)
│       └── mouse_v2_stable/              # v2 stable (권장)
│
├── mvdiffusion/data/
│   ├── fixed_prompt_embeds_6view/        # FaceLift (수평)
│   ├── mouse_prompt_embeds_6view/        # Mouse (경사, text_only)
│   ├── mouse_embeds_with_angles/         # Mouse (경사, with_angles)
│   └── mouse_embeds_from_camera/         # Mouse (카메라 JSON 기반)
│
├── configs/
│   ├── mouse_mvdiffusion_*.yaml          # MVDiffusion configs
│   └── mouse_gslrm_*.yaml                # GS-LRM configs
│
└── outputs/
    ├── compare_embeds/                   # embeds 비교 실험
    ├── pipeline_*/                       # full pipeline 결과
    └── test_*/                           # 개별 테스트
```

---

## 체크포인트 명명 규칙

### MVDiffusion
```
checkpoints/mvdiffusion/mouse/{embed_type}/checkpoint-{step}/
                              │
                              ├── original_facelift_embeds  # FaceLift embeds
                              ├── mouse_cam                  # Mouse embeds
                              └── facelift_cam               # FaceLift camera data
```

### GS-LRM
```
checkpoints/gslrm/{experiment_name}/ckpt_{step}.pt
                  │
                  ├── mouse_finetune      # v1 (기본)
                  ├── mouse_v2            # v2 (높은 loss weight)
                  └── mouse_v2_stable     # v2 stable (낮은 LR)
```

---

## Prompt Embeddings 명명 규칙

```
mvdiffusion/data/{camera_type}_prompt_embeds_{num_views}view_{mode}/
                 │                          │            │
                 │                          │            ├── (없음) = text_only
                 │                          │            ├── with_angles
                 │                          │            └── from_camera
                 │                          │
                 │                          └── 6view, 4view, etc.
                 │
                 ├── fixed = FaceLift (수평)
                 └── mouse = Mouse (경사)
```

### 예시
| 폴더명 | 설명 |
|--------|------|
| `fixed_prompt_embeds_6view` | FaceLift, 6뷰, text_only |
| `mouse_prompt_embeds_6view` | Mouse, 6뷰, text_only |
| `mouse_embeds_6view_with_angles` | Mouse, 6뷰, 정량적 각도 포함 |
| `mouse_embeds_6view_from_camera` | Mouse, 6뷰, 카메라 JSON 기반 |

---

## 출력 폴더 명명 규칙

```
outputs/{experiment_type}_{embeds}_{gslrm_version}/
        │                 │        │
        │                 │        └── v1, v2, v2_stable, pretrained
        │                 │
        │                 └── facelift, mouse, mouse_angles
        │
        └── pipeline, compare_embeds, test_mvdiff, test_gslrm
```

### 예시
| 폴더명 | 설명 |
|--------|------|
| `pipeline_mouse_v1` | Full pipeline, Mouse embeds, GS-LRM v1 |
| `pipeline_mouse_v2_stable` | Full pipeline, Mouse embeds, GS-LRM v2 stable |
| `compare_embeds_facelift` | Embeds 비교, FaceLift |
| `test_mvdiff_mouse_angles` | MVDiff 테스트, Mouse with angles |

---

## Config 명명 규칙

```
{model}_{dataset}_{variant}.yaml
```

| 파일명 | 설명 |
|--------|------|
| `mouse_mvdiffusion_mouse_cam.yaml` | MVDiffusion, Mouse embeds |
| `mouse_mvdiffusion_facelift_cam.yaml` | MVDiffusion, FaceLift embeds |
| `mouse_gslrm_v2.yaml` | GS-LRM v2 |
| `mouse_gslrm_v2_stable.yaml` | GS-LRM v2 stable |

---

## Prompt Embeddings 생성 명령어

### 1. Text Only (기본)
```bash
python scripts/generate_prompt_embeds.py \
    --mode text_only \
    --preset mouse \
    --output_dir mvdiffusion/data/mouse_prompt_embeds_6view
```

### 2. With Angles (정량적 각도)
```bash
python scripts/generate_prompt_embeds.py \
    --mode with_angles \
    --preset mouse \
    --output_dir mvdiffusion/data/mouse_embeds_6view_with_angles
```

### 3. From Camera (카메라 JSON 기반)
```bash
python scripts/generate_prompt_embeds.py \
    --mode from_camera \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --output_dir mvdiffusion/data/mouse_embeds_6view_from_camera
```

---

## 실험 비교 체크리스트

새 실험 시작 전 확인:

- [ ] 체크포인트 저장 경로 고유한가?
- [ ] Prompt embeds 경로 맞는가?
- [ ] Config 파일명 구분되는가?
- [ ] WandB exp_name 고유한가?
- [ ] Output dir 고유한가?

---

## 추천 실험 순서

1. **text_only** embeds로 baseline 확립
2. **with_angles** embeds로 개선 확인
3. **from_camera** embeds로 정밀 조정
4. 각 embeds에 대해 GS-LRM v1 vs v2_stable 비교
