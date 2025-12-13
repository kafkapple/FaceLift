# Mouse-FaceLift 실험 우선순위 계획

## 현재 상황 요약

| 구성요소 | 현재 상태 | 문제점 |
|----------|----------|--------|
| MVDiffusion | FaceLift prompt_embeds 사용 | 수평 뷰 생성 |
| GS-LRM mouse_finetune | Mouse 카메라 (경사 20°) | 카메라 불일치 |
| GS-LRM pretrained | FaceLift 카메라 | ✅ 잘 작동 |

---

## 실험 우선순위

### Phase 1: Option B - Mouse 카메라 통일 (권장, 빠름)

**목표**: Mouse 카메라 설정에 맞춰 전체 파이프라인 통일

| 단계 | 작업 | 예상 시간 | 상태 |
|------|------|----------|------|
| 1.1 | Mouse 카메라용 prompt_embeds 생성 | 1시간 | ⏳ |
| 1.2 | MVDiffusion 재학습 (새 prompt_embeds) | 6-12시간 | ⏳ |
| 1.3 | 결과 검증 (MVDiff + GS-LRM mouse_finetune) | 30분 | ⏳ |

**장점**:
- 기존 GS-LRM mouse_finetune 활용 가능
- 데이터 전처리 불필요

**필요 작업**:
```bash
# 1.1 Mouse prompt_embeds 생성
python scripts/generate_mouse_prompt_embeds.py \
    --camera_json data_mouse/sample_000000/opencv_cameras.json \
    --output_dir mvdiffusion/data/mouse_prompt_embeds_6view

# 1.2 MVDiffusion 재학습
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_mouse_cam.yaml
```

---

### Phase 2: Option A - FaceLift 카메라 통일 (대안)

**목표**: FaceLift 카메라 설정에 맞춰 전체 파이프라인 통일

| 단계 | 작업 | 예상 시간 | 상태 |
|------|------|----------|------|
| 2.1 | 데이터 전처리 (FaceLift 카메라로 변환) | 2-4시간 | ⏳ |
| 2.2 | MVDiffusion 학습 (새 데이터) | 6-12시간 | ⏳ |
| 2.3 | GS-LRM 학습 (새 데이터) | 12-24시간 | ⏳ |
| 2.4 | 결과 검증 | 30분 | ⏳ |

**장점**:
- FaceLift pretrained 모델과 호환
- 기존 prompt_embeds 사용 가능

**필요 작업**:
```bash
# 2.1 데이터 전처리
python scripts/convert_mouse_to_facelift_camera.py \
    --input_dir data_mouse \
    --output_dir data_mouse/facelift_cam

# 2.2 MVDiffusion 학습
python train_diffusion.py \
    --config configs/mouse_mvdiffusion_facelift_cam.yaml

# 2.3 GS-LRM 학습
python train_gslrm.py \
    --config configs/mouse_gslrm_facelift_cam.yaml
```

---

## 체크포인트 명명 규칙

```
checkpoints/
├── mvdiffusion/mouse/
│   ├── mouse_cam/          # Option B
│   │   └── checkpoint-{step}/
│   └── facelift_cam/       # Option A
│       └── checkpoint-{step}/
│
└── gslrm/mouse/
    ├── mouse_cam/          # 기존 mouse_finetune
    │   └── ckpt_*.pt
    └── facelift_cam/       # Option A 새 학습
        └── ckpt_*.pt
```

---

## 결과 비교 매트릭스

| 실험 | MVDiffusion | prompt_embeds | GS-LRM | 카메라 일치 |
|------|-------------|---------------|--------|------------|
| 현재 (문제) | mouse/checkpoint-2000 | FaceLift | mouse_finetune | ❌ |
| Option B | mouse/mouse_cam | **Mouse** | mouse_finetune | ✅ |
| Option A | mouse/facelift_cam | FaceLift | facelift_cam | ✅ |
| Baseline | pretrained | FaceLift | pretrained | ✅ |

---

## 즉시 실행 가능한 작업

1. **Mouse prompt_embeds 생성 스크립트 작성**
2. **Option B로 MVDiffusion 재학습 시작**
3. **결과 비교 스크립트 구현**
