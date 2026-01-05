---
date: 2025-12-12
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, mvdiffusion, gslrm, 3d-reconstruction, domain-gap]
project: mouse-facelift
status: in-progress
generator: ai-assisted
generator_tool: claude-code
---

# Mouse-FaceLift 연구노트 (2025-12-12)

> 생쥐 3D 복원 파이프라인 학습/추론 종합 분석

---

## 1. 오늘의 개요

### 목적
1. MVDiffusion/GSLRM 학습 체크포인트 관리 및 이슈 해결
2. GSLRM 추론 품질 문제 분석 및 도메인 갭 이해
3. 실험 우선순위 정립 및 데이터 전처리 파이프라인 문서화

### 주요 성과
- [x] 체크포인트 경로 버그 수정 (`train_diffusion.py`)
- [x] GSLRM index tensor 순서 버그 수정 (`inference_mouse.py`)
- [x] 도메인 갭 원인 분석 및 해결책 정리
- [x] 실험 우선순위 및 파이프라인 문서화
- [x] MVDiffusion 재학습 시작

---

## 2. 발견된 이슈 및 해결

### 2.1 체크포인트 경로 버그 (Critical)

**문제**: `train_diffusion.py`의 `save_model_hook`에서 경로 중복

```python
# Before (버그)
model.save_pretrained(os.path.join(cfg.checkpoint_prefix, output_dir, "unet"))
# 결과: checkpoints/checkpoints/experiments/.../unet

# After (수정)
model.save_pretrained(os.path.join(output_dir, "unet"))
# output_dir은 accelerator가 전달하는 완전한 경로
```

**결과**: 기존 12,000 steps 체크포인트 손실 → 재학습 필요

### 2.2 GSLRM Index Tensor 순서 버그

**문제**: `inference_mouse.py`의 index 순서가 학습과 불일치

```python
# Before (잘못됨) - [scene_idx, view_idx]
index = torch.stack([
    torch.zeros(num_views).long(),   # scene_idx
    torch.arange(num_views).long()   # view_idx
], dim=-1)

# After (수정됨) - [view_idx, scene_idx]
index = torch.stack([
    torch.arange(num_views).long(),  # view_idx (첫 번째!)
    torch.zeros(num_views).long()    # scene_idx (두 번째)
], dim=-1)
```

**모델 사용**: `index[:,:,0]` → view_id, `index[:,:,-1]` → scene_id

### 2.3 시각화 이미지 얼굴 잘림 현상

**원인 분석**:
- Center crop 문제 아님 (원본 데이터 정상)
- 카메라 각도 문제: 뒤/위에서 촬영한 뷰가 랜덤 선택됨
- `num_input_views: 1` 설정으로 단일 뷰만 시각화

**결론**: Segmentation 문제 아님, 카메라 뷰 선택에 따른 정상 현상

---

## 3. 도메인 갭 분석

### 3.1 학습 vs 추론 데이터 특성

| 항목 | 학습 (합성) | 추론 (실제) |
|------|------------|------------|
| 배경 | 완벽히 분리 (alpha) | Segmentation 필요 |
| 노이즈 | 없음 | 센서 노이즈 |
| 조명 | 일관됨 | 다양함 |
| 카메라 포즈 | 정확함 | 추정 오차 |

### 3.2 현재 파이프라인의 도메인 불일치

```
┌─────────────────────────────────────────────────────────────┐
│  MVDiffusion: 실제 생쥐 이미지로 학습                        │
│  GSLRM: 실제 생쥐로 fine-tune (pretrained는 합성 human)     │
└─────────────────────────────────────────────────────────────┘

추론 시:
  Option A (E2E): Real → MVDiffusion → Synthetic 6-view → GSLRM
                  ✅ 합성 스타일로 변환되어 도메인 일치

  Option B (Direct): Real 6-view → GSLRM(finetuned)
                     ⚠️ 실제 이미지 특성으로 품질 저하 가능
```

### 3.3 해결책

1. **End-to-End 파이프라인 (권장)**: MVDiffusion이 도메인 변환 수행
2. **데이터 증강**: 합성 이미지에 노이즈/블러 추가하여 학습
3. **Mixed 데이터 학습**: 합성 70% + 실제 30%
4. **도메인 적응**: CycleGAN으로 실제→합성 변환

---

## 4. 현재 체크포인트 현황

### GSLRM (Fine-tuned)

| Steps | 경로 | 상태 |
|-------|------|:----:|
| 18,000 | `checkpoints/gslrm/mouse_finetune/ckpt_...18000.pt` | ✅ |
| 19,000 | `checkpoints/gslrm/mouse_finetune/ckpt_...19000.pt` | ✅ |
| 20,000 | `checkpoints/gslrm/mouse_finetune/ckpt_...20000.pt` | ✅ 최신 |

### MVDiffusion

| 경로 | 상태 |
|------|:----:|
| `checkpoints/mvdiffusion/pipeckpts/` | ✅ Pretrained (Human) |
| `checkpoints/mvdiffusion/mouse/` | 🔄 재학습 중 |

---

## 5. 실험 우선순위

### Phase 1: MVDiffusion 학습 (현재)

| 우선순위 | 실험 | 상태 |
|:---:|------|:----:|
| **1** | MVDiffusion fine-tune | 🔄 진행중 |
| **2** | 생성 품질 검증 | ⏳ 대기 |

### Phase 2: End-to-End 검증

| 우선순위 | 실험 | 목적 |
|:---:|------|------|
| **3** | Real → MVDiffusion → GSLRM(pretrained) | 도메인 일치 검증 |
| **4** | Real 6-view → GSLRM(finetuned) | Baseline 비교 |

### Phase 3: GSLRM 전략 결정

| 우선순위 | 전략 | 조건 |
|:---:|------|------|
| **5a** | MVDiffusion 출력으로 GSLRM 학습 | (3) > (4) |
| **5b** | 현재 방식 유지 | (4) ≥ (3) |

---

## 6. 데이터 전처리 요약

### 파이프라인 흐름

```
Raw Videos (6-cam) + SimpleClick Masks + MAMMAL Calibration
                              ↓
                   process_mouse_data.py
                              ↓
            1. 카메라 정규화 (거리 2.7, FOV 50°)
            2. 프레임 균일 샘플링 (2000개)
            3. 마스크 적용 → RGBA
            4. Center crop → 512x512
                              ↓
                     FaceLift Format
          (sample_XXXXXX/images/cam_XXX.png + opencv_cameras.json)
```

### 핵심 정규화

```python
# MAMMAL 거리 (246-414 units) → FaceLift 표준 (2.7 units)
scale_factor = 2.7 / avg_distance  # ~0.008
T_normalized = T * scale_factor
```

---

## 7. 3D 시각화 도구

### Gaussian Splat 뷰어

| 도구 | 특징 |
|------|------|
| **Supersplat** | 웹 브라우저, 설치 불필요 |
| **Blender + Plugin** | 전문적, 플러그인 필요 |
| **Viser/Rerun** | Python 연동 가능 |

### Supersplat 사용법
1. https://playcanvas.com/supersplat 접속
2. `gaussians.ply` 드래그앤드롭
3. 마우스로 조작

---

## 8. 다음 단계

### 즉시
- [ ] MVDiffusion 학습 모니터링 (Wandb: mouse_facelift/mvdiffusion)

### 학습 완료 후
- [ ] End-to-End 파이프라인 테스트
- [ ] 도메인 갭 정량 비교 (PSNR, LPIPS)
- [ ] GSLRM 재학습 전략 결정

### 선택적
- [ ] 다른 뷰 구성 실험 (4/8-view)
- [ ] Joint training 검토

---

## 9. 관련 문서

- [데이터 전처리 가이드](../guides/mouse_data_preprocessing.md)
- [사용법 가이드](../guides/mouse_facelift_usage.md)
- [카메라 파라미터 분석](./241208_camera_parameters_analysis.md)

---

## 변경 이력

| 시간 | 변경 내용 |
|------|----------|
| 오전 | 체크포인트 경로 버그 발견 및 수정 |
| 오후 | GSLRM index 버그 수정, 도메인 갭 분석 |
| 저녁 | 실험 우선순위 정립, 문서 통합 |
