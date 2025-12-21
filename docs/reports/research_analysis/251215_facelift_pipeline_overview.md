---
date: 2025-12-15
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, pipeline, architecture]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# FaceLift Pipeline 구조 분석

> FaceLift 파이프라인의 전체 구조와 각 Stage 역할 분석

## 1. 전체 파이프라인

```
Single Image → MVDiffusion (6-view 생성) → GS-LRM (3D Gaussian)
```

---

## 2. Stage 1: MVDiffusion (Multi-View Generation)

### 목적
단일 이미지에서 일관된 6개 뷰 이미지 생성

### 아키텍처
- **베이스 모델**: Stable Diffusion 2.1
- **구조**: UNet with cross-attention for multi-view consistency
- **Pretrain**: Objaverse 3D 모델 렌더링 이미지

### 입출력 형식
```python
# Input
cond_image: [B, 3, 320, 320]  # 조건 이미지
camera_embeds: [B, 6, 1280]   # 카메라 위치 임베딩

# Output
generated_views: [B, 6, 320, 320, 3]  # 6개 뷰 이미지
```

### 핵심 요소
1. **Prompt Embedding**: CLIP 임베딩으로 뷰 방향 구분
2. **Cross-View Attention**: 뷰 간 일관성 유지
3. **Reference View**: `reference_view_idx: 0` 고정 권장 (random은 mode collapse 유발)

---

## 3. Stage 2: GS-LRM (3D Reconstruction)

### 목적
6개 뷰 이미지에서 3D Gaussian Splatting 모델 생성

### 아키텍처
- **구조**: Transformer + Gaussian decoder
- **학습 방식**: Objaverse pretrain → Domain finetune

### 입출력 형식
```python
# Input
input_views: [B, N, H, W, 4]   # N개 RGBA 이미지
camera_params: dict             # 카메라 파라미터

# Output
gaussians: {
    'means': [N_gaussians, 3],      # 위치
    'scales': [N_gaussians, 3],     # 크기
    'rotations': [N_gaussians, 4],  # 회전 (quaternion)
    'colors': [N_gaussians, 3],     # 색상
    'opacities': [N_gaussians, 1]   # 투명도
}
```

### 핵심 설정
```yaml
training:
  dataset:
    num_views: 6
    num_input_views: 5    # 추론 시 1, 학습 시 5
  losses:
    lpips_loss_weight: 0.0    # 도메인 변화 시 비활성화 권장
```

---

## 4. FaceLift 원 논문 학습 파이프라인

### MVDiffusion (Stage 1)
| 항목 | 내용 |
|------|------|
| 입력 | 단일 정면 얼굴 이미지 |
| 출력 | 6개 뷰 (α, α±45°, α±90°, α+180°) |
| 학습 데이터 | 합성 인간 얼굴 렌더링 데이터 |

### GS-LRM (Stage 2)
| 항목 | 내용 |
|------|------|
| 학습 시 뷰 수 | 8개 (4개 입력 + 4개 supervision) |
| 전체 뷰 수 | 32개 렌더링 (랜덤 HDR 조명) |
| 학습 전략 | Objaverse pretrain → Synthetic Head finetune |

---

## 5. Mouse 적용 시 고려사항

### 데이터 차이
| 항목 | Human (원본) | Mouse |
|------|-------------|-------|
| 데이터 소스 | 합성 렌더링 | 실제 촬영 |
| 3D GT | 있음 | 없음 |
| 객체 위치 | 원점 고정 | 자유 이동 |
| 카메라 | Turntable 균등 | 불균등 배치 |

### 핵심 도전 과제
1. **도메인 정렬**: MVDiffusion → GS-LRM 간 카메라 일치
2. **전처리 정규화**: 객체 크기/위치 균일화
3. **3D 감독 부재**: 2D-only supervision의 한계

---

*통합 출처*:
- `251215_research_mouse_facelift_overview.md`
- `251215_research_stage1_mvdiffusion.md`
- `251215_research_stage2_3d_reconstruction.md`
- `251217_research_facelift_training_pipeline_and_synthesis_strategy.md`
