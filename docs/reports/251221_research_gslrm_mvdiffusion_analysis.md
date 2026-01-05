---
date: 2025-12-21
context_name: "2_Research"
tags: [ai-assisted, gslrm, mvdiffusion, camera, fine-tuning, bug-fix]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# 2025-12-21 연구 일지: GS-LRM & MVDiffusion 종합 분석

> **통합 문서**: 이전 4개 문서를 핵심 내용 중심으로 통합
> - `251221_research_gslrm_camera_analysis.md` (삭제됨)
> - `251221_research_mvdiffusion_checkpoint_path_bug.md` (삭제됨)
> - `251221_research_mvdiffusion_limitations_and_alternatives.md` (삭제됨)
> - `251221_til_gslrm_camera_finetuning.md` (삭제됨)

---

## 1. 핵심 발견: GS-LRM 카메라 의존성

### 1.1 문제 현상
모든 fine-tuning 시도에서 예측이 **흰색/희미한 형태**로 변함

### 1.2 근본 원인: Plücker Ray Encoding

GS-LRM은 **카메라 intrinsics에 강하게 의존**:

```python
ray_direction = normalize(K_inv @ pixel_coord)  # K_inv = intrinsics 역행렬
plucker = (ray_direction, cross(ray_origin, ray_direction))
```

**Pretrained 모델 기대값 vs 실제 마우스 데이터**:

| 항목 | Pretrained (FAKE) | 마우스 (REAL) |
|------|-------------------|---------------|
| fx, fy | 548.99 (고정) | 725~820 (가변) |
| cx, cy | 256.0 (중앙) | 267, 245 (비중앙) |
| distance | ~2.7 units | 200-400mm |

**결론**: 같은 픽셀이라도 다른 ray direction → 모델이 완전히 다른 입력으로 인식

### 1.3 `normalize_distance_to`의 한계

- Translation만 스케일링, **Intrinsics는 변경 안 함**
- 근본 문제 해결 불가

---

## 2. Fine-tuning 실험 결과

### 2.1 실험 A: Real Camera 데이터
- **결과**: Step 1부터 완전히 흰색 (즉시 실패)
- **원인**: Plücker ray가 pretrained 분포와 완전히 다름

### 2.2 실험 B: Freeze All Transformer
| Step | PSNR | 결과 |
|------|------|------|
| 1 | ~14.0 | 형태 보임 (pretrained 지식) |
| 301 | ~13.8 | 일부 희미 |
| 501 | ~13.5 | 대부분 흰색 |
| 901 | ~14.0 | 줄무늬 아티팩트 |

**Gradient Explosion 발생**:
```
Step 894: grad_norm = 2808 (threshold 200의 14배!)
```

### 2.3 핵심 교훈

1. **카메라 포맷은 협상 불가**: Pretrained 모델의 정확한 포맷 필수
2. **Fine-tuning은 위험**: 생성 모델은 fine-tuning에 매우 취약
3. **데이터 전처리가 핵심**: 모델 수정보다 데이터를 모델에 맞추기

---

## 3. MVDiffusion 체크포인트 버그 (수정됨)

### 3.1 버그 상세

```bash
# 잘못된 경로 (존재하지 않음)
--mvdiff_checkpoint checkpoints/mvdiffusion/mouse_centered_real/checkpoint-2000

# 올바른 경로
--mvdiff_checkpoint checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000
```

**결과**: 존재하지 않는 경로 → warning만 출력 → base model(human) 사용 → **생쥐 입력에 사람 얼굴 생성**

### 3.2 교훈
- **Fail-fast 원칙**: Warning 대신 Error로 조기 실패
- **경로 검증**: 체크포인트 경로는 실행 전 명시적 확인 필수

---

## 4. MVDiffusion 근본적 한계

### 4.1 아키텍처 문제

MVDiffusion은 **Discrete View Index** 기반 (0~5 = 균등 60° 간격 가정)

**실제 마우스 카메라 vs MVDiffusion 가정**:

| 뷰 쌍 | 실제 각도 | 가정 | 차이 |
|-------|----------|------|------|
| 0→1 | 13.5° | 60° | -46.5° |
| 5→0 | **208.6°** | 60° | +148.6° |

**문제점**:
- 비균등 카메라 배열에서 뷰 인덱스가 실제 기하학을 반영 못함
- Reference view rotation 증강 불가능

### 4.2 권장 전략

**단기**: 실제 6-view 데이터 직접 사용 (합성 데이터 X)
```yaml
training:
  dataset:
    dataset_path: "data_mouse_centered/data_mouse_train.txt"
```

**중장기**: Camera Pose Conditioning (Plücker Ray Embedding) 도입

---

## 5. 대안 방향

| 방법 | 설명 | 권장도 |
|------|------|--------|
| Zero-shot | Pretrained 그대로, 데이터만 맞춤 | ⭐⭐⭐ |
| LoRA | Low-rank adapter 추가 | ⭐⭐ |
| 다른 모델 | Zero123++, SV3D (카메라 의존성 낮음) | ⭐⭐ |

---

## 관련 파일

- `configs/mouse_gslrm_real_camera.yaml`
- `configs/mouse_gslrm_freeze_all.yaml`
- `scripts/generate_synthetic_data.py` (버그 수정됨)
- 올바른 체크포인트: `checkpoints/mvdiffusion/mouse_pixel_based/checkpoint-2000/`

---

*🤖 Generated with Claude Code - 2025-12-21*
*📝 통합 정리: 2026-01-05*
