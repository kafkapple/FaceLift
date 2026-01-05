---
date: 2026-01-04
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, preprocessing, camera-intrinsics, facelift]
project: Mouse-FaceLift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# Mouse Data Preprocessing: Scaling Strategy Analysis

## Summary

Mouse 데이터 전처리에서 뷰별 객체 크기 불일치 문제 분석 및 FaceLift 파이프라인에 적합한 스케일링 전략 도출.

## Problem Statement

### 현상
`data_mouse_uniform` 전처리 후에도 뷰별 pixel ratio가 **2.4x** 차이:
- 최소: 8.8% (위에서 본 뷰)
- 최대: 25% (측면 뷰)

### 원인
1. **3D 투영의 본질**: 마우스의 비대칭 형태(긴 꼬리) → 시점에 따라 다른 2D 투영 크기
2. **현재 버그**: 뷰마다 다른 `safe_scale` 적용하면서 camera intrinsics 미업데이트

## Option Analysis

### Option A: 뷰별 균일 Pixel Ratio

각 뷰를 독립적으로 스케일링하여 모든 뷰가 동일한 pixel ratio(60%) 달성.

**필요 작업:**
- 뷰마다 다른 scale factor 적용
- Intrinsics 뷰별 업데이트: `fx' = fx * scale_i`

**장점:**
- 모든 뷰에서 일관된 시각적 크기
- 학습 시 attention 균등 분배

**단점:**
- 뷰마다 다른 effective focal length
- Plücker 좌표 불균일 (ray 패턴 불일치)
- Pretrained 모델 분포 벗어남

### Option B: 샘플별 균일 Scale (권장)

샘플 내 모든 뷰에 동일한 scale factor 적용.

**필요 작업:**
- 6개 뷰 중 최소 safe_scale 계산
- 모든 뷰에 동일 scale 적용
- Intrinsics 균일 업데이트

**장점:**
- 3D 기하학적 일관성 유지
- Pretrained 모델 분포와 일치
- Plücker 좌표 균일성 유지
- 구현 단순

**단점:**
- 뷰별 크기 차이 유지 (2.4x)
- 일부 뷰는 작은 객체 (최소 30%+ 보장 필요)

## FaceLift Pipeline Analysis

### 핵심 데이터 흐름

```
Camera Intrinsics (fx, fy, cx, cy)
        ↓
    Pixel Coordinates
        ↓
    Ray Direction = K^(-1) * [u, v, 1]
        ↓
    Plücker Coordinates (d, m = o × d)
        ↓
    Concat with RGB → Transformer
```

### Intrinsics의 역할

- `fx, fy`: Focal length - ray의 수렴/발산 각도 결정
- `cx, cy`: Principal point - ray 방향의 중심점

뷰마다 다른 focal length → 뷰마다 다른 ray 분포 패턴 → 학습 불안정

### Pretrained Model 기대값

FaceLift pretrained model은:
- 모든 뷰에서 유사한 focal length
- Camera distance 2.7로 정규화
- Z-up 좌표계

Option A는 이 분포를 벗어남.

## Recommendation

**Option B (샘플별 균일 스케일) 권장**

### 이유

1. **물리적 정당성**: 뷰별 크기 차이는 3D 투영의 자연스러운 결과
2. **모델 호환성**: Pretrained 모델이 기대하는 입력 분포 유지
3. **Plücker 일관성**: 균일한 intrinsics → 안정적 학습
4. **구현 안정성**: 단순하고 오류 가능성 낮음

### 구현 수정 사항

```python
def process_sample(...):
    # 1. 모든 뷰의 safe_scale 계산
    safe_scales = [calc_safe_scale(bbox, com, ...) for view in views]

    # 2. 최소값을 샘플 전체에 적용
    uniform_scale = min(safe_scales)

    # 3. 모든 뷰에 동일 스케일 적용
    for view in views:
        scaled_image = transform(image, uniform_scale)

        # 4. Intrinsics 업데이트 (핵심!)
        fx_new = fx * uniform_scale
        fy_new = fy * uniform_scale
        cx_new = (cx - w/2) * uniform_scale + w/2
        cy_new = (cy - h/2) * uniform_scale + h/2
```

### 예상 결과

| Metric | 현재 | 수정 후 |
|--------|------|--------|
| 뷰별 크기 차이 | 2.4x | 2.4x (유지) |
| Intrinsics 일관성 | ❌ 불일치 | ✅ 일치 |
| 3D 일관성 | ⚠️ 부분적 | ✅ 완전 |
| Clipping | 0.3% | 0% |

## Additional Findings

### 디스크 공간

| 캐시 | 크기 | 정리 명령어 |
|------|------|-----------|
| wandb | 46GB (삭제됨) | `rm -rf ~/.cache/wandb` |
| huggingface | 24GB | `huggingface-cli cache prune` |
| pip | 6.8GB | `pip cache purge` |
| conda envs | 142GB | `conda env remove -n NAME` |

### 캐시 관리 스크립트

```bash
# 빠른 정리
pip cache purge
conda clean --all -y
rm -rf ~/.cache/wandb
df -h ~
```

## Next Steps

1. `preprocess_uniform_scale.py` 수정: 샘플별 균일 스케일 + intrinsics 업데이트
2. 전처리 재실행
3. 로컬 RTX 3060에서 finetuning 테스트
4. 결과 검증

## References

- `scripts/preprocess_uniform_scale.py`: 현재 전처리 스크립트
- `gslrm/data/mouse_dataset.py`: MouseViewDataset 구현
- `gslrm/model/gslrm.py:1070`: Plücker 좌표 계산
- `docs/reports/260104_mouse_data_clipping_analysis.md`: 클리핑 분석
