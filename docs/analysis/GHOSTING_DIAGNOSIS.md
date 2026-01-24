# GS-LRM Mouse Ghosting 종합 진단 리포트

> Created: 2026-01-24 | Updated: 2026-01-24 | Version: 2.0

## Executive Summary

모든 전처리 방식에서 발생하는 심각한 ghosting 현상의 **근본 원인**은 ray error가 아닌 **object scale 및 centering 불일치**입니다.

| 데이터셋 | PSNR | FG Coverage | Center Offset | Ghosting |
|----------|------|-------------|---------------|----------|
| D3_normalized | **26.5** | **5.28%** | **8px** | 경미 |
| D7_1 | 19.8 | 2.16% | 101px | 심각 |
| D8 | 19.1 | 2.0% | ~100px | 심각 |
| **M3 (예상)** | **25+** | **5%+** | **<15px** | **해결** |

---

## 1. 문제 분석

### 1.1 관찰된 현상
- Turntable 렌더링에서 여러 생쥐가 겹쳐 보임
- 크기가 다른 생쥐들이 같은 위치에 중첩
- Ray error ~0°임에도 ghosting 발생

### 1.2 원인 조사 결과

| 가설 | 검증 결과 |
|------|----------|
| Camera calibration 오류 | ❌ fx=549, dist=2.7 정상 |
| Temporal 비동기화 | ❌ 6대 모두 동기화 확인 |
| Principal point 오류 | ❌ PP=256 정상 |
| **Object scale 불일치** | ✅ **원인 확정** |

---

## 2. 근본 원인: Plucker Ray 분포 불균형

### 2.1 이론적 배경

```
Plucker ray = (direction d, moment m = origin × d)
```

- GS-LRM은 Plucker ray encoding으로 카메라 정보 인코딩
- **작은 객체** → 광선들이 좁은 영역에 집중 → moment 변화 미미
- Transformer가 **뷰 간 차이를 구분하기 어려움**

### 2.2 Objaverse vs Mouse 비교

| 항목 | Objaverse (Pretrained) | Mouse Data |
|------|------------------------|------------|
| FG Coverage | 40-60% | **2-3%** |
| Object Position | 중앙 | 가장자리 |
| Ray Spread | 넓음 | **좁음** |

### 2.3 정량적 비교

| Dataset | FG Coverage | BBox Area | Center Offset |
|---------|-------------|-----------|---------------|
| **D3_normalized** | **5.28%** | ~36K px² | 8px |
| D7_1 | 2.16% | ~14K px² | 101px |
| D8_1 (1.3x zoom) | 3.67% | ~25K px² | 49px |

**핵심 발견**: D3_normalized는 Object-Centered Cropping으로 FG coverage **2.5x 증가** → PSNR 6.7 향상

---

## 3. 해결 방안

### 3.1 M3 전처리 (★ 권장)

```yaml
# D10.3 / M3 프리셋
paradigm: precision_homography    # D7_1 + homography
up_alignment: false               # 미검증, 보류
adaptive_zoom: true
zoom_method: coverage_based
target_fg_coverage: 0.05          # 5%
zoom_after_transform: true        # 변환 후 coverage 계산
```

**예상 결과**:
- 현재: 원본 5% → 변환 후 2.4%
- Zoom: ~1.4x (√(5/2.4) ≈ 1.44)
- 최종: **5% coverage + <15px offset**

### 3.2 실행 명령어

```bash
# M3 전처리
python -m mouse_extensions.preprocessing.preprocess \
    --preset D10.3 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3

# Split 생성
python -m mouse_extensions.preprocessing.split_manager \
    --data-dir /path/to/M3 --preset random
```

---

## 4. 마스크 실험 설정

Ghosting 해결에는 전처리(M3)가 핵심이지만, 마스크 설정도 중요합니다.

### 4.1 권장 설정 (E2_gt_alpha)

```yaml
training:
  losses:
    mask_mode: gt              # GT 마스크 사용 (안정적)
    normalize_by_mask: true    # 작은 전경 보정
    alpha_loss_weight: 0.1     # Shape 수렴 가속
    alpha_loss_type: mse       # BCE보다 안정적
    bg_loss_weight: 0.0
```

### 4.2 실험 우선순위

| Priority | 실험 | 설명 |
|----------|------|------|
| **P0** | M3 + E2_gt_alpha | 전처리 + 권장 마스크 |
| P1 | M3 + E6_lgm_full | Alpha 강화 (1.0) |
| P2 | M3 + E6_combined | 다중 문헌 조합 |

---

## 5. 검증 계획

### Phase 1: M3 전처리 검증
1. M3 데이터셋 생성
2. FG coverage 5%+ 확인
3. Center offset <15px 확인

### Phase 2: 학습 검증
1. M3 + E2_gt_alpha 학습
2. Turntable 렌더링으로 ghosting 확인
3. PSNR 25+ 목표

---

## 관련 문서

- [PREPROCESSING_GUIDE_V2](../preprocessing/PREPROCESSING_GUIDE_V2.md)
- [GHOSTING_SOLUTION_STRATEGY](./GHOSTING_SOLUTION_STRATEGY.md)
- [Mask_Literature_Review](../theory/mask/Mask_Literature_Review.md)
- [Quick Reference](../practical/MOUSE_QUICK_REFERENCE.md)

---

*Report v2.0 | 2026-01-24*
