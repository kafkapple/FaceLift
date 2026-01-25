# PP와 fx 정규화: MVG 이론 및 실험 검증 종합 분석

> **작성일**: 2026-01-25
> **버전**: v2.0 (팩트 체크 완료)
> **관련 문서**: [[HYPOTHESIS_VERIFICATION_260125]], [[M3_SERIES_SPEC]]

---

## 1. 핵심 결론 (TL;DR)

| 요소 | 결론 | 근거 |
|------|------|------|
| **fx** | 549 필수 | Pretrained 호환성 |
| **PP (cx, cy)** | 256 필수 | Pretrained + MVG 일관성 |
| **PP 달성 방법** | Center-Aligned Zoom | 이미지-PP 일관성 유지 |
| **Coverage** | 높을수록 좋음 | D3(74%)→27.0, D7_1(50%)→20.9 |

**⚠️ 중요**: PP를 "강제"하는 것이 아니라, **이미지 변환과 PP가 일관성 있게 256이 되도록** 해야 함.

---

## 2. 실험 데이터 요약

### 2.1 데이터셋별 성능

| Dataset | Coverage | fx | PP | Val PSNR | 상태 |
|---------|----------|-----|-----|----------|------|
| **D3_normalized** | 74% | 549 | 256 (일관) | **27.09** | ✅ 최고 |
| D7_1 | 50% | 549 | 256 | 20.93 | ✅ 안정 |
| D8 | 50% | 549 | 256 | 20.21 | ✅ 안정 |
| M3 (원본) | 80% | **739** ❌ | 가변 | ~17 | ❌ fx 버그 |
| M3_norm | 80% | 549 | **가변** ❌ | ~17 | ❌ PP 불일치 |
| M3_persample | 80% | 549 | **가변** ❌ | ~17 | ❌ PP 불일치 |

### 2.2 핵심 발견

```
Coverage ↑ + fx=549 + PP=256 → PSNR ↑

D3_normalized이 최고인 이유:
1. PP=256 일관성 유지 ✅
2. fx=549 pretrained 호환 ✅
3. 74% Coverage (높음) ✅
```

---

## 3. MVG 이론: Zoom과 Intrinsics

### 3.1 Crop + Resize 시 Intrinsics 변화

원본 이미지: $512 \times 512$, $f_x = 549$

**Zoom = 1.35x 적용:**
1. Crop: $512 / 1.35 = 379 \times 379$
2. Resize: $379 \rightarrow 512$

**새로운 focal length:**
$$f_x^{new} = f_x \times zoom = 549 \times 1.35 = 741$$

**새로운 principal point:**
- **Center-aligned crop**: $c_x^{new} = 256$ (자동 유지)
- **Object-centered crop**: $c_x^{new} = (c_x - crop\_offset) \times zoom \neq 256$

### 3.2 PP 오류의 영향

$$\text{Ray Error} = \arctan\left(\frac{\Delta pp}{f_x}\right)$$

| PP 오류 (px) | fx | Ray Error |
|--------------|-----|-----------|
| 10 | 549 | ~1.0° |
| 50 | 549 | ~5.2° |
| 77 | 549 | ~8.0° |
| 87 | 549 | ~9.0° |

**M3_norm/M3_persample PP 분포:**
```
cx: mean=238, std=19, min=179, max=256
cy: mean=237, std=22, min=169, max=256
최대 PP 오류: 87px → Ray Error ~9° → Ghosting
```

---

## 4. Zoom 정규화 방식 비교

### 4.1 잘못된 방식: Object-Centered + PP 강제

```python
# M3_persample 방식 (문제 있음)
crop_center = mask_centroid  # 객체 중심 (예: 300, 280)
cropped = image[cy-h:cy+h, cx-w:cx+w]  # 객체 중심 기준 crop
zoomed = cv2.resize(cropped, (512, 512))

# PP 계산
cx_new = (256 - crop_offset_x) * zoom  # = 179 (정확한 값)
# 하지만 force_pp_to_target=True로 256 강제 → 불일치!
```

**문제**: 이미지에서 객체는 중앙이 아닌데, PP=256으로 거짓 정보 제공

### 4.2 올바른 방식: Center-Aligned Zoom

```python
# M3_1, M3_2 방식 (올바름)
def apply_zoom_center_aligned(image, zoom):
    size = 512
    crop_size = int(size / zoom)  # = 379
    
    # 핵심: 이미지 중심 기준 crop
    crop_x = (size - crop_size) // 2  # = 66
    crop_y = (size - crop_size) // 2  # = 66
    
    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    zoomed = cv2.resize(cropped, (size, size))
    
    # PP 자동 계산
    # cx_new = (256 - 66) * 1.35 = 256 ✅
    return zoomed
```

**장점**: 이미지 변환과 PP가 자연스럽게 일관성 유지

### 4.3 방식 비교 요약

| 방식 | Crop 기준 | PP 결과 | MVG 일관성 | 권장 |
|------|-----------|---------|------------|------|
| Object-centered + PP 강제 | 객체 중심 | 256 (거짓) | ❌ | ❌ |
| Object-centered + PP 정확 | 객체 중심 | 가변 | ⚠️ 정확하나 | ❌ |
| **Center-aligned** | 이미지 중심 | 256 (진실) | ✅ | ✅ |

---

## 5. fx 정규화 필요성

### 5.1 Pretrained 모델 분포

GS-LRM pretrained (Objaverse):
- $f_x \approx 549$
- $\text{translation\_norm} \approx 2.7$
- $f_x / \text{trans} \approx 203$

### 5.2 fx 불일치 시 문제

```
M3 원본: fx = 549 × 1.35 = 739
→ Pretrained 분포와 불일치
→ 학습 불안정, Val PSNR 저하
```

### 5.3 정규화 흐름

**이중 정규화 (비효율적):**
```
1. fx = 549 (첫 정규화)
2. zoom → fx = 741 (정규화 깨짐)
3. fx = 549 (재정규화)
```

**권장: 단일 정규화**
```
1. 이미지 전처리 (crop, zoom)
2. 마지막에 fx = 549로 정규화
```

---

## 6. D3_normalized 성공 요인 분석

### 6.1 D3 vs 다른 데이터셋

| 요소 | D3_normalized | D7_1, D8 | M3_norm |
|------|---------------|----------|---------|
| fx | 549 | 549 | 549 |
| fy | 552.5 | 549 | 549 |
| fx/fy | 0.9937 | 1.0 | 1.0 |
| PP | 256 (일관) | 256 | **가변** ❌ |
| Coverage | 74% | 50% | 80% |
| Val PSNR | 27.09 | ~20 | ~17 |

### 6.2 성공 이유

1. **PP=256 일관성**: Zoom 없이 자연스럽게 중앙
2. **fx=549**: Pretrained 호환
3. **높은 Coverage (74%)**: 풍부한 정보

**⚠️ 주의**: fx/fy=0.9937 비정방형은 성공 요인이 **아님**. PP 일관성이 핵심.

---

## 7. 권장 설정

### 7.1 데이터셋 (M-Series v2)

| ID | 방식 | fx | PP | Coverage | 상태 |
|----|------|-----|-----|----------|------|
| **M3_1** | Global Zoom + Center-Aligned | 549 | 256 | 80% | ⏳ 전처리 중 |
| **M3_2** | Per-sample Zoom + Center-Aligned | 549 | 256 | 80%+ | ⏳ 전처리 중 |

### 7.2 프리셋 설정

```python
# presets.py - M3_1, M3_2
"M3_1": {
    "paradigm": "precision_homography",
    "adaptive_zoom": False,
    "zoom_factor": 1.35,
    "zoom_method": "center_aligned",  # ★ 핵심
    "normalize_after_zoom": True,     # fx=549 복원
    "force_pp_to_target": False,      # 불필요 (자동 256)
}

"M3_2": {
    "paradigm": "precision_homography",
    "adaptive_zoom": True,
    "target_fg_coverage": 0.05,
    "zoom_method": "center_aligned",  # ★ 핵심
    "normalize_after_zoom": True,
    "force_pp_to_target": False,
}
```

### 7.3 실험 설정

| 우선순위 | 명령어 | 목적 |
|----------|--------|------|
| **P0** | `-d M3_2 -e E0_1_facelift` | D3 재현 (PP=256, Coverage 80%+) |
| P1 | `-d M3_1 -e E0_1_facelift` | Global zoom 기준선 |
| P2 | `-d M1 -e E1_2_alpha` | 안정적 기준선 |

---

## 8. 폐기된 가설

### ~~H: PP 가변이 기하학적으로 더 정확~~

**폐기 이유:**
- MVG 이론상 PP 가변은 정확하나
- GS-LRM pretrained가 PP=256 기대
- 실험 결과: PP 가변 → Val PSNR ~17 (실패)

### ~~H: force_pp_to_target=False 권장~~

**폐기 이유:**
- Object-centered crop + PP 가변 = 여전히 문제
- **해결책은 Center-Aligned Zoom**으로 PP=256을 자연스럽게 달성

---

## 9. 수식 요약

### Zoom 시 Intrinsics 변환

$$K_{new} = \begin{bmatrix} f_x \cdot z & 0 & (c_x - o_x) \cdot z \\ 0 & f_y \cdot z & (c_y - o_y) \cdot z \\ 0 & 0 & 1 \end{bmatrix}$$

- $z$: zoom factor
- $o_x, o_y$: crop offset

### Center-Aligned Zoom (권장)

$$o_x = o_y = \frac{size - size/z}{2}$$

$$c_x^{new} = (c_x - o_x) \cdot z = \left(256 - \frac{size(1 - 1/z)}{2}\right) \cdot z = 256$$

### Ray Error

$$\theta_{error} = \arctan\left(\frac{|c_x^{actual} - c_x^{used}|}{f_x}\right)$$

---

## 10. 관련 문서

- [[M3_SERIES_SPEC]] - M3_1, M3_2 상세 명세
- [[HYPOTHESIS_VERIFICATION_260125]] - 가설 검증 결과
- [[EXPERIMENT_REGISTRY]] - 실험 설정 레지스트리
- [[PREPROCESSING_REGISTRY]] - 전처리 버전 관리

---

*PP_FX_MVG_ANALYSIS v2.0 | 2026-01-25*
*팩트 체크 완료: M3_norm/M3_persample PP 가변 실패 → Center-Aligned Zoom으로 해결*

---

## 11. 버그 발견 및 수정 (2026-01-25)

### 11.1 버그 개요

**발견 시점**: 2026-01-25 전처리 검증 중
**영향 범위**: M3_1, M3_2 데이터셋
**근본 원인**: `normalize_after_zoom` 로직에서 center-aligned zoom 미처리

### 11.2 버그 상세

```python
# 버그 코드 (preprocess.py:650-664)
if getattr(cfg, "normalize_after_zoom", False) and zoom > 1.0:
    renorm_scale = cfg.target_fx / fx  # 549 / 741 = 0.74
    fx = cfg.target_fx
    fy = fy * renorm_scale
    cx = cx * renorm_scale  # ❌ BUG: 256 * 0.74 = 190
    cy = cy * renorm_scale  # ❌ BUG: 256 * 0.74 = 190
```

**문제**: Center-aligned zoom은 PP=256을 유지해야 하는데, `renorm_scale`로 스케일링 적용

### 11.3 영향받은 데이터셋

| 데이터셋 | zoom_center_mode | PP 결과 | 상태 |
|----------|-----------------|---------|------|
| M3_1 | **image** (center) | 190.0 ❌ | 버그 영향 |
| M3_2 | **image** (center) | 254.6 ❌ | 버그 영향 |
| M3_norm | object (기본) | 141.6 | 정상 (object-centered) |
| M3_persample | object (기본) | 253.1 | 정상 (per-sample) |

### 11.4 수정 내용

```python
# 수정 코드 (preprocess.py:650-663)
if getattr(cfg, "normalize_after_zoom", False) and zoom > 1.0:
    renorm_scale = cfg.target_fx / fx
    fx = cfg.target_fx
    fy = fy * renorm_scale
    # ★ BUG FIX (2026-01-25): Center-aligned zoom preserves PP at 256
    if getattr(cfg, "zoom_center_mode", "object") != "image":
        cx = cx * renorm_scale
        cy = cy * renorm_scale
    else:
        # For center-aligned zoom, PP should remain at target (256)
        cx, cy = cfg.target_pp
```

### 11.5 재전처리 명령어

```bash
# 1. 버그 데이터 삭제
rm -rf /home/joon/data/preprocessed/FaceLift_mouse/M3_1
rm -rf /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# 2. 재전처리 (버그 수정 후)
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

# M3_2 (권장)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# M3_1
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_1 \
    --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_1
```

### 11.6 검증 방법

```python
import json, glob, numpy as np

files = glob.glob("/path/to/M3_*/samples/*/opencv_cameras.json")[:50]
cx_list = [json.load(open(f))["frames"][0]["cx"] for f in files]
print(f"cx: mean={np.mean(cx_list):.1f}, std={np.std(cx_list):.1f}")
# 기대값: cx: mean=256.0, std=0.0
```

### 11.7 교훈

1. **MVG 일관성 검증 필수**: 전처리 후 PP, fx, translation 분포 확인
2. **Zoom 방식별 PP 처리 분리**: center-aligned와 object-centered는 다른 로직 필요
3. **단위 테스트 추가**: `test_normalize_after_zoom()` 함수로 회귀 방지

---

*Bug Fix Log: 2026-01-25 | Author: Claude Code*

### 11.8 Per-sample Zoom 버그 (2026-01-25 추가 발견)

**문제**: `zoom_after_transform: True`가 per-sample zoom에서 무시됨

**영향 데이터셋**: M3_2, M3_persample

**증상**:
```
원본 마스크 coverage: ~5% → zoom ≈ 1.0 계산
Homography 적용 후: coverage ~2.77% (감소)
결과: 목표 5% 미달, zoom 거의 없음
```

**수정 코드** (`preprocess.py:955-980`):
```python
if getattr(cfg, "zoom_after_transform", False):
    # Compute coverage on TRANSFORMED mask (not original)
    temp_mask = masks[0].copy()
    M = transforms[0]
    temp_mask = cv2.warpPerspective(...)  # Transform first
    sample_zoom = compute_persample_zoom_coverage(temp_mask, ...)
```

**수정 후 동작**:
```
Transform 후 마스크 coverage: ~2.77%
zoom = sqrt(5% / 2.77%) ≈ 1.34x
결과: 목표 5% 달성
```
