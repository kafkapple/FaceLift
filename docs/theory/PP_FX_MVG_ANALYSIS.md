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
| **Coverage** | 높을수록 좋음 | D3(~6%)→27.0, D7_1(~3%)→20.9 |

**⚠️ 중요**: PP를 "강제"하는 것이 아니라, **이미지 변환과 PP가 일관성 있게 256이 되도록** 해야 함.

---

## 2. 실험 데이터 요약

### 2.1 데이터셋별 성능

| Dataset | Coverage | fx | PP | Val PSNR | 상태 |
|---------|----------|-----|-----|----------|------|
| **D3_normalized** | ~6% | 549 | 256 (일관) | **27.09** | ✅ 최고 |
| D7_1 | ~3% | 549 | 256 | 20.93 | ✅ 안정 |
| D8 | ~3% | 549 | 256 | 20.21 | ✅ 안정 |
| M3 (원본) | ~6% | **739** ❌ | 가변 | ~17 | ❌ fx 버그 |
| M3_norm | ~6% | 549 | **가변** ❌ | ~17 | ❌ PP 불일치 |
| M3_persample | ~6% | 549 | **가변** ❌ | ~17 | ❌ PP 불일치 |

### 2.2 핵심 발견

```
Coverage ↑ + fx=549 + PP=256 → PSNR ↑

D3_normalized이 최고인 이유:
1. PP=256 일관성 유지 ✅
2. fx=549 pretrained 호환 ✅
3. ~6% Coverage (높음) ✅
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
| Coverage | ~6% | ~3% | ~6% |
| Val PSNR | 27.09 | ~20 | ~17 |

### 6.2 성공 이유

1. **PP=256 일관성**: Zoom 없이 자연스럽게 중앙
2. **fx=549**: Pretrained 호환
3. **높은 Coverage (~6%)**: 풍부한 정보

**⚠️ 주의**: fx/fy=0.9937 비정방형은 성공 요인이 **아님**. PP 일관성이 핵심.

---

## 7. 권장 설정

### 7.1 데이터셋 (M-Series v2)

| ID | 방식 | fx | PP | Coverage | 상태 |
|----|------|-----|-----|----------|------|
| **M3_1** | Global Zoom + Center-Aligned | 549 | 256 | ~6% | ⏳ 전처리 중 |
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

---

## See Also

> Parent: [[INDEX]] > Theory

- [[GHOSTING_ANALYSIS]] - Ghosting 원인 및 해결 전략
- [[MASK_GUIDE]] - 마스크 시스템 상세
- [[CENTER_ESTIMATION]] - 3D Triangulation
- [[COORDINATE_SYSTEMS]] - 좌표계 변환
- [[CLIPPING_AND_CENTERING_ANALYSIS]] - 클리핑/센터링

### Archived Camera Theory Documents
이전 camera/ 폴더의 개별 문서들은 이 문서에 통합되었습니다:
- `_archive/theory/camera/PP_FIX_MVG_THEORY.md` - PP 보정 이론 (원본)
- `_archive/theory/camera/error_amplification_analysis.md` - PP 오류 증폭 분석
- `_archive/theory/camera/mvg_virtual_zoom_theory.md` - Virtual zoom MVG 이론

---

*Last updated: 2026-01-27*

## PP Error Amplification Analysis (from archive)

# Ray/PP Error Amplification Analysis for Small Objects
# Generated: 2026-01-21

## 1. Background

When the target object (mouse) occupies a small portion of the image,
geometric errors are amplified relative to the object size.

## 2. Current Mouse Size Statistics (D7_1)

| Metric | Value | % of 512px |
|--------|-------|------------|
| Mean Width | 147 px | 28.7% |
| Mean Height | 162 px | 31.6% |
| Max Width | 222 px | 43.4% |
| Max Height | 295 px | 57.6% |
| Mean Diagonal | ~219 px | 42.8% |

Mouse coverage area: ~(28.7% x 31.6%) = **9.1% of image area**

## 3. Error Amplification Analysis

### 3.1 PP (Principal Point) Error

PP error affects ray direction, especially at image edges.

**D7_1 Current State**: PP = (256, 256) after preprocessing (centered)

**Potential PP errors**:
- Rounding: ~0.01 pixel
- Skew-related: ~0.9 pixel max at edge

**Amplification factor** = Image size / Object size = 512 / 155 = **3.3x**

| PP Error (px) | At Image Level | Relative to Mouse | % of Mouse |
|---------------|----------------|-------------------|------------|
| 0.01 px | 0.002% | 0.0065% | negligible |
| 0.5 px | 0.10% | 0.33% | 0.5 px |
| 0.9 px | 0.18% | 0.59% | **0.9 px** |
| 1.0 px | 0.20% | 0.65% | 1.0 px |

### 3.2 Ray Direction Error

Ray direction affects 3D reconstruction accuracy.

**Formula**: ray_error_deg = arctan(pp_error / focal_length)
With fx = 549, 1 pixel PP error -> 0.104 deg ray error

**Impact on 3D points**:
At distance d=2.7 (normalized), ray error theta causes position error:
position_error = d * tan(theta)

| PP Error | Ray Error | 3D Error at d=2.7 | Relative to Mouse |
|----------|-----------|-------------------|-------------------|
| 0.5 px | 0.052 deg | 0.0024 units | 0.34% |
| 0.9 px | 0.094 deg | 0.0044 units | **0.62%** |
| 1.0 px | 0.104 deg | 0.0049 units | 0.69% |

**Note**: Mouse 3D size ~= 0.7 units (estimated from bounding box)

### 3.3 Skew Error (Currently Ignored)

Original skew values: -5.82 to +1.39 pixels
Max pixel error at edge: ~0.9 pixels

**Skew effects**:
- Non-orthogonal pixel grid
- Error accumulates toward image corners
- Maximum at (0,0) and (512,512)

| Location | Mouse Portion | Skew Effect | Impact |
|----------|---------------|-------------|--------|
| Center | Most frames | less than 0.2 px | Low |
| Edge | Some frames | ~0.9 px | Medium |
| Corner | Rare | ~1.8 px | High |

### 3.4 Scale Anisotropy

Current: scale_x/scale_y = 1.0043 (0.43% difference)
Due to original fx/fy ~= 0.996

**Effect**:
- Vertical stretching of ~0.43%
- For 295 px tall mouse: ~1.3 pixel distortion

### 3.5 Aspect Ratio Change (Rectangle -> Square)

Original: 1152x1024 (ratio 1.125)
Target: 512x512 (ratio 1.0)

**If naive resize** (squish):
- 12.5% vertical compression
- 3D Gaussians would be stretched
- **Not used in D7_1** (crop-based approach)

**D7_1 approach** (PP-centered crop):
- No aspect ratio distortion
- Uses individual scale factors

## 4. Total Error Budget

| Error Source | Max Error | Amplified (3.3x) | Severity |
|--------------|-----------|------------------|----------|
| PP rounding | 0.01 px | 0.03 px | Negligible |
| fx/fy rounding | 0.006 px | 0.02 px | Negligible |
| Skew (ignored) | 0.9 px | **3.0 px** | Medium |
| Scale anisotropy | 1.3 px | **4.3 px** | Medium |

**Total worst-case error**: ~7 pixels relative to mouse (~5% of mouse size)
**Typical error**: ~3 pixels relative to mouse (~2% of mouse size)

## 5. Recommendations

### 5.1 Priority 1: Skew Correction (D8)
- Implement homography-based skew removal
- Expected improvement: -0.9 px max error -> -0 px
- Amplified benefit: **-3.0 px** relative to mouse

### 5.2 Priority 2: Virtual Zoom (D9)
- Crop around mouse, resize to 512x512
- Reduce amplification factor from 3.3x to ~1.5x
- Double benefit: smaller errors AND better resolution

### 5.3 Priority 3: Isotropic Scaling (D8)
- Use single scale factor (average of scale_x, scale_y)
- Eliminates 0.43% anisotropy
- OR: Crop to square region first, then resize

## 6. Proposed D8 vs D9 Comparison

| Feature | D7_1 | D8 | D9 |
|---------|------|----|----|
| Skew | Ignored | Corrected | Corrected |
| fx/fy precision | 549 | 548.994 | 548.994 |
| PP | Centered | Centered | Centered |
| Zoom | 1.0x | 1.0x | **~1.5x** |
| Mouse size | 30% | 30% | **45%** |
| Error amplification | 3.3x | 3.3x | **2.2x** |

## 7. Conclusion

Current D7_1 preprocessing has good geometric accuracy but:
1. Small mouse (30% of image) amplifies any errors by 3.3x
2. Skew causes up to 3 pixel error relative to mouse
3. Virtual zoom (D9) could reduce errors by 50%

**Recommended path**:
D7_1 -> D8 (precision) -> D9 (zoom) for optimal results

---
*Generated by error_amplification_analysis.py*

## MVG Virtual Zoom Theory (from archive)

# MVG Theory: Virtual Zoom for Small Object Enhancement
# Generated: 2026-01-21

## 1. Problem Statement

**Goal**: Make the mouse appear larger in preprocessed images while maintaining
geometric accuracy for 3D reconstruction.

**Constraints**:
- Camera intrinsics must remain geometrically consistent
- No distortion or information loss
- Must work across all 6 views simultaneously

## 2. MVG Background: Image Formation

### 2.1 Pinhole Camera Model

A 3D point X projects to image point x:

```
x = K @ [R | t] @ X

where K = [[fx,  skew, cx],
           [0,   fy,   cy],
           [0,   0,    1 ]]
```

### 2.2 Key Relationship

**Focal length determines FOV**:

```
FOV = 2 * arctan(image_size / (2 * focal_length))
```

For 512x512 image with fx=549: FOV = 50 deg
For 512x512 image with fx=1098: FOV = 26.2 deg (zoomed in 2x)

## 3. Methods for Virtual Zoom

### 3.1 Method A: Crop + Resize (Recommended)

**Principle**: Cropping is equivalent to increasing focal length

**Process**:
1. Crop image from WxH to W'xH' centered at (crop_cx, crop_cy)
2. Resize cropped region to target size (512x512)
3. Update intrinsics:

```python
fx_new = fx * (target_size / crop_size)
fy_new = fy * (target_size / crop_size)
cx_new = (cx - crop_x) * (target_size / crop_size)
cy_new = (cy - crop_y) * (target_size / crop_size)
```

**Geometric accuracy**: PERFECT (no approximation)

**Example** (1.45x zoom):
- Original: 512x512, fx=549, cx=256, cy=256
- Crop: 354x354 centered on mouse at (244, 295)
- After resize to 512x512:
  - fx_new = 549 * (512/354) = 794
  - scale = 512/354 = 1.446

### 3.2 Method B: Virtual Camera Repositioning

**Principle**: Move camera closer while adjusting focal length

**Process**:
1. Scale translation: T_new = T * (1 / zoom_factor)
2. Adjust focal length: fx_new = fx * zoom_factor
3. Image remains unchanged

**Geometric accuracy**:
- PERFECT for planar scenes
- APPROXIMATE for 3D scenes (depth changes)

**Math**:
For point at depth Z, projection is x = fx * X/Z + cx
If we move camera closer by factor k:
- New depth Z' = Z/k
- New projection x' = fx * X/(Z/k) + cx = k * fx * X/Z + cx

This is equivalent to scaling focal length by k.

**Limitation**: Works exactly only if all points are at same depth.
For objects with depth variation (like a mouse), introduces small parallax errors.

### 3.3 Method C: Homographic Zoom (Planar Approximation)

**Principle**: Apply homography that simulates zoom

```
H_zoom = [[s, 0, cx*(1-s)],
          [0, s, cy*(1-s)],
          [0, 0, 1       ]]

where s = zoom_factor
```

**Geometric accuracy**:
- Only exact for planar scenes
- Introduces distortion for 3D objects

**Not recommended** for 3D reconstruction.

## 4. Optimal Strategy for Mouse Data

### 4.1 Recommended: Method A (Crop + Resize)

**Why**:
1. Geometrically exact (no approximation)
2. Works with 3D objects of any depth
3. Simple implementation
4. Consistent with GS-LRM architecture

### 4.2 Implementation Steps

1. **Find global crop region**:
   - For each frame, get mouse bounding box from mask
   - Find the tightest crop that contains mouse in ALL views
   - Add padding (20-30% margin)

2. **Apply crop per view**:
   - Crop region may differ per view (due to 3D geometry)
   - OR use same crop offset for all views (simpler, small loss)

3. **Update intrinsics**:

```python
scale = target_size / crop_size  # e.g., 512/354 = 1.45
fx_new = fx_orig * scale
fy_new = fy_orig * scale
cx_new = (cx_orig - crop_x) * scale
cy_new = (cy_orig - crop_y) * scale
```

4. **Verify GS-LRM compatibility**:
   - GS-LRM expects fx approx 549
   - If fx_new >> 549, need additional normalization
   - Solution: Apply camera distance scaling inversely

### 4.3 Camera Normalization with Zoom

**Problem**: After zoom, fx_new = 794 (not 549)

**Solution**: Keep larger effective focal length
- GS-LRM reads fxfycxcy from data
- NOT hardcoded to 549
- Can handle varying focal lengths

**Verification needed**: Check GS-LRM behavior with fx >> 549

### 4.4 Alternative: Zoom + Distance Compensation

To maintain fx approx 549 while zooming:

```
zoom_factor = 1.45
fx_after_crop = 549 * 1.45 = 796

To normalize back to fx=549:
distance_scale = 549 / 796 = 0.69
T_new = T_orig * 0.69  # Move camera closer
```

**Effect**: Mouse appears larger, fx=549, camera closer
**Warning**: May violate GS-LRM distance assumptions (d=2.7)

## 5. Practical Zoom Limits

### 5.1 Current Statistics

| Metric | Current D7_1 | After 1.45x Zoom |
|--------|--------------|------------------|
| Mouse width | 147 px (29%) | 213 px (42%) |
| Mouse height | 162 px (32%) | 235 px (46%) |
| Error amplification | 3.3x | 2.3x |
| fx | 549 | 796 |
| FOV | 50 deg | 35 deg |

### 5.2 Maximum Practical Zoom

- Largest mouse: 222x295 pixels
- With 20% padding: 266x354 -> crop size 354
- Maximum zoom: 512/354 = **1.45x**

Higher zoom risks cropping the mouse in some frames.

### 5.3 Recommended Zoom: 1.3x (Conservative)

- Crop size: 394x394
- fx_new = 549 * 1.3 = 714
- FOV: 39.5 deg
- Mouse size: 38-41% of image
- Error amplification: 2.5x

## 6. Implementation Plan (D9)

```python
def preprocess_D9_zoom(image, mask, K, R, T, zoom=1.3, target_size=512):
    """
    D9 preprocessing: D8 precision + virtual zoom

    Steps:
    1. Find mouse center from mask
    2. Calculate crop region
    3. Crop image
    4. Resize to target_size
    5. Update intrinsics
    """
    import numpy as np
    import cv2

    # 1. Find mouse center
    ys, xs = np.where(mask > 127)
    mouse_cx = (xs.min() + xs.max()) / 2
    mouse_cy = (ys.min() + ys.max()) / 2

    # 2. Calculate crop size
    crop_size = int(target_size / zoom)
    crop_x = int(mouse_cx - crop_size / 2)
    crop_y = int(mouse_cy - crop_size / 2)

    # Clamp to valid range
    crop_x = max(0, min(crop_x, image.shape[1] - crop_size))
    crop_y = max(0, min(crop_y, image.shape[0] - crop_size))

    # 3. Crop
    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    # 4. Resize
    resized = cv2.resize(cropped, (target_size, target_size))

    # 5. Update intrinsics
    scale = target_size / crop_size
    fx_new = K[0,0] * scale
    fy_new = K[1,1] * scale
    cx_new = (K[0,2] - crop_x) * scale
    cy_new = (K[1,2] - crop_y) * scale

    K_new = np.array([
        [fx_new, K[0,1]*scale, cx_new],
        [0,      fy_new,       cy_new],
        [0,      0,            1     ]
    ])

    return resized, K_new, R, T  # R, T unchanged!
```

## 7. Summary

| Method | Geometric Accuracy | Implementation | Recommendation |
|--------|-------------------|----------------|----------------|
| Crop + Resize | Exact | Simple | RECOMMENDED |
| Virtual Reposition | Approximate | Medium | For planar only |
| Homographic Zoom | Distorted | Complex | Not recommended |

**Final Recommendation**:
- D8: Precision preprocessing (skew correction, exact values)
- D9: D8 + Virtual zoom (1.3x via crop+resize)

---
*MVG Theory Reference: Hartley & Zisserman, "Multiple View Geometry in Computer Vision"*
