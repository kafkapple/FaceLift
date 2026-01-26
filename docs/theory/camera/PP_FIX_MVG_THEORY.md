# PP 고정과 MVG 이론 분석

> **작성일**: 2026-01-25
> **목적**: PP=256 강제가 MVG 이론에 부합하는지 분석 및 올바른 해결책 제시
> **관련 문서**: [[COMPREHENSIVE_ANALYSIS_260125]], [[coordinate_transformation_guide]]

---

## 1. 문제 요약

### 1.1 현재 상황

```
D3_normalized: PP 256 강제 (이미지 변경 없음) → PSNR 27.09 ✅
M3_persample:  PP 가변 (179~256)            → PSNR 17.24 ❌
```

### 1.2 핵심 질문

**PP를 256으로 강제하는 것이 MVG 이론에 맞는가?**

---

## 2. MVG 이론 분석

### 2.1 Principal Point (PP)의 의미

```
PP (cx, cy) = 카메라 광학 축이 이미지 평면과 만나는 점

  광학 축
     │
     │  ← 이 점이 PP
     ▼
┌────●────┐
│         │  이미지 평면
│         │
└─────────┘
```

### 2.2 PP 변경 시 광선(Ray) 방향

카메라 광선 방향 공식:
```
ray_direction = K^(-1) @ [u, v, 1]^T

K = [[fx,  0, cx],
     [ 0, fy, cy],
     [ 0,  0,  1]]
```

**PP(cx, cy)를 변경하면 → 모든 픽셀의 광선 방향이 변경됨**

### 2.3 PP 오류의 영향

```
PP 오류 Δpp = 실제 PP - 사용된 PP

Ray 방향 오류 ≈ arctan(Δpp / fx)

예: Δpp = 77px, fx = 549
    오류 ≈ arctan(77/549) ≈ 8.0°
```

**8°의 광선 오류 → 3D 위치 추정에서 심각한 Ghosting 유발**

---

## 3. D3_normalized는 왜 작동하는가?

### 3.1 D3_normalized 방식

```python
# postcorrect_d3.py
frame['cx'] = 256.0  # 이미지 변경 없이 PP만 256으로
frame['cy'] = 256.0
```

### 3.2 작동 이유 (추정)

| 요인 | 설명 |
|------|------|
| **원본 PP가 256 근처** | D3 원본에서 객체가 이미 중앙 근처 |
| **PP 오류 작음** | 실제 오류 < 10px 추정 |
| **Ray 오류 작음** | < 1° (무시 가능) |

### 3.3 검증 필요

```bash
# D3 원본의 실제 PP 분포 확인
python -c "
import json, glob
for f in glob.glob('/path/to/D3/train/*/opencv_cameras.json')[:10]:
    d = json.load(open(f))
    print(d['frames'][0]['cx'], d['frames'][0]['cy'])
"
```

---

## 4. M3_persample이 실패하는 이유

### 4.1 문제 분석

```
M3_persample PP 분포:
  cx: mean=238.3, std=19.5, min=179.0, max=256.0
  cy: mean=237.1, std=21.9, min=169.0, max=256.0
```

**최대 PP 오류 = 256 - 169 = 87px**
**최대 Ray 오류 = arctan(87/549) ≈ 9.0°**

### 4.2 force_pp_to_target=True의 문제

```
현재 상태:
  - 이미지: zoom crop으로 객체 중심이 ~238,237 위치
  - PP: 256으로 강제

결과:
  - 이미지와 PP가 불일치
  - 광선 방향이 ~8° 틀어짐
  - Ghosting 발생
```

---

## 5. MVG 이론에 부합하는 해결책

### 5.1 Option A: Center-Aligned Zoom (★ 권장)

```python
def apply_zoom_center_aligned(self, image, mask, zoom):
    """이미지 중심 기준 zoom - PP 자동 보존"""
    size = self.config.output_size
    crop_size = int(size / zoom)

    # 핵심: 이미지 중심(256,256) 기준으로 crop
    crop_x = (size - crop_size) // 2
    crop_y = (size - crop_size) // 2

    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    zoomed = cv2.resize(cropped, (size, size))

    # PP는 자동으로 256 유지 (이미지와 일관성 유지)
    return zoomed, (crop_x, crop_y)
```

**장점**: MVG 이론 완벽 준수, PP=256 자동
**단점**: 객체가 가장자리에 있으면 crop에서 잘림

### 5.2 Option B: Object-Centered Zoom + Image Shift

```python
def apply_zoom_object_centered(self, image, mask, zoom):
    """객체 중심 zoom 후 이미지 shift로 중앙 정렬"""
    size = self.config.output_size
    crop_size = int(size / zoom)

    # Step 1: 객체 중심 찾기
    ys, xs = np.where(mask > 127)
    obj_cx = (xs.min() + xs.max()) / 2
    obj_cy = (ys.min() + ys.max()) / 2

    # Step 2: 객체 중심 기준 crop
    crop_x = int(obj_cx - crop_size/2)
    crop_y = int(obj_cy - crop_size/2)
    cropped = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    # Step 3: resize 후 객체가 이미지 중앙에 오도록 확인
    zoomed = cv2.resize(cropped, (size, size))

    # Step 4: 필요시 추가 shift (객체가 정확히 중앙에 오도록)
    # 이미 crop이 객체 중심이므로 resize 후 객체는 중앙에 있음
    # PP = 256, 256 (정확함)

    return zoomed, (crop_x, crop_y)
```

**장점**: 객체가 항상 중앙, MVG 준수
**주의**: crop 후 객체 중심이 resize된 이미지의 중앙인지 검증 필요

### 5.3 Option C: 현재 방식 유지 + PP 정확 기록

```python
# 현재 방식 유지 (PP 가변)
# GS-LRM이 fxfycxcy 텐서로 가변 PP 처리

# 조건: Pretrained 모델이 PP 분산에 적응 가능해야 함
```

**장점**: 코드 변경 최소
**단점**: Pretrained가 PP=256 기대하면 성능 저하

---

## 6. 권장 해결책

### 6.1 즉시 조치 (Quick Fix)

**Option A 적용**: Center-Aligned Zoom

```python
# preprocess.py의 apply_zoom 수정
def apply_zoom(self, image, mask, zoom):
    if zoom <= 1.0:
        return image, mask, (0, 0)

    size = self.config.output_size
    crop_size = int(size / zoom)

    # ★ 변경: 이미지 중심 기준 crop
    crop_x = (size - crop_size) // 2
    crop_y = (size - crop_size) // 2

    cropped_img = image[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]
    cropped_mask = mask[crop_y:crop_y+crop_size, crop_x:crop_x+crop_size]

    zoomed_img = cv2.resize(cropped_img, (size, size))
    zoomed_mask = cv2.resize(cropped_mask, (size, size))

    return zoomed_img, zoomed_mask, (crop_x, crop_y)
```

### 6.2 PP 계산 확인

```python
# compute_camera_params에서
if zoom > 1.0:
    crop_x, crop_y = crop_offset  # = (size - crop_size) // 2
    fx, fy = fx * zoom, fy * zoom

    # Center-aligned crop이면:
    # cx = (256 - crop_x) * zoom = (256 - (256-crop_size/2)) * zoom
    #    = (crop_size/2) * zoom = size/2 = 256
    cx = (cfg.target_pp[0] - crop_x) * zoom  # = 256 자동
    cy = (cfg.target_pp[1] - crop_y) * zoom  # = 256 자동
```

### 6.3 새 프리셋 추가

```python
# presets.py
"M3_persample_centered": {
    "paradigm": "precision_homography",
    ...
    "zoom_method": "center_aligned",  # ★ 새 옵션
    "force_pp_to_target": False,  # 불필요 (자동 256)
    ...
}
```

---

## 7. 검증 계획

### 7.1 수정 후 검증

```bash
# 1. PP 분포 확인
python -c "
for i in range(100):
    f = f'M3_persample_centered/samples/{i:06d}/opencv_cameras.json'
    d = json.load(open(f))
    print(d['frames'][0]['cx'], d['frames'][0]['cy'])
"
# 예상: 모두 256.0, 256.0

# 2. 학습 실험
CUDA_VISIBLE_DEVICES=0 torchrun ... -d M3_persample_centered -e E0_1_facelift
# 예상: Val PSNR 25+
```

---

## 8. 결론

| 방식 | MVG 준수 | PP 결과 | 권장 |
|------|----------|---------|------|
| 현재 (force=False) | ⚠️ | 가변 | ❌ |
| force=True (단순) | ❌ | 256 (불일치) | ❌ |
| **Center-Aligned** | ✅ | 256 (일치) | ★ |
| Object+Shift | ✅ | 256 (일치) | ○ |

**핵심**: PP 강제가 아닌, **이미지 변환과 PP가 일관성 있게** 256이 되어야 함

---

## 관련 문서

- [[COMPREHENSIVE_ANALYSIS_260125]] - 종합 분석 보고서
- [[coordinate_transformation_guide]] - 좌표계 변환 이론
- [[PREPROCESSING_REGISTRY]] - 전처리 버전 관리

---

*PP Fix MVG Theory v1.0 | 2026-01-25*
