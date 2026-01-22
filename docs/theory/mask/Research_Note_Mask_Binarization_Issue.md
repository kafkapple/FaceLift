# Research Note: Mask Binarization Issue in Preprocessing

> D4/D6-1 vs D7 전처리 방식의 마스크 처리 차이 분석
> Created: 2026-01-19

---

## 1. 문제 발견

### 1.1 관찰 현상

`comprehensive_report_v7`의 "1.2 Sample Images with PP Markers" 섹션에서:

| Row | Dataset | 관찰 |
|-----|---------|------|
| D4 | PP=256 forced | 배경에 검은색 무언가 보임 |
| D6-1 | No crop | 배경에 검은색 무언가 보임 (가장 심함) |
| **D7** | PP-centered shift | **정상** (깨끗한 흰색 배경) |

특히 **View 5**에서 두드러지게 나타남.

### 1.2 초기 가설

- 마스크 적용 오류?
- Compositing 로직 버그?
- 전처리 방식 차이?

---

## 2. 조사 과정

### 2.1 Alpha 채널 분석

```python
# View 5 Alpha Channel Analysis

D4:
  Alpha=0 pixels: 236,181 (90.1%)  # Background
  Alpha=255 pixels: 21,136 (8.1%)  # Foreground
  Semi-transparent (0<α<255): 4,827 ← 주목!

D6-1:
  Alpha=0 pixels: 244,990 (93.5%)
  Alpha=255 pixels: 14,634 (5.6%)
  Semi-transparent: 2,520 ← 주목!

D7_test:
  Alpha=0 pixels: 253,318 (96.6%)
  Alpha=255 pixels: 8,826 (3.4%)
  Semi-transparent: 0 ← 완전 binary!
```

**핵심 발견**: D7만 semi-transparent 픽셀이 0개

### 2.2 배경 영역 RGB 분석

```python
# RGB in alpha=0 regions (배경)

D4:    mean=[167, 167, 165], std=[88.6, 86.9, 93.9]
D6-1:  mean=[145, 145, 143], std=[96.0, 94.3, 100.8]  # 가장 어두움
D7:    mean=[194, 194, 193], std=[90.2, 89.0, 93.7]   # 가장 밝음
```

모든 데이터셋에서 **원본 비디오 컨텐츠가 배경 RGB에 남아있음** (High std = 다양한 값)

### 2.3 원본 마스크 비디오 분석

```python
# Original mask video (View 5, Frame 0)

Shape: (1024, 1152)
Unique values: 220  # 이진이 아님!
Semi-transparent (0<v<255): 15,267 pixels
Range: [1, 253]

Histogram:
  0: 1,093,632
  1-127: 7,535      # Semi-transparent (dark)
  128-254: 7,732    # Semi-transparent (bright)
  255: 70,749
```

**발견**: 원본 마스크 비디오 자체가 **anti-aliased** (비디오 압축으로 인한 것으로 추정)

---

## 3. 근본 원인

### 3.1 D7 전처리 코드 (preprocess_D7_pp_centered.py:341)

```python
# D7: 명시적 마스크 이진화
mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)
```

### 3.2 D4/D6-1 전처리 코드 (preprocessor_d6.py)

```python
# D6: 마스크 이진화 없음
mask_resized = cv2.resize(masks[i], (new_w, new_h),
                          interpolation=cv2.INTER_NEAREST)
# INTER_NEAREST는 보간 없이 최근접 값 사용
# 하지만 원본이 이미 semi-transparent이면 그대로 유지됨
```

### 3.3 Compositing 시 영향

```python
# Report visualization (comprehensive_report_v7.py:154-157)
alpha_f = alpha[:, :, np.newaxis] / 255.0
composite = (rgb * alpha_f + bg * (1 - alpha_f)).astype(np.uint8)

# 예: alpha=128인 픽셀
# composite = rgb * 0.5 + white * 0.5
# → 어두운 RGB가 50% 투과되어 회색으로 보임!
```

---

## 4. 결과 비교

### 4.1 Semi-transparent 픽셀 특성

| Dataset | 개수 | Alpha 범위 | Alpha 평균 | 위치 |
|---------|------|-----------|-----------|------|
| D4 | 4,827 | [1, 254] | 128.7 | y=[122-397], x=[16-376] |
| D6-1 | 2,520 | [1, 253] | 119.8 | y=[177-402], x=[161-455] |
| **D7** | **0** | - | - | - |

### 4.2 Semi-transparent 픽셀 분포

- **경계선상**: ~42-44%
- **경계선 외**: ~56-58% (내부/외부 영역)

→ 단순 edge anti-aliasing이 아닌, 원본 마스크 비디오의 압축 아티팩트

---

## 5. 결론

### 5.1 이것은 버그가 아님

- D4/D6-1: 원본 마스크 값을 그대로 보존 (설계 의도)
- D7: 명시적 이진화 적용 (threshold=127)
- 두 접근 모두 유효하나, **시각화 결과가 다름**

### 5.2 학습에 미치는 영향

| 항목 | 영향 |
|------|------|
| **Loss 계산** | 영향 없음 (alpha를 mask로 사용, threshold 적용) |
| **시각화** | D4/D6-1에서 "dirty" 배경 |
| **최종 결과** | 영향 미미 (binary threshold 사용 시) |

### 5.3 권장 사항

1. **신규 전처리 시**: D7 방식의 마스크 이진화 적용

   ```python
   mask_binary = np.where(mask > 127, 255, 0).astype(np.uint8)
   ```

2. **기존 D4/D6-1 데이터**:
   - 학습에는 영향 없으므로 재전처리 불필요
   - 시각화 목적이면 재전처리 고려

3. **Report 시각화**:
   - 마스크 기반 compositing 전 명시적 이진화 권장

---

## 6. 관련 파일

| 파일 | 위치 | 설명 |
|------|------|------|
| D7 전처리 | `Dataset prep/scripts/preprocess_D7_pp_centered.py:341` | 마스크 이진화 포함 |
| D6 전처리 | `mouse_extensions/preprocessing/preprocessor_d6.py` | 이진화 없음 |
| 리포트 생성 | `Dataset prep/scripts/comprehensive_report_v7.py:154` | Compositing 로직 |
| 원본 마스크 | `/home/joon/data/markerless_mouse_1_nerf/simpleclick_undist/` | Anti-aliased video |

---

## 7. 참고

### 7.1 마스크 비디오가 Anti-aliased인 이유

- SimpleClick segmentation 결과를 비디오로 저장 시 압축 적용
- H.264/H.265 등의 lossy 코덱 사용으로 edge smoothing 발생
- 원본 segmentation은 binary였을 가능성 높음

### 7.2 INTER_NEAREST vs Binarization

```python
# INTER_NEAREST: 보간 없이 최근접 값 선택
# - 입력이 binary면 출력도 binary
# - 입력이 anti-aliased면 출력도 anti-aliased 유지

# np.where(mask > 127, 255, 0): 명시적 이진화
# - 입력 값과 무관하게 항상 binary 출력
```

---

*Created: 2026-01-19*
*Investigation: mask_iou IoU=1.0 버그 수정 중 발견*
