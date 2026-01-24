# GS-LRM Mouse Ghosting 해결 전략

> Created: 2026-01-24 | Version: 1.0
> 문헌 근거 기반 종합 분석 및 해결 전략

---

## Executive Summary

GS-LRM에서 발생하는 mouse ghosting 현상의 근본 원인은 **Plucker ray 분포 불균형**입니다. 
문헌 조사 결과, object가 이미지의 2-3%만 차지할 경우 view 간 ray 분포가 유사해져 
Transformer가 view를 구분하기 어려워집니다. 해결책으로 **Object-Centered Zoom** 
전처리와 **Normalized Masked Loss**를 결합한 새로운 파이프라인을 제안합니다.

---

## 1. 문제 분석

### 1.1 현상 관찰
- 모든 전처리 방식(D7_1, D8, D8.2)에서 ghosting 발생
- Ray error ~0°임에도 여러 생쥐가 겹쳐 보임
- 크기와 위치가 조금씩 다른 복수의 생쥐 형상

### 1.2 근본 원인: Plucker Ray 분포 불균형

**Plucker Coordinates 정의** [Light Field Networks, Sitzmann et al.]:
```
P = (d, m) = (d, o × d)
- d: 광선 방향 (3D, normalized)
- m: 모멘트 (3D) = 원점에서 광선까지의 수직 거리 × 방향
```

**GS-LRM 입력 구조** [GS-LRM, Zhang et al. ECCV 2024]:
```
Input = [RGB(3ch) | Plucker(6ch)] = 9 channels per pixel
```

**문제**: 작은 object (2-3% coverage)
- 97% 픽셀의 ray가 배경을 가리킴
- View 간 Plucker 분포가 거의 동일
- Transformer가 view 구분 실패 → 독립적 Gaussian 생성 → Ghosting

---

## 2. 문헌 근거

### 2.1 Object Scale 및 Centering 표준

| 방법론 | Object Coverage | Centering | 출처 |
|--------|-----------------|-----------|------|
| LRM | - | 이미지 중앙 crop | Hong et al., 2023 |
| Objaverse Rendering | 50-80% | 중앙 정렬 | Deitke et al., 2023 |
| Instant-NGP | - | [-1,1]³ 정규화 | Müller et al., 2022 |

**LRM 전처리 프로토콜** [LRM Paper]:
> "For video data, we crop and resize images to center the object with adjusted camera parameters."

### 2.2 Mask Supervision 접근법

| 방법 | Loss 공식 | 가중치 | 출처 |
|------|-----------|--------|------|
| **LGM** | MSE(α_pred, α_gt) | 1.0 | 3DTopia, ECCV 2024 |
| **Object-Centric 2DGS** | γ × mean(α × (1-M)) | γ=0.5 | arxiv 2501.08174 |
| **Pose Splatter** | L1 / sum(mask) | λ=0.5 | NeurIPS 2025 |

**Normalized Masked Loss** [Pose Splatter]:
> "Division by mask sum normalizes gradients by foreground area, providing scale-invariant supervision."

```python
L_color = sum(|pred - gt| * mask) / (3 * sum(mask))
```

### 2.3 Plucker Encoding 장점

**Position Invariance** [Light Field Networks]:
- 동일 광선은 어느 점에서 정의해도 동일한 Plucker 좌표
- Object 위치 변화에 강건

**Smooth Manifold** [PluckeRF, CVPR 2025]:
- 연속적이고 미분 가능
- Neural network 학습에 적합

---

## 3. 최적 해결 전략

### 3.1 전략 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│  D7_1 기하학적 정확성   +   D3 Object Centering/Scaling            │
│         (Ray Error ~0°)         (FG Coverage 5%+)                   │
│                              ↓                                       │
│                    D7_1_centered (새 프리셋)                         │
└─────────────────────────────────────────────────────────────────────┘
```

### 3.2 새 전처리 파이프라인: D7_1_centered

**목표**:
1. FG Coverage: 2% → **5%+** (2.5x 증가)
2. Center Offset: 101px → **<15px**
3. Ray Error: **~0°** 유지
4. fx=549, dist=2.7 유지 (Pretrained 호환)

**파이프라인 단계**:

```
Step 1: Object Center Detection
────────────────────────────────
- 모든 뷰의 mask에서 centroid 계산
- 3D triangulation으로 world-space center 추정
- → object_center_3d

Step 2: Object-Centered Crop
────────────────────────────────
- 각 뷰에서 object_center_3d를 back-project
- 해당 점을 이미지 중앙으로 crop
- Adaptive zoom: target_coverage = 5%

Step 3: Intrinsics Adjustment
────────────────────────────────
# Crop offset 반영
cx_new = cx_orig - crop_offset_x
cy_new = cy_orig - crop_offset_y

# Zoom 반영
fx_zoomed = fx * zoom_factor
fy_zoomed = fy * zoom_factor
cx_zoomed = cx_new * zoom_factor
cy_zoomed = cy_new * zoom_factor

Step 4: Camera Normalization
────────────────────────────────
# fx → 549 정규화
scale = 549.0 / fx_zoomed
fx_final = 549.0
fy_final = fy_zoomed * scale
cx_final = 256.0  # 중앙으로 shift
cy_final = 256.0

# Translation 비례 조정
translation_final = translation * (549.0 / fx_zoomed)
```

### 3.3 훈련 설정 최적화

**Loss 함수 조합** (문헌 기반):

```yaml
training:
  losses:
    # RGB Loss (Pose Splatter style)
    mask_mode: gt
    normalize_by_mask: true  # L / sum(mask)
    
    # Alpha Supervision (LGM style)
    alpha_loss_weight: 0.2
    alpha_loss_type: mse
    
    # Background Penalty (Object-Centric 2DGS style)
    bg_loss_weight: 0.3
    
    # Perceptual Loss
    perceptual_loss_weight: 0.5
```

---

## 4. 구현 계획

### Phase 1: 전처리 파이프라인 구현 (우선순위 P0)

**파일**: `mouse_extensions/preprocessing/presets.py`

```python
PRESETS["D7_1_centered"] = {
    "paradigm": "object_centered_zoom",
    "base_preset": "D7_1",  # 기하학적 정확성 상속
    
    # Object centering
    "center_method": "triangulation",  # 3D triangulation
    "center_tolerance_px": 15,
    
    # Adaptive zoom
    "adaptive_zoom": True,
    "target_fg_coverage": 0.05,  # 5%
    "zoom_range": [1.0, 2.5],
    
    # Normalization (pretrained 호환)
    "target_fx": 549.0,
    "target_cx": 256.0,
    "target_cy": 256.0,
    
    # Quality
    "transform": "homography",  # from D8
    "skew_correction": True,
    "ray_error": "~0 deg",
}
```

**파일**: `mouse_extensions/preprocessing/object_centered_preprocessor.py`

```python
class ObjectCenteredPreprocessor:
    """
    D7_1의 기하학적 정확성과 D3의 object centering을 결합.
    """
    
    def __init__(self, config):
        self.target_coverage = config.get("target_fg_coverage", 0.05)
        self.zoom_range = config.get("zoom_range", [1.0, 2.5])
        self.center_tolerance = config.get("center_tolerance_px", 15)
    
    def compute_object_center_3d(self, masks, cameras):
        """3D triangulation으로 object center 추정"""
        # 각 뷰의 2D centroid 계산
        centroids_2d = [self._compute_centroid(m) for m in masks]
        
        # DLT triangulation
        center_3d = triangulate_point(centroids_2d, cameras)
        return center_3d
    
    def compute_adaptive_zoom(self, masks):
        """Target coverage 달성을 위한 zoom factor 계산"""
        current_coverage = np.mean([m.sum() / m.size for m in masks])
        zoom = np.sqrt(self.target_coverage / current_coverage)
        return np.clip(zoom, *self.zoom_range)
    
    def process_frame(self, images, masks, cameras):
        # Step 1: Object center
        center_3d = self.compute_object_center_3d(masks, cameras)
        
        # Step 2: Adaptive zoom
        zoom = self.compute_adaptive_zoom(masks)
        
        # Step 3-4: Crop, zoom, normalize
        processed = []
        for img, mask, cam in zip(images, masks, cameras):
            result = self._process_single_view(
                img, mask, cam, center_3d, zoom
            )
            processed.append(result)
        
        return processed
```

### Phase 2: 실험 설정 (우선순위 P1)

**파일**: `configs/experiments/E7_centered_optimal.yaml`

```yaml
# E7: Object-Centered + Normalized Masked Loss + Alpha Supervision
experiment:
  name: E7_centered_optimal
  description: "Literature-based optimal configuration for mouse reconstruction"

training:
  losses:
    mask_mode: gt
    normalize_by_mask: true
    alpha_loss_weight: 0.2
    alpha_loss_type: mse
    bg_loss_weight: 0.3
    perceptual_loss_weight: 0.5
    
dataset:
  name: D7_1_centered
  
model:
  num_input_views: 4
```

### Phase 3: 검증 (우선순위 P2)

**검증 항목**:
1. FG Coverage: 5% 이상인지 확인
2. Center Offset: 15px 이내인지 확인
3. Ray Error: ~0° 유지 확인
4. fx/dist 비율: ~203 유지 확인

**비교 실험**:
| 실험 | 데이터셋 | 설정 | 기대 결과 |
|------|----------|------|-----------|
| Baseline | D7_1 | E0_paper | PSNR ~20 |
| Centered | D7_1_centered | E0_paper | PSNR ~24+ |
| Optimal | D7_1_centered | E7_centered_optimal | PSNR ~26+ |

---

## 5. 예상 결과

### 5.1 정량적 개선

| Metric | D7_1 현재 | D7_1_centered 예상 |
|--------|-----------|-------------------|
| Val PSNR | ~20 | **24-26** |
| FG Coverage | 2.16% | **5%+** |
| Center Offset | 101px | **<15px** |
| Ghosting | 심각 | **경미/없음** |

### 5.2 정성적 개선

```
Before (D7_1):
┌─────────────────┐
│     👻👻👻      │  ← 여러 생쥐 겹침
│    (ghosting)   │
└─────────────────┘

After (D7_1_centered):
┌─────────────────┐
│       🐭        │  ← 단일 선명한 생쥐
│   (clear)       │
└─────────────────┘
```

---

## 6. 핵심 참고문헌

1. **GS-LRM**: Zhang et al., "GS-LRM: Large Reconstruction Model for 3D Gaussian Splatting", ECCV 2024
2. **LRM**: Hong et al., "LRM: Large Reconstruction Model for Single Image to 3D", ICLR 2024
3. **LGM**: 3DTopia, "Large Multi-View Gaussian Model", ECCV 2024 Oral
4. **Pose Splatter**: Goffinet et al., "Pose Splatter: 3D Gaussian for Animal Pose", NeurIPS 2025
5. **Object-Centric 2DGS**: "Object-Centric 2D Gaussian Splatting", arXiv 2501.08174
6. **Light Field Networks**: Sitzmann et al., "Light Field Networks", NeurIPS 2021
7. **Instant-NGP**: Müller et al., "Instant Neural Graphics Primitives", SIGGRAPH 2022

---

*Document Version: 1.0 | Created: 2026-01-24*
