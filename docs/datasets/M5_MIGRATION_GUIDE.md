# M5 Dataset Migration Guide

> M3 → M5: Re-centering Paradigm | 2026-01-28

---

## 1. Why M5?

### M3의 Per-view Distance Normalization 버그

M3까지의 카메라 정규화는 **각 카메라를 개별적으로** `dist=2.7`로 강제했음.

```
M3 (버그):
  Camera 0: dist=3.2 → scale to 2.7 (×0.84)
  Camera 1: dist=2.5 → scale to 2.7 (×1.08)
  Camera 2: dist=3.8 → scale to 2.7 (×0.71)
  → 각 카메라가 다른 scale factor → rig 형상 왜곡 최대 34.7%
```

**문제의 본질**: Multi-view 카메라 rig는 상대적 위치 관계가 핵심. 개별 정규화는 이 관계를 파괴하여 3D 재구성 품질을 저하시킴.

### M5의 해결: Uniform Scaling

```
M5 (수정):
  1. 모든 카메라 centroid → origin으로 re-center
  2. 평균 거리 계산: mean_dist = (3.2 + 2.5 + 3.8) / 3 = 3.17
  3. 균일 scale: all cameras × (2.7 / 3.17) = ×0.852
  → 모든 카메라에 동일 scale → rig 형상 완벽 보존
```

---

## 2. M5 Version System

| 버전 | Paradigm | 변환 | Zoom | 권장 |
|------|----------|------|------|------|
| **M5** | RECENTERED_AFFINE | Affine | 없음 | ✅ 기본 |
| **M5h** | RECENTERED_HOMOGRAPHY | Homography | 없음 | 정밀 보정 |
| **M5h_1** | RECENTERED_HOMOGRAPHY | Homo + Global Zoom | Global | 안정적 |
| **M5h_2** | RECENTERED_HOMOGRAPHY | Homo + Per-sample Zoom | Per-sample | ⭐ 권장 |

### Paradigm 설명

- **RECENTERED_AFFINE**: Centroid re-centering + uniform normalization + affine 변환
- **RECENTERED_HOMOGRAPHY**: 위와 동일하되, skew 보정을 위해 homography 변환 사용

### 모든 버전 공통

| 항목 | 값 |
|------|-----|
| PP (cx, cy) | 256 |
| fx | 549 |
| 해상도 | 512×512 |
| Re-centering | ✅ (centroid → origin) |
| Uniform scaling | ✅ (mean dist = 2.7) |

---

## 3. Key Changes from M3

| 항목 | M3 | M5 | 영향 |
|------|-----|-----|------|
| **Distance norm** | Per-view (개별) | Uniform (균일) | rig 형상 보존 |
| **Re-centering** | 없음 | Centroid → origin | 좌표계 일관성 |
| **Rig distortion** | 최대 34.7% | 0% | 3D 품질 향상 |
| **Preset base** | D7.1 (M1) / D8 (M2) | RECENTERED 계열 | 새 preset 체계 |

### 코드 레벨 변경

```python
# M3: Per-view normalization (버그)
for cam in cameras:
    cam.translation *= (2.7 / cam.translation.norm())  # 개별 scale

# M5: Uniform normalization (수정)
centroid = cameras.mean_position()
cameras.recenter(centroid)  # centroid → origin
mean_dist = cameras.mean_distance()
uniform_scale = 2.7 / mean_dist
cameras.scale_all(uniform_scale)  # 동일 scale 적용
```

---

## 4. Data Splits

**데이터 위치**: `/home/joon/data/preprocessed/FaceLift_mouse/M5/`

### FaceLift Style (90/10 Random)

| 파일 | 비율 | 용도 |
|------|------|------|
| `data_mouse_train.txt` | 90% | 학습 |
| `data_mouse_val.txt` | 10% | 검증 |

**특징**: Random split. 동일 시퀀스의 프레임이 train/val에 모두 포함될 수 있음.

### Pose Splatter Style (80/10/10 Temporal)

| 파일 | 비율 | 용도 |
|------|------|------|
| `data_mouse_train_ps.txt` | 80% | 학습 |
| `data_mouse_val_ps.txt` | 10% | 검증 |
| `data_mouse_test_ps.txt` | 10% | 테스트 |

**특징**: Temporal split. 시간 순서 기반 분할로 temporal leakage 방지. Pose Splatter 프로젝트와 동일 기준.

---

## 5. Config Examples

### M5 기본 학습

```yaml
dataset:
  preset: M5
  dataset_path: /home/joon/data/preprocessed/FaceLift_mouse/M5
  train_list: data_mouse_train.txt
  val_list: data_mouse_val.txt
  resolution: 512
  num_views: 5

camera:
  fx: 549
  cx: 256
  cy: 256
```

### M5h_2 (Per-sample Zoom, 권장)

```yaml
dataset:
  preset: M5h_2
  dataset_path: /home/joon/data/preprocessed/FaceLift_mouse/M5h_2
  train_list: data_mouse_train.txt
  val_list: data_mouse_val.txt

zoom:
  zoom_center_mode: "image"  # 반드시 명시 (기본값 object-centered 주의)
  zoom_range: [1.0, 1.8]     # [1.0, 2.5]는 클리핑 위험
```

---

## 6. Verification

### 전처리 후 검증 체크리스트

```bash
cd /home/joon/dev/FaceLift

# 1. 데이터 수 확인
wc -l /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_*.txt

# 2. 카메라 파라미터 검증
python -m mouse_extensions.preprocessing.verify_dataset \
    --dataset_path /home/joon/data/preprocessed/FaceLift_mouse/M5

# 3. 확인 항목:
#    - fx ≈ 549 (pretrained 호환)
#    - cx, cy ≈ 256
#    - mean translation distance ≈ 2.7
#    - rig distortion = 0% (uniform scaling 검증)
#    - per-view distance std가 작지 않음 (개별 정규화 아닌지 확인)
```

### M3 vs M5 비교 검증

```bash
# Per-view distance 분산 비교
# M3: std ≈ 0 (각 카메라 개별 2.7 → 분산 없음 = 버그)
# M5: std > 0 (카메라별 거리 차이 유지 = 정상)
```

**핵심 지표**: Per-view translation distance의 std가 0에 가까우면 per-view normalization 버그가 남아있는 것.

---

*FaceLift M5 Migration Guide | 2026-01-28*
