# 241208: 카메라 파라미터 분석 및 버그 수정

**날짜:** 2024-12-08
**주제:** MAMMAL → FaceLift 카메라 변환, 파라미터 버그 수정
**상태:** ✅ 완료

---

## 핵심 요약

| 항목 | 내용 |
|------|------|
| **문제** | MAMMAL 카메라 데이터 로딩 버그 (LIST vs DICT) |
| **원인** | 카메라 데이터가 LIST인데 DICT로 처리 |
| **해결** | load_camera_params() 수정, 거리 정규화 추가 |
| **결과** | 카메라 거리 2.7, FOV 50°로 정규화 |

---

## 1. 카메라 파라미터 정의

### 좌표계 시각화
```
                    Z (up)
                    |
                    |  elevation (고도각)
                    | /
                    |/_____ Y
                   /\
                  /  \
                 /    azimuth (방위각)
                X

        카메라 ●────────────→ 원점 (피사체 위치)
              └─ distance (거리)
```

### 파라미터 계산
```python
distance = sqrt(x² + y² + z²)
elevation = arcsin(z / distance)
azimuth = arctan2(y, x)
fx = (image_width / 2) / tan(FOV / 2)  # FOV 50° → fx ≈ 549
```

---

## 2. MAMMAL vs FaceLift 비교

| 파라미터 | MAMMAL 원본 | FaceLift 표준 | 정규화 후 |
|---------|-------------|---------------|----------|
| 거리 | 246~414 | 2.7 | ✅ 2.7 |
| 고도각 | 11~31° | 20° (기본) | 원본 유지 |
| 방위각 | 불규칙 | turntable | 원본 유지 |
| fx | 1556~1637 | 549 | ✅ 549 |

### MAMMAL 원본 카메라 (6대)
```
Camera 0: dist=246.1, elev=14.9°, azim=-147.0°, FOV=40.4°
Camera 1: dist=414.5, elev=20.6°, azim=34.0°,   FOV=43.6°
Camera 2: dist=363.7, elev=11.3°, azim=86.1°,   FOV=41.1°
Camera 3: dist=340.0, elev=10.7°, azim=-11.3°,  FOV=39.9°
Camera 4: dist=318.3, elev=26.5°, azim=144.0°,  FOV=43.2°
Camera 5: dist=305.7, elev=30.8°, azim=-64.1°,  FOV=41.1°
```

---

## 3. 발견된 버그 및 수정

### 버그: 카메라 데이터 로딩 오류
```python
# Before (버그) - LIST를 DICT처럼 처리
cam_dict = pickle.load(f)
for i in range(num_views):
    if i in cam_dict:  # ❌ LIST에서 'in'은 값 검색!
        cameras.append(cam_dict[i])

# After (수정)
cam_data = pickle.load(f)
if isinstance(cam_data, list):
    for i in range(min(num_views, len(cam_data))):
        cameras.append(cam_data[i])
```

### 추가 수정: 거리 정규화
```python
# convert_to_facelift_format() 수정
avg_distance = np.mean(distances)
scale_factor = target_distance / avg_distance  # 2.7 / 330 ≈ 0.008
T_normalized = T * scale_factor
```

---

## 4. 스케일 변환 원리

### 3D 장면의 스케일 불변성
```
핀홀 카메라: u = fx * (X / Z) + cx
스케일 변환: P' = s * P, fx' = fx / s

u' = (fx/s) * (sX / sZ) + cx = fx * (X/Z) + cx = u
→ 동일한 이미지 유지!
```

### 결론
- 거리와 fx를 같은 비율로 조정하면 이미지 불변
- 회전(R)은 유지, Translation(T)만 스케일링
- 정보 손실 없음 (가역적 변환)

---

## 5. 데이터 파이프라인

```
MAMMAL 원본 (new_cam.pkl)
         ↓
평균 거리 계산 (avg ≈ 330)
         ↓
스케일 팩터 (2.7 / 330 ≈ 0.008)
         ↓
Translation 정규화 (T × scale)
         ↓
FaceLift Intrinsics (fx=fy=549, cx=cy=256)
         ↓
opencv_cameras.json 저장
```

---

## 관련 파일
- `scripts/process_mouse_data.py` - 데이터 전처리
- `configs/mouse_config.yaml` - 학습 설정

---

*통합 문서: 241208_camera_parameters_analysis.md + 241208_mouse_facelift_camera_fix.md*
