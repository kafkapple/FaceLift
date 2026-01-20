# FaceLift 전처리 변천사: 3단계 축약

> 핵심 실험들 위주로 가설, 실험방법, 결과를 3단계로 정리

---

## Phase 1: 문제 발견 (D1-D4)

### 가설
"Multi-view 일관성을 위해 3D triangulation 기반 center estimation이 필요하다"

### 실험
| 버전 | 방법 | 결과 |
|------|------|------|
| D1 | PP-centered shift (뷰별 독립) | 뷰 간 불일치 |
| D2 | Bbox center | PSNR ~7 |
| D3 | Triangulation + 정규화 없음 | **PSNR 2.11** (실패) |
| D3n | Triangulation + 정규화 | PSNR 26.34 |
| D4 | Triangulation + PP=256 강제 | **PSNR 27.87** (Ghosting) |

### 핵심 발견
1. **카메라 정규화 필수**: D3 vs D3n에서 +24 PSNR 향상
2. **PP=256 강제 문제**: D4에서 높은 PSNR이지만 Ghosting 발생

### 결론
정규화는 필수이나, PP=256 강제는 기하학 오류 유발

---

## Phase 2: 원인 분석 (D4 버그 발견)

### 가설
"D4의 Ghosting은 PP를 256으로 강제 기록하면서 발생한 ray 방향 오류가 원인이다"

### 실험: PP 오차 분석

실제 crop 후 PP vs 기록된 PP:
- 실제: (312, 287)
- 기록: (256, 256)
- 오차: 56px, 31px

Ray direction error:
- theta = arctan(56 / 549) = 5.8도
- 양 축 결합: 11-13도

### D6 시리즈 검증

| 버전 | 방법 | PP | PSNR | Ghosting |
|------|------|-----|------|----------|
| D6-1 | No crop (resize) | 정확 | 23.34 | Reduced |
| D6-3 | Triangulation + 정확 PP | 정확 (가변) | 21.18 | Reduced |

### 핵심 발견
1. PP 정확하면 Ghosting 감소
2. 그러나 PSNR이 D4보다 낮음
3. 원인: PP가 뷰마다 다름 (std ~52px) -> pretrained 분포 불일치

### 결론
기하학 정확성 vs Pretrained 호환성 딜레마 존재

---

## Phase 3: 최종 해결 (D7)

### 가설
"PP를 이미지 중앙으로 shift하면 기하학 정확성과 pretrained 호환성을 동시에 달성할 수 있다"

### 방법: PP-Centered Shift

기존 D4:
- 쥐를 이미지 중앙에 배치
- PP=256으로 강제 기록 (틀림!)

D7:
- PP를 이미지 중앙으로 shift
- PP=256 실제 달성 (맞음!)

수학적 원리:
이미지를 delta만큼 shift하고 PP도 delta만큼 shift하면 기하학적으로 동등
결과: PP가 정확히 (256, 256)에 위치

### D7 특성

| Parameter | 값 | 의미 |
|-----------|-----|------|
| fx, fy | 549, 549 | 정규화됨 |
| cx, cy | 256, 256 | 정확히 달성 |
| 쥐 위치 | 중앙에서 약 30px | 허용 가능 |
| Ray error | 0도 | 기하학 정확 |

### 예상 결과

| 비교 | D4 | D7 |
|------|-----|-----|
| PSNR | 27.87 | >= 27 (예상) |
| Ghosting | Yes | **No (예상)** |
| 기하학 | 틀림 | 정확 |
| PP | 강제 256 | 실제 256 |

---

## 요약 다이어그램

```
Phase 1: 문제 발견
D1-D3 (실패) -> D3n (성공, 정규화) -> D4 (PSNR 높음, Ghosting)
                                          |
Phase 2: 원인 분석                        v
D6 시리즈 -> PP 정확하면 Ghosting 감소, 그러나 PSNR 낮음
            PP 불일치가 원인
                    |
Phase 3: 해결      v
D7 -> PP-centered shift = 기하학 정확 + PP=256 = 최적 해결책
```

---

## 핵심 교훈 3가지

1. **카메라 정규화는 필수**
   - 미적용 시 PSNR 2-3으로 학습 실패

2. **기하학 오류 > 분포 불일치**
   - 기하학 오류는 복구 불가
   - 분포 불일치는 finetuning으로 적응 가능

3. **PSNR != 품질**
   - 높은 PSNR이 Ghosting-free를 보장하지 않음
   - 반드시 시각적 검증 필요

---

*Generated: 2026-01-20*
