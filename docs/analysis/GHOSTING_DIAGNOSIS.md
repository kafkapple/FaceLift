# GS-LRM Mouse Ghosting 종합 진단 리포트

> Created: 2026-01-24 | Author: Claude Code

## Executive Summary

모든 전처리 방식에서 발생하는 심각한 ghosting 현상의 **근본 원인**은 ray error가 아닌 **object scale 및 centering 불일치**입니다.

| 데이터셋 | PSNR | FG Coverage | Object Centering | Ghosting |
|----------|------|-------------|------------------|----------|
| D3_normalized | **26.5** | **5.28%** | **8px** (양호) | 경미 |
| D7_1 | 19.8 | 2.16% | 101px (불량) | 심각 |
| D8 | 19.1 | 2.0% | ~100px | 심각 |

## 1. 문제 정의

### 관찰된 현상
- 모든 turntable 렌더링에서 여러 개의 생쥐가 겹쳐 보임
- 크기가 조금씩 다른 생쥐들이 같은 위치에 중첩
- 이론적 ray error ~0°임에도 불구하고 ghosting 발생

### 원인 가설 (조사 전)
1. ❌ 카메라 calibration 오류
2. ❌ Temporal 비동기화
3. ❌ Principal point 계산 오류
4. ✅ **Object scale/centering 불일치**

## 2. 조사 결과

### 2.1 Temporal Synchronization (검증 완료)
- 6대 카메라 모두 18,000 frames @ 100fps
- 하드웨어 동기화 확인됨
- **결론: Temporal 문제 아님**

### 2.2 Camera Parameters (검증 완료)
| Dataset | fx | cx | cy | dist | Ray Error |
|---------|-----|-----|-----|------|-----------|
| D7_1 | 549 | 256 | 256 | 2.7 | ~0° |
| D8 | 549 | 256 | 256 | 2.7 | ~0° |

- Pretrained 분포(fx=549, dist=2.7)와 일치
- **결론: Camera parameter 문제 아님**

### 2.3 Object Scale 및 Centering (핵심 원인)

#### 비교 분석
| Metric | D3_normalized | D7_1 | 차이 |
|--------|---------------|------|------|
| FG Coverage | 5.28% | 2.16% | **2.5x** |
| BBox Area | ~36,000 px² | ~14,000 px² | **2.5x** |
| Center Distance | 8 px | 101 px | **12.6x** |

#### GS-LRM Pretrained 분포
- Objaverse 데이터: 중심에 위치한 큰 객체
- Object가 이미지의 ~40-60% 차지
- **Mouse 데이터는 이 분포에서 크게 벗어남**

## 3. 근본 원인 분석

### 3.1 Plucker Ray Encoding과 Object Scale
```
Plucker = (direction, moment) = (d, o × d)
```

- moment 벡터 `m = o × d`는 **카메라 원점 기준**으로 계산
- 작은 객체 → 광선들이 좁은 영역에 집중 → moment 변화 미미
- Transformer가 뷰 간 차이를 구분하기 어려움

### 3.2 Self-Attention의 한계
- GS-LRM은 **명시적 Multi-view Consistency Loss 없음**
- 뷰 간 일관성이 오직 Self-Attention에 의존
- 작은/off-center 객체 → attention 신호 약함

### 3.3 Per-View Gaussian 독립 생성
- Pixel-aligned Gaussians가 **뷰별로 독립 생성**
- 동일 3D 점이 여러 뷰에서 다른 Gaussian으로 표현 가능
- → Ghosting artifact

## 4. 해결 방안 (우선순위순)

### P0: Object-Centered Zoom Preprocessing (권장)
D3_normalized 방식 + 기하학적 정확성 결합

```python
# D8.3_centered_zoom 프리셋 제안
{
    paradigm: object_centered_zoom,
    target_fg_coverage: 0.05,      # D3와 동일한 5%
    target_center_offset: 10,       # 최대 10px 오프셋
    zoom_range: [1.0, 2.0],         # 적응적 zoom
    fx_normalization: True,         # fx=549 유지
    translation_normalization: True  # dist 비례 조정
}
```

**핵심**: 
- Object 중심으로 crop → 큰 foreground
- 이후 fx, translation 정규화 → pretrained 호환

### P1: D3_normalized 활용
- 현재 가장 높은 PSNR (26.5)
- 단, fy 변동 있음 (537-559)
- 즉시 적용 가능

### P2: D8 Zoom Factor 증가
- 현재 D8.1: 1.3x zoom → 3.67% coverage
- D8.2: 1.48x zoom → 4.4% coverage
- **필요: ~1.8-2.0x zoom** for 5% coverage

### P3: Multi-view Consistency Loss 추가
- Epipolar constraint loss
- Cross-view Gaussian correspondence loss
- 모델 수정 필요 (장기)

## 5. 검증 계획

### Phase 1: 데이터 검증 (즉시)
1. D3_normalized로 turntable 생성하여 ghosting 수준 비교
2. D8.2 zoom factor를 1.8x로 증가 테스트

### Phase 2: 새 프리셋 구현 (단기)
1. D8.3_centered_zoom 프리셋 구현
2. Object 중심 crop + 정규화 파이프라인

### Phase 3: 모델 개선 (장기)
1. Multi-view consistency loss 추가
2. Cross-view attention layer 추가

## 6. 핵심 교훈

1. **Ray Error ≠ Ghosting 원인**: 기하학적 정확성만으로는 부족
2. **Pretrained 분포 일치 중요**: scale, centering 모두 맞아야 함
3. **FG Coverage 최소 5%**: D3의 5.28%가 기준
4. **Object Centering 필수**: 중심에서 10px 이내

---

*Report generated: 2026-01-24*
