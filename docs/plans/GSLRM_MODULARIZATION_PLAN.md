# gslrm.py 모듈화 계획

## 현황 분석

### 파일 크기 비교
| 파일 | 원본 (upstream) | 현재 | 증가량 |
|------|-----------------|------|--------|
| gslrm.py | 1,552 lines | 2,363 lines | +811 lines (+52%) |

### 추가된 주요 기능
1. **Mask 기반 Loss 계산**: gt_mask, rendered_alpha 활용
2. **Alpha Supervision**: MSE/BCE alpha loss
3. **향상된 시각화**: 에러 맵, mask overlay, 뷰별 비교
4. **Validation 확장**: mask IoU, per-view metrics, turntable

---

## 모듈화 대상

### 1. 즉시 추출 가능 (Low Risk)

| 함수/메서드 | 현재 위치 | 이동 대상 | 라인수 |
|-------------|-----------|-----------|--------|
| `_compute_mask_iou` | gslrm.py:457 | loss_extensions.py | ~50 |
| `_add_error_scale_annotation` | gslrm.py:678 | visualization.py | ~100 |
| `compute_pred_mask_for_visualization` | gslrm.py:2312 | visualization.py | ~50 |

### 2. 리팩토링 후 추출 (Medium Risk)

| 함수/메서드 | 현재 위치 | 이동 대상 | 비고 |
|-------------|-----------|-----------|------|
| Validation 로직 (~280 lines) | gslrm.py:2014-2300 | validation_extensions.py | 큰 함수, 분리 필요 |
| Turntable 렌더링 | gslrm.py:2100-2280 | visualization/turntable.py | 시각화 관련 |

### 3. 유지 (원본 수정 최소화)

| 항목 | 사유 |
|------|------|
| `_compute_l2_loss`, `_perceptual`, `_ssim` | 원본 메서드 확장, mask 파라미터만 추가 |
| `_compute_all_losses` | 핵심 로직, 인터페이스 유지 |
| `forward` | 모델 핵심 |

---

## Dead Code 후보

| 함수 | 라인 | 호출 횟수 | 조치 |
|------|------|-----------|------|
| `set_current_step` | 1004 | 1 (정의만) | 확인 필요 |
| `get_overview` | 1042 | 1 (정의만) | 확인 필요 |
| `save_validations` | 2298 | 1 (정의만) | 확인 필요 |
| `compute_pred_mask_for_visualization` | 2312 | 1 (정의만) | 추출 후 삭제 |

---

## 실행 계획

### Phase 1: 안전한 추출 (즉시)
1. `_compute_mask_iou` → `mouse_extensions/model/loss_extensions.py`
2. `_add_error_scale_annotation` → `mouse_extensions/visualization/`
3. Dead code 정리 (확인 후)

### Phase 2: Validation 리팩토링
1. `save_validation_results` 분리
   - 메트릭 계산 → `validation_extensions.py`
   - 시각화 → `visualization/validation_viz.py`
   - Turntable → `visualization/turntable.py`

### Phase 3: 인터페이스 정리
1. gslrm.py에서 mouse_extensions 호출로 전환
2. 원본 메서드는 wrapper로 유지 (호환성)

---

## 목표 구조

```
FaceLift/
├── gslrm/model/gslrm.py          # 원본 + 최소 수정 (~1600 lines 목표)
│
└── mouse_extensions/
    ├── model/
    │   ├── loss_extensions.py    # 손실 계산 (mask IoU 추가)
    │   ├── validation_extensions.py  # NEW: 검증 로직
    │   └── ...
    ├── visualization/
    │   ├── validation_viz.py     # NEW: 검증 시각화
    │   ├── turntable.py          # 기존 + 확장
    │   └── error_annotation.py   # NEW: 에러 맵 주석
    └── ...
```

---

## 주의사항

1. **호환성 유지**: 기존 실험 설정 작동 보장
2. **점진적 이동**: 한 번에 하나씩, 테스트 후 진행
3. **Import 정리**: 순환 참조 방지

---

*v1.0 | 2026-01-26*

---

## Phase 1 완료 (2026-01-26)

### 변경 사항

| 항목 | Before | After | 감소 |
|------|--------|-------|------|
| gslrm.py | 2,363 lines | 2,159 lines | -204 (-8.6%) |

### 추출된 함수

| 함수 | 원본 위치 | 새 위치 |
|------|-----------|---------|
| `_compute_mask_iou` | gslrm.py | → wrapper → `loss_extensions.compute_mask_iou` |
| `_add_error_scale_annotation` | gslrm.py | → wrapper → `visualization.add_error_scale_annotation` |
| `compute_pred_mask_for_visualization` | gslrm.py (standalone) | → `visualization.error_annotation` |

### 제거된 Dead Code

| 함수 | 사유 |
|------|------|
| `set_current_step` | `_archive/` 코드에서만 사용 |
| `get_overview` | `_archive/` 코드에서만 사용 |

### 신규 파일

- `mouse_extensions/visualization/error_annotation.py`

### 검증

- [x] 구문 검사 통과
- [x] 모듈 import 성공
- [ ] 학습 테스트 (pending)

