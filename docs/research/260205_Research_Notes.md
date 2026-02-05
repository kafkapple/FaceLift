# 260205 Research Notes

> **Date**: 2026-02-05
> **Topics**: H1 진단 실험, 핵심 가설, Master Plan, Deformation Fix

---

## 1. H1 진단 실험 최종 결과 ⭐⭐⭐

### 1.1 버그 수정

**발견된 문제**: GS-LRM only 실험이 실제로는 E2E로 실행됨
| 문제 | 원인 | 해결 |
|------|------|------|
| `--input_view_idx 0` 플래그 | MVDiffusion 모드 강제 활성화 | 플래그 제거 |
| `--model M5t` 자동 완성 | defaults.py가 MVDiffusion 체크포인트 설정 | GS-LRM만 명시 |

### 1.2 최종 결과 (수정됨)

**M5t2 (2880 train, 6.0 epochs)**:
| 실험 | Mode | Split | PSNR |
|------|------|-------|------|
| h1a | GS-LRM only | Train | 20.13 |
| h1b | GS-LRM only | Test | 19.58 |
| h1c | E2E | Train | 20.47 |
| h1d | E2E | Test | 19.63 |

**M5t (1198 train, 20.7 epochs)**:
| 실험 | Mode | Split | PSNR |
|------|------|-------|------|
| h1a | GS-LRM only | Train | 21.98 |
| h1b | GS-LRM only | Test | 20.56 |
| h1c | E2E | Train | 20.95 |
| h1d | E2E | Test | 19.15 |

### 1.3 핵심 발견

**MVDiffusion이 병목 (M5t에서 확인)**:
| Dataset | GS-LRM Test | E2E Test | Gap |
|---------|-------------|----------|-----|
| M5t (1198) | 20.56 | 19.15 | **+1.41** |
| M5t2 (2880) | 19.58 | 19.63 | -0.05 |

**결론**:
1. M5t (undertrained MVDiffusion)에서 **GS-LRM only >> E2E**
2. **데이터 다양성 > 반복 학습** (2880 × 6ep > 1198 × 20ep)

---

## 2. 데이터셋 분할 현황

### Temporal Split 비교

| 데이터셋 | Split | 특징 | Leakage |
|----------|-------|------|---------|
| M5 | 8:1:1 random | 랜덤 셔플 | ⚠️ 우려 |
| M5t | 1:1:1 temporal | 시간순 | ✅ 안전 |
| **M5t2** | 8:1:1 temporal | 시간순, train 최대화 | ✅ 안전 ← **현재 실험 중심** |

### 파일 위치
```
~/data/preprocessed/FaceLift_mouse/M5/
├── data_mouse_train.txt      # M5 random
├── data_mouse_t_train.txt    # M5t temporal 1:1:1
└── data_mouse_t2_train.txt   # M5t2 temporal 8:1:1 ← 현재 사용
```

---

## 3. 뷰 사용 상세

### GS-LRM 학습/평가 뷰
| 구분 | 뷰 개수 | 용도 |
|------|---------|------|
| Input Views | 4 | Forward 입력 |
| Target Views | 6 | 렌더링 대상 (Input 포함) |
| Supervision | 6 | Loss 계산 |

### 6-view Validation 주의
- `target_has_input=True` (기본): same-view 비교 (일반화 평가 아님)
- `target_has_input=False`: target 빈 리스트 → validation 불가
- **6-view는 별도 test set 평가 필수**

---

## 4. 시스템 아키텍처

```
[Input]         [MVDiffusion]         [GS-LRM]        [Deformation]
Single View  →  Multi-view Gen  →  3D Gaussians  →  Temporal Reg
(1 view)        (1→6 views)          (4 input)       (t, t+1 pair)

책임:
- 단일 뷰       - 뷰 일관성         - 3D 품질       - 시간 일관성
                - 프롬프트 영향      - 카메라 정규화  - 드리프트 방지
```

---

## 5. 모듈별 가설

### MVDiffusion
| ID | 가설 | 상태 |
|----|------|------|
| D1 | CFG dropout이 뷰 일관성에 영향 | 🔄 M5t2_cfgr 실험 중 |
| D2 | Full attention > sparse | ✅ M5t2_consistent 완료 |
| D3 | 학습 스텝 부족 (10k → 20k) | 📋 계획 |

### GS-LRM
| ID | 가설 | 상태 |
|----|------|------|
| G1 | 4-view로 충분 (6-view 불필요) | 🔄 View ablation 중 |
| G2 | Per-view norm보다 Batch norm | 📋 계획 |

---

## 6. Deformation MLP Fix Plan

### 문제: Autoregressive Drift
```
Frame N = Frame 0 + Sum(Delta_1..N)
→ 누적 오차 증가 → "녹아내림" 현상
```

### 해결 방향
1. **Per-frame + Regularization**: 각 프레임 독립 재구성 + ARAP loss
2. **Flow Alignment**: Optical flow로 2D-3D motion 정렬
3. **Sliding Window**: Window 단위 joint optimization

---

## 7. 후속 작업 우선순위

### P1 (즉시)
- Cyclic MVDiffusion 학습 완료 대기 (GPU 4)
- View Ablation Training (GPU 5, 6, 7)

### P2 (학습 완료 후)
- View Ablation 비교 분석
- Cyclic vs Baseline 뷰별 비교

---

## 8. 참고: 버그 수정 이력

| 버그 | 원인 | 해결 |
|------|------|------|
| val/loss=1.0 | MetricsComputer.compute_per_view_metrics 누락 | 메서드 추가 |
| Turntable 중복 라벨 | train/val show_overlay 불일치 | config 기반 통일 |

---

*FaceLift Research Notes | 2026-02-05*
