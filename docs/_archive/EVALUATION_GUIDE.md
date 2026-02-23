# FaceLift Evaluation Guide

> 실험 평가 기준, 메트릭 정의, 데이터 분할 전략

Generated: 2026-02-06

---

## 1. 평가 유형 정의

### 1.1 일반화 유형 (Generalization Types)

| 유형 | 정의 | 평가 방법 | 현재 상태 |
|------|------|-----------|-----------|
| **Novel View** | 같은 시간, 다른 카메라 각도 | Hold-out view 렌더링 | ⚠️ 부분적 |
| **Novel Time** | 같은 뷰, 미래 시간 | Temporal val/test split | ✅ Split 있음, 로깅 필요 |
| **Novel Subject** | 완전히 새로운 피사체 | Cross-session 평가 | ❌ 미구현 |

### 1.2 현재 Validation 동작

```
target_has_input=True (기본값):
┌─────────────────────────────────────────────────────────────────┐
│  3-view 실험 예시:                                              │
│  - Input: [0, 2, 4] (3개 뷰)                                    │
│  - Target: 6개 뷰에서 랜덤 샘플 (Input 포함 가능!)               │
│                                                                 │
│  결과: Same-view + Novel-view 혼합 평가                         │
│  → 순수 Novel View 일반화 측정 아님                             │
└─────────────────────────────────────────────────────────────────┘

6-view 실험:
┌─────────────────────────────────────────────────────────────────┐
│  - Input: [0, 1, 2, 3, 4, 5] (전체)                             │
│  - Target: 같은 6개 뷰                                          │
│                                                                 │
│  결과: 100% Same-view (Novel View 평가 불가)                    │
│  → 6-view 실험은 의미 없음, 삭제 권장                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 메트릭 정의

### 2.1 기본 메트릭

| 메트릭 | 단위 | 방향 | 설명 | 좋음/나쁨 기준 |
|--------|------|------|------|----------------|
| **PSNR** | dB | ↑ | Peak Signal-to-Noise Ratio | 🟢 >25 / 🔴 <15 |
| **SSIM** | 0-1 | ↑ | Structural Similarity | 🟢 >0.9 / 🔴 <0.7 |
| **LPIPS** | 0-1 | ↓ | Perceptual Distance (AlexNet) | 🟢 <0.1 / 🔴 >0.3 |
| **Mask IoU** | 0-1 | ↑ | Foreground Mask Overlap | 🟢 >0.85 / 🔴 <0.7 |

> **See also**: [METRICS_PROTOCOL](../theory/METRICS_PROTOCOL.md) — White-BG 계산 프로토콜, PoseSplatter 비교 분석, FG-only vs Full-image PSNR 차이

### 2.2 Temporal 메트릭 (추가 권장)

| 메트릭 | 정의 | 용도 |
|--------|------|------|
| **Temporal Smoothness** | `||∇_t(rgb)||` | 연속 프레임 일관성 |
| **Velocity MAE** | 키포인트 속도 오차 | 동작 정확도 |
| **Cross-Temporal Consistency** | 동일 포즈 다른 시점 RGB 일관성 | 시간 안정성 |

---

## 3. 데이터 분할 전략

### 3.1 M5t2 Temporal Split (권장)

```
전체: 3600 샘플 (frame_jump=5, ~10분 영상)

├── Train (80%): 2880 샘플 (프레임 0-2880)
│   └── 학습용, 데이터 증강 적용
│
├── Val (10%): 360 샘플 (프레임 2881-3240)
│   └── Hyperparameter 튜닝, Early stopping
│   └── 학습 중 100 step마다 평가
│
└── Test (10%): 360 샘플 (프레임 3241-3600)
    └── 최종 평가 (한 번만 사용!)
```

### 3.2 Split 파일 위치

```bash
/home/joon/data/preprocessed/FaceLift_mouse/M5/
├── data_mouse_t2_train.txt  # 2880 lines
├── data_mouse_t2_val.txt    # 360 lines
└── data_mouse_t2_test.txt   # 360 lines
```

### 3.3 Anti-Pattern (피해야 할 것)

| 실수 | 결과 | 해결책 |
|------|------|--------|
| 랜덤 split 사용 | PSNR ~20% 과대평가 | Temporal split 사용 |
| Test set 반복 사용 | Test set 오염 | Val에서만 튜닝 |
| 6-view로 Novel View 평가 | Hold-out 0개 | 5-view 이하 사용 |
| 인접 프레임 버퍼 없음 | 암묵적 leakage | Buffer zone 50-100 프레임 |

---

## 4. View Ablation 실험 가이드

### 4.1 유효한 실험 설정

| 실험 | Input Views | Hold-out Views | Novel View 평가 |
|------|-------------|----------------|-----------------|
| 1-view | 1 | 5 | ✅ 가능 |
| 2-view | 2 | 4 | ✅ 가능 |
| 3-view | 3 | 3 | ✅ 가능 |
| 4-view | 4 | 2 | ✅ 가능 |
| 5-view | 5 | 1 | ✅ 가능 (최소) |
| **6-view** | 6 | **0** | ❌ **불가능** |

### 4.2 현재 결과 (2026-02-06)

| 실험 | Best PSNR (dB) | Epochs | 상태 |
|------|----------------|--------|------|
| 1-view | 11.23 | 7.5 | ✅ |
| 2-view | 17.76 | 4.3 | ✅ |
| 3-view | 20.19 | 2.9 | ✅ |
| 5-view | **23.27** | 3.1 | ✅ **Best** |
| 6-view | 0.00 | 0 | ❌ 삭제 예정 |

### 4.3 권장 Baseline

- **5-view**: 최대 input views (Novel View 평가 가능)
- **3-view**: 적은 뷰에서도 좋은 성능 (MV-Diffusion 대안)

---

## 5. 로깅 권장사항

### 5.1 WandB 로깅 구조

```python
# 학습 중 (매 step)
log_dict = {
    "train/loss": loss,
    "train/psnr": psnr,
    "train/l2_loss": l2_loss,
}

# Validation (매 100 step)
log_dict = {
    "val/psnr": val_psnr,          # Same-view 포함 (현재)
    "val/ssim": val_ssim,
    "val/lpips": val_lpips,        # 추가 권장
}

# Temporal Validation (추가 권장)
log_dict = {
    "temporal_val/psnr": temporal_psnr,  # 미래 시간 평가
    "temporal_val/lpips": temporal_lpips,
}

# 최종 평가 (test set, 1회)
log_dict = {
    "test/psnr": test_psnr,
    "test/ssim": test_ssim,
    "test/lpips": test_lpips,
}
```

### 5.2 평가 빈도

| 평가 유형 | 빈도 | 용도 |
|-----------|------|------|
| Train metrics | 매 step | 학습 진행 확인 |
| Val (same-view) | 100 step | 과적합 감지 |
| Val (temporal) | 500 step | 시간 일반화 확인 |
| Test | 학습 완료 후 1회 | 최종 성능 보고 |

---

## 6. 실험 실행 체크리스트

### 6.1 실험 전

- [ ] Split 파일 확인 (`data_mouse_t2_{train,val,test}.txt`)
- [ ] `num_input_views` 확인 (6 제외)
- [ ] `target_has_input` 확인 (Novel View 평가 시 False 권장)
- [ ] Pretrained checkpoint 경로 확인
- [ ] WandB project 설정 (`FaceLift-Mouse`)

### 6.2 실험 중

- [ ] val/loss 모니터링 (1.0이면 버그)
- [ ] best_psnr.json 생성 확인
- [ ] Turntable 렌더링 확인

### 6.3 실험 후

- [ ] Test set 평가 (1회만!)
- [ ] 레포트 생성 (`generate_experiment_report.py`)
- [ ] GT vs Pred 시각화 생성
- [ ] Memory/docs 업데이트

---

## 7. 다음 단계 (TODO)

### P0 (즉시)

- [ ] 6-view 실험 디렉토리 삭제
- [ ] 5-view를 baseline으로 문서화

### P1 (1일 내)

- [ ] Temporal val 로깅 추가 (validator.py 수정)
- [ ] LPIPS 로깅 활성화

### P2 (1주 내)

- [ ] Novel View 전용 평가 모드 (`target_has_input=False`)
- [ ] Temporal smoothness 메트릭 추가
- [ ] 외삽 평가 파이프라인 구현

---

## 참고 자료

- [Temporal Split Best Practices](https://arxiv.org/html/2512.06932v1)
- [3DGS Evaluation Protocol](https://github.com/graphdeco-inria/gaussian-splatting)
- [Pose Splatter Paper](https://arxiv.org/abs/2505.18342)

---

*FaceLift Project | Evaluation Guide v1.0 | 2026-02-06*
