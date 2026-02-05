# H1 진단 실험 최종 결과 (수정됨)

> **작성**: 2026-02-05
> **상태**: ✅ 완료 (버그 수정 후 재실험)

---

## 1. 배경: 이전 실험 버그

### 1.1 발견된 문제

**GS-LRM only 실험이 실제로는 E2E로 실행됨**

| 문제 | 원인 | 해결 |
|------|------|------|
| `--input_view_idx 0` 플래그 | MVDiffusion 모드 강제 활성화 | 플래그 제거 |
| `--model M5t` 자동 완성 | `defaults.py`가 MVDiffusion 체크포인트 설정 | GS-LRM만 명시 |

### 1.2 수정된 스크립트

```bash
# 기존 (버그): --input_view_idx 0 → use_mvdiffusion = True
# 수정: --input_view_idx 없음 → GS-LRM only (6-view GT 직접 입력)

CUDA_VISIBLE_DEVICES=5 $PYTHON -m mouse_extensions.scripts.inference.run_e2e_inference \
    --data_dir /home/joon/data/preprocessed/FaceLift_mouse/M5 \
    --split /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --num_frames 50 \
    --model M5t2 \
    --skip_preprocess --prefer_ema \
    --output_dir outputs/eval/h1_diagnosis_M5t2/h1a_gslrm_train
```

---

## 2. 최종 결과 (수정됨) ⭐⭐

### 2.1 M5t2 (2880 train samples, 6.0 epochs)

| 실험 | Mode | Split | PSNR ↑ | N |
|------|------|-------|--------|---|
| h1a | **GS-LRM only** | Train | 20.13 | 50 |
| h1b | **GS-LRM only** | Test | **19.58** | 50 |
| h1c | E2E (MVDiff→GS) | Train | 20.47 | 50 |
| h1d | E2E (MVDiff→GS) | Test | **19.63** | 50 |

**Gap (GS-LRM - E2E)**: Test 기준 **-0.05** (무의미)

### 2.2 M5t (1198 train samples, 20.7 epochs)

| 실험 | Mode | Split | PSNR ↑ | N |
|------|------|-------|--------|---|
| h1a | **GS-LRM only** | Train | 21.98 | 50 |
| h1b | **GS-LRM only** | Test | **20.56** | 50 |
| h1c | E2E (MVDiff→GS) | Train | 20.95 | 50 |
| h1d | E2E (MVDiff→GS) | Test | **19.15** | 50 |

**Gap (GS-LRM - E2E)**: Test 기준 **+1.41** ← ⭐ **MVDiffusion이 병목**

---

## 3. 핵심 발견 ⭐⭐⭐

### 3.1 MVDiffusion이 병목 (M5t에서 확인)

| Dataset | GS-LRM Test | E2E Test | Gap |
|---------|-------------|----------|-----|
| **M5t** (1198 train) | 20.56 | 19.15 | **+1.41** |
| **M5t2** (2880 train) | 19.58 | 19.63 | -0.05 |

**결론**: M5t (undertrained MVDiffusion)에서 **GS-LRM only >> E2E**
→ MVDiffusion의 생성 오류가 최종 품질 저하의 원인

### 3.2 데이터 양 vs Epoch 수

| Dataset | Train Samples | Epochs | MVDiff PSNR | 결과 |
|---------|---------------|--------|-------------|------|
| M5t | 1,198 | **20.7** | 낮음 | 품질 저하 |
| M5t2 | 2,880 | **6.0** | 높음 | 품질 유지 |

**결론**: **데이터 다양성 > 반복 학습**
- 더 많은 epoch 학습 (20.7 vs 6.0)보다
- 더 많은 unique 샘플 (2880 vs 1198)이 중요

### 3.3 해석 매트릭스 결과

| 패턴 | 의미 | M5t2 | M5t |
|------|------|------|-----|
| GS-LRM ≈ E2E | MVDiffusion 영향 없음 | ✅ | ❌ |
| GS-LRM >> E2E | MVDiffusion이 노이즈 | ❌ | ✅ |
| Train >> Test | 일반화 문제 | ❌ | ❌ |

---

## 4. 후속 작업 권장

### 4.1 우선순위 P1 (즉시)

| 작업 | 설명 | 상태 |
|------|------|------|
| **Cyclic MVDiffusion** | 모든 뷰에 robust하게 학습 | 🔄 GPU 4 진행 중 |
| **Temporal V1 재시작** | 중단된 학습 재개 | ⏳ 대기 |

### 4.2 우선순위 P2 (학습 완료 후)

| 작업 | 설명 |
|------|------|
| **View Ablation** | Cyclic vs Baseline 뷰별 비교 |
| **M5t MVDiffusion 추가 학습** | epoch 증가 또는 데이터 증강 |

---

## 5. 파일 위치

### 5.1 실험 결과

```
outputs/eval/
├── h1_diagnosis_M5t2/
│   ├── h1a_gslrm_train/   # GS-LRM only (train)
│   ├── h1b_gslrm_test/    # GS-LRM only (test)
│   ├── h1c_e2e_train/     # E2E (train)
│   └── h1d_e2e_test/      # E2E (test)
├── h1_diagnosis_M5t/
│   └── (동일 구조)
└── reports/
    └── h1_diagnosis_comparison.md   # 종합 비교
```

### 5.2 스크립트

```
scripts/
├── run_h1_gslrm_only.sh      # 수정된 GS-LRM only 스크립트
└── run_h1_experiments.sh     # 전체 H1 실험 스크립트
```

---

## 6. 참고: Mask 기반 PSNR 계산

```python
# GT alpha > 0.5 영역만 비교 (foreground only)
mask = (gt_alpha > 0.5).astype(np.float32)
psnr = compute_psnr(rendered * mask, gt * mask)
```

**이유**: 렌더링 결과의 배경(흰색)과 GT 배경(투명)이 다름
→ 배경 포함 시 PSNR ~5 수준으로 측정 오류

---

*FaceLift H1 Diagnosis | Corrected Results | 2026-02-05*
