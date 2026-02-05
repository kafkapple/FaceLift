# FaceLift 핵심 가설 및 실험 설계

> **목적**: 모든 핵심 가설과 검증 실험을 체계적으로 정리
> **작성**: 2026-02-05
> **상태**: 실험 진행 중

---

## 1. 데이터셋 분할 현황

### 1.1 Temporal Split 비교

| 데이터셋 | Train:Val:Test | 특징 | Data Leakage |
|----------|----------------|------|--------------|
| **M5** | 8:1:1 (random) | 랜덤 셔플 | ⚠️ 우려 (시간순 연속 프레임이 train/test에 분리될 수 있음) |
| **M5t** | 1:1:1 (temporal) | 시간순 분할 (앞-중간-뒤) | ✅ 안전 |
| **M5t2** | 8:1:1 (temporal) | 시간순 분할, train 최대화 | ✅ 안전 ← **현재 실험 중심** |

### 1.2 M5t vs M5t2 비교 실험 의미

| 비교 | 목적 | 예상 결과 |
|------|------|----------|
| M5t (1:1:1) vs M5t2 (8:1:1) | 학습 데이터 양의 영향 | M5t2 > M5t (더 많은 학습 데이터) |
| M5 (random) vs M5t2 (temporal) | Data leakage 영향 | M5 과적합 가능, M5t2 더 일반화 |

### 1.3 파일 위치

```
~/data/preprocessed/FaceLift_mouse/M5/
├── data_mouse_train.txt      # M5 random split (8:1:1)
├── data_mouse_val.txt
├── data_mouse_test.txt
├── data_mouse_t_train.txt    # M5t temporal (1:1:1)
├── data_mouse_t_val.txt
├── data_mouse_t_test.txt
├── data_mouse_t2_train.txt   # M5t2 temporal (8:1:1) ← 현재 사용
├── data_mouse_t2_val.txt
└── data_mouse_t2_test.txt
```

---

## 2. 뷰 사용 상세

### 2.1 GS-LRM 학습/평가 뷰

| 구분 | 뷰 개수 | 인덱스 | 용도 |
|------|---------|--------|------|
| **Input Views** | 4 | config.model.num_input_views | Forward에 입력되는 뷰 |
| **Target Views** | 6 | config.model.num_views | 렌더링 대상 (Input 포함) |
| **Supervision** | 6 | 모든 뷰 | Loss 계산 대상 |

```python
# gslrm/data/dataset.py
# 학습 시: 6개 뷰 중 4개를 랜덤 선택 (또는 설정에 따라 고정)
input_indices = random.sample(range(6), num_input_views)
# 나머지 2개는 novel view로 평가
```

### 2.2 MVDiffusion 학습/평가 뷰

| 구분 | 뷰 개수 | 설명 |
|------|---------|------|
| **Input** | 1 | 조건 이미지 (front view) |
| **Output** | 6 | 생성된 multi-view 이미지 |
| **Supervision** | 6 | GT 이미지와 비교 |

### 2.3 문서 업데이트 위치

- **GS-LRM 뷰 설정**: `docs/experiments/EXPERIMENT_CONFIG_GUIDE.md`
- **데이터셋 상세**: `docs/datasets/M5_SERIES_SPEC.md`
- **MVDiffusion 설정**: `configs/mouse/mouse_mvdiffusion_*.yaml`

---

## 3. 핵심 가설 및 실험 설계

### 가설 H1: 최종 품질 오류 원인 진단 ⭐⭐ (P1)

**질문**: GS-LRM 최종 재구성 품질 오류의 주요 원인은?
- (A) MVDiffusion 단계의 생성 오류?
- (B) Test set 처음 보는 것 (일반화 문제)?

**검증 방법**: 2×2 Factorial Design

| 실험 | Pipeline | Dataset | 기대 결과 |
|------|----------|---------|----------|
| **H1-1** | E2E (MV→GS) | Train | MV 품질 확인 (상한선) |
| **H1-2** | E2E (MV→GS) | Test | MV 일반화 확인 |
| **H1-3** | GS-LRM only | Train | GS 재구성 품질 (상한선) |
| **H1-4** | GS-LRM only | Test | **GS 일반화 확인** |

**해석 매트릭스**:

| 결과 패턴 | 의미 | 결론 |
|----------|------|------|
| H1-1 ≈ H1-3 >> H1-2 ≈ H1-4 | Train/Test 차이가 지배적 | 일반화 문제 |
| H1-1 >> H1-3, H1-2 >> H1-4 | E2E가 항상 좋음 | MVDiffusion이 도움 |
| H1-1 << H1-3, H1-2 << H1-4 | E2E가 항상 나쁨 | MVDiffusion이 노이즈 추가 |
| H1-3 ≈ H1-4 >> H1-1 ≈ H1-2 | GS-LRM 단독이 좋음 | GT 입력이 핵심 |

### 가설 H2: 8-layer MLP Deformation 효과 ⭐⭐ (P1)

**질문**: Deformation Network가 temporal consistency를 개선하는가?

**검증**: V1 vs V2 비교 (완료)

| 결과 | V1 (Autoregressive) | V2 (Per-frame Ref) |
|------|---------------------|-------------------|
| Drift (50 frames) | 0.78 | 0.00 |
| Pattern | 누적 | Bounded |

✅ **결론**: V2 방식이 drift 방지에 효과적

### 가설 H3: 카메라 정규화 방식 ⭐ (P2)

**질문**: Batch Uniform vs Per-view vs No Norm 중 어떤 것이 최적?

| 실험 | Preset | 정규화 |
|------|--------|--------|
| H3-1 | M5 | Batch Uniform ← 현재 기준선 |
| H3-2 | M5_4 | No Norm |
| H3-3 | M5_5 | Per-view |

### 가설 H4: 학습 데이터 양의 영향 (P3)

| 실험 | Dataset | Train 비율 |
|------|---------|-----------|
| H4-1 | M5t | 33% (1:1:1) |
| H4-2 | M5t2 | 80% (8:1:1) |

---

## 4. Quick Start: 실험 명령어

### H1: E2E vs GS-LRM × Train/Test (4개 실험)

```bash
# GPU 설정
export CUDA_VISIBLE_DEVICES=6

# H1-1: E2E + Train
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --config configs/mouse/e2e_inference.yaml \
    --data_list ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --output_dir outputs/H1_e2e_train \
    --num_samples 50

# H1-2: E2E + Test
python -m mouse_extensions.scripts.inference.run_e2e_inference \
    --config configs/mouse/e2e_inference.yaml \
    --data_list ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --output_dir outputs/H1_e2e_test \
    --num_samples 50

# H1-3: GS-LRM only + Train (GT 입력)
python -m mouse_extensions.scripts.inference.run_gslrm_inference \
    --config configs/mouse/gslrm_inference.yaml \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_*.pt \
    --data_list ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_train.txt \
    --output_dir outputs/H1_gslrm_train \
    --num_samples 50

# H1-4: GS-LRM only + Test (GT 입력)
python -m mouse_extensions.scripts.inference.run_gslrm_inference \
    --config configs/mouse/gslrm_inference.yaml \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_*.pt \
    --data_list ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_test.txt \
    --output_dir outputs/H1_gslrm_test \
    --num_samples 50
```

### H2: Deformation V2 학습

```bash
# Gaussian 추출 (이미 완료: /node_data/joon/outputs/FaceLift/gaussians/M5t2_sample50)

# V2 학습
python -m mouse_extensions.scripts.train_deform_v2 \
    --config configs/mouse/deform_v2.yaml \
    --gaussians_dir /node_data/joon/outputs/FaceLift/gaussians/M5t2_sample50 \
    --output_dir /node_data/joon/checkpoints/FaceLift/deform_v2/M5t2 \
    --epochs 200
```

### H3: 카메라 정규화 Ablation (Config만 변경)

```bash
# M5 (Batch Uniform) - 이미 학습됨
# M5_4 (No Norm)
python train_gslrm.py -d M5_4 -e E0_1

# M5_5 (Per-view)
python train_gslrm.py -d M5_5 -e E0_1
```

### H4: 데이터 양 비교

```bash
# M5t (1:1:1) - 학습 데이터 33%
# M5t2 (8:1:1) - 학습 데이터 80%, 이미 진행 중
```

---

## 5. 용량 최소화 가이드라인

### 5.1 Gaussian 캐시

| 방식 | 용량 | 권장 |
|------|------|------|
| 전체 2880 프레임 | 158GB | ❌ |
| 샘플 50-100 프레임 | 3-6GB | ✅ |

**원칙**: 실험 목적에 맞는 최소 샘플만 추출

### 5.2 체크포인트

| 항목 | 권장 정책 |
|------|----------|
| 학습 중 | 최근 3개만 유지 |
| 완료 후 | best + final만 보존 |
| 중간 결과 | 즉시 삭제 |

### 5.3 추론 캐시

```bash
# 추론 후 즉시 정리
rm -rf outputs/*/gaussian_cache
rm -rf outputs/*/intermediate
```

---

## 6. 평가 지표

| 지표 | 설명 | 용도 |
|------|------|------|
| **PSNR** | 픽셀 오류 | 이미지 품질 |
| **SSIM** | 구조 유사도 | 지각 품질 |
| **LPIPS** | 지각 거리 | 사람 인식 |
| **TC** | Temporal Consistency | 프레임간 일관성 |
| **IoU** | 마스크 일치도 | 형상 정확도 |

---

## 7. 다음 단계 (우선순위순)

1. **[P1]** H1 실험 (E2E vs GS-LRM × Train/Test) ← 오류 원인 진단
2. **[P1]** H2 V2 Deformation 학습 및 평가
3. **[P2]** H3 카메라 정규화 Ablation 완료
4. **[P3]** H4 데이터 양 비교 (M5t vs M5t2)

---

*FaceLift Mouse Project | Hypothesis Document v1.0 | 2026-02-05*

---

## 6. MVDiffusion View Usage Analysis (260205 추가)

### 6.1 현재 문제점

| 항목 | 현재 설정 | 문제 |
|------|----------|------|
| reference_view_idx | **0 (고정)** | 다른 뷰 입력 시 일반화 저하 |
| 학습 데이터 | 2880 samples | 실질적으로 View 0만 학습 |

### 6.2 개선안: Random Reference View

```yaml
# 변경 전 (모든 config)
reference_view_idx: 0

# 변경 후 (mouse_mvdiffusion_M5t2_randref.yaml)
reference_view_idx: "random"  # 6x data augmentation
```

**효과**:
- 같은 샘플에서 6개 뷰 각각을 reference로 사용
- 2880 × 6 = **17,280 effective samples**
- 어떤 뷰가 입력되어도 robust하게 동작
- **E2E 프레임 겹침 문제 완화 기대**

### 6.3 실험 매트릭스

| Experiment | Ref View | Steps | 목적 |
|------------|----------|-------|------|
| M5t2_cfgr | Fixed (0) | 10k | Baseline |
| **M5t2_randref** | **Random** | 10k | 6x augmentation |
| M5t2_randref_20k | Random | 20k | Extended |

---

## 7. H1 실험 스크립트 (260205 추가)

위치: `/home/joon/dev/FaceLift/scripts/run_h1_experiments.sh`

```bash
# 전체 실행
./scripts/run_h1_experiments.sh all

# 개별 실행
./scripts/run_h1_experiments.sh h1a  # GS-LRM + Train
./scripts/run_h1_experiments.sh h1b  # GS-LRM + Test
./scripts/run_h1_experiments.sh h1c  # E2E + Train
./scripts/run_h1_experiments.sh h1d  # E2E + Test
```


---

## 8. H1 실험 결과 (260205 완료) ✅

### 8.1 실험 완료 상태

| 실험 | 모드 | Split | 상태 | 출력 경로 |
|------|------|-------|------|----------|
| **h1a** | GS-LRM only | Train | ✅ 완료 | `outputs/eval/h1_diagnosis_M5t2/h1a_gslrm_train/` |
| **h1b** | GS-LRM only | Test | ✅ 완료 | `outputs/eval/h1_diagnosis_M5t2/h1b_gslrm_test/` |
| **h1c** | E2E (MVDiff→GS) | Train | ✅ 완료 | `outputs/eval/h1_diagnosis_M5t2/h1c_e2e_train/` |
| **h1d** | E2E (MVDiff→GS) | Test | ✅ 완료 | `outputs/eval/h1_diagnosis_M5t2/h1d_e2e_test/` |

### 8.2 사용된 체크포인트 (M5t2 Best)

| Model | Checkpoint | 비고 |
|-------|-----------|------|
| **GS-LRM** | `M5t2_E0_1_facelift/best_psnr.pt` | Best PSNR 자동 저장 |
| **MVDiffusion** | `mouse_M5t2_cfgr/checkpoint-10000` | CFG=0.05 복원 버전 |

### 8.3 정량 지표 결과 ⭐

> 평가 방법: Foreground-only PSNR (GT alpha mask 사용), View 0 기준
> Reports: `outputs/eval/h1_diagnosis_M5t2/reports/`

#### M5t2 (2880 train samples)

| 실험 | Mode | Split | PSNR ↑ | SSIM ↑ | N |
|------|------|-------|--------|--------|---|
| h1a | GS-LRM | Train | **19.18±1.31** | 0.6239 | 50 |
| h1b | GS-LRM | Test | **19.58±0.90** | 0.6048 | 50 |
| h1c | E2E | Train | **19.34±1.22** | 0.6225 | 50 |
| h1d | E2E | Test | **19.63±0.76** | 0.6037 | 50 |

#### M5t (1198 train samples) - 데이터 양 비교용

| 실험 | Mode | Split | PSNR ↑ | SSIM ↑ | N |
|------|------|-------|--------|--------|---|
| h1a | GS-LRM | Train | **18.85±1.12** | 0.6204 | 50 |
| h1b | GS-LRM | Test | **19.69±1.18** | 0.6010 | 28 |
| h1c | E2E | Train | **18.85±1.12** | 0.6204 | 50 |
| h1d | E2E | Test | **19.15±1.15** | 0.6030 | 50 |

### 8.4 핵심 발견 ⭐⭐

1. **GS-LRM vs E2E 차이 미미** (~0.1-0.2 PSNR)
   - h1a (GS-LRM) = 19.18 vs h1c (E2E) = 19.34
   - MVDiffusion이 노이즈를 추가한다는 초기 가설과 **다름**

2. **Train-Test 차이 없음** (오히려 Test가 약간 높음)
   - Train: 19.18 → Test: 19.58 (GS-LRM)
   - Train: 19.34 → Test: 19.63 (E2E)
   - 일반화 문제 없음

3. **데이터 양 영향 제한적** (M5t2 vs M5t)
   - M5t2 Train: 19.18 vs M5t Train: 18.85 (+0.33)
   - M5t2 Test: 19.63 vs M5t Test: 19.15 (+0.48)

### 8.5 해석

원래 가설 매트릭스 대비:

| 패턴 | 의미 | 본 실험 결과 |
|------|------|-------------|
| H1-1 ≈ H1-3 >> H1-2 ≈ H1-4 | Train/Test 차이 지배적 | ❌ 해당 안됨 |
| H1-1 >> H1-3, H1-2 >> H1-4 | MVDiffusion이 도움 | ❌ 차이 미미 |
| H1-1 << H1-3, H1-2 << H1-4 | MVDiffusion이 노이즈 | ❌ 차이 미미 |
| **H1-3 ≈ H1-4 ≈ H1-1 ≈ H1-2** | 모두 비슷 | ✅ **본 결과** |

**결론**: 현재 파이프라인에서 품질 병목은 단일 요인(MVDiffusion 또는 일반화)이 아님.
모델 자체의 reconstruction capacity가 PSNR ~19 수준에서 수렴.

### 8.6 후속 실험 방향

1. **입력 뷰 ablation**: Reference view index (0-5) 변경 영향
2. **해상도 ablation**: 512 → 384 → 256 영향
3. **Temporal V2**: Multi-frame training으로 consistency 개선

### 8.7 생성된 영상

각 폴더에 다음 파일 생성:
- `turntable_first.mp4` - 첫 프레임 360° 회전
- `time_fixed.mp4` - 시간 변화 (고정 각도)
- `time_rotating.mp4` - 시간 + 회전 동시
- `turntable_grid.jpg` - 6×6 그리드 개요

---

## 9. View Ablation 실험 (260205 진행중)

### 9.1 목적

MVDiffusion 입력 뷰(reference view)에 따른 품질 변화 분석

### 9.2 실험 설계

| 실험 | Input View | GPU | 상태 |
|------|------------|-----|------|
| h1d (기존) | View 0 | - | ✅ 완료 |
| view_1 | View 1 | 5 | ⏳ 대기 |
| view_2 | View 2 | 5 | ⏳ 대기 |
| view_3 | View 3 | 6 | ⏳ 대기 |
| view_4 | View 4 | 6 | ⏳ 대기 |
| view_5 | View 5 | 7 | ⏳ 대기 |

### 9.3 실행 스크립트

```bash
./scripts/run_h1_view_ablation.sh
```

### 9.4 출력 위치

```
outputs/eval/h1_view_ablation/
├── view_1/
├── view_2/
├── view_3/
├── view_4/
└── view_5/
```

---

## 10. Cyclic Reference View Augmentation (260205 학습중)

### 10.1 가설

**고정 reference view (view 0)**로만 학습 시, 다른 뷰 입력에 취약.
→ **Cyclic augmentation**으로 모든 뷰에 robust하게 학습

### 10.2 설정 비교

| 항목 | Baseline (`M5t2_cfgr`) | Cyclic (`M5t2_cyclic`) |
|------|------------------------|------------------------|
| `reference_view_idx` | `0` (고정) | `[0,1,2,3,4,5]` (랜덤) |
| Effective samples | 2880 | 2880 × 6 = 17,280 |
| CFG | 0.05 | 0.05 |
| Steps | 10,000 | 10,000 |

### 10.3 학습 상태

| 항목 | 값 |
|------|-----|
| GPU | 4 |
| 진행 | ~80/10000 steps |
| WandB | `mvdiff_M5t2_cyclic` |
| Config | `configs/mvdiffusion/mouse_mvdiffusion_M5t2_cyclic.yaml` |
| 예상 완료 | ~36시간 |

### 10.4 비교 실험

학습 완료 후:
1. **Baseline**: `mouse_M5t2_cfgr/checkpoint-10000`
2. **Cyclic**: `mouse_M5t2_cyclic/checkpoint-10000`

동일 View Ablation 실험으로 비교 예정

---

## 11. Backlinks

- [[EXPERIMENT_REGISTRY]] - 실험 등록 현황
- [[MOUSE_QUICK_REFERENCE]] - 명령어/체크포인트 Quick Reference
- [[260203_MVDiffusion_CFG_Ablation]] - CFG 복원 실험
- [[INFERENCE_E2E_GUIDE]] - E2E 추론 가이드

---

*Updated: 2026-02-05 16:30*

---

## 12. M5t2 vs M5t 데이터 양 비교 실험 (260205 완료) ✅

### 12.1 목적

학습 데이터 양이 성능에 미치는 영향 검증

### 12.2 데이터셋 비교

| Dataset | Split | Train | Val | Test | 비고 |
|---------|-------|-------|-----|------|------|
| **M5t2** | 80:10:10 | 2880 | 360 | 360 | 학습 최대화 |
| **M5t** | 1:1:1 | 1198 | 1198 | 1204 | Pose-Splatter 호환 |

### 12.3 실험 완료 상태

-  ✅ 완료 (reports/ 포함)
-  ✅ 완료 (reports/ 포함)

### 12.4 체크포인트

| Dataset | GS-LRM | MVDiffusion |
|---------|--------|-------------|
| M5t2 |  |  |
| M5t |  |  |

### 12.5 정량 비교 결과 ⭐

| Metric | M5t2 (2880 train) | M5t (1198 train) | 차이 |
|--------|-------------------|------------------|------|
| **GS-LRM Train PSNR** | 19.18 | 18.85 | +0.33 |
| **GS-LRM Test PSNR** | 19.58 | 19.69 | -0.11 |
| **E2E Train PSNR** | 19.34 | 18.85 | +0.49 |
| **E2E Test PSNR** | 19.63 | 19.15 | +0.48 |

### 12.6 결론

1. **Train PSNR 차이 미미** (0.3-0.5 PSNR)
   - 2.4배 데이터 증가 (1198 → 2880)의 효과가 크지 않음

2. **Test PSNR 차이 거의 없음** (-0.1 ~ +0.5)
   - 일반화 성능은 데이터 양에 크게 의존하지 않음

3. **해석**: 현재 모델은 더 많은 데이터보다는
   - 모델 capacity 또는 training scheme 개선이 필요할 수 있음
   - 데이터 다양성(variety)이 양(quantity)보다 중요할 수 있음

---

*Updated: 2026-02-05*
