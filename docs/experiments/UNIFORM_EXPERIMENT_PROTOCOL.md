> ⚠️ **NOTE (260207)**: 이 문서는 v1 프로토콜 설계 문서입니다. `max_steps`는 dead config로 확인됨 → 실제 종료는 `max_fwdbwd_passes`. 현재 실험은 `base_uniform_v2.yaml` + `H4_VIEW_ABLATION.md` 참조.
>
# Uniform Experiment Protocol

> 모든 View Ablation 실험을 동일 조건에서 공정 비교하기 위한 프로토콜

---

## 1. 현재 문제점

### 1.1 불균일한 종료 조건

| 실험 | Best Step | Current Step | 상태 |
|------|-----------|--------------|------|
| 1-view | 10801 | 12801+ | 수렴 후 계속 진행 |
| 2-view | 6201 | 7001+ | 수렴 후 계속 진행 |
| 3-view | 4201 | 4901+ | 수렴 후 계속 진행 |
| 5-view | 4401 | 4701+ | 수렴 후 계속 진행 |
| 6-view | 1 | 1801+ | Validation 버그 |

**문제**: 각 실험이 다른 step에서 best, 비교 어려움

### 1.2 6-view Validation 버그

- `best_psnr.json`: value=0, step=1
- 원인: 초기화 시 0으로 저장 후 갱신 안 됨
- **수정 필요**: `checkpoint_best` 함수에서 초기값 처리

---

## 2. 균일 실험 프로토콜

### 2.1 Fixed Step 방식 (권장)

모든 실험을 **동일한 step 수**만큼 학습:

```yaml
training:
  schedule:
    max_steps: 15000  # 모든 실험 동일
    val_every: 100    # 더 자주 validation
    
  checkpointing:
    checkpoint_every: 500
    save_best: true
    save_last: true
```

**장점:**
- 완벽한 공정 비교 (동일 컴퓨팅 예산)
- Best와 Last 모두 비교 가능
- 수렴 곡선 전체 비교 가능

### 2.2 Early Stopping 방식 (대안)

```yaml
training:
  schedule:
    max_steps: 50000  # 충분히 크게
    early_stop_patience: 20  # 20 * val_every = 2000 step 동안 개선 없으면 종료
    val_every: 100
```

**장점:**
- 자동 종료 (불필요한 학습 방지)
- 각 실험의 실제 수렴 시점 파악

**단점:**
- 실험마다 총 step 수 다름

### 2.3 Epoch 기반 방식

```yaml
training:
  schedule:
    num_epochs: 10  # 데이터셋 전체를 10번 순회
```

**M5t2 기준**: 2880 samples, batch_size=2 → 1 epoch ≈ 1440 steps
- 10 epochs = 14400 steps

---

## 3. 표준 실험 설정

### 3.1 공통 Config (base_uniform.yaml)

```yaml
# 학습 설정
training:
  schedule:
    max_steps: 15000
    val_every: 100
    early_stop_patience: 30  # Optional
    
  checkpointing:
    checkpoint_every: 500
    save_best: true
    save_last: true

  dataloader:
    batch_size_per_gpu: 2
    num_workers: 8

# 평가 설정
validation:
  target_has_input: true  # 6-view 포함 위해
  compute_per_view: true  # 뷰별 메트릭 분리
  
# 로깅
logging:
  val_metrics:
    - psnr
    - ssim
    - lpips
    - per_view_psnr
```

### 3.2 View별 Config (experiment_Nview.yaml)

```yaml
model:
  num_input_views: N  # 1, 2, 3, 5, 6
  num_views: 6
```

---

## 4. 평가 기준

### 4.1 Primary Metrics

| 메트릭 | 기준 | 비교 방법 |
|--------|------|-----------|
| **Val PSNR @ 15000 step** | Last checkpoint | 동일 step에서 비교 |
| **Best Val PSNR** | Best checkpoint | 수렴 성능 비교 |
| **Convergence Step** | Best step | 학습 효율 비교 |

### 4.2 Secondary Metrics (Per-view 분석)

| 메트릭 | 의미 |
|--------|------|
| Same-view PSNR | Input 뷰 재구성 품질 |
| Novel-view PSNR | Hold-out 뷰 일반화 |
| PSNR Gap (Same - Novel) | 일반화 손실 |

### 4.3 6-view 특수 처리

- Novel-view 메트릭 계산 불가 (hold-out 0개)
- **Same-view PSNR만 비교** (상한선 역할)
- 또는 로 별도 실험 (에러 발생 가능)

---

## 5. 실험 재실행 계획

### 5.1 Phase 1: 버그 수정

1. `checkpoint_best` 초기값 처리 수정
   - 첫 저장 시 -inf 대신 0 사용 문제 해결
   
2. 6-view validation 로직 검증
   - WandB에서 val/psnr 로깅 확인

### 5.2 Phase 2: 균일 재실험

```bash
# 모든 실험 동시 실행 (GPU 분배)
for views in 1 2 3 5 6; do
    CUDA_VISIBLE_DEVICES=-1 python train_gslrm.py \
        --config configs/mouse/uniform/base_uniform.yaml \
        --config configs/mouse/uniform/view.yaml \
        --output_dir checkpoints/uniform_ablation/view
done
```

### 5.3 Phase 3: 결과 수집

```bash
python -m mouse_extensions.scripts.evaluation.generate_experiment_report \
    --experiments uniform_ablation \
    --output_dir outputs/reports/uniform
```

---

## 6. 체크리스트

### 실험 전

- [ ] `checkpoint_best` 버그 수정 확인
- [ ] base_uniform.yaml 생성
- [ ] 각 view별 config 생성
- [ ] GPU 할당 계획

### 실험 중

- [ ] WandB에서 모든 실험 val/psnr 정상 로깅 확인
- [ ] 15000 step 도달 확인

### 실험 후

- [ ] best_psnr.json 모든 실험 확인
- [ ] 동일 step (15000)에서 Last checkpoint 메트릭 추출
- [ ] 통합 비교 레포트 생성

---

*FaceLift | Uniform Experiment Protocol v1.0 | 2026-02-06*
