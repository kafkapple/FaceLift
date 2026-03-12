# NeurIPS 2026 Gap Analysis: BehaviorSplatter

> Created: 2026-03-03 | Target: NeurIPS 2026 Submission Deadline (~May 2026)

---

## 1. Current Status Assessment

### 1.1 Strengths (논문으로 이미 충분한 부분)

| 항목 | 근거 | 강도 |
|------|------|:----:|
| **Fair comparison framework** | 동일 데이터, coverage-aware metrics, 최초 FL vs PS 비교 | ★★★ |
| **Systematic bottleneck analysis** | 86% MVDiff 병목 정량화, view ablation 1-6v | ★★★ |
| **GS-LRM upper bound** | +10 dB over PS (23.84 vs 13.78) — 강력한 ceiling | ★★★ |
| **Plücker conditioning** | 체계적 ablation (Extrinsic → Plucker → Spatial Token) | ★★☆ |
| **Template-free pipeline** | 종 비의존적, 실시간 가능 | ★★☆ |

### 1.2 Weaknesses (NeurIPS 수용 기준 미달)

| 항목 | 현재 | NeurIPS 기대 | Gap 심각도 |
|------|------|-------------|:----------:|
| **E2E 품질** | 9.04 dB (PS 대비 -4.74) | ≥ PS (13.78) 또는 justified gap | 🔴 Critical |
| **데이터 다양성** | 1 mouse, 1 arena | ≥ 2 species, ≥ 2 setups | 🔴 Critical |
| **행동 분석 검증** | Synthetic clustering only | Real behavior GT labels | 🟡 High |
| **Novelty depth** | 기존 components 조합 | 새로운 아키텍처/기법 | 🟡 High |
| **Temporal consistency** | Per-frame (독립) | 시간적 연속성 | 🟢 Medium |

---

## 2. Top 5 Priority Experiments

### Priority 1: Silhouette-Focused Loss (H8) 🔴

**Why**: IoU가 0.954 (GT) → 0.577 (E2E)로 하락 — **실루엣 오류가 PSNR gap의 60%+** 기여. 현재 MVDiffusion 학습은 MSE loss만 사용하여 background pixel에도 동일 가중치 부여. 마우스가 이미지의 ~2.5%만 차지하므로 foreground 학습 신호가 극히 약함.

**How**:
```python
# MVDiffusion training loss modification
L_total = L_mse + λ_sil * L_silhouette + λ_fg * L_foreground_weighted

# Option A: Binary silhouette loss on predicted alpha
L_silhouette = BCE(predicted_alpha, gt_alpha)  # Direct shape supervision

# Option B: Foreground-weighted MSE
weights = gt_mask * (1/fg_ratio) + (1 - gt_mask) * 1.0  # fg_ratio ≈ 0.025
L_fg_weighted = (weights * (pred - gt)^2).mean()

# Option C: Perceptual loss on foreground-only crops
L_fg_perc = VGG_loss(crop_fg(pred), crop_fg(gt))
```

**Expected impact**: IoU 0.577 → 0.65-0.70, PSNR_gt +1.5-3.0 dB
**Timeline**: 1-2일 구현, 2-3일 학습 (10K steps)
**리소스**: GPU 1개 (A6000), MVDiffusion training script 수정

**구체적 실행 계획**:
1. `train_diffusion.py`에 silhouette loss 추가 (gt_alpha 활용)
2. Foreground-weighted MSE 옵션 구현
3. H4b checkpoint에서 fine-tune (추가 5K-10K steps)
4. E2E eval → IoU 변화 중심 비교

---

### Priority 2: Domain Adaptation for GS-LRM (DA1) 🔴

**Why**: GS-LRM은 GT 뷰 입력 시 23.84 dB이지만, MVDiff 합성 뷰 입력 시 9.04 dB. **분포 불일치**(distribution mismatch)가 핵심: GS-LRM은 깨끗한 실제 이미지로 학습되었으나, 추론 시에는 diffusion 합성 이미지(artifact 포함)를 받음.

**How**:
```
Step 1: MVDiff batch inference → 합성 뷰 데이터셋 생성
  - 2,880 training frames × 6 views = 17,280 합성 이미지
  - 현재 218/2,880 완료 (DA1 데이터 생성 중)

Step 2: GS-LRM fine-tuning on synthetic views
  - Input: MVDiff 합성 6뷰 (노이즈, 아티팩트 포함)
  - Target: GT rendering (깨끗한 이미지)
  - Config: domain_adapt_E2_v1.yaml
  - LR: 1e-6 (기존 fine-tune 동일), 5K-10K steps

Step 3: E2E evaluation
  - 동일 파이프라인: 1v → MVDiff → DA-GS-LRM → 3D
  - 기대: GS-LRM이 합성 뷰의 아티팩트에 robust해짐
```

**Expected impact**: PSNR_gt +2-5 dB (9.04 → 11-14 dB). 이론적 근거: view ablation에서 3 GT views = 18.56 dB → 현재 6 합성 뷰가 1 GT view 수준이므로, 합성 뷰 품질에 적응하면 2-3 GT view 수준 도달 가능.
**Timeline**: 데이터 생성 1-2일 + 학습 1-2일 + 평가 0.5일
**리소스**: GPU 2개 (데이터 생성 + 학습 병렬)

---

### Priority 3: H7v2 Spatial Token Completion + Proper E2E 🟡

**Why**: H7v2는 현재 학습 중 (GPU6, step ~165/10K). Spatial token 방식이 Val PSNR에서 H6a_v2를 step 400+부터 지속적으로 상회(+0.26~+1.22 dB). **그러나 E2E 평가가 아직 없음** — pose_injector weights 저장 버그가 수정된 첫 모델이므로, 진정한 Plücker pose conditioning E2E를 최초로 평가할 수 있음.

**How**:
```
1. H7v2 학습 완료 대기 (~35h remaining)
2. Best checkpoint 선정 (val PSNR 기준)
3. E2E inference (반드시 --input_view_idx 0 포함!)
   CUDA_VISIBLE_DEVICES=5 python run_e2e_inference.py \
     --data_dir /path/to/M5t2_test \
     --input_view_idx 0 \
     --mvdiff_ckpt /path/to/H7v2/best_checkpoint \
     --gslrm_ckpt /path/to/6view_v2/best_psnr.pt \
     --use_pose_injector \
     --output_dir outputs/H7v2_e2e
4. Fair eval: fair_comparison.py
5. 검증 프로토콜 (grep 'Batch Path' → "Path 2b" 확인)
```

**Expected impact**: Val PSNR 27.5+ (H6a_v2 27.34 초과 예상). E2E에서 +0.5-1.0 dB 가능 (spatial token이 multi-view consistency 향상 시).
**Timeline**: 학습 완료 대기 (~35h) + E2E eval 8h + 비교 분석 2h
**리소스**: 현재 GPU6에서 이미 실행 중

---

### Priority 4: Multi-Species Evaluation (Rat7M or Fly) 🔴

**Why**: NeurIPS 리뷰어의 가장 예상 가능한 비판: "단일 마우스, 단일 환경에서만 검증". Template-free를 주장하려면 **최소 2종 이상의 동물**에서 작동을 보여야 함.

**How**:

**Option A: Rat7M Dataset (권장)**
- **데이터**: Rat7M [Marshall et al., 2022] — 7M frames, multi-view rat motion capture
- **장점**: 마우스와 유사한 체형 → 최소한의 전처리 수정으로 적용 가능
- **절차**:
  1. Rat7M 다운로드 및 M5 형식으로 전처리 (카메라 정규화, 해상도 맞춤)
  2. GS-LRM zero-shot 평가 (mouse-trained 모델 → rat 데이터)
  3. GS-LRM fine-tune on rat (소량, ~1K frames)
  4. E2E 파이프라인 평가

**Option B: CalMS21 (Mouse Social Behavior)**
- **데이터**: CalMS21 [Sun et al., 2021] — 다수 마우스 사회적 행동 + 행동 라벨
- **장점**: 행동 분석 ground truth 제공 → Priority 5와 시너지
- **단점**: 단일 뷰 top-down만 제공 → 3D 재구성 GT 없음

**Option C: Fly (Drosophila)**
- **데이터**: FlyTracker 등 공개 데이터
- **장점**: 완전히 다른 체형 → 일반화 강력 입증
- **단점**: 매우 작은 객체, 추가 도전

**Expected impact**: "Cross-species generalization" 섹션 추가 → 논문 일반성 크게 강화
**Timeline**: 데이터 준비 2일 + 평가 2일 + 분석 1일 = 5일
**리소스**: GPU 1-2개, 추가 데이터 다운로드

---

### Priority 5: Real Behavior Validation with Ground Truth Labels 🟡

**Why**: 현재 행동 클러스터링은 synthetic 데이터에서만 검증됨 (Silhouette 0.596). 실제 행동 라벨과의 비교가 없으면 downstream task의 유용성을 입증할 수 없음.

**How**:

**Step 1: 행동 라벨 획득**
- CalMS21 (annotated mouse behavior: attack, investigate, mount, other)
- 또는 DANNCE 데이터에 자체 행동 라벨링 (walking, grooming, rearing, resting)
- 최소 4개 행동 카테고리, 각 100+ frames

**Step 2: Embedding 추출**
```python
# Per-frame pipeline
for frame in test_frames:
    gaussians = reconstruct_3d(frame)           # BehaviorSplatter
    views = spherical_render(gaussians, n=32)   # 32 views
    features = resnet18(views)                  # 512D × 32
    sh_embed = spherical_harmonics(features)    # 8192D
    embedding = adv_pca(sh_embed, dim=50)       # 50D
```

**Step 3: 정량 평가**
- **Classification accuracy**: Embedding → logistic regression → 행동 분류 정확도
- **Retrieval**: 같은 행동 embedding이 클러스터링되는지 (NMI, ARI 지표)
- **Comparison**: Keypoint-based vs Visual embedding 비교
- **Ablation**: BehaviorSplatter embedding vs PoseSplatter embedding vs DLC keypoints

**Expected impact**: "Behavior analysis" 섹션의 정량 결과 완성 → 논문의 "behavior" 주장 입증
**Timeline**: 라벨링 2일 + 평가 2일 + 분석 1일 = 5일
**리소스**: GPU 1개 + annotation effort

---

## 3. Priority Matrix

```
                        Impact on Paper
                   Low         Medium        High
              ┌──────────┬──────────┬──────────┐
    Easy      │          │          │ P3: H7v2 │
  (1-3 days)  │          │          │ complete │
              ├──────────┼──────────┼──────────┤
   Medium     │          │ P5: Real │ P1: Sil  │
  (3-5 days)  │          │ behavior │ loss     │
              ├──────────┼──────────┼──────────┤
    Hard      │          │          │ P2: DA1  │
  (5-10 days) │          │ P4: Multi│          │
              │          │ species  │          │
              └──────────┴──────────┴──────────┘
```

## 4. Recommended Execution Order

```
Week 1 (03/03-03/09):
  ├─ [P3] H7v2 학습 완료 대기 → E2E eval (이미 실행 중)
  ├─ [P1] Silhouette loss 구현 + 학습 시작
  └─ [P2] DA1 데이터 생성 완료 + fine-tune 시작

Week 2 (03/10-03/16):
  ├─ [P1] Silhouette loss E2E eval + 비교
  ├─ [P2] DA1 GS-LRM E2E eval
  └─ [P4] Multi-species 데이터 준비 시작

Week 3 (03/17-03/23):
  ├─ [P4] Multi-species (Rat7M) 평가
  ├─ [P5] 행동 라벨링 + embedding 평가
  └─ [논문] 실험 결과 종합 + 논문 개정

Week 4 (03/24-03/30):
  ├─ [논문] 최종 결과 반영 + 논문 polish
  └─ [논문] 공동 저자 리뷰 + 제출 준비
```

## 5. Minimum Viable Paper (보험 전략)

P1-P5 모두 완수하지 못할 경우의 **최소 요건**:

| 항목 | 필수 | 현재 | Action |
|------|:----:|:----:|--------|
| E2E 품질 개선 증거 | ✅ | ❌ | P1 or P2 중 하나 |
| 체계적 ablation | ✅ | ✅ | 이미 완료 |
| Fair comparison | ✅ | ✅ | 이미 완료 |
| 2+ species | ⚠️ | ❌ | P4 (없으면 limitation으로) |
| 행동 분석 정량 | ⚠️ | △ | P5 (synthetic으로 대체 가능) |

**최소 전략**: P1 (silhouette loss)만 성공해도 +2-3 dB 개선으로 "closing the gap" 스토리 가능. P2 추가 시 더 강력. P4 없이도 "future work"로 처리 가능하지만 리뷰어 리스크 있음.

---

## 6. Risk Assessment

| Risk | 확률 | 영향 | 대응 |
|------|:----:|:----:|------|
| P1 silhouette loss 효과 미미 | 30% | High | P2 (DA1)로 대체 — 두 개 병렬 진행 |
| H7v2 학습 발산/불안정 | 15% | Medium | H6a_v2 결과로 fallback |
| Rat7M 전처리 난이도 | 40% | Medium | CalMS21로 전환 (2D only지만 행동 라벨 풍부) |
| E2E PSNR이 PS를 넘지 못함 | 80% | Low | "Feed-forward efficiency + upper bound" 프레이밍으로 대응 |
| NeurIPS 마감까지 시간 부족 | 25% | High | Workshop paper 또는 ICLR 2027로 전환 |

---

*NeurIPS Gap Analysis v1.0 | 2026-03-03*
