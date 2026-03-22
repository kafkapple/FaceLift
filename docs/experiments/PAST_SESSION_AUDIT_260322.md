# Past Session Items Audit — 260322

> **Navigation**: [<- INDEX](../INDEX.md) | [PHASE2_NOVEL_VIEW_ROADMAP](PHASE2_NOVEL_VIEW_ROADMAP.md)
> **Version**: v2.0 | **Created**: 2026-03-22 | **Status**: AUDIT REPORT (bias-corrected)

---

## ⚠️ Critical Warning: Confirmation Bias Found (260322 재검증)

**3-model deliberation (Claude/Gemini/GPT) 결과, 이 문서의 v1.0에 다수의 확증 편향이 발견됨.**

| # | 원래 주장 | 문제 | 수정 |
|---|-----------|------|------|
| **1** | "ARI=0.052 → Gaussian과 KP가 **상보적**" | ❌ ARI=0.052 = **random disagreement**. "상보적" 아님 | "두 feature space가 다름" (행동적 의미 미확인) |
| **2** | "Gaussian features capture behavioral info" | ❌ **Foreground masking 부재** (opacity 97.8%≈0.5, background 포함) | 미검증. Foreground-only 재실험 필요 |
| **3** | "HLAC 실험 ✅ 완료" | ⚠️ 코드 실행됨 ≠ **검증 완료**. Background-contaminated features로 실행 | 조건부 완료 (foreground 재실험 P1) |
| **4** | "NeurIPS 핵심 기여" | ❌ 현재 데이터로는 주장 불가 | 삭제. Foreground-masked 결과 나온 후 재평가 |

**추가 발견된 프로젝트 전반 이슈**:

| Claim | 문제 | Risk |
|-------|------|:----:|
| PSNR=23.84 "strong" | White background inflation 미고려 | 🟡 |
| α=0.3 "best" (PSNR only) | Foreground-only PSNR/LPIPS 미검증 | 🟡 |
| DiFix zero-shot → "FT necessary" | 대안 미탐색 (hyperparams, input format) | 🟡 |
| "Multi-species = key differentiator" | Rat/marmoset 데이터 미완성 상태에서 주장 | 🟡 |
| Canonical set "covers space" | Selection bias (구현 가능한 것 위주) | 🟡 |

> **교훈**: 양의 결과일수록 더 비판적으로. "코드 존재 ≠ 검증 완료", "낮은 ARI ≠ 상보적", "높은 PSNR ≠ 좋은 품질".

---

## Executive Summary

7개 과거 세션 논의 항목을 코드/문서/서버 상태 기준으로 감사한 결과:
- **완료**: 2개 (coordinate_utils, FL Fair 512 평가)
- **조건부 완료**: 1개 (HLAC — foreground masking 재실험 필요)
- **부분 완료**: 3개 (Gaussian quality, deformation, multi-tier)
- **미시작**: 1개 (VAME/Selfee 실험)

NeurIPS 2026 제출 기준 **Top 3 우선순위**: Foreground-masked HLAC 재실험 > Multi-species pipeline > Gaussian quality 체계적 실행.

---

## 1. PoseSplatter Visual Embedding / HLAC Gaussian Test

### 상태: ⚠️ CONDITIONAL — 결과 존재하나 해석 수정 필요 (260322 재검증)

**증거**:
- `mouse_extensions/behavior/hlac_m5t2.py` — Core BehaviorSplatter 실험 스크립트 ✅
- `outputs/m5t2_hlac_analysis/results.json` — **598 frames, K=8 optimal** ✅
- `outputs/m5t2_hlac_analysis/classification_comparison_K8.png` — 시각화 ✅

**결과** (results.json):

| Feature | Gaussian-defined HLACs | KP-defined HLACs |
|---------|:---:|:---:|
| **Gaussian_raw** | **90.8%** | 45.8% |
| **KP_centered** | — | **99.2%** |
| Cluster Agreement (ARI) | 0.052 | NMI=0.140 |

**~~이전 해석~~ (❌ WRONG — confirmation bias)**:
> ~~"Gaussian features는 keypoints와 독립적인 행동 정보를 인코딩 (상보적)"~~

**수정된 해석 (260322 3-model deliberation)**:
- ARI=0.052 = **random disagreement** (0에 가까움). "상보적"이 아니라 "관계 없음"
- 45.8% (Gaussian→KP labels): chance(12.5%) 대비 above-chance이나, **robust하지 않음**
- **치명적 문제**: Gaussian features에 **foreground masking 미적용** — opacity 97.8%≈0.5로 top-K pruning이 사실상 random sampling과 동치
- KP(99.2%) >> Gaussian(45.8% cross) — KP가 행동 분류에 압도적으로 우수

**올바른 주장**:
- "두 feature space의 K-means clustering은 다름 (ARI=0.052)"
- "그러나 이것이 행동적 상보성을 의미하는지는 **foreground-only features + human-annotated labels로 재검증 필요**"

**남은 작업 (P1)**:
1. Foreground-masked Gaussian feature 추출 (opacity threshold sweep / N≥2 filter)
2. s-DANNCE 13-class HLAC labels로 재분류 (K-means가 아닌 supervised)
3. Combined feature (KP+Gaussian) downstream task 성능 비교

---

## 2. Past TODO Items (260318 Handoff)

### 상태: 혼합

| TODO | 상태 | 증거 |
|------|:----:|------|
| **좌표계 guard** (coordinate_utils.py) | ✅ **완료** | 108줄, SSOT, `mammal_to_gslrm()`, `assert_gslrm_space()` 등 구현 |
| **Deformation module 검토** | 🟡 부분 | 6개 파일 존재 (network, trainer v1/v2, inference). 프로젝트 메모리에 "Gaussian identity NOT maintained" 발견 기록 (260319) |
| **메트릭 문서** (BIC, DBCV 등) | 🟡 부분 | gaussian_quality_metrics.py에 일부 포함. 독립 문서 없음 |
| **VAME 실험** | ❌ **미시작** | 코드/설정 파일 없음 |
| **Selfee 설치 + 실험** | ❌ **미시작** | 코드/설정 파일 없음 |

**팩트체크**:
- TODO-1 (coordinate_utils): **완료**. `assert_gslrm_space()`, `assert_mammal_space()` guard 함수 포함.
- TODO-2 (Deformation): **핵심 발견 있음** — "Deformation module: Gaussian identity NOT maintained" (프로젝트 메모리 260319). 이는 dense feature 전략에 중대 영향 → scene flow 기반 접근 필요 확인.
- TODO-4,5 (VAME/Selfee): **미시작**. 현재 Phase 2 우선순위에서 후순위로 밀림.

**우선순위**: TODO-1,2 완료됨. TODO-3 P2. TODO-4,5 **P2** (Phase 2 multi-species에 리소스 집중).

---

## 3. Stage 2 AI Enhancement / Multi-Tier Dataset

### 상태: 🟡 PARTIALLY DONE (전략 수립, 실행 부분)

**증거**:
- `docs/experiments/NOVEL_VIEW_QUALITY_STRATEGY.md` — 3-tier 전략 문서 ✅
- `mouse_extensions/scripts/eval/artifact_metrics.py` — Artifact 측정 코드 ✅
- DiFix 3D+ 학습 중 (이번 세션, Stage 1 Type 3) ✅

**미완료**:
- ❌ 100-sample validation set 미생성
- ❌ Stage 1 false positive 검증 미실행
- ❌ Stage 2 AI enhancement 미적용 (DiFix 학습 중)
- ❌ Artifact masks annotation 미시작

**Red Team 경고 (유효)**:
> "Stage 2 AI enhancement를 벤치마크 GT로 절대 사용하지 말 것" — **이번 DiFix zero-shot 실패가 이를 정확히 확인**. Pretrained DiFix는 우리 도메인에서 이미지를 **악화**시킴.

**Visionary 제안 (유효)**:
> "3-tier dataset (Raw/Cleaned/AI-enhanced) + artifact masks" — NeurIPS Eval Track에서 독보적 기여 가능. **현재 실행 중인 DiFix fine-tuning이 Tier 2 (AI-enhanced) 생성의 첫 단계.**

**우선순위**: **P0** — DiFix 학습 완료 후 validation set 생성 + artifact mask annotation.

---

## 4. HLAC PoC / Covariance Eigenvalues

### 상태: 🔒 BLOCKED → 🟢 UNBLOCKED (이번 세션)

**증거**:
- `mouse_extensions/behavior/extract_covariance_features.py` — 추출 코드 ✅
- h1_sdannce_probe.py에서 "once GS-LRM reconstruction of s-DANNCE data is available" 명시

**상태 변화**:
- **이전**: s-DANNCE → GS-LRM 파이프라인 미구축 → BLOCKED
- **이번 세션**: `sdannce_to_gslrm.py` v2 구현 + SAM2 마스크 파이프라인 완성 → **UNBLOCKED**
- SCN2A_WK1 rat 데이터로 GS-LRM 추론 성공 (v3, v4 smoke test)

**다음 단계**:
1. s-DANNCE mouse 데이터에 GS-LRM 추론 → Gaussian features 추출
2. Covariance eigenvalues 계산
3. HLAC probe 실행: KP vs Gaussian_covariance 비교
4. ERR 체크포인트에서도 covariance 추출 → 등방성 강제 효과 비교

**우선순위**: **P1** — Rat FT 데이터 준비와 병렬 가능. NeurIPS 핵심 실험.

---

## 5. Gaussian Parameter Distribution Visualization

### 상태: 🟡 PARTIALLY DONE (코드 있음, 체계적 실행 미확인)

**증거**:
- `mouse_extensions/scripts/eval/gaussian_quality_metrics.py` — 포괄적 메트릭 ✅
  - Anisotropy Ratio (max/min scale)
  - Isotropy Score (Var(log(s)))
  - Opacity Ambiguity (0.1 < opacity < 0.9 비율)
  - Alpha Entropy (Shannon entropy of alpha histogram)
  - `compute_opacity_distribution()` — histogram 포함
- `outputs/reports/gaussian_quality_512/gaussian_quality_comparison.json` — S18에서 생성 ✅

**추가 발견** (탐색 에이전트):
- `mouse_extensions/behavior/analyze_gaussian_distributions.py` — **전용 분석 스크립트 존재!**
  - Hartigan's Dip Test (unimodal/multimodal 판정)
  - GMM BIC (mixture component 수 결정)
  - Body-part별 분포 분석
- **핵심 발견 (S18)**: Opacity = **0.500** for 97.8% of Gaussians (feed-forward GS-LRM, pruning 없음)

**미완료**:
- ❌ 체계적 unimodal/multimodal 분석 미실행 (코드는 존재)
- ❌ 전체 체크포인트 비교 (baseline, α variants, ERR)의 분포 시각화

**우선순위**: **P1** — 30분 작업. `analyze_gaussian_distributions.py` 실행만 하면 됨.

---

## 6. FL Fair Comparison 512 / DiFix / MAMMAL

### 상태: 대부분 완료

| 항목 | 상태 | 증거 |
|------|:----:|------|
| Fair eval 512 | ✅ **완료** | `fair_eval_512_vs6v.json`, α=0.3 best (PSNR 24.85) |
| 3600f 512 렌더링 | ✅ **완료** | 7종 렌더 (alpha10 + 5개 view ablation, S18) |
| DiFix 512 pairs | ✅ **완료** | 43.2K pairs, Type 3 (1v+2v→6v) |
| DiFix 학습 | 🔄 **진행 중** | PoC Stage 1, GPU 7, ~step 300/2000 |
| MAMMAL Pseudo-GT 512 | ❌ **미완료** | 384 해상도만 존재 |

**팩트체크**:
- 384 기반 PSNR → 512 네이티브 업데이트: **완료** (S18 fair eval)
- MAMMAL pseudo-GT 512: **P1** — 렌더링 필요 (DiFix Stage 1.5에 필요)

**우선순위**: MAMMAL 512 렌더만 남음 → **P1** (DiFix Stage 1.5 전에 필요)

---

## 7. BehaviorSplatter Paper / Feature Canonical Set

### 상태: 🟡 PARTIALLY DONE (프레임워크 수립, 구현 일부)

**증거**:
- 논문 제목: "BehaviorSplatter: Multi-Species Novel-View 3D Reconstruction Benchmark for Animal Behavior Analysis" (PHASE2 roadmap)
- Feature canonical set 논의됨 (deliberation 기록)

| Feature | 상태 | 증거 |
|---------|:----:|------|
| **S1**: Raw KP (66d) | ✅ | h1_sdannce_probe.py, extract_sdannce_features.py |
| **S3**: B-SOiD-style (100+d) | ✅ | h1_sdannce_probe.py에 S3_single/S3_dyadic 구현 |
| **D1**: Body-Part Gaussian (308d) | ✅ | extract_covariance_features.py (22×7 static + 22×9 temporal) |
| **D3**: DINOv2 (768d) | ✅ | **extract_dinov2_features.py 존재!** (ViT-B/14 patch aggregation) |
| **D4**: Scene Flow | ❌ | scene_flow_quiver.png 1장만. 추출 파이프라인 미구현 |

**수정된 평가**: S1~D3 모두 코드 존재! D4 (Scene Flow)만 미구현.
- D4는 Deformation module의 Gaussian identity 미유지 문제로 기존 접근 불가 → 대안 필요

**논문 draft**: `docs/_archive/phase1_root/PAPER_DRAFT_BehaviorSplatter.md` (v1.0) 존재
- Title: "BehaviorSplatter: Template-Free 3D Reconstruction and Behavior Analysis..."
- NeurIPS 2026 Datasets Track primary

**우선순위**: D4 대안 설계 **P2**. 나머지 canonical set은 모두 실행 가능.

---

## Priority Matrix (NeurIPS 2026 기준, bias-corrected v2)

| 순위 | 항목 | 카테고리 | 근거 | 상태 |
|:----:|------|----------|------|:----:|
| **1** | **Foreground-masked HLAC 재실험** | 실험 | 현재 결과 무효 — 논문 핵심 주장의 전제 | ⚠️ P0 |
| **2** | Multi-species pipeline (Rat FT) | 데이터 | NeurIPS "multi-species" 차별화 | 🔄 진행 중 |
| **3** | DiFix fine-tuning + novel view | 품질 | Novel view quality = 논문 기여 | 🔄 진행 중 |
| **4** | Foreground-only PSNR/LPIPS 평가 | 검증 | Background inflation 문제 해결 | 미시작 |
| **5** | Gaussian parameter 분포 (Item 5) | 분석 | `analyze_gaussian_distributions.py` 실행 | 30분 |
| **6** | MAMMAL 512 pseudo-GT (Item 6) | 데이터 | DiFix Stage 1.5 전제 조건 | 2시간 |
| **7** | 100-sample validation set (Item 3) | 평가 | 벤치마크 신뢰성 | 1시간 |
| **8** | D4 Scene Flow 대안 설계 (Item 7) | 설계 | Deformation identity 문제 우회 필요 | 설계 단계 |
| **9** | VAME/Selfee (Item 2) | 실험 | Baseline 확장, 후순위 | 8시간+ |

> **⚠️ Note**: "코드 존재 ≠ 검증 완료". 모든 feature 추출은 foreground masking 적용 후 재평가 필요.
> HLAC 결과는 foreground-masked 재실험 전까지 논문에 사용 불가.

---

## Action Items (이번 세션 이후)

### 즉시 (P0)
- [x] DiFix Stage 1 학습 (진행 중, GPU 7)
- [x] SAM2 1000 frames annotation (진행 중, GPU 5)
- [ ] DiFix 학습 완료 → GT camera + bottom view 평가
- [ ] Rat FT 데이터 변환 (SAM2 완료 후)

### 다음 세션 (P1)
- [ ] HLAC Gaussian probe 실행 (s-DANNCE mouse → GS-LRM → covariance → probe)
- [ ] Gaussian opacity/scale 분포 histogram 체계적 생성
- [ ] MAMMAL pseudo-GT 512 렌더링
- [ ] DiFix Stage 1.5 (Type 2 novel view 혼합)

### 이후 (P2)
- [ ] D3 DINOv2 feature 추출 파이프라인
- [ ] D4 Scene Flow 추출 파이프라인
- [ ] 100-sample validation set + artifact mask annotation
- [ ] VAME/Selfee baseline 실험

---

## Foreground-Masked HLAC Experiment Plan (P0)

### Why
현재 HLAC 결과는 **foreground masking 없이** 추출된 Gaussian features 사용 → background-contaminated → 무효.
올바른 검증을 위해 foreground-only features로 재실험 필요.

### 이미 존재하는 도구

| 도구 | 파일 | 기능 |
|------|------|------|
| **N≥2 Visibility Filter** | `multiview_visibility_filter.py` | Gaussian center를 6 view에 projection → N≥2 view에서 foreground에 속하는 것만 선별 |
| **Capsule Filter** | `capsule_filter.py` | 22 skeleton joints 기반 anatomical capsule → foreground Gaussian 분류 |
| **Covariance Extractor** | `extract_covariance_features.py --n_filter 2` | N≥2 filter 적용 옵션 이미 존재 |

### 실험 프로토콜

```
Step 1: Foreground Gaussian extraction (3 conditions)
  A) Opacity threshold sweep: top-K with threshold 0.6, 0.7, 0.8, 0.9
  B) N≥2 multiview visibility filter (이미 구현, --n_filter 2)
  C) Capsule filter (anatomical, 이미 구현)

Step 2: Feature extraction per condition
  - Gaussian_raw_fg (31d, foreground-only)
  - Covariance_fg (22×7=154d, foreground-only)

Step 3: Classification with GROUND TRUTH labels
  - s-DANNCE HLAC 13-class labels (NOT K-means clusters)
  - Linear SVM + 5-fold CV
  - Compare: KP_centered vs Gaussian_raw_fg vs Covariance_fg vs Combined

Step 4: Complementarity test (proper)
  - CCA (Canonical Correlation Analysis) between KP and Gaussian features
  - Combined vs individual downstream accuracy
  - If Combined > max(KP, Gaussian): evidence for complementarity
  - If Combined ≈ KP: Gaussian is redundant
```

### 명령어 (예상)

```bash
# Step 1B: N>=2 filter + covariance extraction
CUDA_VISIBLE_DEVICES=4 python -m mouse_extensions.behavior.extract_covariance_features \
    --n_filter 2 --output_dir outputs/features/covariance_n2

# Step 3: Classification with s-DANNCE labels
python -m mouse_extensions.behavior.h1_sdannce_probe \
    --feature_dir outputs/features/covariance_n2 \
    --labels_path /home/joon/data/sdannce/mouse/dataverse/HLAC_labels.mat
```

### 예상 소요: 2-3시간 (GPU 1개)

---

## Deliberation Notes

### 3모델 합의 — Confirmation Bias Audit (260322)

**Round 1** (초기):
- Multi-species + novel view quality 우선 (현재 진행 중)
- HLAC probe unblock

**Round 2** (bias correction, 3-model deliberation):
- **HLAC ARI=0.052 해석 ❌ WRONG**: "상보적" 아님, "random disagreement"
- **Foreground masking 부재 = 치명적**: 모든 Gaussian feature 실험 무효화 위험
- **PSNR background inflation**: foreground-only 메트릭 필수 병기
- **"완료" ≠ "검증 완료"**: 코드 존재가 결과 검증을 보장하지 않음
- **프로젝트 전반 6개 주장에서 confirmation bias 발견** (상단 Warning 참조)

---

*FaceLift | Past Session Audit | v2.0 (bias-corrected) | 2026-03-22*
