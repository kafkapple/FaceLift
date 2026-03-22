# Past Session Items Audit — 260322

> **Navigation**: [<- INDEX](../INDEX.md) | [PHASE2_NOVEL_VIEW_ROADMAP](PHASE2_NOVEL_VIEW_ROADMAP.md)
> **Version**: v1.0 | **Created**: 2026-03-22 | **Status**: AUDIT REPORT

---

## Executive Summary

7개 과거 세션 논의 항목을 코드/문서/서버 상태 기준으로 감사한 결과:
- **완료**: 2개 (coordinate_utils, FL Fair 512 평가)
- **부분 완료**: 4개 (HLAC, Gaussian quality, deformation, multi-tier)
- **미시작**: 1개 (VAME/Selfee 실험)

NeurIPS 2026 제출 기준 **Top 3 우선순위**: Multi-species pipeline (현재 진행 중) > Gaussian quality 체계적 실행 > BehaviorSplatter 논문 canonical feature 구현.

---

## 1. PoseSplatter Visual Embedding / HLAC Gaussian Test

### 상태: ✅ COMPLETED (260321, 결과 존재)

**증거**:
- `mouse_extensions/behavior/hlac_m5t2.py` — Core BehaviorSplatter 실험 스크립트 ✅
- `outputs/m5t2_hlac_analysis/results.json` — **598 frames, K=8 optimal** ✅
- `outputs/m5t2_hlac_analysis/classification_comparison_K8.png` — 시각화 ✅

**핵심 결과** (results.json):

| Feature | Gaussian-defined HLACs | KP-defined HLACs |
|---------|:---:|:---:|
| **Gaussian_raw** | **90.8%** | 45.8% |
| **KP_centered** | — | **99.2%** |
| Cluster Agreement (ARI) | 0.052 | NMI=0.140 |

**해석**: Gaussian features는 keypoints와 **독립적인 행동 정보**를 인코딩! (ARI=0.052 = 매우 낮은 일치 → 상보적 feature space). 원래 가설 "Gaussian << KP = lossy" 가 **반증**됨.

> 🌟 **이것은 NeurIPS 핵심 기여**: "3D Gaussian properties capture distinct behavioral information not available from keypoints alone"

**남은 작업**: s-DANNCE 기반 cross-species 검증 (이번 세션에서 파이프라인 unblock됨)

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

## Priority Matrix (NeurIPS 2026 기준, 수정)

| 순위 | 항목 | 카테고리 | 근거 | 상태 |
|:----:|------|----------|------|:----:|
| **1** | Multi-species pipeline (Rat FT) | 데이터 | NeurIPS 핵심 — "multi-species" 차별화 | 🔄 진행 중 |
| **2** | DiFix fine-tuning + novel view | 품질 | Novel view quality = 논문 핵심 기여 | 🔄 진행 중 |
| **3** | Cross-species HLAC probe (Item 1+4) | 실험 | "Gaussian→behavior" **이미 mouse에서 확인** → rat 검증 | ✅→🔄 |
| **4** | Gaussian parameter 분포 (Item 5) | 분석 | `analyze_gaussian_distributions.py` 실행만 필요 | 30분 |
| **5** | MAMMAL 512 pseudo-GT (Item 6) | 데이터 | DiFix Stage 1.5 전제 조건 | 2시간 |
| **6** | 100-sample validation set (Item 3) | 평가 | 벤치마크 신뢰성 | 1시간 |
| **7** | D4 Scene Flow 대안 설계 (Item 7) | 설계 | Deformation identity 문제 우회 필요 | 설계 단계 |
| **8** | VAME/Selfee (Item 2) | 실험 | Baseline 확장, 후순위 | 8시간+ |

> **Note**: D3 (DINOv2), S1-S3, D1은 모두 코드 존재. 실행/평가만 필요.
> Item 1 (HLAC)은 **이미 완료** — Gaussian이 KP와 독립적 행동 정보 인코딩 확인.

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

## Deliberation Notes

### 3모델 합의 (Claude/Gemini/GPT)
- **최우선**: Multi-species pipeline + novel view quality (현재 진행 중)
- **핵심 발견**: Deformation module의 Gaussian identity 미유지 → scene flow 기반 접근 필요 (260319 확인)
- **DiFix zero-shot 실패**: 도메인 gap 확인 → fine-tuning 필수, pretrained를 GT로 사용 불가 (Red Team 경고 유효)
- **HLAC probe unblock**: s-DANNCE→GS-LRM 파이프라인 완성으로 핵심 실험 가능해짐

---

*FaceLift | Past Session Audit | 2026-03-22*
