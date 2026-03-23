# Frame Selection Literature Review

Pose-Diversity 기반 Frame Selection 방법론 문헌 조사

---

## Executive Summary

90K 프레임(50fps rat video, 6-cam)에서 3-5K 학습 프레임을 선별하는 문제는 **coreset selection**, **active learning**, **video summarization** 세 분야의 교차점에 위치한다.

핵심 발견:
1. DeepLabCut의 k-means clustering이 가장 성숙한 실용적 baseline
2. DANNCE 파이프라인에서 pose cluster 기반 stratified sampling이 이미 검증됨
3. FisherRF(ECCV 2024)의 information-theoretic view selection이 3DGS에 직접 적용 가능한 최신 방법론
4. **Gap**: "animal multi-view 3D reconstruction을 위한 pose-diversity frame selection"은 기존 문헌 미개척 영역

**권장 접근**: "keypoint FPS + behavioral stratification" 하이브리드

---

## 1. Core Methods for Intelligent Frame Selection

### 1.1 Keypoint-based Diversity Sampling

#### K-means Clustering in Pose Space

| 항목 | 상세 |
|------|------|
| 대표 구현 | DeepLabCut `extract_frames(algo='kmeans')` |
| 알고리즘 | 프레임 다운샘플링 → flatten → k-means → 클러스터 대표 선택 |
| 장점 | 시각적 다양성 확보, 희소 행동 포착 |
| 한계 | O(n*k*d) per iteration, 극히 희소 이벤트 누락 가능 |
| 권장량 | 100-200 frames (DLC 기준) |
| 신뢰도 | ★★★★★ (수천 논문에서 사용) |

#### Farthest Point Sampling (FPS) in Pose Space

| 항목 | 상세 |
|------|------|
| 핵심 | Eldar et al. (1997), iterative 최원점 선택 |
| 복잡도 | Naive O(NC), 최적화 O(N log N) |
| 장점 | k-means보다 uniform coverage 보장, 파라미터 없음 |
| 한계 | noise/outlier에 민감 |
| 적용성 | 23 kp × 3D = 69-dim, 90K→5K = ~450M ops → 수 초 |
| 변형 | Curvature-Informed FPS (2024), Adjustable FPS (2022) |
| 신뢰도 | ★★★★☆ |

### 1.2 Motion-based Filtering

| 항목 | 상세 |
|------|------|
| 접근 | 연속 프레임 간 keypoint displacement → 정적 프레임 필터링 |
| 관련 | MotionGS (NeurIPS 2024) — optical flow + deformable 3DGS |
| 적용 | `displacement_t = \|\|kp_{t+1} - kp_t\|\|_2` → 임계값 이하 제거 |
| 권장 | **전처리 단계**로 사용 (정적 제거) → diversity를 후처리로 적용 |
| 신뢰도 | ★★★★☆ |

### 1.3 Information-Theoretic Frame Selection

#### FisherRF (ECCV 2024 Oral)

| 항목 | 상세 |
|------|------|
| 논문 | Jiang et al., "Active View Selection and Uncertainty Quantification for Radiance Fields using Fisher Information" |
| 알고리즘 | Fisher Information Matrix → Expected Information Gain (EIG) 최대화 뷰 선택 |
| 속도 | 3DGS backend에서 70 fps view selection |
| 장점 | model-aware, ground truth 불필요, 1 backward pass |
| 한계 | 학습된 모델 필요 (cold-start), view selection ≠ frame selection |
| 적용 | 2-stage: 초기 모델 → FisherRF → 추가 프레임 → 재학습 |
| 신뢰도 | ★★★★★ |

### 1.4 Active Learning Approaches

| 항목 | 상세 |
|------|------|
| 논문 | Taketsugu et al., "Active Transfer Learning for Efficient Video-Specific Human Pose Estimation", WACV 2024 |
| 접근 | model uncertainty 기반 informative frame 선택 |
| 적용성 | DANNCE가 이미 pose 추출 → training data budget 문제로 변환 |

---

## 2. Animal Pose Estimation의 Frame Selection 전략

### 2.1 DANNCE / s-DANNCE

| 항목 | 상세 |
|------|------|
| Frame Selection | **Pose cluster 기반 stratified sampling**: 7 clusters k-means → 각 5 frames |
| PAIR-R24M | 10K timepoints random sampling → 60K frames (6 cam) |
| 행동 레이블 | 11 behavioral labels + 3 interaction categories |
| 신뢰도 | ★★★★★ (Harvard Olveczky Lab, NeurIPS Datasets 2021) |

**핵심**: DANNCE 팀 자체가 "pose cluster → stratified sampling"을 사용. 학습 세트 구성에 직접 적용 가능.

### 2.2 DeepLabCut

| 방법 | 상세 |
|------|------|
| `uniform` | 시간 균등 |
| `kmeans` | 시각적 클러스터링 (width=30 다운샘플) |
| `manual` | GUI 선택 |
| 권장량 | 100-200 frames |
| 신뢰도 | ★★★★★ (10,000+ citations) |

### 2.3 SLEAP

| 항목 | 상세 |
|------|------|
| 접근 | Human-in-the-loop active learning: ~10 frames → 모델 → 불확실 프레임 제안 → 반복 |
| 효율 | <200 labels로 peak accuracy 90% 달성 |
| 신뢰도 | ★★★★★ (Nature Methods 2022) |

### 2.4 Keypoint-MoSeq / B-SOiD

| 항목 | 상세 |
|------|------|
| Keypoint-MoSeq | Datta Lab, Nature Methods 2024 — pose dynamics → behavioral syllables 자동 발견 |
| B-SOiD | keypoint 특징 클러스터링 → 행동 상태 분류 |
| 적용 | 행동 syllable 분류 → 각 category에서 proportional sampling |

---

## 3. 3D Reconstruction Frame Selection 전략

### 3.1 Deformable NeRF 계열

| 논문 | 년도 | 전략 |
|------|------|------|
| Nerfies | 2021 | Per-frame latent code, **전체 프레임 사용** |
| D-NeRF | 2021 | Per-frame deformation, 전체 사용 |
| HyperNeRF | 2021 | Higher-dim representation, 전체 사용 |

**관찰**: 대부분 dense regular sampling 가정 (단일 카메라, 짧은 비디오). 90K×6cam에서는 frame selection 필수.

### 3.2 3D Gaussian Splatting

| 관찰 | 상세 |
|------|------|
| 원본 3DGS | COLMAP 전체 이미지 사용, explicit selection 없음 |
| Sparse-view | multi-height + wide-angle coverage가 핵심 |
| SWinGS (ECCV 2024) | Sliding window temporal partitioning, motion 적응 |
| NBV (Next-Best-View) | Uncertainty 기반, emerging trend |

---

## 4. Coreset Selection Methods

### Method Taxonomy (arXiv 2505.17799, 2025)

| 카테고리 | 대표 방법 | 적용성 |
|----------|----------|--------|
| **Geometry-based** | k-Center Greedy, Herding | ★★★★★ (label 불필요) |
| **Scoring-based** | Forgetting, GraNd | ★★★☆☆ (학습 중 필요) |
| **Diversity (Submodular)** | FASS, PRISM | ★★★★☆ |
| **Gradient Matching** | CRAIG, GradMatch | ★★☆☆☆ (task-specific) |
| **Bilevel Optimization** | GLISTER, RETRIEVE | ★★☆☆☆ (val set 필요) |

### Submodular Optimization

| 항목 | 상세 |
|------|------|
| 이론 | Diminishing returns → diversity + representativeness 모델링 |
| 보장 | Greedy (1-1/e) ≈ 0.63 approximation ratio |
| 적용 | Facility Location → pose diversity coverage 보장 |

---

## 5. Pose Diversity 정량 지표

| 지표 | 정의 | 장점 | 한계 |
|------|------|------|------|
| Mean Pairwise Distance | 선택 프레임 간 pose distance 평균 | 직관적 | 클러스터 구조 미반영 |
| Coverage Radius | 최대 nearest-selected distance (k-Center) | Coverage 직접 측정 | 극단값 민감 |
| Cluster Coverage | K clusters 중 포함 비율 | 행동 다양성 직접 | K 선택 의존 |
| Chamfer Distance | 원본-subset 대칭 거리 | 전체 분포 유사도 | O(n*m) |
| MMD | Kernel 기반 분포 차이 | 이론적 보장 | Kernel 선택 필요 |

**DANNCE 사례**: 7 clusters, 각 균등 추출 → cluster coverage 100% = 실질적 standard.

---

## 6. Gap Analysis & NeurIPS Contribution

**발견된 Gap**: "Animal multi-view 3D reconstruction을 위한 pose-diversity frame selection"은 기존 문헌에서 다루어지지 않은 영역.

이 gap 자체가 NeurIPS Datasets Track contribution의 한 축:
> "90K 프레임 중 pose-diversity aware selection으로 3-5K를 선별하여,
> uniform sampling 대비 X% 향상된 reconstruction quality를 달성"
→ 방법론적 기여

---

## 7. Recommended Pipeline (3-Stage Hybrid)

```
Stage 1: Motion Filtering (90K → ~60K)
├── Keypoint displacement: ||kp_{t+1} - kp_t||_2
├── 정적 프레임 제거 (displacement < threshold)
└── 극단적 motion blur 제거

Stage 2: Behavioral Stratification (60K → category별 할당)
├── Keypoint-MoSeq 또는 k-means로 행동 syllables 분류
├── 각 category에 proportional/equal 할당량
└── 희소 행동 oversampling (√freq weighting)

Stage 3: Pose-Space FPS (category별 → 최종 3-5K)
├── 각 behavioral category 내에서 FPS 수행
├── 69-dim pose vector (23 kp × 3D) 사용
└── Chamfer distance로 분포 유사도 검증
```

### 검증 지표

| 지표 | 목표값 |
|------|--------|
| Cluster Coverage | 100% |
| Chamfer Distance | < uniform sampling |
| PSNR | > uniform sampling baseline |
| Behavioral Gini | < 0.3 |

---

## References

### Animal Pose Estimation & Frame Selection
- SLEAP, Nature Methods 2022
- DeepLabCut, Nature Methods 2022
- DANNCE, GitHub (Harvard Olveczky Lab)
- PAIR-R24M, NeurIPS Datasets 2021
- Keypoint-MoSeq, Nature Methods 2024

### 3D Reconstruction & View Selection
- FisherRF, ECCV 2024 Oral
- SWinGS, ECCV 2024
- MotionGS, NeurIPS 2024
- Nerfies/HyperNeRF, ICCV/SIGGRAPH 2021

### Coreset Selection
- Coreset Selection Survey, arXiv 2025
- Submodular Video Summarization, CVPR 2015
- Submodularity in Data Subset Selection, ICML 2015

### Farthest Point Sampling
- Beyond FPS in Point-Wise Analysis, arXiv 2021
- Curvature-Informed FPS, arXiv 2024

---

*Literature Review v1.0 | 2026-03-23 | FaceLift Phase 2 — Frame Selection Research*

↑ [INDEX](../INDEX.md) | ↔ [PREPROCESSING_REGISTRY](../datasets/PREPROCESSING_REGISTRY.md) | ↔ [PIPELINE_ARCHITECTURE](PIPELINE_ARCHITECTURE.md)
