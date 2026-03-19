# Related Work Survey for NeurIPS 2026 Dataset Track

> Multi-model research analysis | Created: 2026-03-19

---

## Positioning Summary

**Our Contribution Space**: Multi-view feed-forward 3DGS + behavior analysis 통합 benchmark — 현재 부재.

| vs Competitor | Their Approach | Our Differentiation |
|--------------|----------------|---------------------|
| **Pose Splatter** (NeurIPS 2025) | Shape carving + 3DGS, template-free | Richer appearance + novel view quality + behavior benchmark |
| **MoReMouse** (AAAI 2026) | Monocular + mesh | Multi-view + Gaussian, higher quality |
| **MAMMAL** (NatComm 2023) | Articulated mesh fitting | Dense Gaussian + differentiable rendering |
| **Keypoint-MoSeq** (NatMethods 2024) | Sparse keypoints → behavior | Dense 3DGS embedding → richer features |
| **PAIR-R24M** (NeurIPS 2021 D&B) | Rat keypoint dataset | Dense 3D reconstruction + multi-species |

---

## Must-Cite (7 papers)

| Paper | Venue | Role |
|-------|-------|------|
| GS-LRM | ECCV 2024 | Our backbone |
| FaceLift | arXiv 2024 | Original upstream |
| Pose Splatter | NeurIPS 2025 | Direct competitor |
| MAMMAL | NatComm 2023 | Data source + mesh baseline |
| DANNCE | NatMethods 2021 | Keypoint gold standard |
| Keypoint-MoSeq | NatMethods 2024 | Behavior analysis SOTA |
| PAIR-R24M | NeurIPS 2021 D&B | Dataset Track precedent |

## Strong-Cite (8 papers)

L4GM (NeurIPS 2024), GART (CVPR 2024), 3D-Fauna (CVPR 2024), AiM (NeurIPS 2025 D&B), SBeA (NatMachIntell 2024), B-SOiD (NatComm 2021), VAME (CommBiol 2022), MoReMouse (AAAI 2026)

## Context-Cite (10 papers)

4D-GS, GaussianAvatar, SC-GS, Animal3D, AP-10K, DogMo, AniMer, DLC2Action, SUBTLE, AnimalGS

---

## Gap Analysis

| Gap | Evidence | Our Fill |
|-----|----------|----------|
| No multi-view feedforward 3DGS + behavior benchmark | Survey of 25 papers | Dataset + benchmark + baselines |
| No calibrated multi-view 3DGS rodent dataset | PAIR-R24M=keypoints only, DogMo=dog | 6-cam, 3600 frames, Gaussians + keypoints |
| No 3DGS embedding → behavior classification study | All prior work uses keypoints | Dense feature extraction + clustering |

---

## Area 1: 3DGS for Deformable Objects (8 papers)

| Paper | Venue | Relevance |
|-------|-------|:---------:|
| 4D Gaussian Splatting | CVPR 2024 | ★★★ |
| Deformable 3D Gaussians | CVPR 2024 | ★★★ |
| GS-LRM | ECCV 2024 | ★★★★★ |
| GART | CVPR 2024 | ★★★★ |
| GaussianAvatar / 3DGS-Avatar | CVPR 2024 | ★★★ |
| L4GM | NeurIPS 2024 | ★★★★ |
| SC-GS | CVPR 2024 | ★★★ |
| FaceLift | arXiv 2024 | ★★★★★ |

## Area 2: Animal Benchmarks (13 papers)

| Paper | Venue | Relevance |
|-------|-------|:---------:|
| Pose Splatter | NeurIPS 2025 | ★★★★★ |
| MAMMAL | NatComm 2023 | ★★★★★ |
| MoReMouse | AAAI 2026 | ★★★★★ |
| DANNCE | NatMethods 2021 | ★★★★ |
| SLEAP | NatMethods 2022 | ★★★★ |
| DeepLabCut | NatMethods 2018/2022 | ★★★★ |
| PAIR-R24M | NeurIPS 2021 D&B | ★★★★ |
| Animal3D | ICCV 2023 | ★★★ |
| AP-10K | NeurIPS 2021 | ★★★ |
| 3D-Fauna | CVPR 2024 | ★★★★ |
| AniMer | CVPR 2025 | ★★★ |
| DogMo | arXiv 2025 | ★★★★ |
| AiM | NeurIPS 2025 D&B | ★★★★ |

## Area 3: 3D-Based Behavior Analysis (6 papers)

| Paper | Venue | Relevance |
|-------|-------|:---------:|
| Keypoint-MoSeq | NatMethods 2024 | ★★★★★ |
| B-SOiD | NatComm 2021 | ★★★★ |
| VAME | CommBiol 2022 | ★★★★ |
| SBeA | NatMachIntell 2024 | ★★★★ |
| SUBTLE | IJCV 2024 | ★★★★ |
| DLC2Action | bioRxiv 2025 | ★★★ |

---

*Total: 25 key references identified*
*Created: 2026-03-19 | Multi-model research analysis*
