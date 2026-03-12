# Document Audit & Consolidation Report

> Created: 2026-03-03 | Status: Complete

## 1. Audit Scope

4개 문서 소스를 전수 조사:

| Source | Location | Files | Role |
|--------|----------|:-----:|------|
| Obsidian FaceLift | `~/Documents/Obsidian/.../FaceLift/` | ~101 | Research notes, presentations |
| Obsidian PoseSplatter | `~/Documents/Obsidian/.../PoseSplatter/` | ~32 | PS experiment notes |
| FaceLift Project (local) | `/Users/joon/dev/FaceLift/docs/` | ~44 | Authoritative docs (SSOT) |
| gpu03 Server | `/home/joon/dev/FaceLift/` | — | Code + experiments (SSOT for code) |

## 2. Critical Findings

### 2.1 SSOT Conflicts (3개)

| # | Conflict | Obsidian | Project docs | Resolution |
|---|----------|----------|-------------|------------|
| 1 | INDEX.md 버전 | v8.0 (outdated) | v10.0+ (current) | **Project docs가 SSOT** |
| 2 | 실험 결과 수치 | 일부 초기값 (H3 PSNR_fg 8.85) | 최신 fair eval (H4b 9.04) | **Project docs가 SSOT** |
| 3 | H4b E2E 결과 | 없음 | ⚠️ 21.02 (GS-LRM only 오인) | **수정 완료** (E2E_EXPERIMENT_ANALYSIS.md) |

### 2.2 Duplicate Content (7개 클러스터)

| Cluster | Files | 핵심 내용 | Action |
|---------|:-----:|----------|--------|
| Pose conditioning | 4 | H6a, H7, Plucker hypothesis | → `plucker_ray_analysis.md`로 통합 |
| Experiment status | 3 | Lab meeting, status report, registry | → `EXPERIMENT_REGISTRY.md`가 SSOT |
| Diffusion bottleneck | 3 | 260222, improvement plan, master plan | → `E2E_EXPERIMENT_ANALYSIS.md`로 통합 |
| Fair comparison | 2 | FL_vs_PS_comparison, 260222 note | → `FL_vs_PS_comparison.md`가 SSOT |
| Preprocessing | 3 | PP bug, registry, M5 spec | → `PREPROCESSING_REGISTRY.md`가 SSOT |
| Visual embedding | 2 | 260210, 260211 notes | PoseSplatter 측 (Obsidian only) |
| Camera convention | 3 | Convention guide, coordinate error, meeting | → `COORDINATE_SYSTEMS.md` (Obsidian) |

### 2.3 Stale/Outdated Documents (12개)

| Document | 문제 | Status |
|----------|------|--------|
| `_archive/기타_claude_logs/*` (11 files) | Claude 세션 로그, 정제 안됨 | Archive 유지 |
| `260205_FaceLift_Master_Experiment_Plan.md` | Phase 1-3 계획이 현재 Phase 5+ | Historical reference |
| `_Notes/260117_Principal_Point_Bug_Analysis.md` | PP bug 해결됨 | Archive |
| `_archive/docs_outdated/*` (7 files) | 이미 archival됨 | 유지 |
| `Presentation/English - Research - Facelift.md` | 초안 수준, 미완성 | Update needed |

### 2.4 Missing Documents

| 필요 문서 | 현재 상태 | Priority |
|----------|----------|:--------:|
| 종합 논문 초안 | 없음 | **P1** |
| NeurIPS gap analysis | 없음 | **P1** |
| H7/H7v2 spatial token 전용 문서 | plucker_ray_analysis.md에 일부 | P2 |
| PS M5 fair eval 상세 보고서 | 수치만 있음 | P2 |
| Visual embedding downstream task 문서 | PS Obsidian에만 | P2 |

### 2.5 Naming Inconsistencies

| 패턴 | 예시 | 표준 |
|------|------|------|
| H vs E prefix 혼용 | H3=E2+pose, H4=extended, H4b=different extended | H=hypothesis, E=experiment epoch |
| 날짜 형식 | 260224 vs 2026-02-24 | 파일명=YYMMDD, 본문=YYYY-MM-DD |
| PSNR 타입 미명시 | "PSNR 22.34" vs "PSNR_fg" vs "PSNR_gt" | 항상 suffix 명시 필수 |

## 3. Key Numerical Data (Verified Cross-Source)

### 3.1 GS-LRM View Ablation (Fair Eval, M5t2 Test, 360f × 5 eval views)

| Views | PSNR_gt | IoU | Source |
|:-----:|:-------:|:---:|--------|
| 1 | 10.47 | 0.028 | CLAUDE.md, E2E_ANALYSIS |
| 2 | 15.95 | 0.858 | CLAUDE.md, E2E_ANALYSIS |
| 3 | 18.56 | 0.899 | CLAUDE.md, E2E_ANALYSIS |
| 4 | 20.66 | 0.926 | CLAUDE.md, E2E_ANALYSIS |
| 5 | 22.16 | 0.942 | CLAUDE.md, E2E_ANALYSIS |
| **6** | **23.84** | **0.954** | CLAUDE.md, E2E_ANALYSIS |

### 3.2 E2E Results (Verified, All Path 2b)

| Experiment | MVDiff Checkpoint | PSNR_gt | IoU | Pose |
|-----------|-------------------|:-------:|:---:|------|
| baseline_360f | ckpt-5000 (sparse) | 7.93 | 0.474 | None |
| e1_cosine_20k | E1 cosine 20K | 7.90 | 0.528 | None |
| e2_resume_20k | E2 resume 20K | 8.20 | 0.521 | None |
| e3_pose | E3 extrinsic | 8.10 | 0.523 | Extrinsic+Add |
| p1_6view_e2e | ckpt-5000 | 8.44 | 0.495 | None |
| H4b_step20000_e2e | H4b ckpt-20000 | ⏳ 실행 중 | ⏳ | Extrinsic+Add |

### 3.3 PoseSplatter Fair Eval (M5 Test, Same Dataset)

| Config | PSNR_gt_masked | IoU | Type |
|--------|:--------------:|:---:|------|
| 6cam | 13.78 | 0.846 | Per-scene opt. |
| 5cam (holdout 4,5) | 13.92 | 0.849 | Per-scene opt. |
| 5cam_temporal_deform | — | 0.843 | Per-scene + temporal |

### 3.4 MVDiffusion Val PSNR (Best Checkpoints)

| Experiment | Method | Best Val PSNR | Step |
|-----------|--------|:-------------:|:----:|
| E2 baseline | No pose | 26.82 | 15K |
| H3 (E2+pose) | Extrinsic+Add | 26.55 | 7K |
| H4b Extended | Extended 10K | 26.24 | 4.6K |
| **H6a_v2** | **Plucker+Add** | **27.34** | **9K** |
| H7 (original) | Plucker+Spatial | ~26.1 | 3.6K |

## 4. Consolidation Actions Taken

| # | Action | Files |
|---|--------|-------|
| 1 | E2E 실험 분석 통합 문서 생성 | `E2E_EXPERIMENT_ANALYSIS.md` |
| 2 | Pipeline 감사 보고서 생성 | `PIPELINE_INCONSISTENCY_AUDIT.md` |
| 3 | CLAUDE.md에 검증 프로토콜 추가 | `CLAUDE.md` §8 |
| 4 | 본 감사 보고서 작성 | 이 문서 |

## 5. Recommendations

1. **Obsidian → Project docs 단방향 참조**: Obsidian은 개인 연구 노트, Project docs/가 SSOT
2. **PSNR 타입 항상 명시**: `PSNR_gt`, `PSNR_fg`, `PSNR_wh`, `PSNR_int` 구분 필수
3. **H prefix 정리**: H1-H8 hypotheses 체계 유지하되, 변형(H4b, H6a_v2, H7v2)은 명확히 문서화
4. **Presentation 문서 갱신 필요**: 현재 260215 초안 → 최신 결과 반영 필요

---

*FaceLift Document Audit | v1.0 | 2026-03-03*
