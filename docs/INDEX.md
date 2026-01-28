# FaceLift Mouse Documentation Index

> Single hub for all active documentation.
> **177 → 37 files** (2026-01-28 restructuring)

---

## Quick Access

| 문서 | 용도 |
|------|------|
| [[MOUSE_QUICK_REFERENCE]] | 매일 사용하는 명령어/설정 참조 |

---

## Theory (8)

불변 지식: 수학, MVG, 좌표계, 손실 함수

| 문서 | 내용 |
|------|------|
| [[PP_FX_MVG_ANALYSIS]] | ⭐ PP/fx/MVG 종합 + Error Amplification + Virtual Zoom |
| [[MASK_GUIDE]] | 마스크 시스템 + Literature Review |
| [[GHOSTING_ANALYSIS]] | Ghosting 원인/해결 + Floater Mitigation |
| [[FLOATER_ARTIFACT_ANALYSIS]] | Floater 아티팩트 분석 |
| [[CLIPPING_AND_CENTERING_ANALYSIS]] | 클리핑 및 센터링 |
| [[CENTER_ESTIMATION]] | 3D Triangulation 센터 추정 |
| [[COORDINATE_SYSTEMS]] | 좌표계 변환 + Camera Data Pipeline |
| [[GS-LRM_Loss_Formula]] | 손실 함수 수식 + Metrics Guide |

---

## Datasets (7)

| 문서 | 내용 |
|------|------|
| [[PREPROCESSING_REGISTRY]] | ⭐ 전처리 SSOT (D0~M5) — 수정 금지 |
| [[M5_SERIES_SPEC]] | ⭐ 현재 활성 데이터셋 (M5+ablation) |
| [[RAW_DATA]] | 원본 데이터 + 6카메라 배치 |
| [[LEGACY_DATASETS]] | D0-D10 + M3 시리즈 통합 참조 |
| [[M5_MIGRATION_GUIDE]] | M3→M5 마이그레이션 |
| [[E3_SYNTHETIC_RENDERING_GUIDE]] | 합성 데이터 렌더링 |
| [[MAMMAL_UV_TEXTURE_GUIDE]] | MAMMAL UV 텍스처 |

---

## Experiments (6)

| 문서 | 내용 |
|------|------|
| [[EXPERIMENT_CONFIG_GUIDE]] | ⭐ Config 가이드 + Two-Phase Training |
| [[EXPERIMENT_REGISTRY]] | 실험 이력 + Ablation 결과 |
| [[EXPERIMENT_QUICKSTART]] | 실험 빠른 시작 |
| [[MVDIFFUSION_FINETUNE_GUIDE]] | MVDiffusion finetune + Prompt Embeddings |
| [[TRAINING_LOGGING_GUIDE]] | WandB 로깅 |
| [[VISUALIZATION_SETTINGS]] | 시각화 설정 |

---

## Notes (3)

연구 노트, 분석 보고서

| 문서 | 내용 |
|------|------|
| [[260123_Research_Note]] | Rendering Clamping + Turntable Debug |
| [[Camera_Preprocessing_Analysis_Report]] | 카메라 전처리 분석 |
| [[depth_regularization_plan]] | Depth Regularization 연구 방향 |

---

## Plans (3)

| 문서 | 내용 |
|------|------|
| [[SYNTHETIC_DATA_MANUAL]] | 합성 데이터 매뉴얼 + Overview |
| [[MAMMAL_MESH_RENDERING_PLAN]] | MAMMAL 메쉬 렌더링 계획 |
| [[BLENDER_VALIDATION_PLAN]] | Blender 검증 계획 |

---

## Tutorials (8)

| Step | 문서 |
|------|------|
| 0 | Branch Setup |
| 1 | Fork and Setup |
| 2 | Mouse Dataset |
| 3 | Preprocessing |
| 4 | Config Setup |
| 5 | Training |
| - | VSCode Debug Mask Guide |
| - | [[TOOLS_REFERENCE]] |

---

## Local Obsidian (코드 상세)

서버에서 삭제, 로컬 `_module_refactoring/`에 보관:

| 문서 | 내용 |
|------|------|
| GS-LRM_Architecture_Reference | Architecture Guide + Code Walkthrough + Training Pipeline 통합 |
| 260121_Visualization_Modularization_Plan | 시각화 모듈화 계획 |
| 260122_Pipeline_Deep_Dive | 파이프라인 상세 분석 |
| Visualization_Mask_Consistency_Guide | 시각화 마스크 일관성 |

---

*FaceLift Mouse Documentation | Restructured: 2026-01-28*
*Total: 37 files (theory 8 + datasets 7 + experiments 6 + notes 3 + plans 3 + tutorials 8 + top-level 2)*
