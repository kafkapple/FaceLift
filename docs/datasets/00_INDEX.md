# Dataset Documentation Hub

> **Navigation**: [← MoC](../00_MoC_INDEX.md) | [CLAUDE.md](../../CLAUDE.md)
> **SSOT**: 모든 데이터셋 관련 문서의 중앙 허브
> **최종 업데이트**: 2026-01-26

---

## Quick Reference

| 용도 | 권장 | 특징 |
|------|------|------|
| **기준선** | D7_1 (M1) | Affine, PP=256, No zoom |
| **정밀** | D8 (M2) | Homography, PP=256, No zoom |
| **운영** | **M3_2** ⭐ | Per-sample zoom, PP=256 |
| **Hold-out** | M3_3t | Pose-Splatter 1/3 split |

---

## 핵심 문서

| 문서 | 내용 | 우선순위 |
|------|------|----------|
| **[[DATASET_ANALYSIS]]** | 성공/실패 요인 종합, Coverage 정의 | ⭐ 필독 |
| **[[PREPROCESSING_REGISTRY]]** | 프리셋 정의, 전처리 명령어 | ⭐ 필독 |
| [[EXPERIMENT_RESULTS]] | 실험 결과 PSNR 비교 | 참조 |
| [[M3_SERIES_SPEC]] | M3 계열 상세 | 참조 |

---

## 데이터셋 상태

### ✅ Active (사용 권장)

| 데이터셋 | Transform | Zoom | PP | 용도 |
|----------|-----------|------|-----|------|
| **D7_1** (M1) | Affine | ❌ | 256 | 기준선 |
| **D8** (M2) | Homography | ❌ | 256 | 기준선 |
| **M3_2** ⭐ | Homo+Zoom | ✅ | 256 | **권장** |
| **M3_3** | Homo+Zoom | ✅ | 256 | 권장 |
| **M3_3t** | M3_3 + Split | ✅ | 256 | Hold-out test |

### ❌ Deprecated (사용 금지)

| 데이터셋 | 문제 |
|----------|------|
| M4 | PP 가변 (object-centered) |
| M3_norm, M3_persample | PP 가변 |
| D1~D6 | 기하학 손상 |

---

## 핵심 발견

### PP 정합이 핵심
- **PP=256**: GS-LRM pretrained 호환 ✅
- **PP 가변**: Ray error 13-16° → Ghosting ❌

### Transform 차이 미미
- D7_1 (Affine): 20.93 PSNR
- D8 (Homography): 20.21 PSNR
- 차이: +0.72 (Affine이 약간 높음)

### Coverage 효과 (검증 필요)
- 실측: D7_1 ~2-3%, M3_2 ~6%
- 가설: Coverage↑ → PSNR↑ (M3_2 결과로 검증)

---

## Quick Start

```bash
# 전처리 (M3_2)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 --input-dir /home/joon/data/raw/markerless_mouse_1_nerf \
    --output-dir /home/joon/data/preprocessed/FaceLift_mouse/M3_2

# 학습
torchrun --standalone --nproc_per_node=1 train_gslrm.py -d M3_2 -e E1_2_alpha

# Hold-out Test Split 생성
python -m mouse_extensions.preprocessing.create_temporal_split \
    --input M3_3 --output M3_3t --symlink
```

---

## 관련 문서

| 카테고리 | 문서 |
|----------|------|
| 분석 | [[DATASET_ANALYSIS]], [[EXPERIMENT_RESULTS]] |
| 설정 | [[PREPROCESSING_REGISTRY]], [[M3_SERIES_SPEC]] |
| 이론 | [[../theory/PP_FX_MVG_ANALYSIS]] |
| 참조 | [[RAW_DATA]], [[CAMERA_CONFIG]], [[VERSION_SCHEMA]] |

---

*Dataset Hub v5.0 | 2026-01-26 | 간소화, Coverage 수정*
