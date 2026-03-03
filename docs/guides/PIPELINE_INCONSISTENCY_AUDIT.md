# Pipeline Inconsistency Audit Report

> Created: 2026-03-03 | Status: Fixes Applied (Phase 1)

## 1. Background

E2E inference pipeline (`MVDiffusion → GS-LRM`) 및 pose conditioning 코드 전체에 대한
비일관성 조사 결과. 3개 병렬 리뷰 (pose conditioning, E2E pipeline, config YAML)에서
**총 16개 이슈** 발견.

## 2. Critical Issues (7개) — 수정 완료

### C1. `method` 기본값 불일치 ✅ Fixed

| 위치 | 기본값 |
|------|--------|
| `mvdiffusion_pipeline.py:154` | ~~`"plucker"`~~ → `"extrinsic"` |
| `create_pose_injector_from_config:565` | `"spherical"` |

**수정**: 추론 기본값을 `"extrinsic"` (가장 보편적)으로 통일.
모든 현행 config에는 `method`가 명시되어 있으므로 실질적 영향은 방어적.

### C2. `camera_json` 상대 경로 비일관성 ✅ Fixed

| 위치 | 경로 타입 |
|------|----------|
| `_DEFAULT_CAMERA_JSON` (line 31) | 절대 (`Path(__file__).parent / ...`) |
| `pose_config` 기본값 (line 157-159) | ~~상대~~ → 절대 |

**수정**: `str(Path(__file__).parent / "cameras" / "m5_cameras.json")`로 통일.

### C3. `--skip_preprocess` 항상 True ✅ Fixed

- `run_e2e_inference.py:191`: `action="store_true", default=True`
- `action="store_true"`는 이미 `default=False` 내장 → `default=True` 제거

### C4. `camera_indices` 적용 시 views 미 slicing ✅ Fixed

- `end_to_end.py:199-202`: cameras만 slicing, views는 누락
- 6뷰 MVDiff + 4뷰 GS-LRM 조합 시 shape 불일치 → `views = views[camera_indices]` 추가

### C5. Cache stale (train/eval 전환) ✅ Fixed

- `pose_conditioning_integration.py`: `_cached_embeddings`가 `train()` 전환 시 무효화 안됨
- `train()` override 추가: `trainable and mode=True` 시 캐시 초기화

### C6. `n_spatial_tokens` 미저장 ✅ Fixed

- `pose_conditioning_integration.py:198`: 계산되지만 인스턴스에 저장 안됨
- `self.n_spatial_tokens` 추가

### C7. Spatial token pipeline 미지원 ✅ Fixed (이전 커밋)

- `mvdiffusion_pipeline.py`: `inject_spatial` 디스패치 + `spatial_token_size` 전달
- 커밋: `50bb51d`

## 3. Important Issues (9개) — 미수정 (문서화)

### I1. `batch_size` 미전달 (학습)

- `train_diffusion.py:672,773`: `inject()` 호출 시 `batch_size` 미전달
- `BN // n_views`로 추론되므로 동작하지만, `spatial_token` + tensor `ref_view_idx` 조합 시 잠재적 shape mismatch
- **우선순위**: Medium (현재 단일 GPU 학습에서는 문제 없음)

### I2. `ref_view_idx` tensor vs int 판단

- `pose_conditioning_integration.py:374,491`: `isinstance(ref_view_idx, int) and ref_view_idx == 0` 체크
- DataLoader 배치화 시 `ref_view_idx`는 tensor → 캐시 미사용 → 매번 재계산 (성능 이슈)
- **우선순위**: Low (정확성 문제 아닌 효율 문제)

### I3. OmegaConf `.get()` vs `.method` 혼용

- `train_diffusion.py:1003-1004`: DictConfig에서 속성 접근 → 키 누락 시 에러
- **우선순위**: Low (모든 현행 config에 해당 키 존재)

### I4. `accelerator.scaler` 비공개 API 의존

- `train_diffusion.py:1146-1156`: Accelerate 버전 변경 시 호환 깨짐
- **우선순위**: Low (현재 Accelerate 버전에서 동작)

### I5. Fallback prompt embed 하드코딩 경로

- `mvdiffusion_pipeline.py:215`: `"mvdiffusion/data/fixed_prompt_embeds_6view/clr_embeds.pt"` 상대 경로
- zeros fallback 시 경고만 출력, silent 품질 저하
- **우선순위**: Medium

### I6. 하드코딩 서버 절대 경로

- `run_e2e_inference.py:280`: `/home/joon/dev/FaceLift` 하드코딩
- `defaults.py`의 `FACELIFT_ROOT` 미재사용
- **우선순위**: Low (현재 환경에서는 문제 없음)

### I7. Auto output_dir mode 분류 버그

- `run_e2e_inference.py:288`: `--input_image` 사용 시 `"gslrm"`으로 잘못 분류
- **우선순위**: Low (대부분 `--output_dir` 명시 사용)

### I8. `rotation_speed` 기본값 3중 불일치

- argparse=0.3, getattr fallback=1.0, 함수 기본=1.0
- **우선순위**: Low (시각화 관련, 추론 품질 무관)

### I9. `num_views` sync 불완전

- `mvdiffusion_pipeline.py:131-133`: `pipe.num_views` 업데이트 후 하위 컴포넌트 동기화 없음
- 4-view 체크포인트를 6-view base에 로드 시 잠재적 shape 불일치
- **우선순위**: Medium (4-view 모델 E2E 사용 시 영향)

## 4. Config YAML 검증 결과

**26개 config 분석 완료** (11개 pose enabled, 15개 disabled):

| 검증 항목 | 결과 |
|----------|------|
| 필수 파라미터 (method, integration, embed_dim) | ✅ 11/11 |
| camera_json 경로 일관성 | ✅ 11/11 |
| spatial_token_size (H7만) | ✅ 설정됨 (8) |
| plucker_resolution (3개) | ✅ 모두 64 |
| trainable 명시 | ⚠️ H6a만 미설정 (기본 frozen) |

**비참조 파일 없음**: 모든 config가 존재하는 체크포인트/camera를 참조.

## 5. Fix Summary

| 커밋 | 수정 내용 | 파일 수 |
|------|----------|------:|
| `50bb51d` | spatial_token pipeline 지원 | 1 |
| (pending) | C1-C6 Critical fixes | 4 |

**수정된 파일:**
- `mouse_extensions/inference/mvdiffusion_pipeline.py` (C1, C2)
- `mouse_extensions/scripts/inference/run_e2e_inference.py` (C3)
- `mouse_extensions/inference/end_to_end.py` (C4)
- `mouse_extensions/model/pose_conditioning_integration.py` (C5, C6)

## 6. Remaining Work

### Phase 2 (권장)
- I1: `batch_size` 명시적 전달 (train_diffusion.py)
- I5: prompt embed 경로 정리 (defaults.py 기반 통합)
- I9: num_views sync 강화

### Phase 3 (선택)
- I2, I3, I4, I6, I7, I8: 비핵심 개선

---

*FaceLift Pipeline Audit | v1.0 | 2026-03-03*
