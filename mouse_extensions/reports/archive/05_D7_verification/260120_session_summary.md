# 260120 Session Summary: Config Modularization & D7 Variants

> **Date**: 2026-01-20
> **Scope**: Config 시스템 모듈화, D7.1/D7.2 전처리 지원

---

## 1. 완료 작업

### 1.1 모듈화 Config 시스템 구축

**위치**: configs/mouse/_modular/

| 폴더 | 파일 수 | 내용 |
|------|--------|------|
| base/ | 2 | model.yaml, runtime.yaml |
| schemas/ | 5 | 5v_alpha, 4v_alpha, 5v_nomask, 6v_alpha, 4v_random |
| datasets/ | 4 | D7, D7_t, D7_1, D7_2 |

**사용법**:
  cd configs/mouse/_modular
  python generate_config.py --dataset D7_1 --schema 5v_alpha

---

### 1.2 Config 정리

| 분류 | 파일 수 | 위치 |
|------|--------|------|
| Active | 30 | configs/mouse/*.yaml |
| Archived | 104 | configs/mouse/_archive/ |

**아카이브 상세**:
- legacy_gslrm_v/: 36개 (gslrm_v15~v71)
- deprecated_D1_D4/: 38개 (D1~D4)
- deprecated_D6/: 30개 (D6-1/2/3)

---

### 1.3 background_loss_weight 수정

- 모든 config에서 background_loss_weight: 1.0 → 0.0
- 코드 기본값: 0.0 (get default)
- 변경된 파일: 100개

---

### 1.4 D7 스크립트 수정

**파일**: mouse_extensions/scripts/preprocess_D7_pp_centered.py

**추가된 기능**:
- --scale-mode 인자 추가
- fx_only (D7), individual (D7.1), average (D7.2) 지원

**수정된 코드**:
  # D7Config에 scale_mode 추가
  scale_mode: str = "fx_only"
  
  # compute_transform에서 scale 분기 처리
  if cfg.scale_mode == 'individual':
      pass  # use scale_x, scale_y directly
  elif cfg.scale_mode == 'average':
      scale_x = scale_y = (scale_x + scale_y) / 2
  else:  # fx_only
      scale_y = scale_x

**버그 수정**:
- main()에서 scale_mode 전달 누락 → 수정
- scale 변수 undefined 오류 → scale_avg로 대체

---

### 1.5 샘플 데이터 검증

| Dataset | scale_mode | fx | fy | scale_x | scale_y |
|---------|------------|-----|-----|---------|---------|
| D7.1_test | individual | 549.0 | 549.0 | 0.3404 | 0.3382 |
| D7.2_test | average | ~547.3 | ~550.7 | 0.3393 | 0.3393 |

---

## 2. 생성된 문서

| 문서 | 위치 |
|------|------|
| PREPROCESSING_REGISTRY.md v3.0 | docs/ |
| 260120_D7_variants_comparison.md | reports/05_D7_verification/ |
| presets.py (업데이트) | preprocessing/ |
| RUN_D7_PREPROCESSING.md | scripts/ |

---

## 3. 다음 단계

### 즉시 실행 가능
  # D7.1 전체 전처리
  python -m mouse_extensions.scripts.preprocess_D7_pp_centered \
      --data-dir /home/joon/data/markerless_mouse_1_nerf \
      --output-dir /home/joon/data/preprocessed/FaceLift_mouse/D7_1 \
      --camera-pkl /home/joon/data/markerless_mouse_1_nerf/new_cam.pkl \
      --frame-interval 5 \
      --scale-mode individual

### 실험 설정 생성
  cd configs/mouse/_modular
  python generate_config.py --dataset D7_1 --schema 5v_alpha

---

## 4. 핵심 변경사항 요약

1. **Config 133개 → 30개 (104개 아카이브)**
2. **모듈화 시스템**: dataset + schema → config 자동 생성
3. **D7 확장**: scale_mode로 D7.1, D7.2 지원
4. **문서 통합**: PREPROCESSING_REGISTRY.md v3.0

---

*Session completed: 2026-01-20*
