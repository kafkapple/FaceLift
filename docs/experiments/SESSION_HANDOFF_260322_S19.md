# Session Handoff — 260322 S19+

> **Navigation**: [<- INDEX](../INDEX.md) | [PAST_SESSION_AUDIT](PAST_SESSION_AUDIT_260322.md)
> **Created**: 2026-03-22 | **Status**: HANDOFF

---

## 1. 실행 중 프로세스 (gpu03)

| GPU | 작업 | PID/wandb | 로그 | 예상 완료 |
|:---:|------|-----------|------|:---------:|
| **5** | Rat FT v1 (α=0.3, 5000 steps) | wandb: 9mumbw5f | `outputs/rat_ft_v1_r3.log` | 수 시간 |
| **7** | DiFix PoC Type3 (2000 steps) | wandb tracking | `outputs/difix_training/poc_type3/` | 곧 완료 |

### 완료 후 TODO

- **Rat FT 완료 시**: turntable render로 정성 평가 + held-out view PSNR
- **DiFix 완료 시**: GT camera + bottom view 평가 (GPU 7 재활용)

---

## 2. Confirmation Bias 사건 (핵심)

### 발견

이전 세션에서 HLAC ARI=0.052를 "Gaussian과 KP가 상보적"으로 해석 → **3-model deliberation으로 "random disagreement"임이 확인됨**.

### 수정 완료

| 항목 | 파일 |
|------|------|
| 감사 문서 v2.0 | `docs/experiments/PAST_SESSION_AUDIT_260322.md` (Critical Warning 추가) |
| 논문 서사 v4.0 | `memory: project_paper_narrative.md` |
| Feedback 메모리 | `memory: feedback_confirmation_bias.md` |
| Deformation 교차참조 | `memory: project_deformation_finding.md` |

### Foreground HLAC 재실험 결과 (K-means labels)

| Feature | KP-labels Acc | 해석 |
|---------|:------------:|------|
| KP_centered (66d) | **99.2%** | gold standard |
| Cov_N2_static (154d, fg) | **62.5%** | +16.7% vs raw(45.8%) |
| KP+Cov_N2 (220d) | **95.0%** | KP보다 낮음 |

### ⚠️ 3모델 합의: 결론이 성급

| 반론 | 설명 | 보완 실험 |
|------|------|-----------|
| **Circular bias** | KP로 만든 K-means labels → KP에 유리 | s-DANNCE HLAC 13-class labels 필요 |
| **Dimensionality** | 154d 추가 → linear SVM noise 취약 | PCA + RF/MLP |
| **Aggregate** | Body-part별 정보가 전체에서 희석 | per-body-part 분석 |

### 종합 실험 코드 (구현 완료, 실행 대기)

`mouse_extensions/behavior/hlac_comprehensive.py` — 4-part analysis:
1. Multi-classifier (LinearSVM, RF, MLP) + PCA sweep (10, 20, 50)
2. Paired t-test complementarity
3. Body-part analysis (head, torso, forelimb, hindlimb, tail)
4. CCA (Canonical Correlation Analysis)

**블로커**: `keypoints_22_3d.npz` 삭제됨 (§3 참조) → 경로 수정 필요

---

## 3. 스토리지 구조 감사 (다음 세션 P1)

### 삭제된 데이터 (S18 디스크 정리)

| 경로 | 크기 | 영향 |
|------|:----:|------|
| `/node_data/joon/data/results/MAMMAL_mouse/` | 3.4G | **keypoints_22_3d.npz 포함** — 38개 .py 파일 참조 |
| `/node_data/joon/data/results/sam3d_gui_outputs/` | 1.1G | 미확인 |

**원인**: config grep만으로 "orphan" 판단 → Python 코드 내 런타임 경로 참조를 놓침.

**복구 방안**:
1. MAMMAL 파이프라인에서 재생성 (keypoints npz = MAMMAL 추론 결과물)
2. 기존 추출된 features (`outputs/report/clustering/features/`) 직접 사용 (우회)

**영향받는 파일 (38개)**: `grep -r 'keypoints_22_3d' mouse_extensions/ --include='*.py'`
- 핵심: `hlac_m5t2.py`, `hlac_comprehensive.py`, `capsule_filter.py`, `multiview_visibility_filter.py`
- _experiments/ 하위: 16개 (대부분 일회성)
- scripts/: 8개
- analysis/: 2개

### 스토리지 개선 계획 (우선순위순)

#### P1: 즉시 (구조 무결성)

| # | 조치 | 효과 |
|---|------|------|
| 1 | **keypoints_22_3d.npz 복구** (MAMMAL 재추론 or 백업) | 38개 파일 unblock |
| 2 | difix3d symlink: `dev/Difix3D/checkpoints → /node_data/.../difix3d` | 일관성 |
| 3 | models/difix 중복 검증: checksum 비교 후 삭제 | 4.9G 절약 |
| 4 | `node_data/outputs/FaceLift` 역할 확인 → canonical 결정 | 2.8G + 명확성 |
| 5 | Difix3D .gitignore 생성 | 안전 |

#### P2: 단기 (wandb 통합)

| # | 조치 | 효과 |
|---|------|------|
| 6 | WANDB_DIR → node_data_2 설정 (bashrc.local) | 향후 분산 방지 |
| 7 | 기존 wandb 통합 이전 | 5.8G 절약 |

#### P3: 아카이브

| # | 조치 | 효과 |
|---|------|------|
| 8 | data/processed (3.5G) 참조 확인 후 정리 | NVMe 확보 |
| 9 | project_splatter 데이터 별도 보관 | 604M + 정리 |

### 데이터 삭제 안전 체크리스트 (향후 필수)

```
삭제 전 체크:
1. grep -r '{path}' ~/dev/ --include='*.py' --include='*.yaml' --include='*.sh'
2. find ~/dev/ -name '*.py' -exec grep -l '{basename}' {} \;
3. 삭제 대상을 /tmp/로 mv 후 1주일 관찰
4. 삭제 실행 전 사용자 확인
```

---

## 4. 커밋 이력 (이번 세션)

```
71846cb fix: disable wandb.log_code to prevent I/O hang
9c067e6 feat: comprehensive HLAC probe + fix wandb.log_code hang
7dcd91f fix: RAT1 dataset config absolute paths
74335b7 feat: foreground HLAC probe + rat FT configs
d9802e6 docs: bias correction — HLAC reinterpretation + foreground experiment plan
b958b33 docs: past session audit + rat FT guide + stale handoff cleanup
3fb9152 feat(sdannce): rat GS-LRM conversion v2 + DiFix eval scripts
```

---

## 5. 다음 세션 우선순위

| # | 작업 | 전제 조건 |
|---|------|-----------|
| **1** | keypoints_22_3d.npz 복구 (MAMMAL 재추론) | MAMMAL env 확인 |
| **2** | hlac_comprehensive.py 실행 (경로 수정 후) | #1 완료 |
| **3** | Rat FT 결과 확인 + 정성 평가 | 학습 완료 대기 |
| **4** | DiFix eval (GT camera + bottom view) | 학습 완료 대기 |
| **5** | 스토리지 구조 P1 조치 (difix symlink, outputs 통일) | 조사 완료 |
| **6** | s-DANNCE HLAC 13-class labels로 재실험 | mouse s-DANNCE→GS-LRM 필요 |

---

## 6. 핵심 메모리 파일 (갱신됨)

| 파일 | 내용 |
|------|------|
| `feedback_confirmation_bias.md` | ⭐ 모든 실험 해석에서 bias 경계 |
| `project_paper_narrative.md` | v4.0 — HLAC "redundant" 결론, critical path 수정 |
| `project_novel_view_strategy.md` | DiFix FT 진행, zero-shot 실패 확인 |
| `project_rat_pipeline_260322.md` | Rat SAM2+GS-LRM+FT pipeline |
| `project_audit_260322.md` | 7항목 감사 bias-corrected |

---

## 7. 경로 관리 체계 개선안 (P1 — 높은 중요도)

### 현재 문제

38개 Python 파일이 `/node_data/joon/data/results/MAMMAL_mouse/...` 같은 **절대 경로를 하드코딩**. 디스크 정리 시 "참조 없음"으로 오판 → 삭제 → 38개 파일 동시 장애.

### 근본 원인

1. **경로 분산**: 코드 내 절대 경로가 파일마다 다르게 하드코딩
2. **SSOT 부재**: "이 데이터의 canonical 경로"가 어디에도 정의되지 않음
3. **삭제 안전망 부재**: grep 범위를 config만으로 한정 → Python 코드 누락

### 개선안: Centralized Path Registry

**`mouse_extensions/behavior/paths.py`를 SSOT로 확장:**

```python
# paths.py — ALL data paths centralized here
# When data moves, update HERE only. All other files import from here.

from pathlib import Path
import os

# Base directories (environment-aware)
NODE_DATA = Path(os.environ.get("NODE_DATA", "/node_data/joon"))
HOME_DATA = Path(os.environ.get("HOME_DATA", "/home/joon/data"))
PROJECT_ROOT = Path(os.environ.get("FACELIFT_ROOT", "/home/joon/dev/FaceLift"))

# MAMMAL keypoints (primary dependency for behavior analysis)
KP_22_PATH = NODE_DATA / "data/results/MAMMAL_mouse/v012345_kp22_20260126/keypoints_22_3d.npz"

# Preprocessed data
M5_DATA = HOME_DATA / "preprocessed/FaceLift_mouse/M5"
RAT_DATA = PROJECT_ROOT / "outputs/sdannce_rat_ft/gslrm_format"

# Extracted features (already computed, cached)
FEATURES_DIR = PROJECT_ROOT / "outputs/report/clustering/features"
GAUSSIAN_RAW = FEATURES_DIR / "gaussian_raw_features.npz"
COV_N2_STATIC = FEATURES_DIR / "covariance_n2/covariance_static.npy"
COV_N2_TEMPORAL = FEATURES_DIR / "covariance_n2/covariance_temporal.npy"
TEMPORAL_FEATURES = FEATURES_DIR / "temporal/temporal_features.npz"

# Checkpoints
CKPT_DIR = NODE_DATA / "checkpoints/FaceLift/gslrm"

def validate_paths(*paths):
    """Check paths exist, warn on missing."""
    missing = [p for p in paths if not p.exists()]
    if missing:
        import warnings
        for p in missing:
            warnings.warn(f"Path not found: {p}")
    return len(missing) == 0
```

### 마이그레이션 계획

| 단계 | 작업 | 영향 |
|------|------|------|
| 1 | `paths.py` SSOT 확장 (위 설계) | 새 파일 |
| 2 | 38개 파일에서 하드코딩 경로 → `from .paths import KP_22_PATH` | 38 files |
| 3 | 삭제 전 체크 스크립트 작성: `grep -r '{path}' --include='*.py'` | 안전망 |
| 4 | `paths.py`에 `validate_paths()` 추가 → 세션 시작 시 자동 검증 | 조기 발견 |

### 스토리지 계층 규칙 문서화

```
/node_data (NVMe)  = HOT: 학습 중 데이터만. 삭제 시 paths.py 먼저 확인
/node_data_2 (SSD) = WARM: 캐시, wandb, 아카이브
/home (NFS)        = COLD: 코드, 최종 결과물, 장기 보관

규칙:
- 모든 데이터 경로는 paths.py에 등록
- 삭제 전: `grep -r '{basename}' ~/dev/ --include='*.py'` 필수
- 이동 후: paths.py 업데이트 + `validate_paths()` 실행
```

---

*FaceLift | Session Handoff S19+ | 2026-03-22*
