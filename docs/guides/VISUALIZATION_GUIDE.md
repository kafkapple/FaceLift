# Visualization Module Guide

BehaviorSplatter 시각화 파이프라인 사용 가이드.

---

## 1. 모듈 구조

```
mouse_extensions/behavior/
├── cinematic_sequence.py           # 시네마틱 영상 생성 (config 기반)
├── cinematic_default.yaml          # 기본 설정 (흰 BG, 8 segments)
├── cinematic_6view_grid.yaml       # 6-view grid 설정
├── multiview_visibility_filter.py  # N-threshold sweep + temporal grid 영상
├── render_bodypart_gaussians.py    # Body-part 렌더링 (v7, 레거시)
├── run_all_visualizations.sh       # 일괄 배치 실행
├── view_projected_filtering.py     # 2D projected filtering (bbox/radial)
├── analyze_gaussian_distributions.py # Gaussian 분포 분석
└── find_active_segments.py         # High-motion 구간 탐색
```

## 2. Quick Start

```bash
ssh gpu03
cd /home/joon/dev/FaceLift
conda activate facelift

# 기본 시네마틱 영상 (1개)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.cinematic_sequence \
    --config mouse_extensions/behavior/cinematic_default.yaml \
    --output-dir outputs/viz/cinematic/mouse/demo

# 일괄 배치 (white/black BG, keypoint on/off, 6-view grid)
CUDA_VISIBLE_DEVICES=5 bash mouse_extensions/behavior/run_all_visualizations.sh
```

## 3. Cinematic Sequence (cinematic_sequence.py)

### Segment Types

| Type | 시간 흐름 | 카메라 | 설명 |
|------|----------|--------|------|
| `flow_gt` | ✅ advancing | GT view | 원본 RGB 연속 재생 |
| `flow_mask` | ✅ advancing | GT view | FG mask overlay 연속 재생 |
| `flow_render` | ✅ advancing | GT view | GS-LRM 렌더 (매 프레임 inference) |
| `freeze_orbit` | ❄️ frozen | 360° turntable | 정지 + 수평 회전 |
| `freeze_elev` | ❄️ frozen | elevation arc | 정지 + 수직 sweep |
| `flow_orbit` | ✅ advancing | turntable | 움직이면서 회전 |
| `flow_bodypart` | ✅ advancing | turntable | 움직이면서 부위 전환 |
| `freeze_bodypart` | ❄️ frozen | turntable | 정지 + 부위 전환 + orbit |
| `flow_novel` | ✅ advancing | fixed novel | 고정 novel camera에서 temporal flow. `elevation`, `prev_elevation`+`transition_frames`으로 smoothstep sweep 진입 가능 |
| `flow_head_kp` | ✅ advancing | orbit+zoom | 머리 부위 확대 + MAMMAL 22-keypoint overlay (depth-based opacity, 약어 라벨, 우측 legend) |
| `grid_multiview` | ✅ advancing | 3×2 grid | 6-view grid 3단계: GT raw → GS-LRM recon → Novel views (2 elevations × 3 azimuths) |

### Config 옵션 (YAML)

```yaml
global:
  fps: 15                      # 프레임 레이트 (낮을수록 느림)
  resolution: 512              # 렌더링 해상도
  bg_color: [1.0, 1.0, 1.0]   # 배경색 [R,G,B] (0-1)
  n_filter: 2                  # Multi-view visibility threshold (N>=2 권장)
  gt_view: 0                   # GT 카메라 뷰 인덱스 (0-5)
  crossfade: 0.3               # 세그먼트 간 crossfade (초)
  keypoint_overlay: false      # MAMMAL 22-keypoint overlay on/off (per-segment으로 제어 권장)
  mask_border_color: null      # null=투명, [1,1,1]=흰색, [1,0,0]=빨강
  cache_segments: false        # true이면 .seg_cache/에 세그먼트별 프레임 캐시 (재실행 가속)

camera:
  radius: 2.7                  # 카메라 궤도 반경
  hfov: 50                     # 수평 FOV (도)

frame_range: "195:315"         # start:end[:step]

segments:                      # 세그먼트 시퀀스 정의
  - type: flow_gt
    duration: 3.0              # 초
    label: "Ground Truth"      # 화면 좌상단 텍스트

  - type: freeze_orbit
    duration: 4.0
    elevation: 20              # 카메라 고도 (도)
    label: "360° Orbit"

  - type: flow_bodypart
    duration: 6.0
    parts: [face, torso, tail, left_paw, right_paw]
    elevation: 30
    zoom: 2.0                  # body-part 확대 배율 (1.0=원본, 2.0=2배)
    label: "Body-Part Tracking"
```

### CLI Override

```bash
# Config 파일 + CLI override
python -m mouse_extensions.behavior.cinematic_sequence \
    --config cinematic_default.yaml \
    --frame-range 195:255 \
    --fps 10 \
    --output-dir outputs/my_demo

# 세그먼트 캐시 사용 (두 번째 실행부터 크게 빠름)
python -m ... --use-cache

# 캐시 초기화 후 전체 재실행
python -m ... --use-cache --clear-cache
```

### 완료 확인 (원격 실행 시)

```bash
# ✅ 올바른 방법: CINEMATIC_DONE 라인 확인
ssh gpu03 "grep 'CINEMATIC_DONE' ~/dev/FaceLift/outputs/experiments/mouse/cinematic_v5.log"

# ❌ 틀린 방법: [7/7] 라인 = 마지막 세그먼트 '시작'이지 완료가 아님
# ❌ 틀린 방법: ls로 MP4 존재 확인 = 이전 실행 파일일 수 있음
```

> **교훈**: `all_imgs` 수집이 전부 끝난 후 VideoWriter가 실행되므로, 마지막 세그먼트 로그가
> 찍혀도 영상 쓰기(수백 프레임)가 남아있다. 신규 실행 전 이전 MP4를 삭제하면 오인 방지.

### Turntable 시작 방향

Turntable orbit는 **GT 카메라와 동일한 방향에서 시작**합니다.
GT camera의 c2w에서 azimuth를 추출하여 `get_turntable_cameras` 출력을 회전시킵니다.

## 4. Cinematic Demo v6 (configs/mouse/cinematic/)

논문용 데모 영상. 3 variants, 30fps, 768px, ~68s.

### v6 Variants

| Variant | Config | Controls | Extreme Views | 비고 |
|---------|--------|----------|---------------|------|
| **v6_A** | `cinematic_demo_v6_A.yaml` | ON | ON | Full version (icon bar 포함) |
| **v6_B** | (v6_A `--dual-output-dir`) | OFF | ON | v6_A dual output (추가 추론 없음) |
| **v6_C** | `cinematic_demo_v6_C.yaml` | OFF | OFF | Clean standard (moderate views only) |

### 세그먼트 구성 (v6_A, 10 segments)

| # | Type | Duration | 설명 |
|---|------|----------|------|
| 1 | `flow_gt_opener` | 3.0s | GT 6-cam mosaic → zoom to cam 0 |
| 2 | `flow_gt` | 3.0s | GT RGB (raw background) |
| 3 | `flow_mask` | 2.5s | SAM2 foreground segmentation |
| 4 | `flow_render` | 8.0s | GS-LRM 재구성 (α=0.3) |
| 5 | `freeze_orbit` | 10.0s | 360° turntable (SLERP 진입, no_crossfade) |
| 6 | `flow_novel` | 6.0s | Bottom view (-80°, use_prev_azimuth) |
| 7 | `flow_head_kp` | 9.0s | Head close-up + KP (zoom 1.0→0.75→1.0) |
| 8 | `flow_novel_extra` | 8.0s | Extrapolated novel views cycling |
| 9 | `flow_gt_zoom_out` | 5.0s | SLERP → GT cam + zoom-out to mosaic |
| 10 | `grid_novel_6views` | 8.0s | 6 novel views grid (Top/Bot/Frt/Rgt/Rear/Lft) |

### 실행 명령

```bash
# 전체 3 variants 동시 실행 (GPU4 + GPU5)
bash mouse_extensions/scripts/run_cinematic_v6_all.sh

# 또는 개별 실행
CUDA_VISIBLE_DEVICES=5 /home/joon/anaconda3/envs/facelift/bin/python \
    -m mouse_extensions.behavior.cinematic_sequence \
    --config configs/mouse/cinematic/cinematic_demo_v6_A.yaml \
    --output-dir outputs/viz/cinematic/mouse/demo_v6_A \
    --use-cache --save-segments \
    --dual-output-dir outputs/viz/cinematic/mouse/demo_v6_B

# 완료 확인
grep 'CINEMATIC_DONE' outputs/viz/cinematic/mouse/demo_v6_A/run.log

# 다운로드
scp gpu03:~/dev/FaceLift/outputs/viz/cinematic/mouse/demo_v6_{A,B,C}/cinematic_demo.mp4 ~/Downloads/
```

### 소요 시간

- 총 unique inference: ~526 (frame_step=2 적용 후)
- GPU당 inference 속도: ~17s/frame (GPU5 기준, 260329 측정)
- 예상 완료 시간: **약 2.5시간**

### 왜 MP4 크기가 작나? (~7MB)

| 단계 | 크기 | 이유 |
|------|------|------|
| Float32 프레임 | ~10GB | 768²×3×4B×1545f |
| uint8 변환 | ~2.7GB | 4x 절약 |
| **mp4v 압축 후** | **~7MB** | ~385:1 — 흰 배경 + 느린 움직임으로 temporal redundancy 극대화 |

## 5. 세그먼트 캐시 시스템

세그먼트별 렌더링 결과를 `.seg_cache/`에 저장하여 부분 재실행 가속.

### 동작 원리

```
generate()
├── 캐시 없음: handler() 실행 → 프레임 렌더 → .npz 저장
└── 캐시 있음: .npz 로드 → self.fi 커서 복원 → last_fi로 prev_fd 재추론(1회)
```

각 `.npz` 파일에 저장되는 정보:
- `frames`: uint8 압축 프레임 배열 (768×768×3×N)
- `fi_after`: 이 세그먼트 후 프레임 커서 위치
- `last_fi`: 마지막 inference 프레임 인덱스 (다음 세그먼트의 `prev_fd` 복원용)

### 세그먼트 선택적 재실행

```bash
# seg_05_flow_head_kp만 수정 후 재실행
rm outputs/viz/cinematic/mouse/demo_v5/.seg_cache/seg_05_flow_head_kp.npz
python3 -m ... --use-cache
# → seg 0-4 캐시 로드, seg 5-6만 재추론
```

> **주의**: `freeze_orbit`처럼 이전 세그먼트 Gaussians에 의존하는 세그먼트는,
> 앞 세그먼트 캐시 히트 시 마지막 프레임 1회 재추론으로 `prev_fd` 복원.

### 캐시 용량

- Uncompressed uint8: ~2.7GB
- npz 압축 후 (흰 배경 + 느린 움직임): 예상 **500MB~1GB**
- 사용 후 삭제 권장: `rm -rf .seg_cache/`

## 7. Multi-View Visibility Filter (multiview_visibility_filter.py)

### Sweep Mode (정적 비교)

```bash
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.multiview_visibility_filter \
    --frame-idx 0 222 1000 \
    --n-thresholds 0 1 2 3 4 5 6 \
    --output-dir outputs/analysis/mouse/filtering/multiview_filter
```

출력: N별 × white/black BG × 3뷰 비교 grid PNG + body-part balance 차트

### Video Mode (temporal)

```bash
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.behavior.multiview_visibility_filter \
    --mode video \
    --frame-range 195:255 \
    --n-filter 2 \
    --views 0 1 2 3 4 5 \
    --parts face tail torso \
    --output-dir outputs/analysis/mouse/filtering/multiview_filter_video
```

출력: 6-view grid MP4 × (all + 각 body part) × (white + black BG)

## 8. 배치 실행

```bash
# 전체 시각화 일괄 생성
CUDA_VISIBLE_DEVICES=5 bash mouse_extensions/behavior/run_all_visualizations.sh
```

생성되는 영상 (전체 경로):

| 출력 경로 | 내용 |
|-----------|------|
| `outputs/viz/cinematic/mouse/cinematic_white/` | 기본 시네마틱 (흰 BG) |
| `outputs/viz/cinematic/mouse/cinematic_white_kp/` | Keypoint overlay 포함 |
| `outputs/viz/cinematic/mouse/cinematic_black/` | 검정 BG 버전 |
| `outputs/analysis/mouse/filtering/multiview_filter_video/` | N>=2 temporal 6-view grid |

### 기존 실험 결과 위치 (260321 이전)

> 260324 v2 마이그레이션 완료. 모든 결과가 v2 경로로 이동됨.

| 결과 경로 | 내용 |
|-----------|------|
| `outputs/viz/cinematic/mouse/cinematic_final_v2/` | 최종 cinematic (최신) |
| `outputs/viz/cinematic/mouse/highres_768_30fps{,_a03,_a10}/` | ⭐ **최신 고화질** (768px/30fps, α=0.0/0.3/1.0) |
| `outputs/viz/cinematic/mouse/cinematic_white/` | Batch script 기본 출력 |
| `outputs/viz/cinematic/mouse/_archive/` | v1~v9 iterations (아카이브) |
| `outputs/viz/comparison/mouse/6view_grid/` | GT vs GS-LRM 6-view 그리드 |
| `outputs/analysis/mouse/filtering/novel_grid_filtered/` | Novel view orbit (face/torso/tail/all) |
| `outputs/analysis/mouse/filtering/multiview_filter_video/` | Body-part mask filter 9종 |
| `outputs/viz/bodypart/mouse/bodypart_renders_active/` | Body-part 렌더 PNG (최신) |

## 9. 커스텀 설정 예시

### 검정 BG + keypoint + 느린 FPS

```yaml
global:
  fps: 10
  bg_color: [0.0, 0.0, 0.0]
  keypoint_overlay: true
```

### Body-part만 (orbit 없이)

```yaml
segments:
  - type: flow_gt
    duration: 3.0
    label: "GT"
  - type: flow_bodypart
    duration: 10.0
    parts: [face, torso, tail, left_paw, right_paw, all]
    elevation: 25
    zoom: 2.0
    label: "Body Parts"
```

### 빠른 테스트 (짧은 프레임)

```yaml
frame_range: "195:210"
global:
  fps: 10
segments:
  - type: flow_render
    duration: 1.5
    label: "Quick Test"
```

## 10. 주요 발견 사항

### Multi-View Visibility Filtering

- **1M Gaussian 중 75-84%가 배경 노이즈** (N=0, 어떤 뷰에서도 foreground 아님)
- **N>=2**: ~5% 잔존 (40-70K), body-part balance 최적 (max/min ratio 3.2-5.4)
- **N>=6** (완전 교집합): tail 등 가는 부위 소실, N>=2 대비 불리
- Black BG에서 opacity=0.5 + white color "ghost" Gaussian 진단 가능

### 기존 연구 대비

기존 3DGS feature 논문(Feature 3DGS, LangSplat 등)은 전부 학습 기반 pruning에 의존.
Feed-forward GS-LRM의 opacity=0.5 문제는 미탐구 영역 → multi-view consensus가 novel contribution.

## 11. Orientation-Aware Gaussian Filter

Novel bottom view artifact 억제를 위한 post-filter. 상세: [[ORIENTATION_FILTER_GUIDE]]

```bash
# Quick test (before/after comparison)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.test_orientation_filter \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 --n-frames 5

# Multi-view grid video comparison
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.filter_grid_comparison \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 --frame-range 3240:3280
```

## 12. Opacity & Scaling Analysis

Gaussian 분포 분석 도구. 상세: [[../hypotheses/H8_opacity_anisotropy_analysis]]

```bash
# Single checkpoint analysis (4 plots + JSON stats)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.opacity_analysis \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 --n-frames 20

# Multi-checkpoint comparison (α=0.0 vs α=0.3)
CUDA_VISIBLE_DEVICES=5 python -m mouse_extensions.scripts.eval.opacity_analysis \
    --m5-dir /home/joon/data/preprocessed/FaceLift_mouse/M5 --n-frames 10 \
    --checkpoints "6v_a0.0=/path/to/6view/best_psnr.pt" "6v_a0.3=/path/to/alpha03/best_psnr.pt"
```

## Related

- ↑ [[docs/INDEX]] — 문서 허브
- ↓ [[ORIENTATION_FILTER_GUIDE]] — Orientation filter 상세 가이드
- ↔ [[../hypotheses/H8_opacity_anisotropy_analysis]] — Opacity/anisotropy 분석 보고서
- ↔ [[outputs/reports/260321_behaviorsplatter_comprehensive]] — 종합 보고서

---

*BehaviorSplatter | Visualization Guide | 2026-03-26 (updated: §8-9 추가)*
