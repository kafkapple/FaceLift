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

### Config 옵션 (YAML)

```yaml
global:
  fps: 15                      # 프레임 레이트 (낮을수록 느림)
  resolution: 512              # 렌더링 해상도
  bg_color: [1.0, 1.0, 1.0]   # 배경색 [R,G,B] (0-1)
  n_filter: 2                  # Multi-view visibility threshold (N>=2 권장)
  gt_view: 0                   # GT 카메라 뷰 인덱스 (0-5)
  crossfade: 0.3               # 세그먼트 간 crossfade (초)
  keypoint_overlay: false      # MAMMAL 22-keypoint overlay on/off
  mask_border_color: null      # null=투명, [1,1,1]=흰색, [1,0,0]=빨강

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
```

### Turntable 시작 방향

Turntable orbit는 **GT 카메라와 동일한 방향에서 시작**합니다.
GT camera의 c2w에서 azimuth를 추출하여 `get_turntable_cameras` 출력을 회전시킵니다.

## 4. Multi-View Visibility Filter (multiview_visibility_filter.py)

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

## 5. 배치 실행

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
| `outputs/viz/cinematic/mouse/cinematic_v1~v9/`, `cinematic_white/` | 이전 iteration |
| `outputs/viz/comparison/mouse/6view_grid/` | GT vs GS-LRM 6-view 그리드 |
| `outputs/analysis/mouse/filtering/novel_grid_filtered/` | Novel view orbit (face/torso/tail/all) |
| `outputs/analysis/mouse/filtering/multiview_filter_video/` | Body-part mask filter 9종 |
| `outputs/viz/bodypart/mouse/bodypart_renders_active/` | Body-part 렌더 PNG (최신) |

## 6. 커스텀 설정 예시

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

## 7. 주요 발견 사항

### Multi-View Visibility Filtering

- **1M Gaussian 중 75-84%가 배경 노이즈** (N=0, 어떤 뷰에서도 foreground 아님)
- **N>=2**: ~5% 잔존 (40-70K), body-part balance 최적 (max/min ratio 3.2-5.4)
- **N>=6** (완전 교집합): tail 등 가는 부위 소실, N>=2 대비 불리
- Black BG에서 opacity=0.5 + white color "ghost" Gaussian 진단 가능

### 기존 연구 대비

기존 3DGS feature 논문(Feature 3DGS, LangSplat 등)은 전부 학습 기반 pruning에 의존.
Feed-forward GS-LRM의 opacity=0.5 문제는 미탐구 영역 → multi-view consensus가 novel contribution.

## Related

- ↑ [[docs/INDEX]] — 문서 허브
- ↔ [[outputs/reports/260321_behaviorsplatter_comprehensive]] — 종합 보고서 (N>=2 상세, 실험 계획)
- ↔ [[outputs/reports/260320_gaussian_bodypart_analysis]] — Gaussian 분포 분석

---

*BehaviorSplatter | Visualization Guide | 2026-03-24*
