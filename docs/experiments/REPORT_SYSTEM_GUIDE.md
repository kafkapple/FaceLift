# Report System Guide

> **Purpose**: FaceLift 실험 비교 HTML 레포트 생성 시스템의 통합 가이드
> ← [[INDEX]] | **Version**: v1.0 | **Updated**: 2026-02-23

---

## 1. System Overview

Report system은 여러 실험의 정량/정성 비교를 담은 **self-contained HTML 레포트**를 생성한다.
모든 이미지가 base64로 임베딩되어, 단일 HTML 파일만으로 공유 가능하다.

### Architecture

```
YAML Config → cli.py → builder.py → loaders.py → HTML Report
                         ↓              ↓
                    schema.py      visualizers.py
                    (dataclasses)  (PIL grids)
```

### Code Location

```
mouse_extensions/scripts/report/
├── cli.py              # Entry point (argparse)
├── schema.py           # Dataclasses (ExperimentConfig, ProtocolConfig, ReportConfig)
├── loaders.py          # JSON/이미지 로딩 유틸리티
├── visualizers.py      # PIL 기반 비교 그리드/바 차트 생성
├── builder.py          # 메인 오케스트레이터 (Jinja2 렌더링)
├── templates/
│   └── report.html     # Jinja2 HTML 템플릿
└── configs/
    └── 6view_comparison.yaml   # 실험 비교 설정
```

### Why This Architecture?

- **YAML-driven**: 코드 수정 없이 새 실험 추가 가능
- **Self-contained HTML**: 서버 없이 브라우저에서 바로 열림, 이메일/슬랙 공유 용이
- **PIL-only**: matplotlib 의존성 없이 이미지 그리드 생성 (서버 환경 호환성)

---

## 2. Code Modules

### 2.1 cli.py -- Entry Point

CLI 인터페이스. YAML config를 읽고 `build_report()`를 호출한다.

```bash
python -m mouse_extensions.scripts.report.cli --config <yaml> --output <html>
```

| Flag | Description |
|------|-------------|
| `--config` | YAML config 파일 경로 (필수) |
| `--output` | 출력 HTML 파일 경로 (필수) |
| `--no-images` | Metrics-only 모드. 이미지 생략하여 경량 레포트 생성 (39KB vs 1.4MB) |

### 2.2 schema.py -- Data Models

세 가지 핵심 dataclass로 레포트 구조를 정의한다.

```python
@dataclass
class ExperimentConfig:
    name: str           # Display name (e.g., "FL GS-LRM 6v")
    method: str         # Method identifier (e.g., "gslrm_6view")
    metrics_json: str   # Path to fair eval JSON output
    render_dir: str     # Directory containing rendered images
    render_pattern: str # Filename pattern for renders (e.g., "{frame:06d}/cam_{view:03d}.png")
    color: str          # CSS color for charts (e.g., "#4A90D9")

@dataclass
class ProtocolConfig:
    name: str           # Protocol display name (e.g., "Temporal (Same-Camera)")
    type: str           # "overall" or "per_view"
    view_key: str       # None for overall, "view_5" for spatial NVS
    metrics: list[str]  # List of metric names to include

@dataclass
class ReportConfig:
    title: str
    dataset: str
    experiments: list[ExperimentConfig]
    protocols: list[ProtocolConfig]
    # Visualization settings (sample frames, image size, etc.)
```

### 2.3 loaders.py -- Data Loading

실험 결과 JSON과 렌더 이미지를 로딩하는 유틸리티.

| Function | Input | Output | Description |
|----------|-------|--------|-------------|
| `load_metrics(json_path)` | Fair eval JSON 경로 | `dict` | 전체 metric dictionary 로딩 |
| `get_metrics(data, view_key)` | Data dict, view key | `dict` | view_key에 따라 overall 또는 per-view metric 추출 |
| `load_render(dir, pattern, fid, vid)` | 렌더 디렉토리, 패턴, frame/view ID | `PIL.Image` | 렌더 이미지 로딩 |
| `load_gt(gt_dir, fid, vid)` | GT 디렉토리, frame/view ID | `PIL.Image` | GT RGBA를 RGB로 변환하여 반환 |

**`get_metrics` 동작 방식**:

```python
def get_metrics(data, view_key=None):
    if view_key is None:
        return data["metrics"]           # Protocol A: overall/temporal
    else:
        return data["per_view_metrics"][view_key]  # Protocol B: spatial NVS
```

### 2.4 visualizers.py -- Image Generation

PIL 기반으로 비교 이미지를 생성한다. matplotlib 의존성 없음.

#### `create_comparison_grid(experiments, gt, frame_id, views, img_size=256)`

- **구조**: Row = GT + 각 실험, Column = 각 view
- **출력**: PIL.Image (그리드)
- **용도**: 정성적 비교 (qualitative section)

```
         view_0    view_1    view_2    view_3    view_4    view_5
GT       [img]     [img]     [img]     [img]     [img]     [img]
FL 6v    [img]     [img]     [img]     [img]     [img]     [img]
FL E2E   [img]     [img]     [img]     [img]     [img]     [img]
PS M5    [img]     [img]     [img]     [img]     [img]     [img]
```

#### `create_metric_bars(experiments, metrics)`

- **구조**: Metric별 grouped horizontal bar chart
- **색상**: 각 실험의 `color` 필드 사용
- **용도**: 정량적 비교 (protocol section)

### 2.5 builder.py -- Report Assembly

메인 오케스트레이터. 전체 레포트 생성 파이프라인을 관장한다.

```python
def build_report(config: ReportConfig) -> str:
    # 1. Load all experiment JSONs
    exp_data = {exp.method: load_metrics(exp.metrics_json) for exp in config.experiments}

    # 2. For each protocol: extract metrics, find best, mark winners
    protocol_results = []
    for protocol in config.protocols:
        metrics = {}
        for exp in config.experiments:
            metrics[exp.method] = get_metrics(exp_data[exp.method], protocol.view_key)
        best = find_best_values(metrics, protocol.metrics)
        protocol_results.append({"protocol": protocol, "metrics": metrics, "best": best})

    # 3. Generate comparison grids for sample frames
    grids = create_comparison_grids(config, sample_frames)

    # 4. Render Jinja2 template
    html = render_template(protocol_results, grids, config)

    # 5. Images are embedded as base64 (self-contained)
    return html
```

### 2.6 templates/report.html -- Jinja2 Template

HTML 레포트의 뼈대. 아래 섹션 순서로 구성된다:

1. **Header**: 제목, 날짜, 데이터셋 정보
2. **Setup**: 실험 설정 요약 (모델, checkpoint, view 수)
3. **Dataset**: M5t2 split 정보, 프레임 범위
4. **Protocols**: 각 프로토콜별 metric 테이블 + 바 차트
5. **Qualitative**: 샘플 프레임별 비교 그리드
6. **Conclusions**: 주요 발견 요약

**CSS 특성**:
- Responsive layout (모바일 호환)
- Print-friendly (인쇄 시 페이지 나눔)
- Images: base64 data URI로 임베딩

---

## 3. Metric Calculation

### 3.1 Fair Evaluation Script

**파일**: `mouse_extensions/scripts/eval/fair_comparison.py`

모든 metric은 (frame, view) 쌍 단위로 계산한 뒤, **flat aggregate**한다.

```python
def evaluate_fair(render_dir, gt_dir, frame_range):
    all_values = []  # Flat list across all (frame, view) pairs
    for frame_id in frame_range:  # 360 test frames (3240-3599)
        for view_id in range(6):  # 6 views each
            gt = load_rgba(gt_dir / f"{frame_id:06d}/images/cam_{view_id:03d}.png")
            pred = load_rgb(render_dir / ...)

            # Compute per-pair metrics
            metrics = compute_metrics(pred, gt)
            all_values.append(metrics)

    # Flat aggregate: mean/std/min/max over 2160 values (360x6)
    return aggregate(all_values)
```

**Why flat aggregate?** 3DGS/NeRF 문헌의 표준 방식 (PixelSplat, pixelNeRF 등). 프레임별 평균 후 전체 평균하는 2-pass hierarchical 방식이 아닌, 모든 값을 1-pass로 평균한다.

### 3.2 Individual Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **psnr_gt_masked** | `PSNR(pred * gt_mask, gt_rgb * gt_mask)` | GT foreground 영역만 비교. 기본 품질 지표 |
| **psnr_intersection** | `PSNR(pred * intersection, gt * intersection)` where `intersection = pred_mask AND gt_mask` | 양쪽 모두 foreground인 영역만 비교. 순수 색상 정확도 |
| **iou** | `(pred_mask AND gt_mask) / (pred_mask OR gt_mask)` | Silhouette overlap. 형태 정확도 |
| **coverage** | `(pred_mask AND gt_mask) / gt_mask` | GT foreground를 얼마나 커버하는지 |
| **ssim_gt_masked** | `SSIM(pred_bbox, gt_bbox)` on GT mask bounding box crop | Structural similarity. 지각 품질 |

**Metric 해석 가이드**:

- `psnr_gt_masked` 낮고 `psnr_intersection` 높음 → coverage 문제 (형태는 맞지만 빠진 부분 있음)
- `iou` 높고 `coverage` 낮음 → pred가 GT보다 작은 영역 커버
- `psnr_intersection` vs `psnr_gt_masked` 차이 → silhouette 불일치 정도

### 3.3 Mask Extraction

GT와 prediction의 마스크 추출 방식이 다르다는 점에 유의해야 한다.

```python
# GT mask: RGBA alpha channel (exact)
gt_mask = gt_image[:, :, 3]  # Alpha channel, 0 or 1

# Prediction mask: white-background extraction
def extract_foreground_mask(image, threshold=0.98):
    white = (image > threshold).all(dim=0)  # Per-pixel: R, G, B 모두 > 0.98
    return (~white).float()                 # Foreground = 1.0, Background = 0.0
```

| Source | Method | Precision |
|--------|--------|-----------|
| GT | Alpha channel | Exact (binary) |
| Prediction | White-BG extraction (threshold=0.98) | Approximate (threshold-sensitive) |

**주의**: threshold가 낮으면 밝은 foreground도 배경으로 분류될 수 있음. mouse 데이터의 경우 foreground가 이미지의 ~2.5%만 차지하므로 threshold 민감도가 높다.

### 3.4 Averaging Method

**Flat 1-pass average** (NOT hierarchical 2-pass):

```
Protocol A (Temporal): 360 frames x 6 views = 2160 values → mean
Protocol B (Spatial):  360 frames x 1 holdout view = 360 values → mean
```

3DGS/NeRF 문헌 표준. 모든 (frame, view) 쌍을 동일 가중치로 평균한다.

---

## 4. Evaluation Protocols

### 4.1 Protocol A: Temporal (Same-Camera)

| Item | Detail |
|------|--------|
| **Definition** | 학습에 사용된 카메라 각도에서, 시간적으로 hold-out된 프레임 평가 |
| **Data** | Test frames 3240-3599, 모든 6 views |
| **Metric source** | `data["metrics"]` (overall) |
| **Measures** | 알려진 카메라에서의 reconstruction quality, novel time |
| **Sample count** | 360 x 6 = 2160 |

### 4.2 Protocol B: Spatial NVS (Novel View)

| Item | Detail |
|------|--------|
| **Definition** | 학습에 사용되지 않은 view (view_5)에서의 평가 |
| **Data** | Test frames 3240-3599, view_5 only |
| **Metric source** | `data["per_view_metrics"]["view_5"]` |
| **Measures** | 진정한 novel view synthesis 성능 |
| **Sample count** | 360 x 1 = 360 |

### 4.3 Protocol C: Combined (Optional)

Protocol A와 B의 가중 결합. 전체적인 성능 순위 결정에 활용 가능.

### Protocol 선택 기준

- **Reconstruction 비교**: Protocol A (temporal) -- 동일 카메라 재구성 품질
- **NVS 비교**: Protocol B (spatial) -- view_5 hold-out
- **종합 평가**: 두 protocol 모두 보고 (레포트에 둘 다 포함)

---

## 5. JSON Output Format

Fair evaluation script의 출력 JSON 구조. 레포트 시스템의 입력이 된다.

```json
{
  "experiment": "gslrm_6view",
  "metrics": {
    "psnr_gt_masked": {"mean": 23.84, "std": 2.1, "min": 18.5, "max": 30.2, "n": 2160},
    "iou": {"mean": 0.954, "std": 0.03, "min": 0.85, "max": 0.99, "n": 2160},
    "coverage": {"mean": 0.998, "std": 0.001, "min": 0.99, "max": 1.0, "n": 2160},
    "psnr_intersection": {"mean": 23.84, "std": 2.1, "min": 18.5, "max": 30.2, "n": 2160},
    "ssim_gt_masked": {"mean": 0.91, "std": 0.04, "min": 0.80, "max": 0.97, "n": 2160}
  },
  "per_view_metrics": {
    "view_0": {
      "psnr_gt_masked": {"mean": 24.1, "std": 1.9, "min": 19.0, "max": 29.5, "n": 360},
      "iou": {"mean": 0.96, "std": 0.02, "min": 0.88, "max": 0.99, "n": 360}
    },
    "view_5": {
      "psnr_gt_masked": {"mean": 16.81, "std": 3.2, "min": 10.5, "max": 24.0, "n": 360},
      "iou": {"mean": 0.72, "std": 0.08, "min": 0.50, "max": 0.90, "n": 360}
    }
  },
  "per_frame_metrics": {
    "3240": {
      "view_0": {"psnr_gt_masked": 24.5, "iou": 0.97},
      "view_5": {"psnr_gt_masked": 17.2, "iou": 0.75}
    }
  }
}
```

**Key structure**:
- `metrics`: 전체 flat average (Protocol A에 사용)
- `per_view_metrics`: View별 분리 통계 (Protocol B에 사용 -- `view_5`)
- `per_frame_metrics`: 프레임별 상세 (디버깅/분석용)

---

## 6. Report Configs (YAML)

### 6.1 6-View Comparison (Primary)

**파일**: `mouse_extensions/scripts/report/configs/6view_comparison.yaml`

3개 실험 비교:

| Experiment | Method | Description |
|-----------|--------|-------------|
| FL GS-LRM 6v | `gslrm_6view` | GT input upper bound (Stage 2 only) |
| FL E2E 1v | `e2e_1view` | Full pipeline (Stage 1 multi-view diffusion + Stage 2 GS-LRM) |
| PS M5 6v | `ps_m5_6view` | Pose-Splatter per-scene optimization baseline |

2개 protocol:
- **Protocol A** (temporal): `type: overall`, `view_key: null`
- **Protocol B** (spatial): `type: per_view`, `view_key: view_5`

```yaml
title: "6-View Fair Comparison: FaceLift vs Pose-Splatter"
dataset: "M5t2 (3600 frames, 8:1:1 split)"

experiments:
  - name: "FL GS-LRM 6v (GT input)"
    method: gslrm_6view
    metrics_json: "experiments/comparison/tier/gslrm_6view_fair.json"
    render_dir: "outputs/tier_comparison/gslrm_6view_test/samples/"
    render_pattern: "{frame:06d}/cam_{view:03d}.png"
    color: "#4A90D9"

  - name: "FL E2E (Stage 1 + Stage 2)"
    method: e2e_1view
    metrics_json: "experiments/comparison/tier/e2_resume_20k_fair.json"
    render_dir: "outputs/_archive_renders/E2_resume_20k/samples/"
    render_pattern: "{frame:06d}/cam_{view:03d}.png"
    color: "#D94A4A"

  - name: "PS M5 6v (per-scene)"
    method: ps_m5_6view
    metrics_json: "/tmp/ps_m5_fair_results.json"
    render_dir: "/tmp/ps_renders/"
    render_pattern: "{frame:06d}/cam_{view:03d}.png"
    color: "#4AD94A"

protocols:
  - name: "Protocol A: Temporal (Same-Camera)"
    type: overall
    view_key: null
    metrics: [psnr_gt_masked, psnr_intersection, iou, coverage, ssim_gt_masked]

  - name: "Protocol B: Spatial NVS (Novel View)"
    type: per_view
    view_key: view_5
    metrics: [psnr_gt_masked, psnr_intersection, iou, coverage, ssim_gt_masked]
```

### 6.2 4-View Comparison (Planned)

FL GS-LRM 4v + FL E2E + PS 4v 비교. 동일한 YAML 구조로 설정 가능.

---

## 7. Usage

### 7.1 Full Report 생성

이미지 그리드 포함. 파일 크기 ~1.4MB.

```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.report.cli \
    --config mouse_extensions/scripts/report/configs/6view_comparison.yaml \
    --output reports/6view_comparison.html
```

### 7.2 Metrics-Only Report (경량)

이미지 없이 metric 테이블만. 파일 크기 ~39KB. 빠른 확인용.

```bash
python -m mouse_extensions.scripts.report.cli \
    --config mouse_extensions/scripts/report/configs/6view_comparison.yaml \
    --output reports/6view_metrics_only.html \
    --no-images
```

### 7.3 Output Locations

```
reports/
├── 6view_comparison.html          # Full report with images (1.4MB)
├── 6view_metrics_only.html        # Metrics only (39KB)
└── (future: 4view_comparison.html)
```

### 7.4 새 실험 추가 방법

1. Fair eval 실행하여 JSON 생성
2. YAML config에 `experiments` 항목 추가 (name, method, metrics_json, render_dir, color)
3. `python -m mouse_extensions.scripts.report.cli` 재실행

코드 수정 불필요. YAML만 편집하면 된다.

---

## 8. Data Sources

### 8.1 File Paths (gpu03)

| Data | Path |
|------|------|
| FL 6v fair JSON | `experiments/comparison/tier/gslrm_6view_fair.json` |
| FL E2E fair JSON | `experiments/comparison/tier/e2_resume_20k_fair.json` |
| PS M5 fair JSON | `/tmp/ps_m5_fair_results.json` |
| FL 6v renders | `outputs/tier_comparison/gslrm_6view_test/samples/` |
| FL E2E renders | `outputs/_archive_renders/E2_resume_20k/samples/` |
| PS M5 renders | `/tmp/ps_renders/` |
| GT images (RGBA) | `/home/joon/data/preprocessed/FaceLift_mouse/M5/` |
| All fair eval JSONs (17) | `experiments/comparison/tier/` |

### 8.2 M5t2 Data Split

| Split | Frames | Count | Ratio |
|-------|--------|-------|-------|
| Train | 0-2879 | 2880 | 80% |
| Val | 2880-3239 | 360 | 10% |
| Test | 3240-3599 | 360 | 10% |

**Total**: 3600 frames, 6 views each. FL과 PS 동일 split.

### 8.3 Best Checkpoints (M5t2)

| Stage | Checkpoint | Metric |
|-------|-----------|--------|
| Stage 1 (multi-view diffusion) | `mouse_M5t2/checkpoint-5000` | Sparse attn, E2E PSNR_wh=21.29 |
| Stage 2 (GS-LRM) 4-view | `M5t2_E0_1_facelift/best_psnr.pt` | PSNR=22.34 |
| Stage 2 (GS-LRM) 6-view | `6view_v2/best_psnr.pt` | PSNR=24.49 (GT input upper bound) |

---

## 9. Troubleshooting

### JSON not found

```
FileNotFoundError: experiments/comparison/tier/gslrm_6view_fair.json
```

- 경로가 상대경로인 경우, CWD가 FaceLift 루트 (`/home/joon/dev/FaceLift/`)인지 확인
- 절대경로로 YAML config 수정 가능

### Render images missing

```
FileNotFoundError: outputs/tier_comparison/gslrm_6view_test/samples/003240/cam_000.png
```

- `--no-images` 플래그로 metrics-only 모드 사용 가능
- 렌더 디렉토리 구조 확인: `{frame_id:06d}/cam_{view_id:03d}.png`

### Metric 값이 예상보다 낮을 때

1. **coverage 확인**: IoU 0.5 이하면 silhouette 불일치가 PSNR을 끌어내림
2. **psnr_intersection vs psnr_gt_masked 비교**: 차이가 크면 coverage 문제
3. **mask threshold 확인**: mouse의 경우 foreground가 이미지의 ~2.5%라 threshold 민감도 높음

---

## 10. Cross-References

| Topic | Document |
|-------|----------|
| Experiment overview | [[EXPERIMENT_MASTER_GUIDE]] |
| FL vs PS comparison | [[FL_vs_PS_comparison]] |
| Stage 1 bottleneck analysis | [[mvdiffusion_bottleneck_analysis]] |
| Evaluation protocol | [[evaluation_protocol_v1]] |
| Metric theory | [[theory/METRICS_PROTOCOL]] |
| Hypothesis roadmap | [[FaceLift_hypothesis_roadmap]] |

---

*Report System Guide v1.0 | 2026-02-23*
