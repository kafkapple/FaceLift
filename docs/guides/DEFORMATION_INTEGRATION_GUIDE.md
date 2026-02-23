# Deformation Network 통합 가이드

**Date**: 2026-02-04
**Status**: Training Complete → Inference Integration
**Tags**: #facelift #deformation #temporal #video

---

## 1. 개요 (Why)

### 1.1 문제: 프레임별 독립 재구성의 한계

```
Video Frames:  F0   F1   F2   ...  Ft
                |    |    |         |
GS-LRM:        G0   G1   G2   ...  Gt  (각 프레임 독립 처리)

문제: G0 ↔ G1 ↔ G2 간 시간적 연속성 없음
결과: Temporal flickering (깜빡임), 불연속적인 형태 변화
```

### 1.2 해결: Autoregressive Deformation

```
┌──────────────────────────────────────────────────────────┐
│             Temporal Consistency Pipeline                │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  Step 1: GS-LRM으로 각 프레임 Gaussian 생성 (캐시)       │
│          G0, G1, G2, ..., Gt                             │
│                                                          │
│  Step 2: Anchor 프레임 선택 (보통 첫 프레임)             │
│          G0 = Canonical Gaussians                        │
│                                                          │
│  Step 3: Autoregressive 변형 전파                        │
│          G0 → D(G0) → G'1 → D(G'1) → G'2 → ...          │
│                                                          │
│  결과: 시간적으로 부드러운 {G'0, G'1, G'2, ..., G't}     │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 아키텍처 (How)

### 2.1 Deformation Network (8-layer MLP)

```
Input:  Gaussian xyz 좌표 [N, 3]
        │
        ▼
┌───────────────────────────────┐
│   Linear(3 → 256) + ReLU      │
│   Linear(256 → 256) + ReLU    │  × 6 layers
│   Linear(256 → 256) + ReLU    │
│   ...                         │
│   Linear(256 → 5)             │  Output layer (zero-init)
└───────────────────────────────┘
        │
        ▼
Output: 변형 파라미터 [N, 5]
        ├─ Δxyz (position offset)   [N, 3]
        ├─ Δα (opacity offset)      [N, 1]
        └─ Δs (scale offset)        [N, 1]
```

**핵심 설계 특징:**
- **Zero-init output**: 학습 초기에 identity 변형 (Δ=0)
- **Positional input only**: 위치만 입력, 다른 속성은 무시
- **Additive deformation**: 새 파라미터 = 이전 + Δ

### 2.2 GaussianParams 데이터 구조

```python
@dataclass
class GaussianParams:
    xyz: [N, 3]       # 3D 위치
    features: [N, 27] # SH 계수 (sh_degree=2)
    scaling: [N, 3]   # log-scale (활성화 전)
    rotation: [N, 4]  # quaternion
    opacity: [N, 1]   # logit (활성화 전)

    def apply_deformation(self, deform_dict) -> GaussianParams:
        """Δxyz, Δα, Δs를 적용하여 새 파라미터 생성"""
```

### 2.3 Temporal Pipeline

```python
class TemporalGaussianPipeline:
    def forward(self, gaussians_sequence: List[GaussianParams]):
        """
        1. Anchor frame (G0) 유지
        2. Forward propagation: G0 → G'1 → G'2 → ... → G't
        3. Backward propagation (optional): G0 → G'-1 → ...
        """
```

---

## 3. 학습 파이프라인 (What - Training)

### 3.1 2단계 학습 프로세스

```
┌─────────────────────────────────────────────────────────┐
│           Phase 1: Gaussian Cache 생성                  │
├─────────────────────────────────────────────────────────┤
│  for frame_idx in dataset:                              │
│      G = GS-LRM(images, cameras)                        │
│      cache.save(frame_idx, G)  # ~56MB per frame        │
│                                                         │
│  결과: /node_data/.../gaussian_cache/frame_XXXXXX.pt   │
│        Total: 2880 frames × 56MB ≈ 158GB                │
└─────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────┐
│           Phase 2: Deformation Network 학습             │
├─────────────────────────────────────────────────────────┤
│  for step in 10000:                                     │
│      G_t, G_{t+1} = cache.get(frame_t), cache.get(t+1) │
│      G'_{t+1} = G_t + D(G_t.xyz)  # 예측               │
│      Loss = L2(render(G'_{t+1}), render(G_{t+1}))      │
│                                                         │
│  속도: ~4 it/s (GPU 6), ~40분/10K steps                │
└─────────────────────────────────────────────────────────┘
```

### 3.2 학습 명령어

```bash
# 단일 명령어 (캐시 생성 + 학습)
cd /home/joon/dev/FaceLift
CUDA_VISIBLE_DEVICES=6 nohup python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml \
    > logs/deformation_train.log 2>&1 &
```

### 3.3 학습 설정 (default.yaml)

| 항목 | 값 | 설명 |
|------|-----|------|
| `hidden_dim` | 256 | MLP hidden size |
| `num_layers` | 8 | 논문 기준 |
| `max_steps` | 10000 | 총 학습 step |
| `batch_size` | 4 | 프레임 쌍 배치 |
| `lr` | 1e-4 | 학습률 |
| `l2_weight` | 1.0 | 렌더링 L2 loss |
| `perceptual_weight` | 0.1 | VGG loss |

---

## 4. 추론 파이프라인 (What - Inference)

### 4.1 연속 프레임 인식 방식

**현재 구현 상태**: 학습 파이프라인 완료, 추론 통합 예정

**예상 추론 흐름**:
```python
# 옵션 A: 명시적 프레임 리스트
frames = ["frame_0000", "frame_0001", "frame_0002", ...]
pipeline.run_temporal(frames, output_dir)

# 옵션 B: 디렉토리 자동 스캔 (연속 번호 감지)
pipeline.run_temporal(
    input_dir="/path/to/frames/",
    pattern="frame_{:04d}",  # frame_0000, frame_0001, ...
    output_dir=output_dir
)

# 옵션 C: 비디오 입력
pipeline.run_temporal(
    video_path="input.mp4",
    frame_stride=1,  # 매 프레임
    output_dir=output_dir
)
```

### 4.2 프레임 순서 자동 인식 로직 (예정)

```python
def _detect_frame_sequence(self, input_path):
    """연속 프레임 자동 감지"""

    if input_path.endswith('.mp4'):
        # 비디오: 프레임 추출
        return self._extract_video_frames(input_path)

    if os.path.isdir(input_path):
        # 디렉토리: 파일명 패턴 감지
        files = sorted(glob(f"{input_path}/*.png"))

        # 숫자 패턴 추출
        indices = [self._extract_number(f) for f in files]

        # 연속성 검증
        if self._is_consecutive(indices):
            return files
        else:
            raise ValueError("Non-consecutive frames detected")
```

---

## 5. 파일 구조 요약

```
mouse_extensions/
├── model/
│   └── deformation/
│       ├── __init__.py
│       ├── deformation_network.py   # 8-layer MLP
│       ├── deformation_trainer.py   # 학습 로직
│       ├── temporal_pipeline.py     # Autoregressive 파이프라인
│       ├── gslrm_integration.py     # GS-LRM 연동 + GaussianCache
│       └── gaussian_params.py       # GaussianParams 데이터 구조
├── scripts/
│   └── train_deformation.py         # 학습 스크립트
└── inference/
    └── run_temporal_inference.py  # 구현 완료 시간적 E2E 추론
```

---

## 6. GS-LRM vs Deformation 비교

| 항목 | GS-LRM | Deformation Network |
|------|--------|---------------------|
| **목적** | 이미지 → 3D Gaussian | 프레임 간 일관성 |
| **아키텍처** | 24-layer Transformer | 8-layer MLP |
| **파라미터** | ~수천만 | 397,061 |
| **입력** | 6-view 이미지 | Gaussian xyz [N, 3] |
| **출력** | Full Gaussian params | Δxyz, Δα, Δs |
| **학습 시간** | 수 시간/10K steps | ~40분/10K steps |
| **학습 데이터** | 이미지-Gaussian 쌍 | Gaussian 쌍 (캐시) |
| **VRAM** | ~24GB | ~8GB |

---

## 7. 다음 단계

### 7.1 즉시 가능
- [x] Deformation Network 학습 완료 (10K steps)
- [x] Gaussian Cache 생성 (2880 frames)
- [x] Temporal inference pipeline
- [x] Evaluation metrics (jitter 92.7% 감소)
- [x] Comparison video generation
- [ ] Checkpoint 검증 (학습 완료 후)
- [ ] 단일 비디오 시퀀스 테스트

### 7.2 추론 통합 (✅ 구현 완료)
- [ ] `TemporalEndToEndPipeline` 클래스 구현
- [ ] 프레임 시퀀스 자동 감지 로직
- [ ] Turntable 비디오 + Temporal smoothing 시각화
- [ ] CLI 인터페이스 (`run_temporal_inference.py`)

### 7.3 평가
- [ ] Temporal consistency 메트릭 (frame-to-frame difference)
- [ ] Before/After 비교 영상 생성
- [ ] PSNR/SSIM with temporal smoothing ablation

### 7.4 Experiment Priorities

> *Source: TEMPORAL_EXPERIMENTS_PLAN.md (merged 2026-02-11)*

**Identified Issues**: Autoregressive drift (anchor-only start, cumulative error), per-frame GS-LRM output unused.

#### P0: Immediate Fixes

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| T-P0-1 | Per-frame baseline | Per-frame original Gaussian quality check | Pending |
| T-P0-2 | Label fix | Video label clarification | Done |

#### P1: Core Algorithm

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| T-P1-1 | ARAP Loss | Local rigidity regularization | Pending |
| T-P1-2 | Per-frame + Reg | Original preservation + temporal regularization | Pending |
| T-P1-3 | Velocity smooth | Velocity continuity loss | Pending |

#### P2: Enhanced Methods

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| T-P2-1 | Optical Flow | RAFT flow alignment loss | Pending |
| T-P2-2 | Sliding Window | Window-based joint optimization | Pending |
| T-P2-3 | SC-GS style | Sparse control points | Pending |

#### P3: E2E Pipeline

| Exp ID | Name | Goal | Status |
|--------|------|------|--------|
| T-P3-1 | MV-Diffusion → GS-LRM → Temporal | Full pipeline integration | Pending |
| T-P3-2 | Real-time inference | Cache-free real-time inference | Pending |

#### Key Metrics Targets

| Metric | Current | Target | Method |
|--------|---------|--------|--------|
| Temporal Jitter | 0.0012 | < 0.001 | ARAP + velocity |
| PSNR (vs GT) | TBD | > 25 | Per-frame quality |
| Visual Drift | Severe | Minimal | Per-frame baseline |

---

## 8. Quick Reference

### 학습 (현재 가능)
```bash
# Deformation 학습 시작
CUDA_VISIBLE_DEVICES=6 nohup python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml > logs/deformation_train.log 2>&1 &

# 진행 확인
tail -f logs/deformation_train.log
grep -oP '\d+/10000' logs/deformation_train.log | tail -1
```

### 추론 (예정)
```bash
# 비디오 시퀀스 추론 (구현 예정)
python -m mouse_extensions.scripts.inference.run_temporal_inference \
    --input /path/to/frames/ \
    --deform_checkpoint /node_data/.../deformation/default/best.pt \
    --gslrm_checkpoint /node_data/.../gslrm/M5t2_E0_1/best_psnr.pt \
    --output /path/to/output/
```

---

*Engram v2.0 | FaceLift Deformation Integration | 2026-02-04*

