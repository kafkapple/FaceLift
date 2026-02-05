# 8-layer MLP Deformation Network V2 Fix Plan

> **목표**: V1의 autoregressive drift(녹아내림) 문제 해결
> **날짜**: 2026-02-05
> **상태**: 구현 완료, 실험 대기

---

## 핵심 가설 (Core Hypothesis)

### 🎯 주요 가설
**V1 실패 원인**: MLP가 G_t만 입력받아 G_{t+1}를 모르고 변형 → autoregressive 추론 시 오류 누적

**V2 해결책**: MLP에 (G_t, G_{t+1}) 모두 입력 → 원본 참조로 bounded deformation

### 📐 수식 정리

**V1 (broken)**:
```
δ_t = MLP(G_t)                    # G_{t+1} 정보 없음
G'_{t+1} = G'_t + δ_t             # 이전 OUTPUT 사용 → drift 누적
```

**V2 (fixed)**:
```
δ_t = MLP(G_t, G_{t+1})           # 두 프레임 모두 입력
G'_t = G_t + α * δ_t              # 원본 G_t에서 시작, bounded
```

### 🔬 검증 지표

| 지표 | V1 예상 | V2 예상 |
|------|---------|---------|
| Drift (t=50) | > 1.0 (누적) | < 0.1 (bounded) |
| PSNR | 감소 | baseline 유지 |
| Temporal Consistency | 악화 | 개선 |

---

## 실험 대조군 (Experimental Control Groups)

### 실험 A: V1 vs V2 Drift 비교 (P1 - 최우선)

| 조건 | 방법 | 목적 |
|------|------|------|
| **V1 Baseline** | Autoregressive MLP(G_t) | Drift 누적 확인 |
| **V2 Per-frame** | MLP(G_t, G_{t+1}) | Bounded drift 증명 |
| **No Deform** | 원본 GS-LRM | 기준선 |

**측정**: Frame별 drift, 최종 PSNR, 시각적 품질

### 실험 B: Loss 함수 Ablation (P2)

| 조건 | ARAP | Velocity | 예상 효과 |
|------|------|----------|----------|
| **B1** | ✅ | ✅ | 최적 (구조+부드러움) |
| **B2** | ✅ | ❌ | 구조 보존, jitter 있음 |
| **B3** | ❌ | ✅ | 부드러움, 구조 왜곡 가능 |
| **B4** | ❌ | ❌ | Param loss만, baseline |

### 실험 C: Blend Alpha 비교 (P3)

| Alpha | 의미 | 예상 |
|-------|------|------|
| 0.0 | 원본만 | 최고 품질, temporal jitter |
| 0.3 | 약한 보정 | 균형 |
| 0.5 | 중간 | 권장 |
| 0.7 | 강한 보정 | smooth, 약간 blur |
| 1.0 | 변형만 | 최고 smooth, 품질 손실 |

---

## 우선순위별 실행 계획

### P1: 핵심 증명 (V1 vs V2 Drift)

```
1. Gaussian 추출 (GS-LRM → frame_*.pt)
2. V1 추론 실행 (temporal_pipeline.py)
3. V2 추론 실행 (temporal_deform_inference.py)
4. Drift 측정 및 시각화
```

### P2: V2 네트워크 학습

```
1. DeformationNetworkV2 학습 (train_deform_v2.py)
2. ARAP + Velocity loss 효과 확인
3. Checkpoint 저장
```

### P3: 전체 파이프라인 통합

```
1. GS-LRM + Deformation V2 통합 추론
2. Turntable 영상 생성
3. 정량적 평가 (PSNR, SSIM, TC)
```

---

## 구현 현황

### ✅ 완료

| 파일 | 역할 |
|------|------|
| `deformation_network.py` | DeformationNetworkV2 (8-layer, input=110) |
| `deformation_trainer_v2.py` | ARAP + Velocity loss |
| `temporal_deform_inference.py` | V2 inference pipeline |
| `train_deform_v2.py` | Training script |
| `extract_gaussians.py` | GS-LRM → Gaussian 추출 |
| `deform_v2.yaml` | Training config |

### 🔲 대기

- 실제 데이터로 Drift 비교 실험
- V2 학습 및 평가
- Turntable 영상 비교

---

## Quick Start (빠른 실행 가이드)

### 환경 설정

```bash
# gpu03 서버 접속
ssh gpu03
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh
conda activate facelift
```

### P1: Drift 비교 (V1 vs V2) - 10분

```bash
# 기존 gaussian_cache 사용 (이미 존재)
CACHE_DIR=/home/joon/dev/FaceLift/checkpoints/deformation/default/gaussian_cache

# V1 추론 (autoregressive - drift 발생)
python -c "
from mouse_extensions.model.deformation import TemporalGaussianPipeline, TemporalConfig, GaussianParams
import torch

# Load frames
frames = []
for i in range(20):
    data = torch.load(f'{CACHE_DIR}/frame_{i:06d}.pt', map_location='cpu')
    g = GaussianParams(**{k: v for k, v in data.items() if v is not None})
    frames.append(g)

# V1 inference
pipeline = TemporalGaussianPipeline(TemporalConfig())
v1_out = pipeline(frames)

# Measure drift
for t in [0, 10, 19]:
    drift = (frames[t].xyz - v1_out[t].xyz).norm(dim=-1).mean()
    print(f'V1 Frame {t}: drift = {drift:.4f}')
"

# V2 추론 (per-frame reference - bounded)
python -c "
from mouse_extensions.model.deformation import TemporalDeformInference, InferenceConfig, DeformationNetworkV2, DeformationConfigV2, GaussianParams
import torch

CACHE_DIR='/home/joon/dev/FaceLift/checkpoints/deformation/default/gaussian_cache'

# Load frames
frames = []
for i in range(20):
    data = torch.load(f'{CACHE_DIR}/frame_{i:06d}.pt', map_location='cpu')
    g = GaussianParams(**{k: v for k, v in data.items() if v is not None})
    frames.append(g)

# V2 inference (untrained network - shows bounded behavior)
net = DeformationNetworkV2(DeformationConfigV2())
pipeline = TemporalDeformInference(net, InferenceConfig(blend_alpha=0.5, device='cpu'))
v2_out = pipeline.process_sequence(frames)

# Measure drift
for t in [0, 10, 19]:
    drift = (frames[t].xyz - v2_out[t].xyz).norm(dim=-1).mean()
    print(f'V2 Frame {t}: drift = {drift:.4f}')
"
```

### P2: V2 네트워크 학습 - 2시간

```bash
# GPU 설정 (A6000 사용)
export CUDA_VISIBLE_DEVICES=6

# 학습 실행
python -m mouse_extensions.scripts.train_deform_v2 \
    --config configs/mouse/deform_v2.yaml \
    --gaussians_dir /home/joon/dev/FaceLift/checkpoints/deformation/default/gaussian_cache \
    --output_dir /node_data/joon/checkpoints/FaceLift/deform_v2/M5t2 \
    --epochs 200 \
    --lr 1e-4
```

### P3: 시각화

```bash
# Drift 비교 시각화
python -m mouse_extensions.scripts.visualize_deform_comparison \
    --output_dir outputs/deform_comparison \
    --num_frames 50

# 결과 확인
ls outputs/deform_comparison/
# drift_comparison.png, trajectory_3d.png
```

---

## 삭제 가능 파일 (491GB 확보)

```bash
# 안전 삭제 (시각화 보존)
rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/default/e2e_test
rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/default/inference_v2
rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/default/temporal_inference_v1
rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/test_cache
rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/test_run

# 선택 삭제 (재생성 가능, 158GB)
# rm -rf /home/joon/dev/FaceLift/checkpoints/deformation/default/gaussian_cache

# 보존
# - inference_results/ (8.9MB, 시각화)
# - temporal_inference_v2/ (30MB, V2 결과)
# - checkpoint_*.pt (~50MB, 학습 모델)
```

---

## 예상 결과

### Drift 패턴

| Frame | V1 (Autoregressive) | V2 (Per-frame Ref) |
|-------|---------------------|-------------------|
| 0 | 0.00 | ~0.01 |
| 10 | 0.50+ | ~0.01 |
| 20 | 1.00+ | ~0.01 |
| 50 | 2.50+ | ~0.01 |

**결론**: V1은 누적, V2는 bounded

---

*Created: 260205 | Last Updated: 260205 15:30*

---

## 7. Gaussian Sampling Strategy (문헌 기반)

### 7.1 현재 문제점

```python
# extract_gaussians.py (현재)
data_paths = data_paths[:num_frames]  # 단순히 처음 N개 순차 추출
```

**문제**: 전체 시퀀스의 모션 분포를 대표하지 못함

### 7.2 문헌 근거

| 논문 | 샘플 수 | 샘플링 전략 | 용도 |
|------|---------|-------------|------|
| **Dynamic 3DGS** (CVPR24) | 100-300 | Uniform stride | 학습 |
| **4D-GS** (CVPR24) | 50-150 | Consecutive | 시간 일관성 |
| **SC-GS** (CVPR24) | ~100 | Motion-aware | 학습 |
| **Deformable 3DGS** | 200+ | Full sequence | 학습 |

### 7.3 권장 전략

| 용도 | 전략 | 샘플 수 | 정당성 |
|------|------|---------|--------|
| **드리프트 검증** | Sequential | 50 | 누적 오류 관찰에 충분 |
| **학습** | Stratified | 200+ | 전체 분포 커버 |
| **최종 평가** | Motion-aware | 100+ | 정보량 최대화 |

### 7.4 개선된 샘플링 코드

```python
def sample_frames_improved(data_paths: List[str], num_samples: int, strategy: str = 'uniform'):
    total = len(data_paths)
    
    if strategy == 'uniform':
        # 전체 시퀀스에서 균등 간격
        stride = max(1, total // num_samples)
        indices = list(range(0, total, stride))[:num_samples]
        
    elif strategy == 'stratified':
        # K개 구간에서 균등 샘플링
        indices = []
        for i in range(num_samples):
            start = i * total // num_samples
            end = (i + 1) * total // num_samples
            indices.append((start + end) // 2)
            
    elif strategy == 'motion_aware':
        # 사전 계산된 motion score 기반
        motion_scores = load_motion_scores(data_paths)
        indices = np.argsort(motion_scores)[-num_samples:]
        indices = sorted(indices)
    
    return [data_paths[i] for i in indices]
```

