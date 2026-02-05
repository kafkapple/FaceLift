# FaceLift Deformation Network 분석 및 구현 계획

**Date**: 2026-02-03
**Status**: Analysis Complete → Implementation Planning
**Tags**: #facelift #deformation #4d #temporal-consistency

---

## 1. Executive Summary

### 1.1 논문 분석 결과

FaceLift 논문 Appendix 3.5 "Applying FaceLift on Videos"에서 소개된 Deformation Network는:
- **목적**: 프레임별 독립 재구성으로 인한 temporal flickering 해결
- **구조**: 8-layer MLP
- **방식**: Autoregressive generation (이전 프레임 → 다음 프레임 예측)

### 1.2 현재 코드베이스 분석

| 항목 | 상태 | 비고 |
|------|------|------|
| Deformation Network | ❌ **미구현** | 논문의 확장 기능, 코드 미공개 |
| 8-layer MLP | ❌ 없음 | utils_transformer.py의 MLP는 Transformer용 |
| Autoregressive | ❌ 없음 | 현재 프레임별 독립 처리 |
| Canonical Gaussians | ❌ 없음 | 앵커 프레임 개념 없음 |

### 1.3 결론

**Deformation Network는 구현되어 있지 않음** → mouse_extensions에 새로 구현 필요

---

## 2. 논문 상세 분석

### 2.1 문제 정의

**현재 FaceLift의 한계**:
```
Video: [F₀, F₁, F₂, ..., Fₜ]
         ↓    ↓    ↓        ↓
FaceLift: G₀   G₁   G₂  ... Gₜ  (각 프레임 독립 처리)

문제: G₀ ↔ G₁ ↔ G₂ 간 시간적 연속성 없음
결과: Temporal flickering, 불연속적인 형태 변화
```

### 2.2 Deformation Network 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    Deformation Network (Dₜ)                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input: Gaussian positions (x, y, z) from Gₜ              │
│          └─ Shape: [N_gaussians, 3]                        │
│                                                             │
│   Architecture: 8-layer MLP                                 │
│          ┌──────────────────────────────┐                  │
│          │ Linear(3 → hidden_dim)       │                  │
│          │ ReLU                          │                  │
│          │ Linear(hidden_dim → hidden)   │ × 6 layers      │
│          │ ReLU                          │                  │
│          │ Linear(hidden → output_dim)   │                  │
│          └──────────────────────────────┘                  │
│                                                             │
│   Output: Deformation parameters                            │
│          ├─ Δx, Δy, Δz  (position offset)                  │
│          ├─ Δα          (opacity change)                   │
│          └─ Δs          (scale change)                     │
│          └─ Shape: [N_gaussians, 5] or [N_gaussians, 8]    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 2.3 Autoregressive Pipeline

```
Step 1: 초기 Gaussian 생성 (FaceLift)
        F₀ → G₀, F₁ → G₁, ..., Fₜ → Gₜ

Step 2: Anchor Frame 선택
        G₀ = Canonical Gaussians (기준)

Step 3: Autoregressive Deformation
        G₀ + D₀(G₀) → G'₁
        G'₁ + D₁(G'₁) → G'₂
        ...
        G'ₜ₋₁ + Dₜ₋₁(G'ₜ₋₁) → G'ₜ

결과: 시간적으로 연속적인 {G'₀, G'₁, ..., G'ₜ}
```

### 2.4 학습 방법

**Supervision Strategy**:
```python
# Pseudo Ground Truth 생성
target_images = render_6views(G_{t+1})  # FaceLift 출력

# 예측 이미지 생성
G'_{t+1} = apply_deformation(G_t, D_t(G_t.positions))
pred_images = render_6views(G'_{t+1})

# 손실 계산 (추정)
loss = L2_loss(pred_images, target_images) 
     + perceptual_loss(pred_images, target_images)
```

### 2.5 논문에서 명시되지 않은 사항

| 항목 | 상태 | 추정/해결 방안 |
|------|------|----------------|
| Hidden dimension | ❓ | 256 또는 512 (일반적 설정) |
| Activation | ❓ | ReLU (표준) |
| Loss weights | ❓ | L2:1.0, Perceptual:0.1 |
| 학습 데이터셋 | ❓ | 비디오 데이터 일반 |
| 정량적 평가 | ❌ | 정성적 결과만 제시 |

---

## 3. 구현 계획

### 3.1 모듈 구조

```
mouse_extensions/
├── model/
│   └── deformation/
│       ├── __init__.py
│       ├── deformation_network.py    # 8-layer MLP
│       ├── deformation_trainer.py    # 학습 로직
│       └── temporal_pipeline.py      # Autoregressive 파이프라인
├── data/
│   └── temporal_dataset.py           # 연속 프레임 데이터셋
└── scripts/
    └── train_deformation.py          # 학습 스크립트
```

### 3.2 핵심 클래스 설계

#### 3.2.1 DeformationNetwork

```python
class DeformationNetwork(nn.Module):
    """
    8-layer MLP for Gaussian deformation prediction.
    
    Input: Gaussian 3D positions [N, 3]
    Output: Deformation parameters [N, output_dim]
            - position: Δx, Δy, Δz
            - opacity: Δα
            - scale: Δs (or Δsx, Δsy, Δsz)
    """
    
    def __init__(
        self,
        input_dim: int = 3,
        hidden_dim: int = 256,
        output_dim: int = 5,  # Δxyz + Δα + Δs
        num_layers: int = 8,
        activation: str = "relu",
    ):
        super().__init__()
        
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize to predict zero deformation initially
        self._init_weights()
    
    def forward(self, positions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            positions: [N, 3] Gaussian centers
        Returns:
            deformations: [N, output_dim] predicted offsets
        """
        return self.mlp(positions)
```

#### 3.2.2 TemporalGaussianPipeline

```python
class TemporalGaussianPipeline:
    """
    Autoregressive pipeline for temporal Gaussian generation.
    """
    
    def __init__(
        self,
        gslrm_model: GSLRM,
        deform_net: DeformationNetwork,
        anchor_frame_idx: int = 0,
    ):
        self.gslrm = gslrm_model
        self.deform_net = deform_net
        self.anchor_idx = anchor_frame_idx
    
    def generate_initial_gaussians(
        self, 
        video_frames: List[torch.Tensor]
    ) -> List[GaussianParams]:
        """Step 1: Generate independent Gaussians for each frame."""
        gaussians = []
        for frame in video_frames:
            G = self.gslrm(frame)  # Single-frame inference
            gaussians.append(G)
        return gaussians
    
    def apply_deformation(
        self,
        gaussian: GaussianParams,
        deformation: torch.Tensor,
    ) -> GaussianParams:
        """Apply predicted deformation to Gaussian parameters."""
        new_gaussian = gaussian.clone()
        
        # Position offset
        new_gaussian.positions += deformation[:, :3]
        
        # Opacity offset (with clamping)
        new_gaussian.opacities += deformation[:, 3:4]
        new_gaussian.opacities = torch.clamp(new_gaussian.opacities, 0, 1)
        
        # Scale offset
        new_gaussian.scales += deformation[:, 4:]
        new_gaussian.scales = torch.clamp(new_gaussian.scales, min=1e-6)
        
        return new_gaussian
    
    def generate_temporal_sequence(
        self,
        initial_gaussians: List[GaussianParams],
    ) -> List[GaussianParams]:
        """Step 3: Autoregressive deformation."""
        T = len(initial_gaussians)
        
        # Start from anchor
        deformed = [None] * T
        deformed[self.anchor_idx] = initial_gaussians[self.anchor_idx]
        
        # Forward propagation (anchor → T)
        for t in range(self.anchor_idx + 1, T):
            prev_G = deformed[t - 1]
            deformation = self.deform_net(prev_G.positions)
            deformed[t] = self.apply_deformation(prev_G, deformation)
        
        # Backward propagation (anchor → 0)
        for t in range(self.anchor_idx - 1, -1, -1):
            next_G = deformed[t + 1]
            # Use negative deformation for backward
            deformation = -self.deform_net(next_G.positions)
            deformed[t] = self.apply_deformation(next_G, deformation)
        
        return deformed
```

### 3.3 학습 파이프라인

```python
class DeformationTrainer:
    """
    Train deformation network using FaceLift outputs as pseudo GT.
    """
    
    def __init__(self, config):
        self.deform_net = DeformationNetwork(
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
        )
        self.renderer = GaussianRenderer()
        self.perceptual_loss = PerceptualLoss()
        
    def train_step(
        self,
        G_t: GaussianParams,      # Current frame Gaussian
        G_t1_target: GaussianParams,  # Next frame Gaussian (pseudo GT)
        cameras: List[Camera],
    ) -> Dict[str, torch.Tensor]:
        """Single training step."""
        
        # Predict deformation
        deformation = self.deform_net(G_t.positions)
        
        # Apply deformation
        G_t1_pred = self.apply_deformation(G_t, deformation)
        
        # Render 6 views
        pred_images = self.render_views(G_t1_pred, cameras)
        target_images = self.render_views(G_t1_target, cameras)
        
        # Compute losses
        l2_loss = F.mse_loss(pred_images, target_images)
        perc_loss = self.perceptual_loss(pred_images, target_images)
        
        total_loss = l2_loss + 0.1 * perc_loss
        
        return {
            "loss": total_loss,
            "l2_loss": l2_loss,
            "perceptual_loss": perc_loss,
        }
```

### 3.4 구현 단계 (Phase Plan)

| Phase | 내용 | 예상 기간 | 의존성 |
|-------|------|----------|--------|
| **P1** | DeformationNetwork 클래스 구현 | 1일 | - |
| **P2** | GaussianParams 데이터 구조 정의 | 0.5일 | P1 |
| **P3** | Deformation 적용 함수 구현 | 0.5일 | P2 |
| **P4** | TemporalDataset 구현 | 1일 | - |
| **P5** | DeformationTrainer 구현 | 1일 | P1-P4 |
| **P6** | 학습 스크립트 및 Config | 0.5일 | P5 |
| **P7** | 추론 파이프라인 통합 | 1일 | P1-P6 |
| **P8** | 테스트 및 검증 | 1일 | P7 |

**총 예상 기간**: 6-7일

---

## 4. 생쥐 데이터 특화 고려사항

### 4.1 생쥐 움직임 특성

| 특성 | 값 | 영향 |
|------|-----|------|
| 프레임간 이동 | 평균 99px, 최대 195px | 큰 deformation 필요 |
| 관절 변형 | 다리, 꼬리 움직임 | Local deformation 중요 |
| 털 표면 | 미세한 텍스처 변화 | Scale/opacity 변화 |

### 4.2 생쥐 특화 확장

```python
class MouseDeformationNetwork(DeformationNetwork):
    """
    Mouse-specific deformation network with:
    - Larger hidden dim for complex movements
    - Positional encoding for fine-grained control
    - Optional keypoint conditioning
    """
    
    def __init__(self, use_positional_encoding=True, **kwargs):
        super().__init__(**kwargs)
        
        if use_positional_encoding:
            self.pos_encoder = PositionalEncoding(
                input_dim=3,
                freq_bands=10,
            )
            # Adjust first layer input
            self.mlp[0] = nn.Linear(
                3 + 3 * 2 * 10,  # pos + positional encoding
                kwargs.get('hidden_dim', 256)
            )
    
    def forward(self, positions):
        if hasattr(self, 'pos_encoder'):
            positions = self.pos_encoder(positions)
        return self.mlp(positions)
```

### 4.3 ARAP (As-Rigid-As-Possible) Loss 추가

```python
def arap_loss(
    positions_prev: torch.Tensor,
    positions_curr: torch.Tensor,
    neighbors: torch.Tensor,
) -> torch.Tensor:
    """
    Preserve local rigidity during deformation.
    Encourages consistent motion within local neighborhoods.
    """
    # Compute relative positions
    rel_prev = positions_prev[neighbors] - positions_prev.unsqueeze(1)
    rel_curr = positions_curr[neighbors] - positions_curr.unsqueeze(1)
    
    # Distance preservation loss
    dist_prev = torch.norm(rel_prev, dim=-1)
    dist_curr = torch.norm(rel_curr, dim=-1)
    
    return F.mse_loss(dist_prev, dist_curr)
```

---

## 5. 평가 계획

### 5.1 정량적 지표

| 지표 | 측정 방법 | 목표 |
|------|----------|------|
| **Temporal Consistency** | 연속 프레임 간 LPIPS 변화량 | ↓ 낮을수록 좋음 |
| **Flickering Score** | 픽셀 강도 변화 분산 | ↓ 낮을수록 좋음 |
| **PSNR** | GT 대비 렌더링 품질 | ≥ 기존 유지 |
| **SSIM** | 구조적 유사도 | ≥ 기존 유지 |

### 5.2 정성적 평가

- Turntable 비디오 부드러움 시각 확인
- A/B 비교 (Deformation 적용 전/후)
- 프레임 간 형태 변화 연속성

---

## 6. 파일 위치 및 참조

| 문서 | 경로 |
|------|------|
| 이 분석 문서 | `docs/research/260203_Deformation_Network_Analysis.md` |
| FaceLift 논문 | https://arxiv.org/pdf/2412.17812 |
| Quick Reference | `docs/MOUSE_QUICK_REFERENCE.md` |

---

---

## 7. 구현 진행 현황 (260203)

### 완료된 Phase

| Phase | 내용 | 상태 | 테스트 |
|-------|------|------|--------|
| **P1** | DeformationNetwork | ✅ 완료 | 397K params, zero-init |
| **P2** | GaussianParams | ✅ 완료 | round-trip, apply_deform |
| **P3** | TemporalPipeline | ✅ 완료 | autoregressive, gradient |
| **P4** | TemporalDataset | ✅ 완료 | 2880→2879 pairs |
| **P5** | DeformationTrainer | ✅ 완료 | checkpoint save/load |
| **P6** | Training Script | ✅ 완료 | dry run passed |
| **P7** | GS-LRM Integration | 🔄 진행 예정 | - |
| **P8** | E2E Test | 🔄 진행 예정 | - |

### 생성된 파일



### 다음 단계

1. [ ] **P7**: GS-LRM 추론과 통합 (pseudo GT 생성)
2. [ ] **P8**: 전체 파이프라인 E2E 테스트
3. [ ] 실제 학습 실행 및 평가

---

*Created: 2026-02-03 | FaceLift Mouse Project | Research Notes*

---

## 11. Data Naming Convention (260204 추가)

### M5 / M5t / M5t2 구분

| 이름 | 타입 | 설명 |
|------|------|------|
| **M5** | 전처리 데이터 폴더 | `/home/joon/data/preprocessed/FaceLift_mouse/M5/` |
| **M5t** | Config/실험 이름 | M5 + **1to1 split** (1:1:1, 균등 배분) |
| **M5t2** | Config/실험 이름 | M5 + **t2 split** (80:10:10, 학습 최대화) |

### Split 파일 위치
모든 split 파일은 **M5 폴더 안에** 있음:
```
M5/
├── data_mouse_1to1_{train,val,test}.txt  ← M5t용
└── data_mouse_t2_{train,val,test}.txt    ← M5t2용
```

### Split 비율
| Split | Train | Val | Test | 권장 용도 |
|-------|-------|-----|------|----------|
| 1to1 | 1198 | 1198 | 1204 | 균등 비교 실험 |
| t2 | 2880 | 360 | 360 | **프로덕션 학습** ✅ |

### 중요: --split 파라미터 필수
```bash
# ❌ split 없으면 전체 샘플 사용 → train/val/test 구분 안됨
--data_dir ~/data/preprocessed/FaceLift_mouse/M5 --num_frames 20

# ✅ split 명시해야 올바른 데이터 사용
--data_dir ~/data/preprocessed/FaceLift_mouse/M5 --split data_mouse_t2_train.txt --num_frames 20
```

---

## 12. Training Commands (260204 Updated)

### 권장: 2단계 순차 실행 (한 번에)
```bash
cd /home/joon/dev/FaceLift

# Step 1 (캐시 ~2h) + Step 2 (학습 ~4h) 순차 실행
export CUDA_VISIBLE_DEVICES=6 && nohup bash -c '
    echo "=== Step 1: Precompute Gaussian Cache ===" && \
    python -m mouse_extensions.scripts.train_deformation \
        --config configs/deformation/default.yaml \
        --precompute_cache && \
    echo "=== Step 2: Train Deformation Network ===" && \
    python -m mouse_extensions.scripts.train_deformation \
        --config configs/deformation/default.yaml
' > logs/deformation_full.log 2>&1 &

# 진행 확인
tail -f logs/deformation_full.log
```

### 개별 실행 (필요시)
```bash
# Step 1만
export CUDA_VISIBLE_DEVICES=5 && python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml --precompute_cache

# Step 2만 (캐시 완료 후)
export CUDA_VISIBLE_DEVICES=5 && python -m mouse_extensions.scripts.train_deformation \
    --config configs/deformation/default.yaml
```

### 캐시 위치
`/node_data/joon/checkpoints/FaceLift/deformation/default/gaussian_cache/`

### Config 필수 요소 (260204 Fix)

`configs/deformation/default.yaml`에서 GS-LRM 설정 필수:

```yaml
gslrm:
  checkpoint: "/path/to/best_psnr.pt"  # GS-LRM 체크포인트
  config: "/path/to/config.yaml"       # GS-LRM config (필수!)
  resolution: 512                       # 이미지 해상도
```

⚠️ **주의**: `gslrm.config` 누락 시 GSLRMInference 로딩 실패 (silent crash)

---

