---
date: 2024-12-21
context_name: "2_Research"
tags: [ai-assisted, mvdiffusion, camera-pose-conditioning, mouse-facelift, gslrm]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# MVDiffusion 한계 분석 및 대안 전략

## 요약

MVDiffusion의 **고정된 카메라 배치 가정**이 비균등 카메라 배열의 생쥐 데이터셋에서 심각한 한계를 보임.
합성 데이터 품질 저하로 인해 **실제 6-view 데이터 직접 사용**이 최선의 단기 전략이며,
중장기적으로는 **Camera Pose Conditioning** 방식의 diffusion 모델 도입을 권장.

## 1. MVDiffusion의 근본적 한계

### 1.1 아키텍처 분석

MVDiffusion은 **Discrete View Index** 기반 설계:

```python
# MVDiffusion의 뷰 처리 방식
class MVDiffusionUNet:
    def __init__(self, num_views=6):
        # 고정된 6개 뷰에 대한 cross-attention
        self.cross_view_attention = CrossViewAttention(num_views)
        # 뷰 인덱스별 임베딩 (0, 1, 2, 3, 4, 5)
        self.view_embeddings = nn.Embedding(num_views, embed_dim)
```

**핵심 문제**: 뷰 인덱스 0~5가 **균등한 60° 간격**을 암묵적으로 가정

### 1.2 생쥐 카메라 배열 vs MVDiffusion 가정

| 뷰 쌍 | 실제 각도 간격 | MVDiffusion 가정 | 차이 |
|-------|--------------|-----------------|------|
| 0→1 | 13.5° | 60° | -46.5° |
| 1→2 | 22.5° | 60° | -37.5° |
| 2→3 | 36.0° | 60° | -24.0° |
| 3→4 | 27.0° | 60° | -33.0° |
| 4→5 | 52.4° | 60° | -7.6° |
| 5→0 | **208.6°** (gap) | 60° | +148.6° |

**문제점**:
1. View 0→1 간격이 13.5°로 매우 좁음 → 중복 정보
2. View 5→0 간격이 208.6°로 거대한 gap 존재
3. 뷰 인덱스가 실제 기하학적 관계를 반영하지 않음

### 1.3 Reference View Rotation의 불가능

균등 배열에서는 reference view를 순환시켜 데이터 증강 가능:
```
# 균등 60° 배열의 경우
ref=0: [0°, 60°, 120°, 180°, 240°, 300°]
ref=1: [60°, 120°, 180°, 240°, 300°, 0°]  # 단순 순환
```

비균등 배열에서는 이것이 **불가능**:
```
# 생쥐 카메라의 경우
ref=0: [0°, 13.5°, 36°, 72°, 99°, 151.4°]
ref=1 시도: prompt embedding 0 (front view)이 13.5° 뷰에 매칭 → semantic mismatch
```

## 2. 합성 데이터 vs 실제 데이터 전략 비교

### 2.1 합성 데이터 (MVDiffusion) 접근법

```
[실제 1-view 이미지] → MVDiffusion → [합성 6-view 이미지] → GS-LRM
```

**장점**:
- 이론적으로 무한 데이터 생성 가능
- View-consistent 생성 (동일 diffusion 과정)

**단점 (현재 발견)**:
- 카메라 배열 불일치로 geometry 왜곡
- 합성 이미지 품질 저하 (특히 occluded regions)
- Base model (human) bias 잔존

### 2.2 실제 6-view 데이터 직접 사용

```
[실제 6-view 이미지] → GS-LRM (직접 학습)
```

**장점**:
- **최고 품질**: 실제 캡처된 multi-view consistency
- **정확한 camera calibration**: 보정된 extrinsics/intrinsics
- **도메인 갭 없음**: 학습과 추론이 동일 분포

**단점**:
- 데이터 양 제한 (현재 ~1800 프레임)
- 증강 옵션 제한적

### 2.3 결론: 실제 데이터 직접 사용 권장

생쥐 데이터셋은 이미 **synchronized 6-camera setup**으로 촬영됨.
MVDiffusion을 통한 합성 데이터 생성은 불필요하며 오히려 품질을 저하시킴.

```yaml
# 권장 설정: data_mouse_centered 직접 사용
training:
  dataset:
    dataset_path: "data_mouse_centered/data_mouse_train.txt"
    num_views: 6
    num_input_views: 5
```

## 3. Camera Pose Conditioning 전략 (중장기)

### 3.1 개념

**핵심 아이디어**: 뷰 인덱스(0-5) 대신 **연속적인 카메라 포즈**를 조건으로 제공

```
기존: condition = view_index ∈ {0, 1, 2, 3, 4, 5}
개선: condition = camera_pose ∈ SE(3)  # 연속 공간
```

### 3.2 구현 방법론

#### 방법 1: Plücker Ray Embedding (권장)

```python
def plucker_embedding(camera_origin, ray_direction):
    """
    6D Plücker coordinates: 카메라 위치와 시선 방향을 인코딩

    Args:
        camera_origin: [B, 3] - 카메라 위치 (world coordinates)
        ray_direction: [B, H, W, 3] - 각 픽셀의 ray direction

    Returns:
        plucker: [B, H, W, 6] - Plücker coordinates
    """
    # Plücker line representation: (d, m) where m = o × d
    moment = torch.cross(
        camera_origin.unsqueeze(1).unsqueeze(1).expand(-1, H, W, -1),
        ray_direction
    )
    plucker = torch.cat([ray_direction, moment], dim=-1)
    return plucker
```

**장점**:
- 픽셀 단위 기하학적 정보 인코딩
- 임의의 카메라 배열에 일반화
- Zero123++, SV3D에서 검증됨

#### 방법 2: Spherical Coordinate Encoding

```python
class SphericalPoseEncoder(nn.Module):
    """구면 좌표계 기반 카메라 포즈 인코딩"""

    def __init__(self, embed_dim=256, num_freq_bands=64):
        super().__init__()
        self.freq_bands = 2 ** torch.linspace(0, 10, num_freq_bands)
        self.mlp = nn.Sequential(
            nn.Linear(num_freq_bands * 6, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim)
        )

    def fourier_encode(self, x, bands):
        """Fourier feature encoding for continuous values"""
        x_proj = x.unsqueeze(-1) * bands  # [B, num_bands]
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    def forward(self, azimuth, elevation, distance):
        """
        Args:
            azimuth: [B] - 방위각 (radians)
            elevation: [B] - 앙각 (radians)
            distance: [B] - 카메라-원점 거리
        """
        az_enc = self.fourier_encode(azimuth, self.freq_bands)
        el_enc = self.fourier_encode(elevation, self.freq_bands)
        dist_enc = self.fourier_encode(distance, self.freq_bands)

        features = torch.cat([az_enc, el_enc, dist_enc], dim=-1)
        return self.mlp(features)
```

#### 방법 3: Camera Extrinsic Direct Encoding

```python
class ExtrinsicEncoder(nn.Module):
    """카메라 외부 파라미터 직접 인코딩"""

    def __init__(self, embed_dim=256):
        super().__init__()
        # R: 3x3 rotation, t: 3 translation → 12 params
        self.mlp = nn.Sequential(
            nn.Linear(12, 128),
            nn.SiLU(),
            nn.Linear(128, embed_dim)
        )

    def forward(self, rotation, translation):
        """
        Args:
            rotation: [B, 3, 3] - rotation matrix
            translation: [B, 3] - translation vector
        """
        rot_flat = rotation.flatten(start_dim=1)  # [B, 9]
        extrinsic = torch.cat([rot_flat, translation], dim=-1)  # [B, 12]
        return self.mlp(extrinsic)
```

### 3.3 MVDiffusion 수정 방안

```python
# 기존 MVDiffusion
class MVDiffusionUNet:
    def forward(self, x, timestep, view_idx, prompt_embed):
        view_embed = self.view_embeddings(view_idx)  # Discrete
        ...

# Camera Pose Conditioning 적용
class MVDiffusionUNet_PoseConditioned:
    def __init__(self):
        self.pose_encoder = SphericalPoseEncoder(embed_dim=1024)
        # view_embeddings 제거

    def forward(self, x, timestep, camera_pose, prompt_embed):
        pose_embed = self.pose_encoder(
            camera_pose['azimuth'],
            camera_pose['elevation'],
            camera_pose['distance']
        )
        # pose_embed을 cross-attention에 통합
        ...
```

### 3.4 구현 우선순위

| 순위 | 방법 | 난이도 | 기대 효과 | 비고 |
|-----|------|-------|---------|------|
| 1 | Plücker Ray | 중 | 높음 | Zero123++ 방식, 검증됨 |
| 2 | Spherical Coord | 하 | 중 | 구현 간단, 구면 배열에 적합 |
| 3 | Extrinsic Direct | 하 | 중-하 | 가장 단순, 일반화 한계 |

## 4. 권장 전략 로드맵

### 단기 (즉시)
- **실제 centered 데이터로 GS-LRM 학습**
- MVDiffusion 합성 데이터 의존도 제거
- 현재 데이터로 baseline 성능 확립

### 중기 (1-2주)
- **Plücker Ray Conditioning** 구현
- MVDiffusion UNet 수정
- 비균등 카메라 배열 지원

### 장기 (추후)
- Zero123++ 스타일 모델 탐색
- Single-view → 임의 viewpoint 생성
- 카메라 배열 독립적 일반화

## 5. 관련 파일

- `configs/mouse_gslrm_real_data.yaml`: 실제 데이터 학습 설정
- `data_mouse_centered/`: 전처리된 실제 6-view 데이터
- `mvdiffusion/models/unet_mv2d_condition.py`: UNet 수정 대상

## 6. 참고 문헌

- Zero123++: https://arxiv.org/abs/2310.15110
- SV3D: https://arxiv.org/abs/2403.12008
- MVDiffusion: https://arxiv.org/abs/2307.01097
