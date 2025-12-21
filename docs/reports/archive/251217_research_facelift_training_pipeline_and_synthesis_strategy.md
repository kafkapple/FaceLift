---
date: 2024-12-17
context_name: "2_Research"
tags: [ai-assisted, mouse-facelift, gs-lrm, training-pipeline, view-synthesis, 3dgs, 2dgs]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# FaceLift 학습 파이프라인 분석 및 Mouse 합성 데이터 생성 전략

## 1. FaceLift 원 논문 정확한 학습 파이프라인

### 1.1 MVDiffusion (Stage 1)

| 항목 | 내용 |
|------|------|
| **입력** | 단일 정면 얼굴 이미지 |
| **출력** | **6개 뷰** (α, α±45°, α±90°, α+180°) |
| **학습 데이터** | 합성 인간 얼굴 렌더링 데이터 |
| **Base 모델** | Stable Diffusion V2-1-unCLIP |

### 1.2 GS-LRM (Stage 2)

| 항목 | 내용 |
|------|------|
| **학습 시 뷰 수** | **8개** (4개 입력 + 4개 supervision) |
| **전체 데이터 뷰 수** | **32개** 렌더링 (랜덤 HDR 조명) |
| **학습 전략** | **Objaverse pretrain → Synthetic Head finetune** |

```
┌─────────────────────────────────────────────────────────────────────┐
│                    FaceLift GS-LRM 학습 파이프라인                    │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Stage 1: Objaverse Pretrain (일반 3D prior)                        │
│  ───────────────────────────────────────────                        │
│  - ~80K 3D 모델에서 렌더링                                          │
│  - 32개 뷰 렌더링                                                    │
│  - 일반적인 3D geometry prior 학습                                  │
│                                                                      │
│            ↓                                                        │
│                                                                      │
│  Stage 2: Synthetic Head Finetune (도메인 적응)                     │
│  ──────────────────────────────────────────────                     │
│  - 합성 인간 헤드 3D 모델에서 렌더링                                 │
│  - 32개 뷰 렌더링 (랜덤 HDR 조명)                                    │
│  - 매 step: 8개 뷰 랜덤 샘플 (4 input + 4 supervision)              │
│  - Human face 도메인 적응                                            │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 핵심 인사이트

> **GS-LRM이 32개 뷰로 학습하는 이유**: 모든 뷰가 동일한 3D 모델에서 렌더링되어 완벽한 multi-view consistency 보장. 이 경우 2D loss만으로도 올바른 3D 학습이 가능.

---

## 2. Diffusion 기반 View Interpolation 연구 동향

### 2.1 관련 최신 연구

| 모델 | 특징 | 발표 |
|------|------|------|
| **3DiM** | 단일 뷰 → 임의 각도 뷰 생성, geometry-free | Google Research |
| **4DiM** | 연속 카메라 궤적 + 시간 지원, 보간/외삽 가능 | 2024 |
| **ViewNeTI** | 연속 카메라 파라미터 → view token → diffusion | ECCV 2024 |
| **SplatDiff** | Depth splatting + diffusion 결합 | SIGGRAPH 2024 |
| **ViewFusion** | Multi-view consistency via interpolated denoising | CVPR 2024 |

### 2.2 ViewNeTI 핵심 아이디어

```python
# ViewNeTI: 연속적인 카메라 파라미터로 임의 각도 생성
camera_params = (azimuth=37.5, elevation=15)  # 임의의 연속 값
view_token = neural_mapper(camera_params)      # 학습된 매퍼
generated_image = diffusion(view_token)        # 해당 각도 이미지 생성

# 장점:
# - 60°, 120° 뿐 아니라 75°, 97° 등 임의 각도 생성 가능
# - View interpolation 자연스럽게 지원
# - Scene-specific optimization 가능
```

---

## 3. Mouse 데이터 32뷰 합성 전략

### 3.1 전략 비교

| 전략 | 방법 | 품질 | 복잡도 | 권장 |
|------|------|------|--------|------|
| **A: 간이 3DGS → 렌더링** | 6뷰로 3DGS 최적화 → 32뷰 렌더링 | 중-상 | 중 | ◎ 1순위 |
| **B: 2DGS → 렌더링** | 기하학적 정확도 향상된 2DGS | 상 | 중-상 | ◎ 1순위 |
| **C: Zero123++ 활용** | 각 뷰에서 인접 각도 생성 | 중 | 낮음 | O |
| **D: ViewNeTI 적용** | MVDiffusion 연속 각도 확장 | 중-상 | 높음 | △ 장기 |

### 3.2 1순위 전략: 간이 3DGS/2DGS → 32뷰 렌더링

```
┌─────────────────────────────────────────────────────────────────────┐
│                    6뷰 → 32뷰 합성 데이터 파이프라인                   │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  입력: 6개 실제 마우스 뷰 (0°, 60°, 120°, 180°, 240°, 300°)          │
│                                                                      │
│            ↓                                                        │
│                                                                      │
│  Step 1: 간이 3DGS/2DGS 최적화 (per-scene)                          │
│  ────────────────────────────────────────                            │
│  - 6개 뷰 + 카메라 파라미터 입력                                     │
│  - Gaussian Splatting 최적화 (500-2000 iterations)                  │
│  - 완벽하지 않아도 됨 (대략적인 형태만 필요)                          │
│                                                                      │
│            ↓                                                        │
│                                                                      │
│  Step 2: 32개 카메라 위치에서 렌더링                                 │
│  ────────────────────────────────────────                            │
│  - FaceLift와 동일한 카메라 배치 (11.25° 간격)                       │
│  - elevation 20°, distance 2.7                                      │
│  - 32개 RGB 이미지 + 카메라 파라미터 저장                            │
│                                                                      │
│            ↓                                                        │
│                                                                      │
│  Step 3: GS-LRM 학습용 데이터셋 구축                                 │
│  ────────────────────────────────────────                            │
│  - opencv_cameras.json 형식으로 저장                                │
│  - Objaverse와 동일한 형식                                          │
│                                                                      │
│  출력: 32뷰 합성 데이터셋 (~1,600 scenes × 32 views)                │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 4. 간이 3DGS 모듈 구현 계획

### 4.1 설계 원칙

```yaml
목표:
  - 6개 sparse 뷰에서 빠르게 3DGS 최적화
  - 완벽한 품질 X, 대략적인 형태 재구성 O
  - Novel view 렌더링이 가능한 수준

핵심 설정:
  - Gaussian 수: 10,000 ~ 50,000 (소규모)
  - Iterations: 500 ~ 2,000 (빠른 최적화)
  - Loss: L1 + SSIM (단순)
  - 정규화: 없음 또는 최소 (속도 우선)
```

### 4.2 모듈 구조

```
scripts/
└── synthesis/
    ├── __init__.py
    ├── sparse_3dgs.py          # 간이 3DGS 최적화
    ├── sparse_2dgs.py          # 간이 2DGS 최적화 (옵션)
    ├── render_novel_views.py   # 새 각도 렌더링
    ├── create_dataset.py       # 데이터셋 생성
    └── config.yaml             # 설정 파일
```

### 4.3 Sparse3DGS 클래스 설계

```python
# scripts/synthesis/sparse_3dgs.py

class Sparse3DGS:
    """
    6개 sparse 뷰에서 빠른 3DGS 최적화.

    설계 결정:
    - 적은 Gaussian 수 (10K-50K) → 빠른 최적화
    - 짧은 iteration (500-2000) → 대략적 형태만 필요
    - 단순 loss (L1 + SSIM) → 복잡한 정규화 제외
    """

    def __init__(self, config):
        self.num_gaussians = config.get("num_gaussians", 20000)
        self.iterations = config.get("iterations", 1000)
        self.lr_position = config.get("lr_position", 0.001)
        self.lr_feature = config.get("lr_feature", 0.01)
        self.lr_opacity = config.get("lr_opacity", 0.05)
        self.lr_scaling = config.get("lr_scaling", 0.005)
        self.lr_rotation = config.get("lr_rotation", 0.001)

    def optimize(self, images, cameras):
        """
        Args:
            images: [6, H, W, 3] - 6개 뷰 이미지
            cameras: [6, 4, 4] - 카메라 extrinsics (c2w)
            intrinsics: [4] - fx, fy, cx, cy

        Returns:
            GaussianModel: 최적화된 3DGS 모델
        """
        # 1. Point cloud 초기화 (SfM 또는 랜덤)
        # 2. Gaussian 파라미터 초기화
        # 3. 최적화 루프
        # 4. 반환
        pass

    def render(self, camera):
        """새로운 카메라 위치에서 렌더링"""
        pass
```

### 4.4 구현 선택지

| 옵션 | 장점 | 단점 | 구현 난이도 |
|------|------|------|-------------|
| **A: 기존 3DGS 라이브러리 활용** | 검증됨, 빠름 | 의존성 | 낮음 |
| **B: gsplat 라이브러리** | 경량, 유연 | 학습 필요 | 중간 |
| **C: 직접 구현** | 완전 제어 | 시간 소요 | 높음 |

**권장**: gsplat 라이브러리 활용 (CUDA 커널 포함, 설치 간편)

---

## 5. 2DGS 옵션 구현 계획

### 5.1 2DGS vs 3DGS 비교

| 측면 | 3DGS | 2DGS |
|------|------|------|
| **Gaussian 형태** | 3D 타원체 | 2D 디스크 (표면 정렬) |
| **기하학적 정확도** | 중간 | **높음** |
| **표면 표현** | 볼륨 기반 | **표면 기반** |
| **Sparse 뷰 성능** | 좋음 | **더 좋음** |
| **구현 복잡도** | 낮음 | 중간 |

### 5.2 왜 2DGS가 Mouse에 더 적합할 수 있는가?

```
마우스 특성:
- 표면이 명확함 (털로 덮인 외피)
- 내부 볼륨보다 표면 형태가 중요
- 6개 sparse 뷰로 표면 추정이 더 안정적

2DGS 장점:
- 표면에 정렬된 Gaussian → 더 정확한 geometry
- Sparse 뷰에서 floater/artifact 감소
- Novel view 품질 향상
```

### 5.3 Sparse2DGS 클래스 설계

```python
# scripts/synthesis/sparse_2dgs.py

class Sparse2DGS:
    """
    2D Gaussian Splatting for sparse view reconstruction.

    핵심 차이점 (vs 3DGS):
    - Gaussian이 표면에 정렬된 2D 디스크
    - Normal 방향 학습 필요
    - Depth regularization 중요
    """

    def __init__(self, config):
        self.num_gaussians = config.get("num_gaussians", 20000)
        self.iterations = config.get("iterations", 1500)  # 2DGS는 좀 더 필요

        # 2DGS 특수 파라미터
        self.depth_weight = config.get("depth_weight", 0.1)
        self.normal_weight = config.get("normal_weight", 0.05)

    def optimize(self, images, cameras):
        """2DGS 최적화"""
        pass

    def render(self, camera, return_depth=False, return_normal=False):
        """렌더링 + 선택적 depth/normal 출력"""
        pass
```

---

## 6. 임의 각도 생성을 위한 기술적 세부사항

### 6.1 카메라 파라미터 생성

```python
# 32개 카메라 위치 생성 (FaceLift 스타일)
def generate_camera_poses(num_views=32, elevation=20, distance=2.7):
    """
    FaceLift와 동일한 카메라 배치 생성.

    Args:
        num_views: 생성할 뷰 수 (기본 32)
        elevation: 카메라 elevation 각도 (기본 20°)
        distance: 원점에서 카메라까지 거리 (기본 2.7)

    Returns:
        cameras: [num_views, 4, 4] - c2w 행렬들
    """
    cameras = []
    for i in range(num_views):
        azimuth = i * (360 / num_views)  # 0, 11.25, 22.5, ...

        # Spherical to Cartesian
        x = distance * np.cos(np.radians(elevation)) * np.cos(np.radians(azimuth))
        y = distance * np.cos(np.radians(elevation)) * np.sin(np.radians(azimuth))
        z = distance * np.sin(np.radians(elevation))

        # Look-at matrix 계산
        c2w = look_at(eye=[x, y, z], center=[0, 0, 0], up=[0, 0, 1])
        cameras.append(c2w)

    return np.stack(cameras, axis=0)
```

### 6.2 렌더링 파이프라인

```python
# scripts/synthesis/render_novel_views.py

def render_32_views(gaussian_model, intrinsics, output_dir):
    """
    최적화된 Gaussian 모델에서 32개 뷰 렌더링.

    Args:
        gaussian_model: Sparse3DGS 또는 Sparse2DGS 모델
        intrinsics: [fx, fy, cx, cy]
        output_dir: 출력 디렉토리

    Output:
        output_dir/
        ├── images/
        │   ├── cam_000.png
        │   ├── cam_001.png
        │   └── ... (32개)
        └── opencv_cameras.json
    """
    cameras = generate_camera_poses(num_views=32)

    os.makedirs(f"{output_dir}/images", exist_ok=True)
    frames = []

    for i, c2w in enumerate(cameras):
        # 렌더링
        rgb = gaussian_model.render(c2w)

        # 저장
        Image.fromarray((rgb * 255).astype(np.uint8)).save(
            f"{output_dir}/images/cam_{i:03d}.png"
        )

        # 카메라 정보
        w2c = np.linalg.inv(c2w)
        frames.append({
            "w": 512, "h": 512,
            "fx": intrinsics[0], "fy": intrinsics[1],
            "cx": intrinsics[2], "cy": intrinsics[3],
            "w2c": w2c.tolist(),
            "file_path": f"images/cam_{i:03d}.png"
        })

    # JSON 저장
    with open(f"{output_dir}/opencv_cameras.json", "w") as f:
        json.dump({"frames": frames}, f, indent=2)
```

---

## 7. 전체 구현 로드맵

### Phase 1: 기반 구축 (1-2일)

```
[ ] gsplat 라이브러리 설치 및 테스트
[ ] 카메라 파라미터 생성 유틸리티 구현
[ ] 기본 렌더링 파이프라인 구현
```

### Phase 2: Sparse 3DGS 구현 (2-3일)

```
[ ] Sparse3DGS 클래스 구현
[ ] 6개 뷰 최적화 테스트
[ ] 32개 뷰 렌더링 테스트
[ ] 품질 평가 (실제 뷰와 비교)
```

### Phase 3: 2DGS 옵션 구현 (2-3일)

```
[ ] 2DGS 라이브러리 조사 (surfel-gs, 2d-gaussian-splatting)
[ ] Sparse2DGS 클래스 구현
[ ] 3DGS vs 2DGS 품질 비교
```

### Phase 4: 대규모 데이터셋 생성 (1-2일)

```
[ ] 배치 처리 스크립트 구현
[ ] 1,600 scenes × 32 views 생성
[ ] 데이터셋 검증
```

### Phase 5: GS-LRM 학습 (3-5일)

```
[ ] Objaverse pretrained 체크포인트 확보
[ ] 혼합 데이터셋 구성 (Objaverse + Mouse 합성)
[ ] Fine-tuning 실험
[ ] 평가 및 비교
```

---

## 8. 예상 결과물

### 8.1 디렉토리 구조

```
FaceLift/
├── scripts/
│   └── synthesis/
│       ├── sparse_3dgs.py
│       ├── sparse_2dgs.py
│       ├── render_novel_views.py
│       ├── create_dataset.py
│       ├── batch_process.py
│       └── config.yaml
├── data_mouse_synthetic/
│   ├── scene_0000/
│   │   ├── images/
│   │   │   ├── cam_000.png
│   │   │   └── ... (32개)
│   │   └── opencv_cameras.json
│   ├── scene_0001/
│   └── ... (~1,600 scenes)
└── configs/
    └── mouse_gslrm_synthetic.yaml
```

### 8.2 최종 학습 설정

```yaml
# configs/mouse_gslrm_synthetic.yaml
training:
  dataset:
    type: "MixedDataset"
    datasets:
      - name: "objaverse"
        path: "data/objaverse_80k_train.txt"
        weight: 0.7  # 70% Objaverse (3D prior 유지)
      - name: "mouse_synthetic"
        path: "data_mouse_synthetic/train.txt"
        weight: 0.3  # 30% Mouse 합성 데이터

    num_views: 8
    num_input_views: 4
```

---

## 9. 참고 자료

### 라이브러리
- [gsplat](https://github.com/nerfstudio-project/gsplat) - CUDA 가속 3DGS
- [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) - 2DGS 공식 구현
- [surfel-gs](https://github.com/turandai/gaussian_surfels) - Surface-aligned Gaussian

### 논문
- [3D Gaussian Splatting](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/)
- [2D Gaussian Splatting](https://surfsplatting.github.io/)
- [ViewNeTI (ECCV 2024)](https://github.com/jmhb0/view_neti)
- [4DiM](https://4d-diffusion.github.io/)

### 이전 보고서
- `251217_research_gslrm_3d_learning_analysis.md`
- `251215_research_stage2_3d_reconstruction.md`
