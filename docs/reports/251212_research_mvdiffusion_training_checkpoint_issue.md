---
date: 2025-12-12
context_name: "2_Research"
tags: [ai-assisted, mvdiffusion, gslrm, checkpoint, training, mouse-facelift]
project: mouse-facelift
status: completed
generator: ai-assisted
generator_tool: claude-code
---

# MVDiffusion 학습 체크포인트 이슈 및 뷰 학습 분석

## 목적
1. MVDiffusion fine-tuning 체크포인트 관리 중 발생한 이슈 분석
2. Multi-view diffusion 모델의 뷰 구성 학습 메커니즘 이해
3. 카메라 포즈 요구사항 정리

---

## 주요 발견사항

### 1. 체크포인트 경로 버그 (Critical)

**문제**: `train_diffusion.py`의 `save_model_hook`에서 경로 중복 발생

```python
# 버그 코드
model.save_pretrained(os.path.join(cfg.checkpoint_prefix, output_dir, "unet"))
# output_dir이 이미 전체 경로를 포함하는데 checkpoint_prefix를 다시 추가
# 결과: checkpoints/checkpoints/experiments/.../unet (중복)
```

**수정**:
```python
# 수정된 코드
model.save_pretrained(os.path.join(output_dir, "unet"))
# output_dir은 accelerator가 전달하는 전체 경로
```

**교훈**:
- Accelerator의 hook 함수에서 전달되는 경로는 이미 완전한 경로
- 설정값을 중복 적용하지 않도록 주의

### 2. 삭제된 파일 복구 불가

| 항목 | 상태 |
|-----|------|
| 파일시스템 | NFS (10.10.1.1:/data/joon) |
| 휴지통 | 없음 (rm -rf 직접 삭제) |
| 복구 도구 | extundelete/testdisk 미설치 |
| **결론** | 복구 불가능, 재학습 필요 |

---

## MVDiffusion 뷰 학습 메커니즘

### 학습된 뷰 구성에 최적화

MVDiffusion은 **고정된 뷰 구성**으로 학습됩니다:

```python
# inference_mouse.py:86-87
MVDIFFUSION_AZIMUTHS = [0, 60, 120, 180, 240, 300]  # 6개 뷰, 60° 간격
MVDIFFUSION_ELEVATION = 0  # 동일 고도
```

**특성**:
- 학습된 각도 조합에 최적화
- 뷰 간 interpolation 불가 (discrete diffusion)
- 학습 분포 외 뷰는 품질 저하

### 다른 뷰 구성으로 학습

**가능합니다.** 설정 변경 필요:

```yaml
# configs/mouse_mvdiffusion.yaml
n_views: 8  # 뷰 개수 변경

# UNet config
unet_from_pretrained_kwargs:
  num_views: 8  # 반드시 일치

# Prompt embeddings (새로 생성 필요)
prompt_embed_path: "mvdiffusion/data/fixed_prompt_embeds_8view/clr_embeds.pt"
```

**체크리스트**:
- [ ] 데이터셋을 새 뷰 구성으로 준비
- [ ] `n_views` 설정 변경
- [ ] `unet_from_pretrained_kwargs.num_views` 변경
- [ ] Prompt embeddings 새로 생성
- [ ] 처음부터 학습 (pretrained에서 fine-tune)

---

## 카메라 포즈 요구사항

### MVDiffusion vs GSLRM

| 모델 | 포즈 요구사항 | 설명 |
|------|-------------|------|
| **MVDiffusion** | 상대적 뷰 방향 | 고정 구성 (azimuth 기반) |
| **GSLRM** | 정확한 c2w 행렬 | 3D 복원에 필수 |

### 데이터 유형별 포즈 획득

#### 합성 데이터 (Blender 등)
```python
# 렌더링 시 사용한 카메라 파라미터 그대로 사용
camera_data = {
    "w2c": np.array(camera.matrix_world).tolist(),  # world-to-camera
    "fx": focal_length_x,
    "fy": focal_length_y,
    "cx": image_width / 2,
    "cy": image_height / 2
}
```

#### 실제 데이터
- **COLMAP**: Structure-from-Motion으로 포즈 추정
- **ARKit/ARCore**: 디바이스에서 직접 획득
- **Known setup**: 고정 카메라 리그 사용

---

## 추론 파이프라인

### 전체 흐름

```
[Single Image]
    ↓ MVDiffusion (fine-tuned)
[6-View Images] (고정 각도: 0°, 60°, 120°, 180°, 240°, 300°)
    ↓ GSLRM (fine-tuned)
[3D Gaussian Splats]
    ↓ Rendering
[Novel Views] (임의 각도/거리 가능)
```

### 실행 명령어

```bash
# 단일 이미지 → 6뷰 → 3D GS
python inference_mouse.py \
    --input_image examples/mouse.png \
    --mvdiffusion_checkpoint checkpoints/mvdiffusion/mouse/checkpoint-XXXXX \
    --checkpoint checkpoints/gslrm/mouse_finetune/ \
    --output_dir outputs/
```

### 임의 뷰 렌더링

MVDiffusion의 6개 뷰 외 다른 각도가 필요하면:
1. GSLRM으로 3D Gaussian Splats 생성
2. 3D GS에서 임의 카메라로 렌더링 (turntable 등)

```python
# 3D GS에서 turntable 렌더링
from gslrm.model.gaussians_renderer import render_turntable
vis_image = render_turntable(
    gaussians,
    rendering_resolution=512,
    num_views=120  # 3° 간격 360° 회전
)
```

---

## 체크포인트 현황

### MVDiffusion
| 경로 | 상태 |
|-----|------|
| `checkpoints/mvdiffusion/pipeckpts/` | ✅ Pretrained (Human) |
| `checkpoints/mvdiffusion/mouse/checkpoint-12000/` | ❌ 손상 (UNet 삭제됨) |

### GSLRM
| 경로 | 상태 |
|-----|------|
| `checkpoints/gslrm/ckpt_0000000000021125.pt` | ✅ Pretrained (Human) |
| `checkpoints/gslrm/mouse_finetune/ckpt_*` | ✅ Fine-tuned (Mouse) |

---

## 다음 단계

1. **MVDiffusion 재학습**
   ```bash
   cd ~/FaceLift
   python train_diffusion.py --config configs/mouse_mvdiffusion.yaml
   ```

2. **학습 완료 후 추론 테스트**
   - 단일 이미지 → 6뷰 생성
   - 6뷰 → GSLRM → 3D 재구성

3. **다른 뷰 구성 실험** (선택사항)
   - 4뷰, 8뷰 등 다양한 구성 테스트
   - 비균등 각도 배치 실험

---

## 참고

- FaceLift Paper: Multi-view diffusion + feed-forward 3D reconstruction
- MVDiffusion: Fixed view configuration, direction-conditioned
- GSLRM: Gaussian Splat Large Reconstruction Model
