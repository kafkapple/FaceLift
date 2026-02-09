# Turntable Visualization Guide

Turntable 영상 생성 및 검증 가이드.

---

## Quick Test (Inference + TurntableRenderer)

체크포인트에서 모델을 로드하여 시각화만 실행 (training resume 없음):

```bash
cd /home/joon/dev/FaceLift
nohup bash scripts/verify_turntable.sh 6 > ./logs/verify_turntable.log 2>&1 &
tail -f ./logs/verify_turntable.log
```

Custom checkpoint:
```bash
bash scripts/verify_turntable.sh 6 /path/to/checkpoint.pt
```

### 테스트 내용

| Step | 경로 | 검증 대상 |
|------|------|-----------|
| 1/2 | render_from_checkpoint.py | Inference path (orbit only) |
| 2/2 | TurntableRenderer.render_all() | Train/Val path (orbit + view_traj + grid) |

## 개별 실행

### Inference (standalone)

```bash
CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000009200.pt \
    --config configs/base/gslrm_mouse.yaml \
    --data_path /home/joon/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir outputs/verify_turntable/inference \
    --mode turntable --num_samples 1
```

### Training (vis_every=100 간격으로 turntable 자동 생성)

```bash
CUDA_VISIBLE_DEVICES=6 /home/joon/anaconda3/envs/facelift/bin/torchrun \
    --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    --set training.schedule.max_fwdbwd_passes 100
```

## 결과 확인

```bash
# Quick test 결과
ls outputs/verify_turntable/inference/
ls outputs/verify_turntable/renderer/

# Mac으로 복사
scp -r gpu03:~/dev/FaceLift/outputs/verify_turntable/ .
```

## Output 파일 체계

### A. Single-Frame (Train/Val/Inference)

| Filename | 내용 |
|----------|------|
| turntable_orbit_{uid}.mp4 | 360 synthetic orbit (physical CCW) |
| turntable_orbit_with_input_{uid}.mp4 | orbit + labeled input strip |
| turntable_view_with_input_{uid}.mp4 | 6 cam trajectory + hold(15f) + input strip |
| turntable_{uid}.jpg | 6x6 grid image |
| turntable_6view_{uid}.mp4 | 2x3 multiview grid (GT top + Pred bottom) |

### B. Temporal (배치 추론 전용)

| Filename | 내용 |
|----------|------|
| time_fixed.mp4 | 고정 뷰, 시간 변화 |
| time_rotating.mp4 | 시간 + 회전 동시 변화 |

## Config 설정

`configs/base/gslrm_mouse.yaml`:

```yaml
visualization:
  turntable:
    smooth_trajectory: true   # CubicSpline + RotationSpline
    hold_frames: 15           # 각 카메라에서 1.5초 정지 (10fps)
    trajectory_fps: 10        # View trajectory FPS
    orbit_fps: 15             # Orbit FPS
    orbit_views: 120          # Orbit 프레임 수
```

## Rotation Direction

| 함수 | 좌표계 | 기본 동작 |
|------|--------|-----------|
| get_turntable_cameras | cos->x, sin->y (from +X) | clockwise=True (physical CCW) |
| compute_camera_order | atan2(x,y) (from +Y) | Physical CCW order |

rotation_direction: "ccw" (default) -> 모든 영상이 physical CCW (위에서 반시계) 방향.

---

*Created: 2026-02-09 | Updated: 2026-02-10*
