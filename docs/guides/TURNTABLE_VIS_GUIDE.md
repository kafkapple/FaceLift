# Turntable Visualization Guide

Turntable 영상 생성 및 검증 가이드.

---

## Quick Test (Train + Val + Inference)

```bash
# 백그라운드 실행 (GPU 6)
cd /home/joon/dev/FaceLift
nohup bash scripts/verify_turntable.sh 6 > /tmp/verify_turntable.log 2>&1 &

# 로그 확인
tail -f /tmp/verify_turntable.log
```

## 개별 실행

### Training (vis_every=100 간격으로 turntable 자동 생성)

```bash
CUDA_VISIBLE_DEVICES=6 torchrun --standalone --nproc_per_node=1 \
    train_gslrm.py -d M5t2 -e E0_1_facelift \
    --set training.schedule.max_fwdbwd_passes 9302
```

결과: `/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/iter_XXXXX/`

### Inference (standalone)

```bash
CUDA_VISIBLE_DEVICES=6 python mouse_extensions/scripts/inference/render_from_checkpoint.py \
    --checkpoint /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/ckpt_0000000000009200.pt \
    --config configs/base/gslrm_mouse.yaml \
    --data_path ~/data/preprocessed/FaceLift_mouse/M5/data_mouse_t2_val.txt \
    --output_dir outputs/verify_turntable/inference \
    --mode turntable --num_samples 1
```

## 결과 확인

```bash
# Training + Validation
ls /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift/iter_*/turntable_*.mp4

# Inference
ls outputs/verify_turntable/inference/

# Mac으로 복사
scp gpu03:~/dev/FaceLift/outputs/verify_turntable/*.mp4 .
```

## Output 파일 체계

### A. Single-Frame (Train/Val/Inference)

| Filename | 내용 |
|----------|------|
| turntable_orbit_{uid}.mp4 | 360 synthetic orbit (physical CCW) |
| turntable_orbit_with_input_{uid}.mp4 | orbit + labeled input strip |
| turntable_view_with_input_{uid}.mp4 | 6 cam trajectory + hold(15f) + input strip |
| turntable_{uid}.jpg | 6x6 grid image |

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

*Created: 2026-02-09*
