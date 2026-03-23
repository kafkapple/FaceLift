# MAMMAL Mesh Re-fitting Handoff

FaceLift Neural Texture PoC → MAMMAL 23 Bad Frames 재피팅 요청.

---

## 1. Background

FaceLift Neural Texture PoC에서 MAMMAL mesh OBJ를 사용해 per-frame texture 학습.
100프레임 중 **23프레임이 IoU < 0.7** (cam_003 projection 기준).
증상: 팔다리 뒤틀림, 꼬리 관통 등 articulation 오류.

**원인**: 기존 OBJ는 `fast` optim (step1_iters=50)으로 피팅 → iteration 부족.
**해결**: `accurate` optim (step1_iters=200)으로 23프레임만 재피팅.

## 2. Environment

```bash
ssh gpu03
cd /home/joon/dev/MAMMAL_mouse       # ⚠️ "MAMMAL" 아님
conda activate mouse                  # ⚠️ "mammal_stable" 아님 (미존재)
```

- GPU: `CUDA_VISIBLE_DEVICES=4` 또는 `7`
- Experiment config: `conf/experiment/accurate_6views.yaml` (6view + keypoints + optim=accurate)

## 3. Bad Frames (23개)

### MAMMAL Frame IDs

```
720  1320  1920  2040  2160  2760  3600
5160  5520  5880  6000  6120  6960  7200
8280  8400  9360  9480  9840  10080
10680  10800  11880
```

### Frame ID Mapping

| ID 체계 | 설명 | 변환 | 예시 |
|---------|------|------|------|
| **M5 frame** | FaceLift 데이터셋 프레임 ID | — | 144 |
| **MAMMAL frame** | MAMMAL internal ID | M5 × 5 | 720 |
| **OBJ 파일명** | `step_2_frame_{MAMMAL:06d}.obj` | — | `step_2_frame_000720.obj` |

**Source**: `/home/joon/dev/FaceLift/outputs/neural_texture/good_frames.npy` (77 good M5 frame IDs)

## 4. Execution

MAMMAL fitter는 `np.arange(start, end, interval)`만 지원 — **frame list 직접 지정 불가**.
따라서 per-frame 루프로 실행:

```bash
cd /home/joon/dev/MAMMAL_mouse
conda activate mouse

BAD_FRAMES=(720 1320 1920 2040 2160 2760 3600 5160 5520 5880 \
            6000 6120 6960 7200 8280 8400 9360 9480 9840 10080 \
            10680 10800 11880)

for F in "${BAD_FRAMES[@]}"; do
    END=$((F + 1))
    echo "=== Re-fitting frame $F ==="
    CUDA_VISIBLE_DEVICES=4 python fitter_articulation.py \
        +experiment=accurate_6views \
        fitter.start_frame=$F \
        fitter.end_frame=$END \
        fitter.interval=1 \
        --output_dir results/fitting/refit_accurate_23
done
```

### Important

- **별도 output_dir 사용** (`refit_accurate_23/`) — 기존 결과 보존
- 예상 소요: 23 프레임 × (step0=20 + step1=200 + step2=50) ≈ **30분~1시간**
- 전체 100프레임 재피팅 금지 — 77 good frames 품질 저하 위험

## 5. Post-fit Verification (필수)

재피팅 후 반드시 IoU 검증:

```bash
# 검증 스크립트 (FaceLift repo)
cd /home/joon/dev/FaceLift
python mouse_extensions/scripts/neural_texture/verify_mesh_quality.py
```

이 스크립트는 cam_003 기준으로 mesh를 projection하여 GT mask와 IoU를 계산.
**기준: IoU >= 0.7** 통과 시 해당 프레임 사용 가능.

검증 시 OBJ 경로를 새 결과로 변경 필요:
- 스크립트 내 `obj_dir` 변수를 `results/fitting/refit_accurate_23/obj/`로 수정

## 6. Result Copy (검증 통과 후)

```bash
cd /home/joon/dev/MAMMAL_mouse

BAD_FRAMES=(720 1320 1920 2040 2160 2760 3600 5160 5520 5880 \
            6000 6120 6960 7200 8280 8400 9360 9480 9840 10080 \
            10680 10800 11880)

for F in "${BAD_FRAMES[@]}"; do
    SRC="results/fitting/refit_accurate_23/obj/step_2_frame_$(printf '%06d' $F).obj"
    DST="/home/joon/data/synthetic/textured_obj/step_2_frame_$(printf '%06d' $F).obj"
    if [ -f "$SRC" ]; then
        cp "$SRC" "$DST"
        echo "Copied frame $F"
    else
        echo "WARNING: Missing $SRC"
    fi
done
```

### File Locations

| Item | Path |
|------|------|
| MAMMAL repo | `/home/joon/dev/MAMMAL_mouse/` |
| OBJ source (current) | `/home/joon/data/synthetic/textured_obj/` (100 files) |
| Re-fit output | `results/fitting/refit_accurate_23/obj/` |
| Good frames list | `/home/joon/dev/FaceLift/outputs/neural_texture/good_frames.npy` |
| Verify script | `/home/joon/dev/FaceLift/mouse_extensions/scripts/neural_texture/verify_mesh_quality.py` |
| Neural texture code | `/home/joon/dev/FaceLift/mouse_extensions/scripts/neural_texture/` |
| Pseudo-GT (77 good) | `/home/joon/dev/FaceLift/outputs/neural_texture/pseudo_gt_clean/` |

## 7. After Completion

Re-fitting + 검증 완료 후, FaceLift에서:
1. 23프레임 neural texture 재생성 (`perframe_100.py`)
2. nvdiffrast 3-stage (tex→geom→tex) 100프레임 overnight
3. Novel view pseudo-GT downstream 평가

---

*Created: 2026-03-23 | FaceLift S33*
