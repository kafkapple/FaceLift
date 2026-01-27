# E3: Synthetic Rendering Guide

> **Created**: 2026-01-28
> **Purpose**: MAMMAL fitting mesh → Blender 렌더링 → GS-LRM inference 데이터 생성
> **Related**: `MAMMAL_UV_TEXTURE_GUIDE.md` (UV 이론), `M5_SERIES_SPEC.md` (E1 전처리)

---

## 1. Overview

E3는 **완벽한 카메라 파라미터**를 가진 synthetic 데이터로 GS-LRM 동작을 검증하는 실험입니다.

```
MAMMAL fitting OBJ → UV transplant → Blender 32-view render → GS-LRM inference
                     (optional)       (known cameras)          (reconstruction)
```

### 왜 필요한가?

| 문제 | 실제 데이터 (E1) | Synthetic (E3) |
|------|-----------------|----------------|
| 카메라 파라미터 | 추정값 (노이즈 있음) | **GT** (정확) |
| 이미지 품질 | 실제 촬영 | 렌더링 (깨끗) |
| 배경 | 복잡 | 단색/제거 가능 |
| 검증 목표 | 전처리 효과 측정 | **모델 자체 능력 확인** |

E3에서 GS-LRM이 실패하면 → 모델/학습 문제
E3에서 성공하고 E1에서 실패하면 → 전처리/카메라 문제

---

## 2. Pipeline

### 2.1 전체 흐름

```
Step 0: 준비 (fitting 결과 확인)
    │
Step 1: UV Transplant (텍스처 UV 이식) ← optional, 텍스처 렌더링 시
    │
Step 2: Blender Render (32-view orbit)
    │
Step 3: GS-LRM Inference (재구성)
    │
Step 4: 결과 분석 (turntable, mesh)
```

### 2.2 파일 위치

| 항목 | 경로 |
|------|------|
| Fitting OBJ | `/home/joon/dev/MAMMAL_mouse/results/fitting/<run>/obj/` |
| Textured OBJ (참조) | `/home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj` |
| Texture PNG | `/home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png` |
| Transplant script | `mouse_extensions/scripts/blender/transplant_uv.py` |
| Render script | `mouse_extensions/scripts/blender/render_mammal_32view_v2.py` |
| Blender binary | `/home/joon/blender-4.0.2-linux-x64/blender` |

---

## 3. Step 1: UV Transplant

MAMMAL fitting OBJ에는 UV 좌표가 없으므로, textured export에서 UV를 이식합니다.
(fur material로 렌더링할 경우 이 단계 생략 가능)

### 3.1 단일 프레임

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

python mouse_extensions/scripts/blender/transplant_uv.py \
    --fitting /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/step_2_frame_000000.obj \
    --reference /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --output /home/joon/data/synthetic/textured_obj/step_2_frame_000000.obj
```

### 3.2 다중 프레임 (Batch)

```bash
python mouse_extensions/scripts/blender/transplant_uv.py \
    --fitting-dir /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/ \
    --reference /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --output-dir /home/joon/data/synthetic/textured_obj/ \
    --max-frames 100 \
    --frame-step 24
```

| 옵션 | 설명 | 예시 |
|------|------|------|
| `--max-frames` | 최대 프레임 수 | 100 |
| `--frame-step` | N개마다 1개 선택 | 24 → 2382개 중 ~100개 |
| `--fitting-dir` | fitting OBJ 디렉토리 | `step_2_frame_*.obj` 자동 탐색 |

---

## 4. Step 2: Blender Rendering

### 4.1 단일 mesh 렌더링 (Quick test)

하나의 mesh를 32-view로 렌더링:

```bash
CUDA_VISIBLE_DEVICES=7 /home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/E3_test \
    --mesh /home/joon/data/synthetic/textured_obj/step_2_frame_000000.obj \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_views 32 \
    --num_samples 1
```

### 4.2 다중 프레임 렌더링 (Full dataset)

#### 방법 A: `--mesh` 반복 (같은 mesh를 N번)

같은 포즈를 여러 번 렌더링 (augmentation 테스트용):

```bash
/home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/E3_single_pose \
    --mesh /home/joon/data/synthetic/textured_obj/step_2_frame_000000.obj \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_samples 10
```

→ `sample_00000/` ~ `sample_00009/` 생성 (모두 같은 pose)

#### 방법 B: `--mammal_results` (다른 pose 순차 렌더링)

transplant된 OBJ 폴더를 `obj/` 하위 구조로 배치:

```bash
# 1. OBJ 폴더 구조 준비
mkdir -p /home/joon/data/synthetic/textured_mammal/obj
cp /home/joon/data/synthetic/textured_obj/step_2_frame_*.obj \
   /home/joon/data/synthetic/textured_mammal/obj/

# 2. 렌더링 (각 OBJ가 하나의 sample)
/home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/E3_multi_pose \
    --mammal_results /home/joon/data/synthetic/textured_mammal \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_samples 100
```

→ 각 sample이 **다른 포즈**의 생쥐 렌더링

#### 방법 C: Shell loop (가장 유연한 방법)

개별 OBJ를 하나씩 렌더링하여 최대 제어:

```bash
#!/bin/bash
# render_multi_frame.sh

BLENDER=/home/joon/blender-4.0.2-linux-x64/blender
SCRIPT=mouse_extensions/scripts/blender/render_mammal_32view_v2.py
TEXTURE=/home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png
OBJ_DIR=/home/joon/data/synthetic/textured_obj
OUTPUT_BASE=/home/joon/data/synthetic/E3_multi_pose

cd /home/joon/dev/FaceLift

IDX=0
for OBJ in $(ls ${OBJ_DIR}/step_2_frame_*.obj | sort | head -100); do
    SAMPLE_DIR=$(printf "sample_%05d" $IDX)
    OUTPUT_DIR="${OUTPUT_BASE}/${SAMPLE_DIR}"

    echo "[${IDX}] Rendering $(basename ${OBJ}) → ${SAMPLE_DIR}"

    ${BLENDER} --background \
        --python ${SCRIPT} -- \
        --experiment MAMMAL_CENTER \
        --output_dir ${OUTPUT_DIR} \
        --mesh ${OBJ} \
        --texture ${TEXTURE} \
        --num_views 32 \
        --num_samples 1

    # sample_00000/sample_00000/ 구조 → sample_XXXXX/ 로 flatten
    if [ -d "${OUTPUT_DIR}/sample_00000" ]; then
        mv ${OUTPUT_DIR}/sample_00000/* ${OUTPUT_DIR}/
        rmdir ${OUTPUT_DIR}/sample_00000
    fi

    IDX=$((IDX + 1))
done

echo "Done: ${IDX} samples rendered"

# data_train.txt 생성
find ${OUTPUT_BASE} -name "opencv_cameras.json" -exec dirname {} \; | sort > ${OUTPUT_BASE}/data_train.txt
echo "data_train.txt: $(wc -l < ${OUTPUT_BASE}/data_train.txt) entries"
```

**사용법:**

```bash
chmod +x render_multi_frame.sh
bash render_multi_frame.sh
```

### 4.3 Material 옵션

| 옵션 | 설명 | 언제 사용 |
|------|------|-----------|
| `--texture <path>` | UV 텍스처 이미지 적용 | UV transplant 후 |
| `--material fur` | Procedural fur (noise-based) | UV 없는 OBJ, 기본값 |
| `--material simple` | 단색 brown | 빠른 테스트 |

`--texture` 지정 시 `--material` 무시됨 (텍스처 우선).

### 4.4 Render 파라미터

| 파라미터 | 값 | 비고 |
|----------|-----|------|
| Image size | 512 x 512 | GS-LRM 입력 크기 |
| fx, fy | 548.99 | Pretrained 호환 |
| cx, cy | 256 | Image center |
| Distance | 2.7 | 정규화 기준 |
| Elevation | 20° | GS-LRM 학습 설정 |
| Azimuth spacing | 11.25° (32-view) | 360°/32 |
| Output format | RGBA PNG + opencv_cameras.json | |

---

## 5. Step 3: GS-LRM Inference

### 5.1 단일 sample

```bash
cd /home/joon/dev/FaceLift
source ~/anaconda3/etc/profile.d/conda.sh && conda activate facelift

python inference_mouse.py \
    --sample_dir /home/joon/data/synthetic/E3_multi_pose/sample_00000 \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3_synthetic/ \
    --save_turntable --save_mesh
```

### 5.2 전체 dataset (Batch)

```bash
python inference_mouse.py \
    --data_dir /home/joon/data/synthetic/E3_multi_pose/ \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3_synthetic_batch/ \
    --save_turntable --save_mesh
```

---

## 6. Output Structure

### 6.1 렌더링 출력

```
E3_multi_pose/
├── sample_00000/          # frame 000000
│   ├── images/
│   │   ├── cam_000.png    # 512x512 RGBA
│   │   ├── cam_001.png
│   │   └── ... (32 views)
│   └── opencv_cameras.json
├── sample_00001/          # frame 000024
│   └── ...
├── ...
└── data_train.txt         # sample 경로 목록
```

### 6.2 opencv_cameras.json 구조

```json
{
  "frames": [
    {
      "fx": 548.99, "fy": 548.99,
      "cx": 256.0, "cy": 256.0,
      "w": 512, "h": 512,
      "w2c": [[...4x4...]],
      "c2w": [[...4x4...]],
      "file_path": "images/cam_000.png"
    },
    ...
  ]
}
```

### 6.3 GS-LRM 출력

```
outputs/e3_synthetic/
├── turntable_sample_00000.mp4   # 360° turntable video
├── mesh_sample_00000.ply        # 3D Gaussian mesh
└── ...
```

---

## 7. Experiments

### 7.1 E3-A: 단일 포즈 (Quick validation)

**목적**: GS-LRM이 완벽한 카메라에서 mouse mesh를 재구성할 수 있는지 확인

```bash
# 1. 단일 프레임 렌더링
/home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/E3A_single \
    --mesh /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/step_2_frame_000000.obj \
    --material fur \
    --num_views 32 --num_samples 1

# 2. Inference
python inference_mouse.py \
    --sample_dir /home/joon/data/synthetic/E3A_single/sample_00000 \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3a/ --save_turntable --save_mesh
```

**UV transplant 불필요** — fur material 사용.

### 7.2 E3-B: 텍스처 렌더링 (Realistic)

**목적**: 실제 텍스처로 렌더링하여 더 현실적인 입력 생성

```bash
# 1. UV transplant
python mouse_extensions/scripts/blender/transplant_uv.py \
    --fitting-dir /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/ \
    --reference /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --output-dir /home/joon/data/synthetic/textured_obj/ \
    --max-frames 10 --frame-step 240

# 2. 렌더링 (방법 B)
mkdir -p /home/joon/data/synthetic/textured_mammal/obj
cp /home/joon/data/synthetic/textured_obj/step_2_frame_*.obj \
   /home/joon/data/synthetic/textured_mammal/obj/

/home/joon/blender-4.0.2-linux-x64/blender --background \
    --python mouse_extensions/scripts/blender/render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /home/joon/data/synthetic/E3B_textured \
    --mammal_results /home/joon/data/synthetic/textured_mammal \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_samples 10

# 3. Inference
python inference_mouse.py \
    --data_dir /home/joon/data/synthetic/E3B_textured/ \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3b/ --save_turntable --save_mesh
```

### 7.3 E3-C: 대규모 다중 포즈 (Full evaluation)

**목적**: 100개 포즈로 통계적 평가

```bash
# 1. UV transplant (100 frames)
python mouse_extensions/scripts/blender/transplant_uv.py \
    --fitting-dir /home/joon/dev/MAMMAL_mouse/results/fitting/markerless_mouse_1_nerf_v012345_kp22_20260126_025249/obj/ \
    --reference /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --output-dir /home/joon/data/synthetic/textured_obj_100/ \
    --max-frames 100 --frame-step 24

# 2. 렌더링 (Shell loop — 방법 C 권장)
bash mouse_extensions/scripts/blender/render_multi_frame.sh

# 3. Batch inference
python inference_mouse.py \
    --data_dir /home/joon/data/synthetic/E3C_full/ \
    --checkpoint checkpoints/gslrm/D8_E0_paper_original/ \
    --config configs/base/gslrm_mouse.yaml \
    --output_dir outputs/e3c/ --save_turntable --save_mesh
```

---

## 8. Troubleshooting

| 문제 | 원인 | 해결 |
|------|------|------|
| Blender segfault | GPU 드라이버 | `CUDA_VISIBLE_DEVICES=7` 변경 또는 CPU 렌더링 |
| 텍스처 안 보임 | UV 없는 OBJ | Step 1 (UV transplant) 실행 확인 |
| 검은 이미지 | 조명 부족 | `--material fur`는 조명 자동 설정 |
| opencv_cameras.json 없음 | 렌더링 실패 | Blender 로그 확인 |
| Inference OOM | GPU 메모리 | `CUDA_VISIBLE_DEVICES` 큰 GPU 사용 |

---

## 9. References

| 문서 | 내용 |
|------|------|
| `MAMMAL_UV_TEXTURE_GUIDE.md` | UV 좌표 이론, topology, transplant 원리 |
| `M5_SERIES_SPEC.md` | E1 전처리 (re-centering + uniform norm) |
| `coordinate_systems_reference.md` | MAMMAL 좌표계 (-Y up → Blender Z-up) |

---

*E3 Synthetic Rendering Guide | Created: 2026-01-28*
