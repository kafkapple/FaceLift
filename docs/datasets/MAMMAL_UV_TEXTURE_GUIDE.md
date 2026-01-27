# MAMMAL UV Texture & Mesh Correspondence Guide

> **Created**: 2026-01-28
> **Purpose**: MAMMAL fitting mesh와 UV texture map의 관계, 텍스처 렌더링 방법

---

## 1. OBJ File Format 기초

### 1.1 OBJ 구성 요소

```
v  x y z          # vertex position (3D 좌표)
vt u v            # texture coordinate (UV, 0~1 범위)
vn nx ny nz       # vertex normal
f  v1/vt1/vn1 v2/vt2/vn2 v3/vt3/vn3   # face (삼각형)
```

### 1.2 Face Format 변형

| Format | 의미 | 예시 |
|--------|------|------|
| `f 1 2 3` | vertex만 | UV 없음 (fitting OBJ) |
| `f 1/4 2/5 3/6` | vertex/UV | 텍스처 매핑 가능 |
| `f 1/4/7 2/5/8 3/6/9` | vertex/UV/normal | 완전한 형태 |

---

## 2. MAMMAL Mesh Topology

### 2.1 Mesh 사양

| 항목 | 값 | 비고 |
|------|-----|------|
| **Vertices** | 14,522 | 모든 프레임 동일 |
| **Faces** | 28,800 | 삼각형, 모든 프레임 동일 |
| **UV coords** | 15,399 | textured export에만 존재 |
| **좌표계** | MAMMAL (-Y up) | X=body length, Y=height, Z=width |
| **단위** | mm | 원점에서 ~99mm 오프셋 |

### 2.2 핵심 발견: Topology 불변성

```
┌──────────────────────────────────────────────────────────────┐
│  MAMMAL은 parametric model (SMAL 기반)                       │
│  → 모든 프레임이 동일한 vertex 개수 + face topology          │
│  → vertex INDEX가 항상 같은 해부학적 위치에 대응              │
│  → vertex POSITION만 pose에 따라 변화                        │
└──────────────────────────────────────────────────────────────┘
```

**검증 결과:**

| 비교 | Vertex count | Face count | Face topology |
|------|-------------|------------|---------------|
| frame_000 vs frame_500 | 14522 = 14522 | 28800 = 28800 | **IDENTICAL** |
| frame_000 vs textured_export | 14522 = 14522 | 28800 = 28800 | **IDENTICAL** |

### 2.3 UV 좌표와 Vertex의 관계

UV 좌표 수(15,399) > Vertex 수(14,522)인 이유:

```
UV "seam" vertices: mesh를 2D로 펼칠 때 경계선에서
같은 3D vertex가 UV 공간에서 2개 이상의 좌표를 가짐

예: vertex #100 (코 부분)
   → UV seam에 위치
   → UV coord #200 (왼쪽 UV island)
   → UV coord #350 (오른쪽 UV island)
```

**통계:**

| 항목 | 수 |
|------|-----|
| UV 1개인 vertex | 13,665 (94.1%) |
| UV 2개+ (seam) | 857 (5.9%) |
| 총 UV 좌표 | 15,399 |

UV 범위: U=[0.018, 0.978], V=[0.018, 0.988] → 거의 전체 텍스처 활용

---

## 3. 파일 위치 및 구조

### 3.1 Fitting Results (UV 없음)

```
MAMMAL_mouse/results/fitting/<run_name>/
├── obj/
│   ├── step_2_frame_000000.obj   # V=14522, F=28800, VT=0
│   ├── step_2_frame_000005.obj   # 동일 topology, 다른 pose
│   └── ... (2382 files)
├── params/                        # Pose parameters (pkl)
└── render/                        # Debug renders
```

**Face format**: `f 6 21 5` (vertex index만, UV 없음)

### 3.2 Textured Export (UV 포함)

```
MAMMAL_mouse/exports/
├── mouse_frame0_textured.obj      # V=14522, VT=15399, F=28800 ★
└── sequence/
    ├── mouse_frame0.obj           # UV 없는 export
    └── ...
```

**Face format**: `f 6/6924 21/13982 5/7478` (vertex/UV index)

### 3.3 Texture Map

```
MAMMAL_mouse/results/sweep/run_wild-sweep-9/
├── texture_final.png   # 512x512 RGB - UV texture map ★
├── texture.pt          # PyTorch tensor [3, 512, 512]
├── uv_mask.png         # 512x512 grayscale - UV 유효 영역
├── confidence.png      # Per-vertex 텍스처 신뢰도
├── render_front.png    # 검증용 렌더링
├── render_side.png
└── render_diagonal.png
```

### 3.4 Template Meshes (다른 해상도)

```
MAMMAL_mouse/mouse_model/
├── mouse_reduced_face_1800.obj   # V=1942, F=1800 (low-res)
├── mouse_reduced_face_3600.obj   # V=1942, F=3600
└── mouse_reduced_face_7200.obj   # V=3761, F=7200
```

⚠️ Template mesh는 fitting mesh와 **다른 해상도** → UV 직접 호환 불가

---

## 4. UV Transplant 방법

### 4.1 원리

```
textured_export (frame 0):
  v 71.15  29.09  9.10    ← vertex position (frame 0 pose)
  vt 0.52  0.38           ← UV coordinate (topology 기반, pose 무관)
  f 6/6924 21/13982 5/7478 ← vertex/UV mapping

fitting_frame_500 (frame 500):
  v 85.23  31.45  12.67   ← 다른 position (frame 500 pose)
  (vt 없음)
  f 6 21 5                ← 같은 face topology!

UV transplant:
  → textured_export의 vt 줄과 face의 /vt_idx 를
  → fitting OBJ에 그대로 복사
  → vertex position은 그대로 유지
```

### 4.2 왜 가능한가?

UV 좌표는 **mesh topology** (어떤 vertex가 어떤 face에 속하는지)에 의존하며,
**vertex position** (3D 좌표)과는 무관합니다.

```
Analogy: 옷의 재단 패턴(UV)은 옷의 디자인(topology)에 의존하지,
        입는 사람의 체형(position/pose)에는 의존하지 않음.
        같은 디자인의 옷은 다른 사람에게도 동일한 패턴으로 입힐 수 있음.
```

수학적으로:
- UV 좌표 $\mathbf{t}_i = (u_i, v_i)$는 vertex index $i$에 바인딩
- Face $f = (v_a, v_b, v_c)$의 UV는 $(t_a, t_b, t_c)$
- Vertex position $\mathbf{p}_i$가 변해도 $(v_a, v_b, v_c)$ 관계 불변
- 따라서 UV mapping 불변

### 4.3 Transplant 스크립트

```python
# transplant_uv.py
def transplant_uv(fitting_obj_path, textured_obj_path, output_path):
    """
    fitting OBJ의 vertex positions + textured OBJ의 UV/face format 결합.
    
    Args:
        fitting_obj_path: UV 없는 fitting OBJ (임의 프레임)
        textured_obj_path: UV 포함된 textured OBJ (참조용)
        output_path: UV가 추가된 출력 OBJ
    """
    # 1. fitting OBJ에서 vertex positions 추출
    vertices = []
    with open(fitting_obj_path) as f:
        for line in f:
            if line.startswith('v '):
                vertices.append(line.strip())
    
    # 2. textured OBJ에서 UV coords와 faces 추출
    uv_lines = []     # vt lines
    face_lines = []   # f lines (with v/vt format)
    with open(textured_obj_path) as f:
        for line in f:
            if line.startswith('vt '):
                uv_lines.append(line.strip())
            elif line.startswith('f '):
                face_lines.append(line.strip())
    
    # 3. 결합하여 출력
    with open(output_path, 'w') as f:
        f.write("# UV-transplanted MAMMAL mesh\n")
        for v in vertices:
            f.write(v + "\n")
        for vt in uv_lines:
            f.write(vt + "\n")
        for face in face_lines:
            f.write(face + "\n")
```

### 4.4 Batch Transplant (전체 프레임)

```python
from pathlib import Path

fitting_dir = Path("results/fitting/<run>/obj/")
textured_ref = "exports/mouse_frame0_textured.obj"
output_dir = Path("exports/textured_sequence/")

for obj_path in sorted(fitting_dir.glob("step_2_frame_*.obj")):
    out_path = output_dir / obj_path.name
    transplant_uv(str(obj_path), textured_ref, str(out_path))
```

---

## 5. Blender 렌더링에서의 사용

### 5.1 단일 프레임 (텍스처 렌더링 테스트)

```bash
blender --background --python render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /path/to/output \
    --mesh /home/joon/dev/MAMMAL_mouse/exports/mouse_frame0_textured.obj \
    --texture /home/joon/dev/MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png \
    --num_samples 1
```

### 5.2 전체 시퀀스 (UV transplant 후)

```bash
# 1. UV transplant (전 프레임)
python transplant_uv_batch.py \
    --fitting_dir results/fitting/<run>/obj/ \
    --textured_ref exports/mouse_frame0_textured.obj \
    --output_dir exports/textured_sequence/

# 2. 렌더링 (textured sequence 사용)
blender --background --python render_mammal_32view_v2.py -- \
    --experiment MAMMAL_CENTER \
    --output_dir /path/to/output \
    --mammal_results exports/textured_sequence/ \
    --texture /path/to/texture_final.png \
    --num_samples 100
```

### 5.3 Blender Material 설정

UV 텍스처가 있는 OBJ를 로드할 때:

```python
# Blender Python
mat = bpy.data.materials.new("MouseTextured")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]

# Image Texture node → UV에서 자동 매핑
tex = mat.node_tree.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load("texture_final.png")
mat.node_tree.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])

# UV Map은 OBJ 로드 시 자동 생성됨
# Blender가 vt 좌표를 UV Map으로 변환
```

---

## 6. 텍스처 품질 참고

### 6.1 texture.pt vs texture_final.png

| 파일 | 형식 | 해상도 | 용도 |
|------|------|--------|------|
| `texture.pt` | torch [3,512,512] float32 | 512x512 | 프로그래밍용 |
| `texture_final.png` | PNG RGB 8bit | 512x512 | Blender 렌더링용 |

### 6.2 Confidence Map

`confidence.png`: per-vertex 텍스처 추출 신뢰도
- 밝은 영역 = 카메라에서 잘 보인 부분 (텍스처 정확)
- 어두운 영역 = 가려진 부분 (텍스처 추정/보간)

### 6.3 UV Mask

`uv_mask.png`: UV 공간에서 mesh가 차지하는 영역
- 흰색 = mesh surface (텍스처 유효)
- 검정 = 빈 공간 (padding)

---

## 7. Quick Reference

### UV Transplant 가능 조건

```
✅ 같은 vertex 수 (14522)
✅ 같은 face 수 (28800)
✅ 같은 face topology (vertex index 순서 동일)
✅ Parametric model → 모든 프레임 topology 불변
```

### 파일 경로 요약

| 용도 | 경로 |
|------|------|
| Fitting OBJ (UV 없음) | `MAMMAL_mouse/results/fitting/<run>/obj/step_2_frame_*.obj` |
| Textured OBJ (UV 참조) | `MAMMAL_mouse/exports/mouse_frame0_textured.obj` |
| Texture Map | `MAMMAL_mouse/results/sweep/run_wild-sweep-9/texture_final.png` |
| UV Mask | `MAMMAL_mouse/results/sweep/run_wild-sweep-9/uv_mask.png` |
| Confidence | `MAMMAL_mouse/results/sweep/run_wild-sweep-9/confidence.png` |

---

*MAMMAL UV Texture Guide | Created: 2026-01-28*
