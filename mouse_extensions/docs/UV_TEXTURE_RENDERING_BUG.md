# UV Texture Rendering Bug: "Skeleton" Distortion

> **Type**: Critical Bug Analysis & Postmortem
> **Date**: 2026-03-11
> **Status**: ✅ Resolved
> **Files Modified**: `mouse_extensions/scripts/eval/poc_mesh_gs_pairs.py` (`_load_textured_mesh`)

---

## 1. 증상 (Symptom)

텍스처 메시 렌더링에서 생쥐가 "말라 비틀어진" 해골 형태로 보임.
- 살이 움푹 파진 부분이 곳곳에 존재
- Flat mesh (텍스처 없음)는 정상 형태
- 모든 뷰, 모든 프레임에서 동일한 왜곡 발생

## 2. 근본 원인 (Root Cause)

### OBJ v/vt Face 구조와 trimesh 내부 재인덱싱의 불일치

**OBJ 파일 구조** (template OBJ):
```
v  x y z       ← 14,522 vertices (원본 geometry)
vt u v         ← 15,399 UV coords (UV seam에서 vertex 분리)
f  v1/vt1 v2/vt2 v3/vt3   ← 28,800 faces (v와 vt 각각 다른 인덱스)
```

**trimesh가 OBJ를 로드할 때**:
- 각 고유한 `(v_idx, vt_idx)` 쌍에 대해 새 vertex를 생성
- 14,522 → 15,399개로 확장 (UV seam 경계에서 분리)
- **자체적인 순서**로 재인덱싱 → faces도 이 새 인덱스를 참조

### 잘못된 구현 (v1)

```python
# ❌ Bug: 수동으로 faces_vt를 face indices로 사용
uv_to_v = np.zeros(len(t_uvs), dtype=np.int32)
for fv, ft in zip(t_faces_v, t_faces_vt):
    for vi, ti in zip(fv, ft):
        uv_to_v[ti] = vi  # UV index → original vertex index

expanded_verts = frame_verts[uv_to_v]  # [15399, 3]
mesh = trimesh.Trimesh(
    vertices=expanded_verts,
    faces=faces_vt,  # ❌ OBJ의 vt 인덱스를 face로 사용
)
```

**왜 틀린가**:
- `faces_vt`의 인덱스 순서 ≠ trimesh의 내부 face 인덱스 순서
- 검증 결과: `faces_vt[0] = [6923, 13981, 7477]` vs `tmesh.faces[0] = [10, 35, 8]`
- → vertex 위치가 완전히 뒤섞여 삼각형이 엉뚱한 점을 연결

### 올바른 구현 (v2)

```python
# ✅ Fix: trimesh의 native 로딩 + face 대응으로 매핑 구축
template_mesh = trimesh.load(TEXTURED_OBJ, process=False)  # 15,399 verts

# OBJ 원본 face vertex indices 파싱
faces_v_orig = parse_face_v_indices(TEXTURED_OBJ)  # [28800, 3]

# face 대응: tmesh.faces[i][j] (expanded) ↔ faces_v_orig[i][j] (original)
expanded_to_orig = np.full(15399, -1, dtype=np.int32)
for i in range(28800):
    for j in range(3):
        expanded_to_orig[tmesh.faces[i][j]] = faces_v_orig[i][j]

# Per-frame: vertex swap
mesh = template_mesh.copy()
mesh.vertices = frame_verts[expanded_to_orig]  # 정확한 매핑
```

## 3. 왜 이 실수가 발생했는가

| 원인 | 설명 |
|------|------|
| **OBJ 스펙 오해** | `f v/vt` 형식에서 vt 인덱스가 곧 expanded vertex 인덱스라고 가정 |
| **trimesh 블랙박스** | trimesh의 내부 재인덱싱 로직을 확인하지 않고 수동 구현 |
| **Frame 0 우연의 일치** | Frame 0에서 template ≈ frame mesh이므로 왜곡이 덜 눈에 띔 |
| **검증 부재** | flat mesh와 textured mesh의 bounding box만 비교 (동일) → 내부 구조 차이 미확인 |

### 교훈

1. **라이브러리의 native 기능을 우선 사용**: trimesh가 OBJ v/vt를 이미 처리하므로, 수동 구현 대신 trimesh의 결과를 활용
2. **Face-level 검증 필수**: vertex count와 bounding box가 같아도 face topology가 다르면 완전히 다른 mesh
3. **다중 프레임 검증**: Frame 0만 확인하면 우연의 일치에 속을 수 있음

## 4. 정량적 Before/After 검증 (2026-03-12)

Before/After 비교를 통해 버그 수정 효과를 정량적으로 검증.
**둘 다 텍스처가 적용된 상태**에서 비교 (buggy mapping vs fixed mapping).

### 비교 방법

- **BEFORE (buggy)**: `mesh.vertices = frame_verts[uv_to_v]` — UV 순서를 trimesh 순서 template에 적용 → 불일치
- **AFTER (fixed)**: `mesh.vertices = frame_verts[expanded_to_orig]` — face correspondence 기반 정확한 매핑
- **Reference**: Flat mesh (텍스처 없음, 동일 geometry) + GT RGB (alpha mask 기반)
- **Metrics**: `fair_comparison.py`와 동일한 mask 기준 (GT: alpha > 127, pred: white-BG < 0.98)

### 결과 요약 (10 frames × 3 cameras = 30 samples)

| Metric | BEFORE (buggy) | AFTER (fixed) | Delta |
|--------|:--------------:|:-------------:|:-----:|
| IoU(GT) | 0.665 | 0.754 | **+0.088** |
| IoU(flat mesh) | 0.653 | 1.000 | **+0.347** |
| PSNR_masked(GT) | 8.75 | 10.42 | **+1.67 dB** |
| FG pixel ratio | ~1.5-2x larger | compact | — |

### 정성적 차이

- **Buggy**: collapsed diamond/crystal 형태, 전체적으로 skeleton-like distortion
- **Fixed**: proper dark mouse, flat mesh와 동일한 silhouette (IoU=1.0)
- **Evidence**: `diagnostics/bugfix_before_after/` (30 comparison grids + JSON metrics)

### 검증 코드

```python
# Quick check: faces가 일치하는지 확인
manual_faces = faces_vt  # 수동 구성
trimesh_faces = tmesh.faces  # trimesh 로딩
assert np.array_equal(manual_faces, trimesh_faces), "Face mismatch!"

# Vertex 매핑 검증
assert (expanded_to_orig >= 0).all(), "Unmapped vertices exist!"
assert len(np.unique(expanded_to_orig)) == 14522, "Not all original verts covered!"
```

## 5. Related Documents

- ↑ [[COORDINATE_SYSTEMS]] — Issue Log §4 (textured mesh 해골 왜곡)
- ↔ [[DATASET_FRAME_INDEXING]] — §3.4 UV Texture Assets (v/vt 구조)
- ↔ `poc_mesh_gs_pairs.py:_load_textured_mesh()` — 수정된 구현

---

*FaceLift | UV Texture Rendering Bug Analysis | 2026-03-11*
