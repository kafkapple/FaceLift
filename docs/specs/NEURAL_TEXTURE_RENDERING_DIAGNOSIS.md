# Neural Texture Rendering Quality Diagnosis

> **Version**: v1.0 | **Created**: 2026-03-23 | **Status**: ACTIVE
> **Navigation**: [← INDEX](../INDEX.md) | [NEURAL_TEXTURE_ANALYSIS](NEURAL_TEXTURE_ANALYSIS.md)
> **Audit**: 3-model audit (Claude Sonnet + Gemini 2.5 Pro + GPT-4o) conducted 2026-03-23
> **Audience**: FaceLift + MAMMAL project teams

---

## 0. Executive Summary

Neural texture renders exhibit visual artifacts ("bizarre" appearance). A 3-model audit identified **mesh fitting quality as the most likely primary contributor**, with rendering pipeline issues as secondary factors. Key evidence: global model L1=0.105 vs per-frame L1=0.029 (3.6× gap) — this gap is **consistent with** cross-frame geometric inconsistency, though it also partially reflects the inherent difficulty of encoding multiple poses into a single texture MLP.

> ⚠️ **Confounding factor**: Per-frame models have much higher effective capacity per sample than the global model. Even with perfect mesh alignment, a global model averaging 77 poses will have higher L1 than per-frame. The 3.6× gap likely reflects BOTH mesh quality AND multi-pose averaging. §6 diagnostic protocol is designed to disentangle these.

> ⚠️ **IoU ≠ UV accuracy**: Silhouette IoU measures mesh boundary overlap with GT mask. Low IoU means the mesh is wrong SIZE/SHAPE — but pixels the mesh DOES cover may have correct UV. IoU < 0.7 indicates severe misalignment where both boundary AND internal UV are likely wrong.

**Root Cause Ranking** (3-model consensus, to be verified by §6 diagnostic):

| Rank | Cause | Confidence | Evidence |
|:---:|:---|:---:|:---|
| **1** | MAMMAL mesh tracking inconsistency + multi-pose averaging | HIGH | 3.6× L1 gap (confounded — §6 will separate) |
| **2** | Low-IoU frames polluting training | HIGH | 23% frames < IoU 0.7 |
| **3** | UV-as-color interpolation inaccuracy | MEDIUM | Mathematical: color interp ≠ UV interp |
| **4** | pyrender FLAT shading behavior (needs verification) | MEDIUM | FLAT may affect normals only, not vertex color interpolation |
| **5** | 8-bit UV quantization | LOW | At Nyquist limit of highest Fourier frequency (marginal) |

---

## 1. Priority 1: Low-IoU Frame Filtering (Cost: 0, Impact: HIGH)

### Problem

23 of 100 training frames have MAMMAL mesh IoU < 0.7 against GT silhouettes. These frames contribute noisy UV→RGB mappings where the mesh surface doesn't align with the actual mouse body — the MLP is trained on **incorrect UV→color associations** from misaligned regions.

### Why This Is the Primary Cause

The 3.6× L1 improvement from per-frame training is the smoking gun:
- **Per-frame model** (L1=0.029): Each frame gets its own MLP → mesh misalignment only affects pixels within that frame's misaligned region.
- **Global model** (L1=0.105): ALL frames share one MLP → misaligned frames teach the MLP contradictory UV→color mappings → the MLP learns an averaged, blurry, or nonsensical texture.

If the cause were UV encoding or pyrender (frame-invariant factors), both models would have similar L1.

### Specific Bad Frames (MAMMAL indices)

```
720  1320  1920  2040  2160  2760  3600  5160  5520  5880
6000  6120  6960  7200  8280  8400  9360  9480  9840  10080
10680  10800  11880
```

(Source: `verify_mesh_quality.py`, camera 003, IoU threshold = 0.7)

### Action

```python
# In train.py / train_v2.py, add filtering:
GOOD_FRAMES = [f for f in all_frames if frame_iou[f] >= 0.7]  # 77/100 frames
dataset = UVTextureDataset(frames=GOOD_FRAMES, ...)
```

**Expected improvement**: Significant L1 reduction for global model. The per-frame model already avoids this issue implicitly.

### Verification

```bash
# Re-train global model with IoU ≥ 0.7 filter
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.neural_texture.train_v2 \
    --min_iou 0.7 \
    --output_dir outputs/neural_texture/filtered_global
```

Compare L1 with unfiltered: if filtered global L1 approaches per-frame L1 (0.029), mesh quality is confirmed as the primary cause.

---

## 2. Priority 2: MAMMAL Re-fitting (Cost: MEDIUM, Impact: HIGH)

### Problem

Current MAMMAL fitting used `fast` optimization (50 iterations for step_1). This produces IoU < 0.7 for 23% of frames — mainly frames with rapid articulation (running, turning, rearing).

### Solution: Re-fit with `accurate` Config

| Parameter | fast | accurate |
|-----------|:----:|:--------:|
| step_0 iterations | 20 | 20 |
| **step_1 iterations** | **50** | **200** |
| step_2 iterations | 50 | 50 |
| Total per frame | ~120 | ~270 |
| Expected time (23 frames) | — | ~30-60 min |

### MAMMAL Project Action Items

**Config**: Use `accurate_6views` preset (or equivalent with step1_iters=200).

**Scope**: Only re-fit the 23 bad frames listed above. Good frames (IoU ≥ 0.7) do NOT need re-fitting.

**Output path convention**:
```
/home/joon/dev/MAMMAL_mouse/results/fitting/
  {experiment_name}/obj/step_2_frame_{XXXXXX}.obj
```

**Quality target (hypothesis)**: We expect re-fitting to achieve IoU ≥ 0.75, but this is NOT guaranteed — some frames may fail due to occlusion, extreme poses, or fundamental MAMMAL limitations.

**Validation**: After re-fitting, run IoU verification:
```bash
cd /home/joon/dev/FaceLift
python -m mouse_extensions.scripts.neural_texture.verify_mesh_quality \
    --obj_dir /path/to/refit/obj/ \
    --frames 720 1320 1920 ... \
    --threshold 0.75
```

### Why IoU 0.75 Is Still Not Enough for Texture Mapping

Even at IoU 0.75, **25% of pixels** have incorrect UV projection. For texture mapping:
- Misaligned mesh region → UV coords point to wrong anatomical location
- MLP learns: "UV(0.3, 0.7) = brown" from one frame, "UV(0.3, 0.7) = white" from another
- Result: averaged/blurry/contradictory texture

**Ideal target**: IoU ≥ 0.85. Per the audit, texture mapping has stricter alignment requirements than silhouette evaluation because even small spatial shifts map to completely different UV regions.

### For MAMMAL Team

The following information may help prioritize:

| Symptom in FaceLift | MAMMAL Root Cause | Severity |
|:---|:---|:---:|
| Blurry global texture | Cross-frame vertex inconsistency | HIGH |
| "Ghost" artifacts at limb boundaries | Limb articulation error (penetration, detachment) | HIGH |
| Ventral surface distortion | Belly region under-constrained by overhead cameras | MEDIUM |
| Tail/ear texture smearing | Thin structure tracking failure | MEDIUM |

---

## 3. Priority 3: Float UV Passthrough (Cost: MEDIUM, Impact: MEDIUM)

### Problem

Current pipeline encodes UV coordinates as 8-bit vertex colors:
```python
# Current: UV → uint8 vertex colors
colors[:, 0] = np.clip(uv[:, 0] * 255, 0, 255).astype(np.uint8)  # U → R
colors[:, 1] = np.clip(uv[:, 1] * 255, 0, 255).astype(np.uint8)  # V → G
```

Two issues:
1. **Quantization**: 255 discrete levels per axis (precision = 0.004)
2. **Interpolation domain mismatch**: Barycentric interpolation of quantized colors ≠ interpolation of continuous UV coordinates

### Why It Matters (But Is Secondary)

- Quantization (1/255 ≈ 0.004) sits at the **Nyquist limit** of the highest Fourier frequency (2^7=128 → Nyquist=0.004). This makes it marginal, not definitively safe or problematic.
- However, the **interpolation domain mismatch** (Gemini's finding) can introduce non-linear warping at UV seam boundaries — this IS a potential source of subtle artifacts.

### Solution Options

| Option | Approach | Difficulty | Quality |
|:---:|:---|:---:|:---:|
| **A** | Face-ID map → UV lookup | Medium | Best (exact intra-triangle UV, edge aliasing at seams) |
| **B** | nvdiffrast float rasterization | Medium | Best (differentiable) |
| **C** | 16-bit vertex colors | Low | Good (65,536 levels) |
| **D** | Keep 8-bit, document limitation | None | Acceptable |

**Option A (Recommended)**:
```python
# 1. Render face ID map (each face gets unique color)
face_ids = np.arange(len(mesh.faces))
face_colors = face_ids_to_colors(face_ids)  # Encode face ID as RGB
# Assign same color to all 3 vertices of each face
vertex_colors = face_colors[face_per_vertex]
# Render → decode face ID per pixel

# 2. For each pixel, look up triangle + compute barycentric coords
# 3. Interpolate UV precisely from triangle vertices
uv_precise = bary_w0 * uv[tri[0]] + bary_w1 * uv[tri[1]] + bary_w2 * uv[tri[2]]
```

**Option B**: Already partially implemented in `train_geom.py` using nvdiffrast. Requires CUDA.

### Current Code Location

`mouse_extensions/scripts/neural_texture/precompute_uv_maps.py` lines 34-93

---

## 4. Priority 4: FLAT Shading Verification (Cost: LOW, Impact: MEDIUM)

### Problem

pyrender `RenderFlags.FLAT` controls **normal computation** (per-face vs per-vertex), not necessarily vertex color interpolation. In OpenGL, FLAT shading forces constant face normals for lighting, but vertex colors may still be interpolated across triangles. However, this needs **empirical verification** — some pyrender configurations may also flatten vertex colors.

### Verification Test

```python
import pyrender, trimesh, numpy as np

# Create simple triangle with distinct vertex colors
verts = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]], dtype=np.float32)
faces = np.array([[0, 1, 2]])
colors = np.array([[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255]], dtype=np.uint8)

mesh = trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=colors)
mesh_py = pyrender.Mesh.from_trimesh(mesh, smooth=False)

scene = pyrender.Scene(ambient_light=[1, 1, 1])
scene.add(mesh_py)
# ... add camera, render with FLAT flag

# If output is a single solid color → FLAT = no interpolation (PROBLEM)
# If output shows RGB gradient → FLAT only affects lighting, not colors (OK)
```

### Fix (if FLAT = no interpolation)

```python
# Option 1: Switch to smooth=True
mesh_py = pyrender.Mesh.from_trimesh(uv_mesh, smooth=True)

# Option 2: Use default render flags (remove FLAT)
color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
```

### Current Code Location

`precompute_uv_maps.py` line ~85: `flags=pyrender.RenderFlags.FLAT | pyrender.RenderFlags.RGBA`

---

## 5. Priority 5: Training Pipeline Improvements (Cost: LOW, Impact: LOW-MEDIUM)

### 5a. Remove B=255 Constant from MLP Input

The blue channel (B=255) provides zero spatial information but consumes 16 Fourier encoding dimensions (8 bands × sin/cos).

```python
# Current (neural_texture.py):
# Input: UV = (u, v) from color channels R, G → 2D
# The B channel is already NOT used as MLP input (only R, G extracted)
```

**Verification needed**: Confirm that MLP input is actually 2D (u, v) and not 3D (r, g, b). If 2D, this is already correct and the audit finding is moot.

### 5b. Add LPIPS Co-metric

L1=0.029 ≈ 7.4 intensity units mean error — perceptually significant (human JND = 2-4 units). L1 alone cannot diagnose structural/spatial artifacts.

```python
# In train_v2.py, add alongside L1:
import lpips
lpips_fn = lpips.LPIPS(net='vgg').cuda()

# In validation loop:
lpips_val = lpips_fn(pred_patch, gt_patch).item()
print(f"L1={l1:.4f}  LPIPS={lpips_val:.4f}")
```

**Threshold**: LPIPS > 0.1 with L1 < 0.03 → structural artifacts present despite low pixel error.

### 5c. Fourier Bands (Deferred)

Current: 8 bands. Possible Gibbs ringing at sharp texture boundaries. Increase to 12-16 after mesh/UV issues resolved.

---

## 6. Diagnostic Protocol (Step-by-Step)

Before implementing fixes, run this diagnostic to confirm root causes.

> ⚠️ **Visual examples**: Diagnostic outputs should be shared with MAMMAL team as reference images showing the specific artifacts. Include side-by-side: GT image | neural texture render | mesh silhouette overlay.

### Step 1: Correlate "Bizarre" Regions with IoU

```bash
# Render neural texture for 5 high-IoU and 5 low-IoU frames
# Compare visual quality → if low-IoU frames are always worse, mesh = primary cause
python -m mouse_extensions.scripts.neural_texture.visualize \
    --frames 0 500 1000 1500 2000 \
    --frames_bad 720 1320 1920 2040 2160 \
    --output_dir outputs/neural_texture/diagnostic/
```

**Decision tree**:
- If |Pearson r| between per-frame IoU and L1 > 0.4 → **mesh quality is a major factor** → proceed with Priority 1-2
- If |r| < 0.4 AND both high/low-IoU frames look bad → **systemic pipeline issue** → reprioritize to Priority 3-4 (UV/shading)
- If only low-IoU frames look bad → **mesh quality is THE primary factor**
- If only high-IoU frames look bad → **pipeline issue, mesh is not the cause** → reprioritize entirely

### Step 2: Verify FLAT Shading Behavior

Run the triangle test from §4 above. Takes 5 minutes.

### Step 3: Compare 8-bit vs Float UV

```python
# Render same frame with:
# A) Current 8-bit vertex colors
# B) Direct float UV from mesh (bypass pyrender)
# Compare: if identical → quantization is not the issue
```

### Step 4: Check MLP Input Dimensionality

```python
# Confirm MLP receives 2D (u, v), not 3D (r, g, b)
model = NeuralTextureMLP()
print(f"Input dim: {model.encoder.input_dim}")  # Should be 2
```

### Contingency: If Mesh Hypothesis Fails

If §6 Step 1 shows weak IoU-quality correlation:
1. **Notify MAMMAL team** that re-fitting is deprioritized
2. Focus on Priority 3 (float UV passthrough) + Priority 4 (shading verification)
3. Investigate whether the MLP architecture itself is the bottleneck (capacity, loss function, training data preprocessing)

---

## 7. Summary: Action Items by Team

### FaceLift Team (Immediate)

| # | Action | Files | Time |
|:---:|:---|:---|:---:|
| 1 | Filter IoU < 0.7 frames from training | `train_v2.py` | 30min |
| 2 | Run diagnostic protocol (§6) | `visualize.py` | 1h |
| 3 | Verify FLAT shading behavior | `precompute_uv_maps.py` | 5min |
| 4 | Add LPIPS metric | `train_v2.py` | 30min |

### MAMMAL Team (Requested)

> Note: IoU verification will be performed by FaceLift team (requires FaceLift codebase). MAMMAL team only needs to re-fit and provide output OBJs.

| # | Action | Config | Time |
|:---:|:---|:---|:---:|
| 1 | Re-fit 23 bad frames with `accurate` config | step1_iters=200 | ~1h |
| 2 | Provide re-fit OBJs to FaceLift team for validation | Output: `obj/step_2_frame_XXXXXX.obj` | — |
| 3 | (Stretch) If IoU still < 0.75, investigate per-frame fitting failures | May need manual intervention | Variable |

**What MAMMAL team needs to know** (non-jargon):
- FaceLift uses MAMMAL meshes to create a "texture lookup map" — for each pixel, we find where on the 3D surface it corresponds, then use a neural network to predict color.
- If the mesh doesn't align well with the actual mouse body (IoU < 0.7), the lookup map points to wrong body parts → wrong colors → "bizarre" renders.
- We need 23 specific frames re-fit at higher quality. The mesh topology stays the same — only vertex positions change.

### Joint (After Re-fit)

| # | Action | Expected Outcome |
|:---:|:---|:---|
| 1 | Re-train global model with ALL frames (post re-fit) | L1 < 0.05 (vs current 0.105) |
| 2 | If L1 still high → implement float UV (Priority 3) | Eliminate UV encoding artifacts |
| 3 | If L1 approaches per-frame (0.029) → mesh was the sole cause | Close this investigation |

---

## Related Documents

| Document | Relationship |
|:---|:---|
| ↑ [[../INDEX]] | Document hub |
| ↔ [[NEURAL_TEXTURE_ANALYSIS]] | Initial 3-model deliberation + roadmap |
| ↔ [[../../mouse_extensions/docs/UV_TEXTURE_RENDERING_BUG]] | v/vt index bug postmortem |
| ↔ MAMMAL_REFIT_HANDOFF.md | Re-fitting protocol for bad frames |
| ↓ `mouse_extensions/scripts/neural_texture/` | Implementation code |

---

*Neural Texture Rendering Quality Diagnosis | v1.0 | 2026-03-23*
*3-Model Audit: Claude Sonnet 4.6 + Gemini 2.5 Pro + GPT-4o*
