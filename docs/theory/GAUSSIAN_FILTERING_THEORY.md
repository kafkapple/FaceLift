# Gaussian Filtering & Count-Match — Method SSOT

> Single entry point for every post-hoc Gaussian selection step in the mouse pipeline
> (FaceLift `mouse_extensions` owns the code; BehaviorSplatter consumes the outputs by path).
> Referenced by: `scripts/filter_ply.py`, `scripts/make_maskcarve16k.py`,
> `behavior/cinematic_sequence.py::build_count_match_mask`.
> Numbers/claims live elsewhere — see §6. Written 2026-08-13, reverse-verified against code.

## 1. Terminology (three distinct operations — do not conflate)

| Term | When | What | Code |
|---|---|---|---|
| **Pruning** | training-time | adaptive density control (Kerbl 2023) + ghost-gaussian removal | `model/gaussian_pruning.py` |
| **Filtering** | post-hoc | opacity thresholding on saved PLYs | `scripts/filter_ply.py` |
| **Count-match** | post-hoc | multi-view visibility vote + top-K opacity → fixed gaussian budget | `behavior/cinematic_sequence.py::build_count_match_mask`, CLI `scripts/make_maskcarve16k.py` |

## 2. Count-match method (the "5/6 vote + top-K" rule)

Given per-frame GS-LRM output (~100k gaussians, logit-opacity PLY):

1. **Project** each gaussian center into all 6 camera views
   (`behavior/multiview_visibility_filter.py::compute_visibility_counts`).
2. **Foreground test** per view: does the projected center land inside the
   foreground mask (`alpha > 128`)? Count the passing views → `vis_counts ∈ [0,6]`.
3. **Vote gate**: keep candidates with `vis_counts >= n_filter` (default **5 of 6**).
   Removes background floaters/ghosts that only one or two views support.
4. **Budget**: among candidates, keep the **top-K by opacity** (default **K=16,000**).
   If candidates ≤ K, keep them all (no padding).

Defaults are code defaults (`n_filter=5, top_k=16000`) — identical in
`build_count_match_mask` and the standalone CLI.

### Why 16k (design intent)
PoseSplatter uses a fixed 16k-gaussian representation; GS-LRM emits ~100k.
Count-match equalizes the gaussian budget so paradigm comparisons (PS↔GS, round22)
measure representation quality, not point count. It doubles as floater cleanup for
cinematic rendering.

### Provenance tag `w_α=0.3`
Directory names like `M5t2_6view_alpha03_v3_maskcarve16k` encode the source
checkpoint's `alpha_loss_weight=0.3` (experiment E5 lineage — see
`docs/experiments/COMMANDS.md`). The count-match step itself has no alpha parameter;
the tag is checkpoint provenance, not a filter setting.

## 3. Toolchain (each stage = standalone CLI)

| Stage | CLI | Output naming convention |
|---|---|---|
| 1. ckpt → per-frame PLY | `python -m mouse_extensions.scripts.export.batch_ply_export --config ... --output-dir ...` | `ply_a<alpha>/...` |
| 2. opacity filter | `python -m mouse_extensions.scripts.filter_ply --source <ply_dir> --output <.../filtered/a0.3_t0.2> --opacity_threshold 0.2` | `filtered/a<alpha>_t<thr>` |
| 3. count-match | `python -m mouse_extensions.scripts.make_maskcarve16k --ply-dir <ply_dir> --sample-dir-base <preprocessed>/M5 --output <.../*_maskcarve16k>` | `<ckpt-tag>_maskcarve16k` |

Notes: stage-2 opacity is sigmoid-space (script auto-detects logit PLYs);
stage-3 ranking is monotonic in either space. `--limit N` on stage 3 = smoke test.
Smoke-verified 2026-08-13: 99,250–101,383 → exactly 16,000 per frame.

## 4. Known artifact directories (FaceLift_mouse_6view/gaussians/)

| Dir | Produced by | Consumed by |
|---|---|---|
| `filtered/a0.3_t0.2` | stage 2 | BS hosts config `fl_filtered` |
| `M5t2_6view_alpha03_v3_maskcarve16k` | stages 1+3 (M5t2 α=0.3 ckpt) | BS `ve_pipeline` (`gs_gaussian_dir`), FaceLift cinematic v11 |
| `M5t2_6view_alpha10_v3_maskcarve16k` | stages 1+3 (α=1.0 ckpt) | cinematic v11 artifact demo |
| `ply_a0.3` | stage 1 | BS `visualization/canonical_compare.py` |
| `match_16k` | stage 3 (early run) | none found (intermediate) |
| `base_uniform_v2_6view_v2` | stage 1 full-res (281 frames) | none found — regenerable from ckpt |

## 5. History note
An earlier write-up (`PARADIGM_COMPARISON_SSOT.md`, ICML-workshop era) described this
method in §36 but is no longer in any tree; code comments that cited it now point here.

## 6. Cross-references (numbers live there, not here)
- Cinematic parameters: `docs/specs/CINEMATIC_V11_SPEC.md`
- Experiment lineage (E5, α sweep): `docs/experiments/COMMANDS.md`
- Paradigm-comparison numbers: Obsidian `260724_paradigm_TRUTH_round22.md` (round22 CSV = 유일 수치 정본)
- Visualization configs consuming these dirs: `configs/mouse/cinematic/cinematic_v11_FINAL_*.yaml`
