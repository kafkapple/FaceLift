# s-DANNCE Video Data Availability Report

> **Date**: 2026-03-22 | **Investigator**: Claude + Web Research Agent
> **Purpose**: BehaviorSplatter NeurIPS 2026 — BALB/c mouse multi-view video availability

---

## Executive Summary

BALB/c mouse multi-view video data is **NOT publicly downloadable**. Harvard Dataverse MOUSE cohort (DVN/VKJHTD) contains only 128 .mat files (keypoints + behavior labels, 1.9 GB total). Video data exists physically (s-DANNCE inference was run on it) but was never uploaded. **Author email request is the only path.**

---

## 1. Harvard Dataverse Full Survey (17 Datasets)

| Type | Datasets | Description Pattern | Video? |
|------|---------|-------------------|:------:|
| **RAT w/ Video** | SCN2A_SOC1, SOC3, WK1 | "Raw movies and 3D postural tracking" | **Yes** (MP4) |
| **RAT Keypoints** | ARID1B, CHD8, CNTNAP2, FMR1, GRIN2B, NRXN1, SCN2A, LONG EVANS, BEDDING, TRIADS | "keypoint tracking and behavioral mapping" | No (.mat) |
| **MOUSE Keypoints** | MOUSE (DVN/VKJHTD) | "keypoint tracking and behavioral mapping" | **No** (.mat) |

### Size Evidence

| Cohort | Files | Size | Video? |
|--------|:-----:|-----:|:------:|
| SCN2A_SOC1 (rat, w/ video) | 504 | 25.5 GB | Yes |
| SCN2A_WK1 (rat, w/ video) | 878 | 37.1 GB | Yes |
| **MOUSE** (128 recordings) | 128 | **1.9 GB** | **No** |

If 128 mouse recordings included video: 6 cams × ~15 MB × 128 = ~11.5 GB minimum. Actual 1.9 GB matches .mat-only.

---

## 2. Alternative Sources Checked

| Source | Result |
|--------|--------|
| **Duke Box** (s-DANNCE demo) | Rat-only demo data |
| **Original DANNCE repo** | `markerless_mouse_1/2` = C57BL/6, single-animal, not social BALB/c |
| **s-DANNCE GitHub** | No mouse video download links |
| **Cell 2025 Data Availability** | Paywall — likely "upon reasonable request" (Cell Press standard) |

---

## 3. Author Contacts

| Name | Email | Role |
|------|-------|------|
| Ugne Klibaite | klibaite@fas.harvard.edu | Lead contact |
| Bence P. Olveczky | olveczky@fas.harvard.edu | PI (Harvard) |
| Timothy W. Dunn | timothy.dunn@duke.edu | PI (Duke) |

**Request**: "MOUSE cohort raw multi-view video recordings (6-camera MP4) + camera calibration files for BehaviorSplatter NeurIPS 2026 submission"

---

## 4. Implications for BehaviorSplatter

| Data | Video | Calibration | Keypoints | Behavior Labels | GS-LRM Feasible? |
|------|:-----:|:-----------:|:---------:|:---------------:|:-----------------:|
| **Rat SCN2A** (gpu03) | ✅ | ✅ | ✅ (23j) | ✅ (HLAC via Dataverse .mat) | **Testing now** |
| **BALB/c Mouse** | ❌ | ❌ | ✅ (23j) | ✅ (HLAC) | ❌ No video |
| **M5t2 Mouse** | ✅ | ✅ | ✅ (22j) | ❌ None | ✅ Trained |

### Strategy

1. **Rat-first**: Use SCN2A_SOC1 session (complete data on gpu03) for GS-LRM feasibility
2. **Email authors**: Request BALB/c video (parallel, timeline uncertain)
3. **Fallback**: Rat as primary species for dense features + behavior labels

---

## 5. MOUSE Cohort .mat Structure (Verified)

```python
# Example: MOUSE_W4_20240426_0028_L.mat (Lone BALB/c session)
sdannce.group       = 'mouse'
sdannce.mousestrain = 'BALB/c'
sdannce.issoc       = 0 (lone)
sdannce.m1          = (30000, 3, 23)  # 10 min @ 50fps
sdannce.m2          = empty
sdannce.hlac        = (30000,)        # behavior labels
sdannce.llac        = (30000,)        # fine-grained labels
sdannce.cz_action   = (30000, 2)      # t-SNE embedding
```

Mouse sessions: 30,000 frames (10 min) vs Rat: 90,000 frames (30 min).
BALB/c = W (White), C57BL/6 = B (Black).

---

*BehaviorSplatter | s-DANNCE Video Availability | 2026-03-22*

Related:
- ↑ [INDEX](../INDEX.md)
- ↔ [SDANNCE_DATA_CONVENTION](../../CLAUDE.md#sdannce-data-convention)
- ↔ [sdannce-poc dataset_catalog](~/dev/sdannce-poc/docs/data/dataset_catalog.md)
