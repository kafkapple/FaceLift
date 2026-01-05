---
date: 2025-12-20
dataset: data_mouse_uniform_v2
generator: evaluate_preprocessing.py
---

# Preprocessing Quality Report: data_mouse_uniform_v2

## Overview

| Metric | Value |
|--------|-------|
| Dataset Path | `data_mouse_uniform_v2` |
| Samples Analyzed | 100 |
| Total Views | 600 |
| Quality Grade | **D (Needs Improvement)** |

## Size Ratio Statistics (Pixel-Based)

> Size ratio = sqrt(object_pixels / total_pixels)
> Target: Uniform across all views

| Metric | Value |
|--------|-------|
| Mean | 0.3010 (30.1%) |
| Std | 0.0007 |
| CV (Coefficient of Variation) | **0.23%** |
| Min | 0.2993 (29.9%) |
| Max | 0.3034 (30.3%) |
| Range | 0.0041 |

### Interpretation
- CV < 1%: Excellent uniformity
- CV 1-3%: Good uniformity
- CV 3-5%: Acceptable
- CV > 5%: Needs improvement

## Center of Mass (CoM) Statistics

> CoM offset = distance from image center to object center of mass
> Target: Near zero (object centered)

| Metric | Value |
|--------|-------|
| Mean Offset | **45.8px** (8.94% of image) |
| Std | 25.1px |
| Max Offset | 95.4px |
| 95th Percentile | 89.1px |
| 99th Percentile | 94.1px |

### Interpretation
- Mean < 5px: Excellent centering
- Mean 5-10px: Good centering
- Mean 10-20px: Acceptable
- Mean > 20px: Needs improvement

## Cross-View Consistency

| Metric | Value |
|--------|-------|
| Avg Intra-Sample Size CV | 0.14% |
| Avg Intra-Sample CoM Offset Std | 23.2px |

## Quality Distribution

### Size Ratio Histogram
![Size Ratio Distribution](./data_mouse_uniform_v2_quality_report.png)

### Sample Visualizations
![Sample Visualizations](./data_mouse_uniform_v2_sample_visualization.png)

## Recommendations

- **Centering**: CoM-based centering may need adjustment. Consider using Visual Hull-based 3D center
- **Outliers**: 227 views have offset > 50px. Inspect these samples manually

## Generated
- Date: 2025-12-20 15:58:46
- Script: `scripts/evaluate_preprocessing.py`
