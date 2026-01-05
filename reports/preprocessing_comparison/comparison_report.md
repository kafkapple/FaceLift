# Preprocessing Methods Comparison Report

Date: 2025-12-20 16:04

## Summary Table

| Dataset | Samples | Size Mean | Size CV | CoM Offset | Grade |
|---------|---------|-----------|---------|------------|-------|
| data_mouse_pixel_based | 50 | 30.6% | 0.16% | 1.5px | A  |
| data_mouse | 50 | 23.3% | 7.31% | 119.2px | D  |
| data_mouse_centered | 50 | 26.6% | 12.82% | 36.0px | D  |
| data_mouse_normalized | 50 | 26.6% | 12.82% | 36.0px | D  |
| data_mouse_uniform | 50 | 29.9% | 8.92% | 38.3px | D  |
| data_mouse_uniform_v2 | 50 | 30.1% | 0.20% | 42.0px | D  |

## Grading Criteria

| Grade | Size CV | CoM Offset | Description |
|-------|---------|------------|-------------|
| A | < 1% | < 5px | Excellent - Ready for training |
| B | < 3% | < 10px | Good - Minor improvements possible |
| C | < 5% | < 20px | Fair - Needs improvement |
| D | >= 5% | >= 20px | Poor - Requires reprocessing |

## Detailed Analysis

### data_mouse_pixel_based

- **Grade**: A
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 30.6%
  - Std: 0.05%
  - CV: 0.16%
- **Center of Mass Offset**:
  - Mean: 1.5px
  - Std: 0.3px
  - Max: 1.9px

### data_mouse

- **Grade**: D
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 23.3%
  - Std: 1.70%
  - CV: 7.31%
- **Center of Mass Offset**:
  - Mean: 119.2px
  - Std: 13.8px
  - Max: 145.0px

### data_mouse_centered

- **Grade**: D
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 26.6%
  - Std: 3.41%
  - CV: 12.82%
- **Center of Mass Offset**:
  - Mean: 36.0px
  - Std: 6.3px
  - Max: 48.3px

### data_mouse_normalized

- **Grade**: D
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 26.6%
  - Std: 3.41%
  - CV: 12.82%
- **Center of Mass Offset**:
  - Mean: 36.0px
  - Std: 6.3px
  - Max: 48.3px

### data_mouse_uniform

- **Grade**: D
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 29.9%
  - Std: 2.66%
  - CV: 8.92%
- **Center of Mass Offset**:
  - Mean: 38.3px
  - Std: 6.4px
  - Max: 50.7px

### data_mouse_uniform_v2

- **Grade**: D
- **Samples Analyzed**: 50
- **Size Ratio**:
  - Mean: 30.1%
  - Std: 0.06%
  - CV: 0.20%
- **Center of Mass Offset**:
  - Mean: 42.0px
  - Std: 9.3px
  - Max: 53.6px

## Recommendations

**Recommended Dataset**: `data_mouse_pixel_based` (Grade A)

This dataset has the best combination of size uniformity (CV 0.16%) and centering accuracy (offset 1.5px).
