# Fair Comparison: FaceLift vs Pose-Splatter

> Generated: 2026-02-15 23:09

## Fairness Guarantees

| Guarantee | Status |
|-----------|--------|
| Test-only frames | Yes |
| GT alpha masks | Yes |
| Identical metric functions | Yes |
| Foreground-only evaluation | Yes |
| Coverage-aware metrics | Yes |

## Overall Results

| Metric | FaceLift | Pose-Splatter | Gap | Winner |
|--------|----------|---------------|-----|--------|
| PSNR (GT-masked) | 7.83 | 16.80 | -8.97 | PS |
| PSNR (intersection) | 15.44 | 20.48 | -5.04 | PS |
| SSIM (GT-masked) | 0.7509 | 0.8583 | -0.1074 | PS |
| L1 (GT-masked) | 0.2966 | 0.0969 | +0.1997 | PS |
| L1 (intersection) | 0.1247 | 0.0733 | +0.0514 | PS |
| IoU | 0.5184 | 0.8274 | -0.3090 | PS |
| Coverage | 71.4% | 91.9% | -20.5% | PS |
| Precision | 63.4% | 89.2% | -25.9% | PS |

## Coverage Analysis

- FL coverage: 71.4% of GT foreground
- PSNR on covered region: 15.44 dB
- PSNR on full GT mask: 7.83 dB
- Gap from coverage: 7.61 dB

## Interpretation

- **PSNR (GT-masked)**: Standard evaluation using GT foreground mask.
  Penalizes both color error and missing coverage.
- **PSNR (intersection)**: Only evaluates where BOTH models have foreground.
  Isolates color accuracy from coverage.
- **Coverage**: What fraction of GT mouse the model reconstructs.
- **IoU**: Overlap between predicted and GT silhouettes.
