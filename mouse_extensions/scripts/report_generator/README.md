# Report Generator Manual

> **Version**: v1.0 (2026-01-21)
> **Location**: `mouse_extensions/scripts/report_generator/`

---

## Quick Start

```bash
cd /home/joon/dev/FaceLift

# Generate preprocessing comparison report
/home/joon/anaconda3/envs/facelift/bin/python \
    mouse_extensions/scripts/report_generator/unified_report_generator.py \
    --type preprocessing \
    --datasets v13,D1,D7_1,D7_1_t

# Quick dataset summary
/home/joon/anaconda3/envs/facelift/bin/python \
    mouse_extensions/scripts/report_generator/unified_report_generator.py \
    --type summary \
    --datasets D7_1_t
```

---

## Available Scripts

| Script | Description |
|--------|-------------|
| `unified_report_generator.py` | Main report generator (preprocessing, summary) |
| `generate_report.py` | Legacy report generator with figures |
| `coordinate_report_generator.py` | Coordinate system analysis |
| `config.py` | Common configuration |

---

## unified_report_generator.py

### Usage

```bash
python unified_report_generator.py [OPTIONS]
```

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--type` | preprocessing | Report type: `preprocessing`, `summary`, `experiment` |
| `--datasets` | D7_1,D7_1_t | Comma-separated dataset names |
| `--output` | reports/09_unified | Output directory |
| `--base-path` | (auto) | Dataset base path |

### Supported Datasets

| Name | Status | Description |
|------|--------|-------------|
| `v13` | DEPRECATED | Legacy PP=256 forced (whitening) |
| `D1` | DEPRECATED | PP-centered crop (ghosting) |
| `D4` | DEPRECATED | Triangulation + PP=256 |
| `D7` | CURRENT | PP-shift + fy=549 |
| `D7_1` | RECOMMENDED | Individual scale |
| `D7_1_t` | RECOMMENDED | D7.1 + temporal split |
| `D7_2` | ALTERNATIVE | Average scale |

### Output

Reports are saved to:
```
mouse_extensions/reports/09_unified/
├── 260121_preprocessing_comparison.md
└── 260121_summary.md
```

---

## Examples

### 1. Compare All Deprecated vs Recommended

```bash
python unified_report_generator.py \
    --type preprocessing \
    --datasets v13,D1,D4,D7_1,D7_1_t
```

### 2. Quick Check Single Dataset

```bash
python unified_report_generator.py \
    --type summary \
    --datasets D7_1_t
```

### 3. Custom Output Directory

```bash
python unified_report_generator.py \
    --type preprocessing \
    --datasets D7_1,D7_2 \
    --output /path/to/output/
```

---

## Adding New Datasets

Edit `DATASET_REGISTRY` in `unified_report_generator.py`:

```python
DATASET_REGISTRY = {
    "new_dataset": {
        "name": "new_dataset",
        "path": "path_under_base",
        "description": "Description",
        "status": "EXPERIMENTAL",
        "ray_error": "~0 deg",
        "issue": "None"
    },
    ...
}
```

---

## Report Structure

### Preprocessing Report
1. Summary Table (status, ray error, sample counts)
2. Camera Parameter Statistics (fx, fy, cx, cy)
3. Dataset Details (description, path, issues)

### Summary Report
- Quick overview of each dataset
- Status and sample counts

---

## Automation Integration

### Cron Job (Daily Report)

```bash
# Add to crontab
0 9 * * * cd /home/joon/dev/FaceLift && /home/joon/anaconda3/envs/facelift/bin/python mouse_extensions/scripts/report_generator/unified_report_generator.py --type preprocessing --datasets D7_1,D7_1_t >> logs/report_cron.log 2>&1
```

### After Training Completion

```bash
# In training script or post-hook
python unified_report_generator.py --type experiment --checkpoints checkpoints/gslrm/...
```

---

*Generated: 2026-01-21*
