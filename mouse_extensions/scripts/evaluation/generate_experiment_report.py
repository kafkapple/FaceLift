#!/usr/bin/env python3
"""
Generate comprehensive experiment analysis report.

This script analyzes FaceLift experiments and generates:
1. Quantitative metrics (PSNR, SSIM, LPIPS) for train/val/test
2. Qualitative visualizations (GT vs Prediction)
3. Hypothesis validation and conclusions

Usage:
    python -m mouse_extensions.scripts.evaluation.generate_experiment_report \
        --output_dir outputs/reports \
        --experiments view_ablation h1_diagnosis
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mouse_extensions.evaluation import (
    ExperimentReportGenerator,
    ExperimentHypothesis,
    ExperimentCondition,
    SplitMetrics,
    AggregatedMetrics,
    VisualizationGenerator,
)


# Experiment configurations
EXPERIMENTS = {
    "view_ablation": {
        "title": "View Ablation Experiment Report",
        "description": """
This experiment tests how the number of input views affects reconstruction quality.

**Research Question**: What is the optimal number of input views for GS-LRM on mouse data?

**Hypothesis**: More input views should improve reconstruction quality, but with diminishing returns.
        """,
        "checkpoint_base": "/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_view_ablation",
        "dataset_root": "/home/joon/data/preprocessed/FaceLift_mouse/M5",
        "variants": {
            "E0_1_1view": {"n_input_views": 1},
            "E0_1_2view": {"n_input_views": 2},
            "E0_1_3view": {"n_input_views": 3},
            "E0_1_5view": {"n_input_views": 5},
            "E0_1_6view": {"n_input_views": 6},
        },
    },
    "h1_diagnosis": {
        "title": "H1 Diagnosis: MVDiffusion Bottleneck Analysis",
        "description": """
This experiment diagnoses whether MVDiffusion is a bottleneck in the E2E pipeline.

**Research Question**: Does the MVDiffusion stage limit overall reconstruction quality?

**Hypothesis (H1)**: When MVDiffusion is undertrained, E2E performance is significantly worse than GS-LRM alone.
        """,
        "datasets": {
            "M5t2": {
                "checkpoint_base": "/node_data/joon/checkpoints/FaceLift/gslrm/M5t2_E0_1_facelift",
                "train_samples": 2880,
            },
            "M5t": {
                "checkpoint_base": "/node_data/joon/checkpoints/FaceLift/gslrm/M5t_E0_1_facelift",
                "train_samples": 1198,
            },
        },
    },
}


def load_best_psnr(checkpoint_dir: Path) -> Tuple[float, int]:
    """Load best PSNR from checkpoint directory."""
    best_json = checkpoint_dir / "best_psnr.json"
    if best_json.exists():
        with open(best_json) as f:
            data = json.load(f)
            return data.get("value", 0.0), data.get("step", 0)
    return 0.0, 0


def load_wandb_run_id(checkpoint_dir: Path) -> Optional[str]:
    """Load WandB run ID from checkpoint directory."""
    wandb_file = checkpoint_dir / "wandb_run_id.txt"
    if wandb_file.exists():
        return wandb_file.read_text().strip()
    return None


def generate_view_ablation_report(
    output_dir: Path,
    checkpoint_base: Path,
    dataset_root: Path,
    variants: Dict[str, Dict],
) -> Path:
    """Generate view ablation experiment report."""
    generator = ExperimentReportGenerator(output_dir)
    generator.set_report_info(
        title="View Ablation Experiment Report",
        description="""
## Overview

This experiment analyzes how the number of input views affects GS-LRM reconstruction quality on mouse data.

**Dataset**: M5t2 (80:10:10 temporal split)
- Train: 2880 samples
- Val: 360 samples
- Test: 360 samples

**Evaluation Criterion**: Val PSNR (primary), Test PSNR (final)

**Note**: Metrics shown are from training validation (same-view reconstruction).
For true generalization, see the separate test set evaluation.
        """
    )

    # Collect results from each variant
    for name, config in variants.items():
        ckpt_dir = checkpoint_base / name
        if not ckpt_dir.exists():
            print(f"Skipping {name}: directory not found")
            continue

        psnr, step = load_best_psnr(ckpt_dir)
        wandb_id = load_wandb_run_id(ckpt_dir)

        n_views = config["n_input_views"]

        # Create hypothesis
        hypothesis = ExperimentHypothesis(
            name=f"{n_views}-view Input Reconstruction",
            description=f"Test reconstruction quality with {n_views} input views",
            experimental_group=f"{n_views} input views",
            control_group="6-view (baseline)" if n_views != 6 else None,
            expected_outcome=f"{'Lower' if n_views < 6 else 'Baseline'} PSNR expected"
        )

        # Create condition
        condition = ExperimentCondition(
            n_input_views=n_views,
            n_target_views=6,
            dataset="M5t2",
            split_ratio="80:10:10",
            checkpoint_step=step,
        )

        # Create metrics (val from training)
        # Note: We only have training val PSNR currently
        val_metrics = SplitMetrics(
            split="val",
            metrics=AggregatedMetrics(
                psnr_mean=psnr,
                psnr_std=0.0,  # Not available from single value
                ssim_mean=0.0,  # Need to compute
                ssim_std=0.0,
                n_samples=0,  # Unknown
            ),
            sample_results=[],
        )

        generator.add_full_experiment(
            name=f"{n_views}-view",
            hypothesis=hypothesis,
            condition=condition,
            val_metrics=val_metrics,
            wandb_run_id=wandb_id,
            checkpoint_path=str(ckpt_dir / "best_psnr.pt"),
            notes=f"Training stopped at step {step}. Best val PSNR: {psnr:.2f} dB"
        )

    # Generate reports
    return generator.generate_full_report("view_ablation_report")


def generate_h1_diagnosis_report(
    output_dir: Path,
    datasets_config: Dict,
) -> Path:
    """Generate H1 diagnosis report."""
    generator = ExperimentReportGenerator(output_dir)
    generator.set_report_info(
        title="H1 Diagnosis: MVDiffusion Bottleneck Analysis",
        description="""
## Overview

This experiment tests whether MVDiffusion is a bottleneck in the End-to-End (E2E) pipeline.

### Hypothesis (H1)
> When MVDiffusion is undertrained relative to the data size, E2E (MVDiffusion → GS-LRM)
> performance is significantly worse than GS-LRM alone (using GT images).

### Experimental Design

| Dataset | Train Samples | MVDiffusion Epochs | Expected Outcome |
|---------|---------------|-------------------|------------------|
| M5t2    | 2880          | ~6 epochs         | Small gap (sufficient training) |
| M5t     | 1198          | ~20 epochs        | Large gap (undertrained) |

### Evaluation Modes

1. **GS-LRM Only**: Input = GT images (multi-view capture)
2. **E2E**: Input = MVDiffusion generated views (single-view → multi-view)

### Key Metric

**Gap = GS-LRM PSNR - E2E PSNR**
- Gap ≈ 0: MVDiffusion not a bottleneck
- Gap > 1.0 dB: MVDiffusion is bottleneck
        """
    )

    # Load existing H1 results
    h1_results_path = output_dir / "h1_diagnosis_comparison.json"
    if h1_results_path.exists():
        with open(h1_results_path) as f:
            h1_data = json.load(f)

        for dataset_name, modes in h1_data.get("datasets", {}).items():
            train_samples = datasets_config.get(dataset_name, {}).get("train_samples", 0)

            # Add GS-LRM result
            gslrm_test = modes.get("GS-LRM_test", {})
            if gslrm_test:
                generator.add_full_experiment(
                    name=f"{dataset_name} GS-LRM",
                    hypothesis=ExperimentHypothesis(
                        name="GS-LRM Upper Bound",
                        description="GS-LRM performance with GT input images",
                        experimental_group="GT images as input",
                        control_group=None,
                    ),
                    condition=ExperimentCondition(
                        n_input_views=6,
                        n_target_views=6,
                        dataset=dataset_name,
                        split_ratio="80:10:10" if dataset_name == "M5t2" else "33:33:33",
                        extra={"train_samples": train_samples},
                    ),
                    test_metrics=SplitMetrics(
                        split="test",
                        metrics=AggregatedMetrics(
                            psnr_mean=gslrm_test.get("psnr_mean", 0),
                            psnr_std=gslrm_test.get("psnr_std", 0),
                            ssim_mean=gslrm_test.get("ssim_mean", 0),
                            ssim_std=0,
                            lpips_mean=gslrm_test.get("lpips_mean"),
                            lpips_std=0,
                            n_samples=gslrm_test.get("n_samples", 50),
                        ),
                        sample_results=[],
                    ),
                )

            # Add E2E result
            e2e_test = modes.get("E2E_test", {})
            if e2e_test:
                generator.add_full_experiment(
                    name=f"{dataset_name} E2E",
                    hypothesis=ExperimentHypothesis(
                        name="E2E Pipeline Performance",
                        description="Full pipeline: MVDiffusion + GS-LRM",
                        experimental_group="MVDiffusion generated views",
                        control_group="GT images (GS-LRM only)",
                    ),
                    condition=ExperimentCondition(
                        n_input_views=1,  # Single input view for MVDiffusion
                        n_target_views=6,
                        dataset=dataset_name,
                        split_ratio="80:10:10" if dataset_name == "M5t2" else "33:33:33",
                        extra={"train_samples": train_samples, "mvdiffusion": True},
                    ),
                    test_metrics=SplitMetrics(
                        split="test",
                        metrics=AggregatedMetrics(
                            psnr_mean=e2e_test.get("psnr_mean", 0),
                            psnr_std=e2e_test.get("psnr_std", 0),
                            ssim_mean=e2e_test.get("ssim_mean", 0),
                            ssim_std=0,
                            lpips_mean=e2e_test.get("lpips_mean"),
                            lpips_std=0,
                            n_samples=e2e_test.get("n_samples", 50),
                        ),
                        sample_results=[],
                    ),
                )

    # Generate reports
    return generator.generate_full_report("h1_diagnosis_report")


def generate_combined_report(output_dir: Path) -> Dict[str, Path]:
    """Generate combined report with all experiments."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # View Ablation
    view_config = EXPERIMENTS["view_ablation"]
    view_results = generate_view_ablation_report(
        output_dir,
        Path(view_config["checkpoint_base"]),
        Path(view_config["dataset_root"]),
        view_config["variants"],
    )
    results["view_ablation"] = view_results

    # H1 Diagnosis
    h1_config = EXPERIMENTS["h1_diagnosis"]
    h1_results = generate_h1_diagnosis_report(
        output_dir,
        h1_config["datasets"],
    )
    results["h1_diagnosis"] = h1_results

    # Summary report
    summary = {
        "generated": datetime.now().isoformat(),
        "reports": {k: {kk: str(vv) for kk, vv in v.items()} for k, v in results.items()},
    }

    summary_path = output_dir / "report_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("Report Generation Complete")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"\nGenerated reports:")
    for name, paths in results.items():
        print(f"\n  {name}:")
        for fmt, path in paths.items():
            print(f"    {fmt}: {path}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Generate comprehensive experiment analysis reports"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--experiments",
        type=str,
        nargs="+",
        choices=["view_ablation", "h1_diagnosis", "all"],
        default=["all"],
        help="Which experiments to include in report",
    )
    parser.add_argument(
        "--generate_vis",
        action="store_true",
        help="Also generate visualization images",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    if "all" in args.experiments:
        results = generate_combined_report(output_dir)
    else:
        results = {}
        for exp_name in args.experiments:
            if exp_name == "view_ablation":
                config = EXPERIMENTS["view_ablation"]
                results[exp_name] = generate_view_ablation_report(
                    output_dir,
                    Path(config["checkpoint_base"]),
                    Path(config["dataset_root"]),
                    config["variants"],
                )
            elif exp_name == "h1_diagnosis":
                config = EXPERIMENTS["h1_diagnosis"]
                results[exp_name] = generate_h1_diagnosis_report(
                    output_dir,
                    config["datasets"],
                )

    if args.generate_vis:
        print("\nGenerating visualizations...")
        vis_dir = output_dir / "images"
        vis_dir.mkdir(exist_ok=True)

        # View ablation visualizations
        view_config = EXPERIMENTS["view_ablation"]
        vis = VisualizationGenerator(vis_dir)

        checkpoint_dirs = {
            name: Path(view_config["checkpoint_base"]) / name
            for name in view_config["variants"]
        }

        sample_ids = ["000000", "000100", "000500", "001000", "002000"]
        vis.create_view_ablation_grid(
            checkpoint_dirs,
            Path(view_config["dataset_root"]),
            sample_ids,
            filename="view_ablation_comparison.png"
        )
        print(f"  Saved: {vis_dir / 'view_ablation_comparison.png'}")


if __name__ == "__main__":
    main()
