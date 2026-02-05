"""
Report generation module for FaceLift evaluation.

Generates markdown reports and JSON summaries from evaluation metrics.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union
from datetime import datetime
from dataclasses import dataclass

from .metrics import MetricsComputer, MetricResult, AggregatedMetrics


@dataclass
class ExperimentResult:
    """Container for experiment evaluation results."""
    name: str
    split: str
    metrics: AggregatedMetrics
    sample_results: List[MetricResult]
    config: Optional[Dict] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ReportGenerator:
    """
    Generate evaluation reports in markdown and JSON formats.

    Usage:
        generator = ReportGenerator(output_dir="outputs/eval/reports")

        # Single experiment
        generator.add_experiment(result)

        # Generate reports
        generator.generate_markdown_report()
        generator.generate_json_summary()
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[ExperimentResult] = []

    def add_experiment(self, result: ExperimentResult) -> None:
        """Add an experiment result to the report."""
        self.experiments.append(result)

    def add_from_metrics(
        self,
        name: str,
        split: str,
        metrics: AggregatedMetrics,
        sample_results: List[MetricResult],
        config: Optional[Dict] = None,
    ) -> None:
        """Convenience method to add experiment from raw metrics."""
        result = ExperimentResult(
            name=name,
            split=split,
            metrics=metrics,
            sample_results=sample_results,
            config=config,
        )
        self.add_experiment(result)

    def generate_markdown_report(
        self,
        filename: str = "evaluation_report.md",
        include_samples: bool = False,
    ) -> Path:
        """
        Generate a markdown evaluation report.

        Args:
            filename: Output filename
            include_samples: Whether to include per-sample details

        Returns:
            Path to generated report
        """
        report_path = self.output_dir / filename

        lines = [
            "# Evaluation Report",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "---",
            "",
        ]

        # Summary table
        lines.extend([
            "## Summary",
            "",
            "| Experiment | Split | PSNR | SSIM | LPIPS | N |",
            "|------------|-------|------|------|-------|---|",
        ])

        for exp in self.experiments:
            m = exp.metrics
            lpips_str = f"{m.lpips_mean:.4f}" if m.lpips_mean is not None else "-"
            lines.append(
                f"| {exp.name} | {exp.split} | "
                f"{m.psnr_mean:.2f}±{m.psnr_std:.2f} | "
                f"{m.ssim_mean:.4f}±{m.ssim_std:.4f} | "
                f"{lpips_str} | {m.n_samples} |"
            )

        lines.extend(["", "---", ""])

        # Detailed sections for each experiment
        for exp in self.experiments:
            lines.extend([
                f"## {exp.name}",
                "",
                f"- **Split**: {exp.split}",
                f"- **Samples**: {exp.metrics.n_samples}",
                f"- **Timestamp**: {exp.timestamp}",
                "",
                "### Metrics",
                "",
                f"| Metric | Mean | Std |",
                f"|--------|------|-----|",
                f"| PSNR | {exp.metrics.psnr_mean:.2f} | {exp.metrics.psnr_std:.2f} |",
                f"| SSIM | {exp.metrics.ssim_mean:.4f} | {exp.metrics.ssim_std:.4f} |",
            ])

            if exp.metrics.lpips_mean is not None:
                lines.append(
                    f"| LPIPS | {exp.metrics.lpips_mean:.4f} | {exp.metrics.lpips_std:.4f} |"
                )

            lines.append("")

            # Config section
            if exp.config:
                lines.extend([
                    "### Configuration",
                    "",
                    "```yaml",
                ])
                for key, val in exp.config.items():
                    lines.append(f"{key}: {val}")
                lines.extend(["```", ""])

            # Per-sample results
            if include_samples and exp.sample_results:
                lines.extend([
                    "### Per-Sample Results",
                    "",
                    "| Sample | PSNR | SSIM | LPIPS |",
                    "|--------|------|------|-------|",
                ])
                for r in exp.sample_results[:50]:  # Limit to 50 samples
                    lpips_str = f"{r.lpips:.4f}" if r.lpips is not None else "-"
                    lines.append(
                        f"| {r.sample_id} | {r.psnr:.2f} | {r.ssim:.4f} | {lpips_str} |"
                    )
                if len(exp.sample_results) > 50:
                    lines.append(f"| ... | ({len(exp.sample_results) - 50} more) | | |")
                lines.append("")

            lines.extend(["---", ""])

        # Write report
        report_path.write_text("\n".join(lines))
        return report_path

    def generate_json_summary(
        self,
        filename: str = "evaluation_summary.json",
    ) -> Path:
        """
        Generate a JSON summary of all experiments.

        Args:
            filename: Output filename

        Returns:
            Path to generated JSON
        """
        json_path = self.output_dir / filename

        summary = {
            "generated": datetime.now().isoformat(),
            "experiments": []
        }

        for exp in self.experiments:
            exp_data = {
                "name": exp.name,
                "split": exp.split,
                "timestamp": exp.timestamp,
                "metrics": exp.metrics.to_dict(),
                "config": exp.config,
            }
            summary["experiments"].append(exp_data)

        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return json_path

    def generate_comparison_table(
        self,
        group_by: str = "split",
    ) -> str:
        """
        Generate a comparison table grouped by split or experiment type.

        Args:
            group_by: 'split' or 'type'

        Returns:
            Markdown table string
        """
        if group_by == "split":
            # Group by train/test
            train_exps = [e for e in self.experiments if "train" in e.split.lower()]
            test_exps = [e for e in self.experiments if "test" in e.split.lower()]

            lines = [
                "## Train vs Test Comparison",
                "",
                "### Train Split",
                "",
                "| Experiment | PSNR | SSIM |",
                "|------------|------|------|",
            ]
            for e in train_exps:
                lines.append(f"| {e.name} | {e.metrics.psnr_mean:.2f} | {e.metrics.ssim_mean:.4f} |")

            lines.extend([
                "",
                "### Test Split",
                "",
                "| Experiment | PSNR | SSIM |",
                "|------------|------|------|",
            ])
            for e in test_exps:
                lines.append(f"| {e.name} | {e.metrics.psnr_mean:.2f} | {e.metrics.ssim_mean:.4f} |")

        else:
            # Group by type (GS-LRM vs E2E)
            gslrm_exps = [e for e in self.experiments if "gslrm" in e.name.lower()]
            e2e_exps = [e for e in self.experiments if "e2e" in e.name.lower()]

            lines = [
                "## GS-LRM vs E2E Comparison",
                "",
                "### GS-LRM Only",
                "",
                "| Experiment | Split | PSNR | SSIM |",
                "|------------|-------|------|------|",
            ]
            for e in gslrm_exps:
                lines.append(f"| {e.name} | {e.split} | {e.metrics.psnr_mean:.2f} | {e.metrics.ssim_mean:.4f} |")

            lines.extend([
                "",
                "### End-to-End (MVDiffusion + GS-LRM)",
                "",
                "| Experiment | Split | PSNR | SSIM |",
                "|------------|-------|------|------|",
            ])
            for e in e2e_exps:
                lines.append(f"| {e.name} | {e.split} | {e.metrics.psnr_mean:.2f} | {e.metrics.ssim_mean:.4f} |")

        return "\n".join(lines)


def generate_h1_comparison_report(
    experiments_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    dataset_name: str = "M5t2",
    compare_all_views: bool = False,
    compute_lpips: bool = True,
) -> Dict[str, Path]:
    """
    Generate complete H1 experiment comparison report.

    Args:
        experiments_dir: Directory containing H1 experiment outputs
        dataset_root: Dataset root for GT lookup
        output_dir: Output directory for reports
        dataset_name: Dataset name for report title
        compare_all_views: If True, compare all 6 views; otherwise only view 0
        compute_lpips: Whether to compute LPIPS (slower)

    Returns:
        Dict with paths to generated reports
    """
    computer = MetricsComputer(compute_lpips=compute_lpips)
    generator = ReportGenerator(output_dir)

    # H1 experiment mapping
    h1_experiments = [
        ("h1a_gslrm_train", "train"),
        ("h1b_gslrm_test", "test"),
        ("h1c_e2e_train", "train"),
        ("h1d_e2e_test", "test"),
    ]

    views = [0, 1, 2, 3, 4, 5] if compare_all_views else [0]

    for exp_name, split in h1_experiments:
        exp_dir = experiments_dir / exp_name
        if not exp_dir.exists():
            print(f"Skipping {exp_name}: directory not found")
            continue

        print(f"Computing metrics for {exp_name} ({len(views)} views)...")
        results = computer.compute_for_h1_experiment(
            exp_dir, dataset_root, split, views_to_compare=views
        )

        if results:
            metrics = computer.aggregate(results)
            generator.add_from_metrics(
                name=exp_name,
                split=split,
                metrics=metrics,
                sample_results=results,
                config={
                    "dataset": dataset_name,
                    "type": "gslrm" if "gslrm" in exp_name else "e2e",
                    "views_compared": views,
                }
            )
            print(f"  {exp_name}: PSNR={metrics.psnr_mean:.2f}±{metrics.psnr_std:.2f}, "
                  f"SSIM={metrics.ssim_mean:.4f}, N={metrics.n_samples}")
        else:
            print(f"Warning: No results for {exp_name}")

    # Generate reports
    md_path = generator.generate_markdown_report(
        filename=f"h1_evaluation_{dataset_name}.md",
        include_samples=True
    )
    json_path = generator.generate_json_summary(
        filename=f"h1_evaluation_{dataset_name}.json"
    )

    # Generate comparison tables
    comparison_md = generator.generate_comparison_table(group_by="type")
    comparison_path = output_dir / f"h1_comparison_{dataset_name}.md"
    comparison_path.write_text(comparison_md)

    print(f"\nReports generated:")
    print(f"  Markdown: {md_path}")
    print(f"  JSON: {json_path}")
    print(f"  Comparison: {comparison_path}")

    return {
        "markdown": md_path,
        "json": json_path,
        "comparison": comparison_path,
    }
