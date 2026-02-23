"""
Report generation module for FaceLift evaluation.

Generates markdown reports and JSON summaries from evaluation metrics.

Improvements over v1:
- train/val/test 3-way split support
- Metric units and descriptions
- Experiment hypothesis and conditions sections
- Per-view visualization support
- CLI interface

Usage:
    # From Python
    from mouse_extensions.evaluation.report_generator import ExperimentReportGenerator

    generator = ExperimentReportGenerator(output_dir="outputs/reports")
    generator.add_experiment(experiment_config)
    generator.generate_full_report()

    # From CLI
    python -m mouse_extensions.evaluation.report_generator --checkpoint_dir /path/to/checkpoints
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .metrics import MetricsComputer, MetricResult, AggregatedMetrics


class MetricDirection(Enum):
    """Indicates whether higher or lower is better."""
    HIGHER_BETTER = "↑"
    LOWER_BETTER = "↓"


# Metric definitions with units and descriptions
METRIC_INFO = {
    "psnr": {
        "name": "PSNR",
        "unit": "dB",
        "direction": MetricDirection.HIGHER_BETTER,
        "description": "Peak Signal-to-Noise Ratio. Measures pixel-level reconstruction quality.",
        "good_threshold": 25.0,
        "bad_threshold": 15.0,
    },
    "ssim": {
        "name": "SSIM",
        "unit": "0-1",
        "direction": MetricDirection.HIGHER_BETTER,
        "description": "Structural Similarity Index. Measures structural similarity.",
        "good_threshold": 0.9,
        "bad_threshold": 0.7,
    },
    "lpips": {
        "name": "LPIPS",
        "unit": "0-1",
        "direction": MetricDirection.LOWER_BETTER,
        "description": "Learned Perceptual Image Patch Similarity. Measures perceptual distance.",
        "good_threshold": 0.1,
        "bad_threshold": 0.3,
    },
}


@dataclass
class ExperimentHypothesis:
    """Hypothesis being tested in an experiment."""
    name: str
    description: str
    experimental_group: str
    control_group: Optional[str] = None
    expected_outcome: Optional[str] = None


@dataclass
class ExperimentCondition:
    """Experimental conditions and settings."""
    n_input_views: int
    n_target_views: int
    dataset: str
    split_ratio: str  # e.g., "80:10:10"
    checkpoint_step: Optional[int] = None
    training_steps: Optional[int] = None
    batch_size: Optional[int] = None
    learning_rate: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SplitMetrics:
    """Metrics for a single split (train/val/test)."""
    split: str
    metrics: AggregatedMetrics
    sample_results: List[MetricResult]
    n_samples: int = 0

    def __post_init__(self):
        if self.n_samples == 0:
            self.n_samples = self.metrics.n_samples


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


@dataclass
class FullExperimentResult:
    """Complete experiment result with train/val/test splits."""
    name: str
    hypothesis: Optional[ExperimentHypothesis] = None
    condition: Optional[ExperimentCondition] = None
    train_metrics: Optional[SplitMetrics] = None
    val_metrics: Optional[SplitMetrics] = None
    test_metrics: Optional[SplitMetrics] = None
    timestamp: str = ""
    wandb_run_id: Optional[str] = None
    checkpoint_path: Optional[str] = None
    notes: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_primary_metric(self, metric: str = "psnr", split: str = "val") -> Optional[float]:
        """Get primary metric value for comparison."""
        split_metrics = getattr(self, f"{split}_metrics", None)
        if split_metrics is None:
            return None
        return getattr(split_metrics.metrics, f"{metric}_mean", None)


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
                "### End-to-End (Multi-view Diffusion + GS-LRM)",
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


class ExperimentReportGenerator:
    """
    Enhanced report generator with train/val/test support and detailed metrics.

    Features:
    - 3-way split (train/val/test) support
    - Metric units and descriptions
    - Experiment hypothesis and conditions
    - Comparison tables
    - JSON and Markdown output

    Usage:
        generator = ExperimentReportGenerator(output_dir="outputs/reports")

        # Add experiment with full details
        generator.add_full_experiment(
            name="view_ablation_3view",
            hypothesis=ExperimentHypothesis(
                name="Fewer views may improve quality",
                description="Test if 3 input views outperform 4+ views",
                experimental_group="3-view input",
                control_group="6-view input (baseline)"
            ),
            condition=ExperimentCondition(
                n_input_views=3,
                n_target_views=6,
                dataset="M5t2",
                split_ratio="80:10:10"
            ),
            train_metrics=...,
            val_metrics=...,
            test_metrics=...
        )

        # Generate reports
        generator.generate_full_report("view_ablation_report")
    """

    def __init__(self, output_dir: Union[str, Path]):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.experiments: List[FullExperimentResult] = []
        self.title: str = "Experiment Report"
        self.description: str = ""

    def set_report_info(self, title: str, description: str = "") -> None:
        """Set report title and description."""
        self.title = title
        self.description = description

    def add_full_experiment(
        self,
        name: str,
        hypothesis: Optional[ExperimentHypothesis] = None,
        condition: Optional[ExperimentCondition] = None,
        train_metrics: Optional[SplitMetrics] = None,
        val_metrics: Optional[SplitMetrics] = None,
        test_metrics: Optional[SplitMetrics] = None,
        wandb_run_id: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        notes: Optional[str] = None,
    ) -> None:
        """Add a complete experiment result."""
        result = FullExperimentResult(
            name=name,
            hypothesis=hypothesis,
            condition=condition,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            test_metrics=test_metrics,
            wandb_run_id=wandb_run_id,
            checkpoint_path=checkpoint_path,
            notes=notes,
        )
        self.experiments.append(result)

    def _format_metric_value(
        self,
        metric_name: str,
        mean: float,
        std: Optional[float] = None,
        include_unit: bool = True,
    ) -> str:
        """Format metric value with optional std and unit."""
        info = METRIC_INFO.get(metric_name, {})
        unit = info.get("unit", "")

        if std is not None:
            value_str = f"{mean:.2f}±{std:.2f}"
        else:
            value_str = f"{mean:.2f}"

        if include_unit and unit:
            return f"{value_str} {unit}"
        return value_str

    def _get_metric_indicator(self, metric_name: str, value: float) -> str:
        """Get quality indicator emoji based on threshold."""
        info = METRIC_INFO.get(metric_name, {})
        good = info.get("good_threshold")
        bad = info.get("bad_threshold")
        direction = info.get("direction", MetricDirection.HIGHER_BETTER)

        if good is None or bad is None:
            return ""

        if direction == MetricDirection.HIGHER_BETTER:
            if value >= good:
                return "🟢"
            elif value <= bad:
                return "🔴"
            return "🟡"
        else:  # LOWER_BETTER
            if value <= good:
                return "🟢"
            elif value >= bad:
                return "🔴"
            return "🟡"

    def generate_metric_legend(self) -> List[str]:
        """Generate metric description legend."""
        lines = [
            "## Metric Definitions",
            "",
            "| Metric | Unit | Direction | Description |",
            "|--------|------|-----------|-------------|",
        ]
        for key, info in METRIC_INFO.items():
            direction = info["direction"].value
            lines.append(
                f"| **{info['name']}** | {info['unit']} | {direction} better | "
                f"{info['description']} |"
            )
        lines.extend([
            "",
            "**Quality Indicators**: 🟢 Good | 🟡 Moderate | 🔴 Poor",
            "",
        ])
        return lines

    def generate_summary_table(self, primary_split: str = "val") -> List[str]:
        """Generate summary comparison table across all experiments."""
        lines = [
            "## Summary",
            "",
            f"Primary evaluation split: **{primary_split}**",
            "",
            "| Experiment | Input Views | PSNR (dB) | SSIM | LPIPS | Status |",
            "|------------|-------------|-----------|------|-------|--------|",
        ]

        for exp in self.experiments:
            split_metrics = getattr(exp, f"{primary_split}_metrics", None)
            if split_metrics is None:
                continue

            m = split_metrics.metrics
            n_views = exp.condition.n_input_views if exp.condition else "?"

            psnr_str = self._format_metric_value("psnr", m.psnr_mean, m.psnr_std, False)
            ssim_str = f"{m.ssim_mean:.4f}"
            lpips_str = f"{m.lpips_mean:.4f}" if m.lpips_mean else "-"

            indicator = self._get_metric_indicator("psnr", m.psnr_mean)

            lines.append(
                f"| {exp.name} | {n_views} | {psnr_str} | {ssim_str} | {lpips_str} | {indicator} |"
            )

        lines.append("")
        return lines

    def generate_split_comparison_table(self) -> List[str]:
        """Generate table comparing train/val/test across experiments."""
        lines = [
            "## Train / Val / Test Comparison",
            "",
            "| Experiment | Split | PSNR (dB) | SSIM | LPIPS | N |",
            "|------------|-------|-----------|------|-------|---|",
        ]

        for exp in self.experiments:
            for split in ["train", "val", "test"]:
                split_metrics = getattr(exp, f"{split}_metrics", None)
                if split_metrics is None:
                    continue

                m = split_metrics.metrics
                psnr_str = self._format_metric_value("psnr", m.psnr_mean, m.psnr_std, False)
                ssim_str = f"{m.ssim_mean:.4f}"
                lpips_str = f"{m.lpips_mean:.4f}" if m.lpips_mean else "-"

                lines.append(
                    f"| {exp.name} | {split} | {psnr_str} | {ssim_str} | {lpips_str} | {m.n_samples} |"
                )

        lines.append("")
        return lines

    def generate_experiment_details(self, exp: FullExperimentResult) -> List[str]:
        """Generate detailed section for a single experiment."""
        lines = [
            f"## {exp.name}",
            "",
        ]

        # Hypothesis
        if exp.hypothesis:
            lines.extend([
                "### Hypothesis",
                "",
                f"**{exp.hypothesis.name}**",
                "",
                f"{exp.hypothesis.description}",
                "",
                f"- **Experimental group**: {exp.hypothesis.experimental_group}",
            ])
            if exp.hypothesis.control_group:
                lines.append(f"- **Control group**: {exp.hypothesis.control_group}")
            if exp.hypothesis.expected_outcome:
                lines.append(f"- **Expected outcome**: {exp.hypothesis.expected_outcome}")
            lines.append("")

        # Conditions
        if exp.condition:
            c = exp.condition
            lines.extend([
                "### Experimental Conditions",
                "",
                f"| Setting | Value |",
                f"|---------|-------|",
                f"| Input views | {c.n_input_views} |",
                f"| Target views | {c.n_target_views} |",
                f"| Dataset | {c.dataset} |",
                f"| Split ratio | {c.split_ratio} |",
            ])
            if c.checkpoint_step:
                lines.append(f"| Checkpoint step | {c.checkpoint_step} |")
            if c.training_steps:
                lines.append(f"| Training steps | {c.training_steps} |")
            if c.batch_size:
                lines.append(f"| Batch size | {c.batch_size} |")
            if c.learning_rate:
                lines.append(f"| Learning rate | {c.learning_rate} |")
            for key, val in c.extra.items():
                lines.append(f"| {key} | {val} |")
            lines.append("")

        # Metrics table
        lines.extend([
            "### Quantitative Results",
            "",
            "| Split | PSNR (dB) | SSIM | LPIPS | N |",
            "|-------|-----------|------|-------|---|",
        ])

        for split in ["train", "val", "test"]:
            split_metrics = getattr(exp, f"{split}_metrics", None)
            if split_metrics is None:
                continue

            m = split_metrics.metrics
            indicator = self._get_metric_indicator("psnr", m.psnr_mean)
            psnr_str = f"{m.psnr_mean:.2f}±{m.psnr_std:.2f}"
            ssim_str = f"{m.ssim_mean:.4f}±{m.ssim_std:.4f}"
            lpips_str = f"{m.lpips_mean:.4f}" if m.lpips_mean else "-"

            lines.append(
                f"| {split} | {psnr_str} {indicator} | {ssim_str} | {lpips_str} | {m.n_samples} |"
            )

        lines.append("")

        # Metadata
        if exp.wandb_run_id or exp.checkpoint_path:
            lines.extend(["### Metadata", ""])
            if exp.wandb_run_id:
                lines.append(f"- **WandB Run ID**: {exp.wandb_run_id}")
            if exp.checkpoint_path:
                lines.append(f"- **Checkpoint**: `{exp.checkpoint_path}`")
            lines.append(f"- **Timestamp**: {exp.timestamp}")
            lines.append("")

        # Notes
        if exp.notes:
            lines.extend([
                "### Notes",
                "",
                exp.notes,
                "",
            ])

        lines.extend(["---", ""])
        return lines

    def generate_markdown_report(
        self,
        filename: str = "experiment_report.md",
        include_legend: bool = True,
        include_details: bool = True,
        primary_split: str = "val",
    ) -> Path:
        """Generate full markdown report."""
        report_path = self.output_dir / filename

        lines = [
            f"# {self.title}",
            "",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        if self.description:
            lines.extend([self.description, ""])

        lines.extend(["---", ""])

        # Metric legend
        if include_legend:
            lines.extend(self.generate_metric_legend())

        # Summary table
        lines.extend(self.generate_summary_table(primary_split))

        # Split comparison
        lines.extend(self.generate_split_comparison_table())

        # Detailed sections
        if include_details:
            for exp in self.experiments:
                lines.extend(self.generate_experiment_details(exp))

        report_path.write_text("\n".join(lines))
        return report_path

    def generate_json_report(self, filename: str = "experiment_report.json") -> Path:
        """Generate JSON report with full data."""
        json_path = self.output_dir / filename

        def metrics_to_dict(m: Optional[SplitMetrics]) -> Optional[Dict]:
            if m is None:
                return None
            return {
                "split": m.split,
                "n_samples": m.n_samples,
                "psnr": {"mean": m.metrics.psnr_mean, "std": m.metrics.psnr_std},
                "ssim": {"mean": m.metrics.ssim_mean, "std": m.metrics.ssim_std},
                "lpips": {"mean": m.metrics.lpips_mean, "std": m.metrics.lpips_std}
                if m.metrics.lpips_mean else None,
            }

        data = {
            "title": self.title,
            "description": self.description,
            "generated": datetime.now().isoformat(),
            "metric_definitions": {
                k: {
                    "name": v["name"],
                    "unit": v["unit"],
                    "direction": v["direction"].value,
                    "description": v["description"],
                }
                for k, v in METRIC_INFO.items()
            },
            "experiments": [],
        }

        for exp in self.experiments:
            exp_data = {
                "name": exp.name,
                "timestamp": exp.timestamp,
                "hypothesis": {
                    "name": exp.hypothesis.name,
                    "description": exp.hypothesis.description,
                    "experimental_group": exp.hypothesis.experimental_group,
                    "control_group": exp.hypothesis.control_group,
                }
                if exp.hypothesis else None,
                "condition": {
                    "n_input_views": exp.condition.n_input_views,
                    "n_target_views": exp.condition.n_target_views,
                    "dataset": exp.condition.dataset,
                    "split_ratio": exp.condition.split_ratio,
                    **exp.condition.extra,
                }
                if exp.condition else None,
                "metrics": {
                    "train": metrics_to_dict(exp.train_metrics),
                    "val": metrics_to_dict(exp.val_metrics),
                    "test": metrics_to_dict(exp.test_metrics),
                },
                "wandb_run_id": exp.wandb_run_id,
                "checkpoint_path": exp.checkpoint_path,
                "notes": exp.notes,
            }
            data["experiments"].append(exp_data)

        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        return json_path

    def generate_full_report(self, base_filename: str = "report") -> Dict[str, Path]:
        """Generate both markdown and JSON reports."""
        md_path = self.generate_markdown_report(f"{base_filename}.md")
        json_path = self.generate_json_report(f"{base_filename}.json")

        print(f"Reports generated:")
        print(f"  Markdown: {md_path}")
        print(f"  JSON: {json_path}")

        return {"markdown": md_path, "json": json_path}


def load_experiment_from_checkpoint(
    checkpoint_dir: Union[str, Path],
    dataset_root: Union[str, Path],
    experiment_name: Optional[str] = None,
    compute_metrics: bool = True,
    splits: List[str] = None,
) -> FullExperimentResult:
    """
    Load experiment result from a checkpoint directory.

    Args:
        checkpoint_dir: Directory containing checkpoints and config
        dataset_root: Dataset root for metric computation
        experiment_name: Name for the experiment (default: dir name)
        compute_metrics: Whether to compute metrics from renders
        splits: Which splits to evaluate (default: ["val", "test"])

    Returns:
        FullExperimentResult with metrics
    """
    checkpoint_dir = Path(checkpoint_dir)
    dataset_root = Path(dataset_root)

    if experiment_name is None:
        experiment_name = checkpoint_dir.name

    if splits is None:
        splits = ["val", "test"]

    # Load config
    config_path = checkpoint_dir / "config.yaml"
    config = {}
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f)

    # Load WandB run ID
    wandb_id_path = checkpoint_dir / "wandb_run_id.txt"
    wandb_run_id = None
    if wandb_id_path.exists():
        wandb_run_id = wandb_id_path.read_text().strip()

    # Load best checkpoint info
    best_psnr_path = checkpoint_dir / "best_psnr.json"
    best_info = {}
    if best_psnr_path.exists():
        with open(best_psnr_path) as f:
            best_info = json.load(f)

    # Extract condition from config
    condition = None
    if config:
        dataset_cfg = config.get("dataset", {})
        training_cfg = config.get("training", {})

        condition = ExperimentCondition(
            n_input_views=dataset_cfg.get("n_views_input", 6),
            n_target_views=dataset_cfg.get("n_views_target", 6),
            dataset=dataset_cfg.get("dataset", "unknown"),
            split_ratio="80:10:10",  # Default
            checkpoint_step=best_info.get("step"),
            training_steps=training_cfg.get("runtime", {}).get("num_steps"),
            batch_size=training_cfg.get("runtime", {}).get("batch_size"),
            learning_rate=training_cfg.get("schedule", {}).get("base_lr"),
        )

    # Compute metrics if requested
    train_metrics = val_metrics = test_metrics = None

    if compute_metrics:
        computer = MetricsComputer(compute_lpips=True)
        # This would need to be implemented based on how renders are stored
        # For now, we just return the structure

    result = FullExperimentResult(
        name=experiment_name,
        condition=condition,
        train_metrics=train_metrics,
        val_metrics=val_metrics,
        test_metrics=test_metrics,
        wandb_run_id=wandb_run_id,
        checkpoint_path=str(checkpoint_dir / "best_psnr.pt"),
    )

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment reports")
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        help="Checkpoint directory or parent directory containing multiple experiments",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/joon/data/preprocessed/FaceLift_mouse/M5",
        help="Dataset root for metric computation",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reports",
        help="Output directory for reports",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="FaceLift Experiment Report",
        help="Report title",
    )

    args = parser.parse_args()

    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
        output_dir = Path(args.output_dir)

        generator = ExperimentReportGenerator(output_dir)
        generator.set_report_info(args.title)

        # Check if single experiment or multiple
        if (checkpoint_dir / "config.yaml").exists():
            # Single experiment
            result = load_experiment_from_checkpoint(
                checkpoint_dir, args.dataset_root
            )
            generator.experiments.append(result)
        else:
            # Multiple experiments
            for exp_dir in sorted(checkpoint_dir.iterdir()):
                if exp_dir.is_dir() and (exp_dir / "config.yaml").exists():
                    result = load_experiment_from_checkpoint(
                        exp_dir, args.dataset_root
                    )
                    generator.experiments.append(result)

        generator.generate_full_report("experiment_report")
    else:
        print("Usage: python -m mouse_extensions.evaluation.report_generator --checkpoint_dir /path/to/checkpoints")
        print("\nExample:")
        print("  python -m mouse_extensions.evaluation.report_generator \\")
        print("    --checkpoint_dir /node_data/joon/checkpoints/FaceLift/gslrm/M5t2_view_ablation \\")
        print("    --output_dir outputs/reports \\")
        print("    --title 'View Ablation Experiment Report'")
