"""
FaceLift Evaluation Module.

Provides metrics computation, report generation, and visualization for model evaluation.

Usage:
    from mouse_extensions.evaluation import (
        MetricsComputer,
        ExperimentReportGenerator,
        VisualizationGenerator,
    )

    # Compute metrics
    computer = MetricsComputer(compute_lpips=True)
    results = computer.compute_from_dirs(render_dir, gt_dir)
    metrics = computer.aggregate(results)

    # Generate report
    generator = ExperimentReportGenerator(output_dir="outputs/reports")
    generator.add_full_experiment(name="exp1", ...)
    generator.generate_full_report()

    # Generate visualizations
    vis = VisualizationGenerator(output_dir="outputs/reports/images")
    vis.create_gt_vs_pred_grid(gt_images, pred_images, sample_id="001")
"""

from .metrics import (
    MetricsComputer,
    MetricResult,
    AggregatedMetrics,
    compute_metrics_for_experiment,
)
from .report_generator import (
    ReportGenerator,
    ExperimentResult,
    ExperimentReportGenerator,
    FullExperimentResult,
    ExperimentHypothesis,
    ExperimentCondition,
    SplitMetrics,
    METRIC_INFO,
    generate_h1_comparison_report,
    load_experiment_from_checkpoint,
)
from .visualization import (
    VisualizationGenerator,
    generate_report_visualizations,
)

__all__ = [
    # Metrics
    'MetricsComputer',
    'MetricResult',
    'AggregatedMetrics',
    'compute_metrics_for_experiment',
    # Report Generator (Legacy)
    'ReportGenerator',
    'ExperimentResult',
    'generate_h1_comparison_report',
    # Report Generator (Enhanced)
    'ExperimentReportGenerator',
    'FullExperimentResult',
    'ExperimentHypothesis',
    'ExperimentCondition',
    'SplitMetrics',
    'METRIC_INFO',
    'load_experiment_from_checkpoint',
    # Visualization
    'VisualizationGenerator',
    'generate_report_visualizations',
]
