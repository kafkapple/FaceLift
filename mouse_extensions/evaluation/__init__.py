"""
FaceLift Evaluation Module.

Provides metrics computation and report generation for model evaluation.
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
    generate_h1_comparison_report,
)

__all__ = [
    'MetricsComputer',
    'MetricResult',
    'AggregatedMetrics',
    'compute_metrics_for_experiment',
    'ReportGenerator',
    'ExperimentResult',
    'generate_h1_comparison_report',
]
