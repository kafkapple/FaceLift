"""
FaceLift Mouse Preprocessing Report Generators

Modules:
- unified_report: Main report generator combining all analysis
- modules/: Modular components (theory, camera_viz, pp_analysis, dataset_comparison)
- templates/: HTML templates

Usage:
    python -m mouse_extensions.reports.unified_report --output-dir ./reports
    
    # Programmatic usage
    from mouse_extensions.reports import UnifiedReportGenerator
    generator = UnifiedReportGenerator(output_dir='./reports')
    generator.generate_report()
"""

from .unified_report import UnifiedReportGenerator
from .modules import (
    TheoryModule,
    CameraVisualizationModule,
    PPAnalysisModule,
    DatasetComparisonModule,
)

__all__ = [
    'UnifiedReportGenerator',
    'TheoryModule',
    'CameraVisualizationModule',
    'PPAnalysisModule',
    'DatasetComparisonModule',
]
