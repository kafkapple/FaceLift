"""
Report Generator Configuration Schema
=====================================

Defines the structure for dataset preprocessing reports.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path

@dataclass
class DatasetConfig:
    """Configuration for a single dataset."""
    name: str
    path: str
    version: str
    scale_mode: str = "fx_only"  # fx_only, individual, average
    description: str = ""
    status: str = "active"  # active, deprecated
    
@dataclass
class FormulaSection:
    """Mathematical formula section."""
    title: str
    latex: str
    explanation: str
    
@dataclass
class VerificationResult:
    """Verification result for a dataset."""
    dataset: str
    fx_mean: float
    fx_std: float
    fy_mean: float
    fy_std: float
    cx_mean: float
    cy_mean: float
    ray_error_mean: float
    ray_error_max: float
    scale_x_mean: float = 0.0
    scale_y_mean: float = 0.0
    
@dataclass
class ReportConfig:
    """Full report configuration."""
    title: str
    version: str
    date: str
    datasets: List[DatasetConfig]
    
    # Section toggles
    include_overview: bool = True
    include_formulas: bool = True
    include_verification: bool = True
    include_comparison: bool = True
    include_recommendations: bool = True
    
    # Output settings
    output_dir: str = "reports"
    output_format: str = "html"  # html, markdown, both
    generate_figures: bool = True
