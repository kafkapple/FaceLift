"""
Dataset Comparison Module - Compare preprocessing methods and datasets.

Includes M-Series analysis, hypothesis tracking, and recommendation tables.
"""

import json
import pickle
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class DatasetInfo:
    """Information about a preprocessed dataset."""
    name: str
    alias: str  # M1, M2, M3_1, etc.
    paradigm: str
    transform: str
    fx_mean: float
    fx_std: float
    fy_mean: float
    fy_std: float
    cx_mean: float
    cx_std: float
    cy_mean: float
    cy_std: float
    distance: float
    zoom: str  # None, global, per_sample
    zoom_center_mode: str  # image, object
    pp_fixed: bool
    skew_corrected: bool
    ray_error_deg: float
    coverage_pct: float
    clipping_pct: float
    status: str  # active, deprecated, hypothesis
    description: str


# Preset definitions with M-series info
DATASET_SPECS = {
    # ==========================================================================
    # BASELINE (No Zoom)
    # ==========================================================================
    "D7.1": DatasetInfo(
        name="D7.1", alias="M1",
        paradigm="pp_centered_shift", transform="affine",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="None", zoom_center_mode="N/A",
        pp_fixed=True, skew_corrected=False, ray_error_deg=0.0,
        coverage_pct=3.0, clipping_pct=0.0, status="active",
        description="Affine transform baseline. Stable, validated."
    ),
    "D8": DatasetInfo(
        name="D8", alias="M2",
        paradigm="precision_homography", transform="homography",
        fx_mean=548.99, fx_std=0.0, fy_mean=548.99, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="None", zoom_center_mode="N/A",
        pp_fixed=True, skew_corrected=True, ray_error_deg=0.0,
        coverage_pct=3.0, clipping_pct=0.0, status="active",
        description="Homography + skew correction. Precision baseline."
    ),
    
    # ==========================================================================
    # PRODUCTION (Adaptive Zoom, PP=256)
    # ==========================================================================
    "M3_1": DatasetInfo(
        name="M3_1", alias="M3_1",
        paradigm="precision_homography", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="global", zoom_center_mode="image",
        pp_fixed=True, skew_corrected=True, ray_error_deg=0.0,
        coverage_pct=4.5, clipping_pct=0.0, status="active",
        description="Global zoom + center-aligned crop. Safe choice."
    ),
    "M3_2": DatasetInfo(
        name="M3_2", alias="M3_2",
        paradigm="precision_homography", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="per_sample", zoom_center_mode="image",
        pp_fixed=True, skew_corrected=True, ray_error_deg=0.0,
        coverage_pct=5.0, clipping_pct=6.5, status="recommended",
        description="Per-sample zoom + center-aligned crop. Recommended."
    ),
    
    # ==========================================================================
    # HYPOTHESIS TESTING
    # ==========================================================================
    "M3_2b": DatasetInfo(
        name="M3_2b", alias="M3_2b",
        paradigm="precision_homography", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="per_sample", zoom_center_mode="image",
        pp_fixed=True, skew_corrected=True, ray_error_deg=0.0,
        coverage_pct=4.0, clipping_pct=0.0, status="hypothesis",
        description="H2 baseline: Conservative zoom [1.0, 1.5]"
    ),
    "M3_3": DatasetInfo(
        name="M3_3", alias="M3_3",
        paradigm="precision_homography", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="per_sample + safe", zoom_center_mode="image",
        pp_fixed=True, skew_corrected=True, ray_error_deg=0.0,
        coverage_pct=5.0, clipping_pct=0.0, status="hypothesis",
        description="H2 test: Safe zoom (0% clipping guaranteed)"
    ),
    "M4": DatasetInfo(
        name="M4", alias="M4",
        paradigm="object_centered_mvg", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=0.0, cx_std=50.0, cy_mean=0.0, cy_std=30.0,  # Variable
        distance=2.7, zoom="per_sample", zoom_center_mode="object",
        pp_fixed=False, skew_corrected=True, ray_error_deg=11.8,
        coverage_pct=6.0, clipping_pct=0.0, status="hypothesis",
        description="H1 test: Object-centered + PP correction"
    ),
    
    # ==========================================================================
    # DEPRECATED (Reference Only)
    # ==========================================================================
    "D10.3": DatasetInfo(
        name="D10.3", alias="M3 (deprecated)",
        paradigm="precision_homography", transform="homography",
        fx_mean=739.0, fx_std=50.0, fy_mean=739.0, fy_std=50.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="global", zoom_center_mode="object",
        pp_fixed=True, skew_corrected=True, ray_error_deg=8.0,
        coverage_pct=5.0, clipping_pct=5.0, status="deprecated",
        description="fx=739 bug, not normalized after zoom."
    ),
    "M3_persample": DatasetInfo(
        name="M3_persample", alias="M3_persample (deprecated)",
        paradigm="precision_homography", transform="homography",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=185.0, cx_std=52.0, cy_mean=202.0, cy_std=40.0,
        distance=2.7, zoom="per_sample", zoom_center_mode="object",
        pp_fixed=False, skew_corrected=True, ray_error_deg=17.4,
        coverage_pct=5.0, clipping_pct=0.0, status="deprecated",
        description="Object-centered zoom → PP varies → Ghosting"
    ),
    "D4": DatasetInfo(
        name="D4", alias="D4 (deprecated)",
        paradigm="object_centered", transform="affine",
        fx_mean=549.0, fx_std=0.0, fy_mean=549.0, fy_std=0.0,
        cx_mean=256.0, cx_std=0.0, cy_mean=256.0, cy_std=0.0,
        distance=2.7, zoom="None", zoom_center_mode="N/A",
        pp_fixed=True, skew_corrected=False, ray_error_deg=5.6,
        coverage_pct=3.0, clipping_pct=0.0, status="deprecated",
        description="PP=256 forced after crop → Ray error 5.6°"
    ),
}


class DatasetComparisonModule:
    """Generates dataset comparison tables and analysis."""
    
    def __init__(self, datasets: List[str] = None):
        if datasets is None:
            datasets = list(DATASET_SPECS.keys())
        self.datasets = datasets
    
    def get_quick_reference_html(self) -> str:
        """Generate quick reference table."""
        return r'''
<h3>Quick Reference - Recommended Datasets</h3>
<table>
<tr>
    <th>Rank</th><th>Dataset</th><th>Alias</th><th>PP</th><th>fx</th>
    <th>Zoom</th><th>Coverage</th><th>Clipping</th><th>Status</th>
</tr>
<tr class="good">
    <td>⭐1</td><td><strong>M3_2</strong></td><td>-</td><td>256</td><td>549</td>
    <td>Per-sample</td><td>~5%</td><td>6.5%</td><td><strong>Recommended</strong></td>
</tr>
<tr class="good">
    <td>2</td><td><strong>M3_1</strong></td><td>-</td><td>256</td><td>549</td>
    <td>Global</td><td>~4.5%</td><td>0%</td><td>Safe</td>
</tr>
<tr>
    <td>3</td><td>D7.1</td><td>M1</td><td>256</td><td>549</td>
    <td>None</td><td>~3%</td><td>0%</td><td>Baseline (Affine)</td>
</tr>
<tr>
    <td>4</td><td>D8</td><td>M2</td><td>256</td><td>549</td>
    <td>None</td><td>~3%</td><td>0%</td><td>Baseline (Homography)</td>
</tr>
</table>
'''
    
    def get_hypothesis_section_html(self) -> str:
        """Generate hypothesis testing section."""
        return r'''
<h3>Hypothesis Testing Datasets</h3>

<div class="key-point">
<h4>H1: Can PP Correction enable Object-Centered Zoom?</h4>
<table>
<tr><th>Dataset</th><th>zoom_center_mode</th><th>PP</th><th>PP Correction</th><th>Expected</th></tr>
<tr>
    <td><strong>M4</strong></td>
    <td><code>object</code></td>
    <td>Variable → Corrected</td>
    <td>✅ Applied</td>
    <td>Test if max coverage achievable with PP fix</td>
</tr>
</table>
<p><strong>Rationale:</strong> Object-centered zoom maximizes coverage but breaks PP=256. 
M4 tests if explicit PP correction can recover geometry while keeping max coverage.</p>
</div>

<div class="key-point">
<h4>H2: Can Safe Zoom eliminate Clipping without Coverage Loss?</h4>
<table>
<tr><th>Dataset</th><th>zoom_range</th><th>Safe Mode</th><th>Expected Clipping</th></tr>
<tr><td><strong>M3_2b</strong> (baseline)</td><td>[1.0, 1.5]</td><td>❌</td><td>~0%</td></tr>
<tr><td><strong>M3_3</strong> (test)</td><td>[1.0, 2.5]</td><td>✅</td><td>0% (guaranteed)</td></tr>
</table>
<p><strong>Rationale:</strong> M3_2 has 6.5% clipping. Compare conservative zoom (M3_2b) vs 
safe zoom algorithm (M3_3) to find optimal balance.</p>
</div>
'''
    
    def get_m_series_comparison_html(self) -> str:
        """Generate M-series detailed comparison."""
        return r'''
<h3>M-Series Evolution</h3>

<table>
<tr>
    <th>Dataset</th><th>Transform</th><th>Zoom</th><th>zoom_center_mode</th>
    <th>PP</th><th>Ray Error</th><th>Coverage</th><th>Status</th>
</tr>
<tr>
    <td><strong>M1</strong> (D7.1)</td><td>Affine</td><td>None</td><td>N/A</td>
    <td class="good">256</td><td class="good">0°</td><td>~3%</td><td>Baseline</td>
</tr>
<tr>
    <td><strong>M2</strong> (D8)</td><td>Homography</td><td>None</td><td>N/A</td>
    <td class="good">256</td><td class="good">0°</td><td>~3%</td><td>Baseline</td>
</tr>
<tr class="bad">
    <td><strong>M3</strong> (D10.3)</td><td>Homography</td><td>Global</td><td>object</td>
    <td class="good">256</td><td class="bad">~8°</td><td>~5%</td><td>⛔ fx bug</td>
</tr>
<tr class="good">
    <td><strong>M3_1</strong></td><td>Homography</td><td>Global</td><td class="good">image</td>
    <td class="good">256</td><td class="good">0°</td><td>~4.5%</td><td>✅ Safe</td>
</tr>
<tr class="good">
    <td><strong>M3_2</strong></td><td>Homography</td><td>Per-sample</td><td class="good">image</td>
    <td class="good">256</td><td class="good">0°</td><td>~5%</td><td>⭐ Recommended</td>
</tr>
<tr>
    <td><strong>M3_2b</strong></td><td>Homography</td><td>Per-sample (low)</td><td class="good">image</td>
    <td class="good">256</td><td class="good">0°</td><td>~4%</td><td>🔬 H2 baseline</td>
</tr>
<tr>
    <td><strong>M3_3</strong></td><td>Homography</td><td>Per-sample + safe</td><td class="good">image</td>
    <td class="good">256</td><td class="good">0°</td><td>~5%</td><td>🔬 H2 test</td>
</tr>
<tr class="warning">
    <td><strong>M4</strong></td><td>Homography</td><td>Per-sample</td><td class="bad">object</td>
    <td>Variable</td><td class="bad">~12°</td><td>~6%</td><td>🔬 H1 test</td>
</tr>
</table>

<div class="formula-box">
<h4>Key Insight: zoom_center_mode is Decisive</h4>
<p><strong>✅ Success:</strong> <code>zoom_center_mode = "image"</code> → PP = 256 fixed → Ray Error = 0°</p>
<p><strong>❌ Failure:</strong> <code>zoom_center_mode = "object"</code> → PP varies → Ray Error = 11-17°</p>
</div>
'''
    
    def get_m1_vs_m2_html(self) -> str:
        """M1 (D7.1) vs M2 (D8) detailed comparison."""
        return r'''
<h3>M1 (D7.1) vs M2 (D8) - Baseline Comparison</h3>

<table>
<tr><th>Aspect</th><th>M1 (D7.1)</th><th>M2 (D8)</th><th>Winner</th></tr>
<tr>
    <td>Transform</td>
    <td>Affine</td>
    <td>Homography</td>
    <td>-</td>
</tr>
<tr>
    <td>Skew Correction</td>
    <td>❌ Ignored (~0.9px edge error)</td>
    <td class="good">✅ Corrected (0px error)</td>
    <td>M2</td>
</tr>
<tr>
    <td>fx Precision</td>
    <td>549.0</td>
    <td class="good">548.9937744140625</td>
    <td>M2</td>
</tr>
<tr>
    <td>PP</td>
    <td class="good">256</td>
    <td class="good">256</td>
    <td>Tie</td>
</tr>
<tr>
    <td>Interpolation</td>
    <td>LINEAR</td>
    <td class="good">LANCZOS4</td>
    <td>M2</td>
</tr>
<tr>
    <td>PSNR (Observed)</td>
    <td>~20.3</td>
    <td>~21.0</td>
    <td>M2 (+0.7)</td>
</tr>
<tr>
    <td>Computation</td>
    <td class="good">Faster</td>
    <td>Slightly slower</td>
    <td>M1</td>
</tr>
</table>

<p><strong>Recommendation:</strong> Use <strong>M2 (D8)</strong> for production, M1 (D7.1) for quick experiments.</p>
'''
    
    def get_preprocessing_commands_html(self) -> str:
        """Generate preprocessing commands section."""
        return r'''
<h3>Preprocessing Commands</h3>

<pre>
cd ~/dev/FaceLift

# M1 (D7.1) - Affine baseline
python -m mouse_extensions.preprocessing.preprocess \
    --preset D7.1 --input-dir /path/to/raw --output-dir /path/to/D7_1

# M2 (D8) - Homography baseline  
python -m mouse_extensions.preprocessing.preprocess \
    --preset D8 --input-dir /path/to/raw --output-dir /path/to/D8

# M3_1 - Global zoom (Safe)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_1 --input-dir /path/to/raw --output-dir /path/to/M3_1

# M3_2 - Per-sample zoom (Recommended)
python -m mouse_extensions.preprocessing.preprocess \
    --preset M3_2 --input-dir /path/to/raw --output-dir /path/to/M3_2
</pre>
'''
    
    def generate_full_comparison_section(self) -> str:
        """Generate complete comparison section."""
        return f'''
<h2 id="comparison">3. Dataset Comparison</h2>

{self.get_quick_reference_html()}
{self.get_m_series_comparison_html()}
{self.get_m1_vs_m2_html()}

<h2 id="hypothesis">4. Hypothesis Testing</h2>

{self.get_hypothesis_section_html()}

<h2 id="commands">5. Preprocessing Commands</h2>

{self.get_preprocessing_commands_html()}
'''
