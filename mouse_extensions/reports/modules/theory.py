"""
MVG Theory Module - Mathematical foundation for preprocessing reports.

Contains coordinate transformation theory, ray direction calculations,
and error quantification formulas with MathJax-compatible output.
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


@dataclass
class RayErrorAnalysis:
    """Ray direction error analysis result."""
    pp_offset_x: float
    pp_offset_y: float
    ray_error_deg: float
    risk_level: str  # LOW, MEDIUM, HIGH


class TheoryModule:
    """Generates MVG theory section with MathJax formulas."""
    
    # GS-LRM pretrained model expectations
    TARGET_FX = 548.9937744140625
    TARGET_PP = 256.0
    TARGET_DIST = 2.7
    
    @staticmethod
    def get_coordinate_transformation_html() -> str:
        """Generate coordinate system transformation section."""
        return r'''
<div class="formula-box">
<h3>Coordinate System Transformations</h3>

<p><strong>Step 1: World → Camera (Extrinsic)</strong></p>
\[
\mathbf{P}_c = \mathbf{R} \cdot \mathbf{P}_w + \mathbf{T}
\]

<p><strong>Step 2: Camera → Normalized (Perspective Division)</strong></p>
\[
x = X_c / Z_c, \quad y = Y_c / Z_c
\]

<p><strong>Step 3: Normalized → Pixel (Intrinsic)</strong></p>
\[
u = f_x \cdot x + c_x, \quad v = f_y \cdot y + c_y
\]

<p><strong>Combined (Full Projection):</strong></p>
\[
\begin{bmatrix} u \\ v \\ 1 \end{bmatrix} \sim
\underbrace{\begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix}}_{\mathbf{K}}
\cdot
\underbrace{\begin{bmatrix} R & T \end{bmatrix}}_{[R|T]}
\cdot
\begin{bmatrix} X_w \\ Y_w \\ Z_w \\ 1 \end{bmatrix}
\]
<p><em>Note: s is skew parameter (usually 0). D8+ methods correct for non-zero skew.</em></p>
</div>
'''
    
    @staticmethod
    def get_ray_direction_html() -> str:
        """Generate ray direction calculation section."""
        return r'''
<div class="formula-box">
<h3>Ray Direction Calculation</h3>

<p>For Novel View Synthesis, ray direction from pixel (u, v) in camera coordinates:</p>
\[
\mathbf{d}_c = \mathbf{K}^{-1} \cdot \begin{bmatrix} u \\ v \\ 1 \end{bmatrix}
= \begin{bmatrix} (u - c_x) / f_x \\ (v - c_y) / f_y \\ 1 \end{bmatrix}
\]

<div class="warning">
<strong>Critical:</strong> If \(c_x, c_y\) (Principal Point) are wrong, 
the ray direction is wrong, causing <strong>multi-view inconsistency</strong> 
and <strong>ghosting artifacts</strong>.
</div>
</div>
'''
    
    @staticmethod
    def get_pp_error_impact_html() -> str:
        """Generate PP error impact analysis section."""
        return r'''
<div class="formula-box">
<h3>PP Error Impact Analysis</h3>

<p>When PP deviates from (256, 256), ray direction error at center pixel:</p>
\[
\theta_{\text{error}} = \arctan\left(\sqrt{\left(\frac{256 - c_x}{f_x}\right)^2 + \left(\frac{256 - c_y}{f_y}\right)^2}\right)
\]

<p><strong>Risk Levels:</strong></p>
<ul>
    <li><span style="color:green; font-weight:bold">LOW (&lt;0.5°)</span>: Negligible impact</li>
    <li><span style="color:orange; font-weight:bold">MEDIUM (0.5°-2°)</span>: Minor artifacts possible</li>
    <li><span style="color:red; font-weight:bold">HIGH (&gt;2°)</span>: Significant ghosting expected</li>
</ul>

<p><strong>Example Calculations:</strong></p>
<table>
<tr><th>PP Offset</th><th>Ray Error</th><th>Risk</th><th>Example Dataset</th></tr>
<tr class="good"><td>0 px</td><td>0.00°</td><td>LOW</td><td>M3_1, M3_2, D7.1, D8</td></tr>
<tr class="warning"><td>37 px</td><td>~3.9°</td><td>HIGH</td><td>D4 (forced PP=256)</td></tr>
<tr class="bad"><td>119 px</td><td>~12°</td><td>HIGH</td><td>M3_norm (object-centered)</td></tr>
</table>
</div>
'''
    
    @staticmethod
    def get_zoom_center_mode_html() -> str:
        """Generate zoom_center_mode explanation section."""
        return r'''
<div class="key-point">
<h3>zoom_center_mode: Critical Parameter ⭐</h3>

<table>
<tr><th>Mode</th><th>Crop Center</th><th>PP Result</th><th>Ray Error</th><th>Status</th></tr>
<tr class="good">
    <td><code>"image"</code></td>
    <td>Image center (256, 256)</td>
    <td>PP = 256 (fixed)</td>
    <td>0°</td>
    <td>✅ MVG-Correct</td>
</tr>
<tr class="bad">
    <td><code>"object"</code></td>
    <td>Object centroid (varies)</td>
    <td>PP varies per view/sample</td>
    <td>11-17°</td>
    <td>❌ Ghosting</td>
</tr>
</table>

<div class="formula-box">
<p><strong>Mathematical Explanation:</strong></p>
<ul>
    <li><strong>Image-centered crop:</strong> crop_offset = (512 - crop_size) / 2 → PP remains at 256</li>
    <li><strong>Object-centered crop:</strong> crop_offset = obj_center - crop_size/2 → PP = 256 - crop_offset</li>
</ul>
</div>
</div>
'''
    
    @staticmethod
    def get_homography_vs_affine_html() -> str:
        """Generate affine vs homography comparison section."""
        return r'''
<div class="formula-box">
<h3>Affine vs Homography Transform</h3>

<p><strong>Affine Transform (D7.x / M1):</strong></p>
\[
\begin{bmatrix} u' \\ v' \end{bmatrix} = 
\begin{bmatrix} s_x & 0 \\ 0 & s_y \end{bmatrix}
\begin{bmatrix} u \\ v \end{bmatrix} + 
\begin{bmatrix} \Delta_x \\ \Delta_y \end{bmatrix}
\]
<p><em>Preserves parallelism. Ignores camera skew. ~0.9px edge error.</em></p>

<p><strong>Homography Transform (D8.x / M2 / M3.x):</strong></p>
\[
\mathbf{H} = \mathbf{K}_{\text{target}} \cdot \mathbf{K}_{\text{orig}}^{-1}
\]
<p>where:</p>
\[
\mathbf{K}_{\text{target}} = \begin{bmatrix} 548.99 & 0 & 256 \\ 0 & 548.99 & 256 \\ 0 & 0 & 1 \end{bmatrix}
\]
<p><em>Full perspective correction including skew. 0px geometric error.</em></p>

<table>
<tr><th>Aspect</th><th>Affine (M1/D7.1)</th><th>Homography (M2+/D8+)</th></tr>
<tr><td>Skew Correction</td><td>❌ Ignored</td><td class="good">✅ Corrected</td></tr>
<tr><td>Edge Error</td><td>~0.9px</td><td class="good">~0px</td></tr>
<tr><td>Computation</td><td>Faster</td><td>Slightly slower</td></tr>
<tr><td>Use Case</td><td>Baseline</td><td class="good">Production</td></tr>
</table>
</div>
'''
    
    @classmethod
    def compute_ray_error(cls, cx: float, cy: float, fx: float = None) -> RayErrorAnalysis:
        """Compute ray direction error for given PP values."""
        if fx is None:
            fx = cls.TARGET_FX
        
        offset_x = cls.TARGET_PP - cx
        offset_y = cls.TARGET_PP - cy
        
        error_rad = np.arctan(np.sqrt((offset_x / fx)**2 + (offset_y / fx)**2))
        error_deg = np.degrees(error_rad)
        
        if error_deg < 0.5:
            risk = "LOW"
        elif error_deg < 2.0:
            risk = "MEDIUM"
        else:
            risk = "HIGH"
        
        return RayErrorAnalysis(
            pp_offset_x=offset_x,
            pp_offset_y=offset_y,
            ray_error_deg=error_deg,
            risk_level=risk
        )
    
    @classmethod
    def generate_full_theory_section(cls) -> str:
        """Generate complete theory section HTML."""
        return f'''
<h2 id="theory">2. MVG Theory Foundation</h2>

{cls.get_coordinate_transformation_html()}
{cls.get_ray_direction_html()}
{cls.get_pp_error_impact_html()}
{cls.get_zoom_center_mode_html()}
{cls.get_homography_vs_affine_html()}
'''
