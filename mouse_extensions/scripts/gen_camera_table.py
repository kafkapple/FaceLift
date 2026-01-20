#!/usr/bin/env python3
"""Generate camera parameter comparison table."""

import sys
sys.path.insert(0, '/home/joon/dev/FaceLift')

import numpy as np
from pathlib import Path

from mouse_extensions.preprocessing.data_loader import DataLoader


def main():
    data_dir = '/home/joon/data/markerless_mouse_1_nerf'
    loader = DataLoader(data_dir, source_type='raw', num_views=6)
    cameras = loader.cameras
    n_views = len(cameras)
    
    # FaceLift default assumption
    facelift_default = {'fx': 500.0, 'fy': 500.0, 'cx': 256.0, 'cy': 256.0}
    
    print("""
=============================================================
Camera Parameter Comparison Table
=============================================================

| View | Original fx | Original fy | Original cx | Original cy |
|------|------------|------------|------------|------------|
""")
    
    for i, cam in enumerate(cameras):
        print(f"|  {i}   |   {cam['fx']:.1f}    |   {cam['fy']:.1f}    |   {cam['cx']:.1f}    |   {cam['cy']:.1f}    |")
    
    print("""
-------------------------------------------------------------

| Setting | fx | fy | cx | cy | Note |
|---------|-----|-----|-----|-----|------|
| Original (from raw) | varies | varies | varies | varies | Per-view actual values |
| D0/v13 (PP bug) | orig | orig | **256** | **256** | cx,cy forced to 256 ❌ |
| D1 (PP-centered) | orig | orig | **256** | **256** | Crop shifts object to center |
| D2 (Correct PP) | orig | orig | actual | actual | No shift, use real PP |
| D3 (Triangulation) | orig | orig | actual | actual | + 3D center estimation |
| FaceLift default | 500 | 500 | 256 | 256 | Pretrained model assumption |

-------------------------------------------------------------
Key Insights:
- Original cx, cy vary significantly per view (not centered!)
- D0/v13 bug: forced cx=cy=256 but didn't shift image → 13° ray error
- D1: shifts image to center object → PP is now correct (256, 256)
- D2/D3: keep actual PP values → model must handle off-center PP
- FaceLift pretrained on centered data → D1 most compatible
""")
    
    # Generate HTML table
    html = """
<h3>Camera Parameter Comparison</h3>
<table border="1" style="border-collapse: collapse; font-size: 12px;">
<tr style="background-color: #f0f0f0;">
<th>View</th><th>Original fx</th><th>Original fy</th><th>Original cx</th><th>Original cy</th>
</tr>
"""
    for i, cam in enumerate(cameras):
        html += f"<tr><td>{i}</td><td>{cam['fx']:.1f}</td><td>{cam['fy']:.1f}</td><td>{cam['cx']:.1f}</td><td>{cam['cy']:.1f}</td></tr>\n"
    
    html += "</table>\n"
    html += """
<br>
<h4>Dataset PP Methods</h4>
<table border="1" style="border-collapse: collapse; font-size: 12px;">
<tr style="background-color: #f0f0f0;">
<th>Dataset</th><th>fx</th><th>fy</th><th>cx</th><th>cy</th><th>Description</th>
</tr>
<tr><td>D0/v13 (Bug)</td><td>orig</td><td>orig</td><td><b>256</b></td><td><b>256</b></td><td style="color:red">PP forced but no shift → Ray error!</td></tr>
<tr><td>D1 (PP-centered)</td><td>orig</td><td>orig</td><td><b>256</b></td><td><b>256</b></td><td style="color:green">Shift image → PP correct</td></tr>
<tr><td>D2/D3 (Correct PP)</td><td>orig</td><td>orig</td><td>actual</td><td>actual</td><td>Real PP, model handles it</td></tr>
<tr><td>FaceLift Default</td><td>500</td><td>500</td><td>256</td><td>256</td><td>Pretrained assumption</td></tr>
</table>
"""
    
    output_path = Path('/home/joon/dev/FaceLift/mouse_extensions/reports/center_analysis_v4/camera_params.html')
    output_path.write_text(html)
    print(f"\nHTML table saved to: {output_path}")

if __name__ == '__main__':
    main()
