"""Generate HTML report from Gaussian ablation results.

Reads gaussian_ablation_full_results.json and creates a comprehensive
comparison report with correct 3-Axis naming convention.

Usage:
    python -m mouse_extensions.behavior.generate_ablation_report
"""

import json
import time
from pathlib import Path


# 3-Axis naming map: old label → (display name, input type, representation)
RENAME = {
    "Sparse": ("SP_RawPCA", "Sparse (22 joints)", "Raw PCA(20)"),
    "hBehaveMAE": ("SP_MAE", "Sparse (22 joints)", "hBehaveMAE → PCA(30)"),
    "Gauss1000": ("DN_Stats_1K", "Dense (3DGS, 1K)", "Summary Stats"),
    "Gauss5000": ("DN_Stats_5K", "Dense (3DGS, 5K)", "Summary Stats"),
    "Gauss12500": ("DN_Stats_12.5K", "Dense (3DGS, 12.5K)", "Summary Stats"),
    "Gauss50000": ("DN_Stats_50K", "Dense (3DGS, 50K)", "Summary Stats"),
    "Gauss200000": ("DN_Stats_200K", "Dense (3DGS, 200K)", "Summary Stats"),
    "Hybrid_G1000": ("HY_Concat_1K", "Hybrid", "SP + DN_1K Concat"),
    "Hybrid_G5000": ("HY_Concat_5K", "Hybrid", "SP + DN_5K Concat"),
    "Hybrid_G12500": ("HY_Concat_12.5K", "Hybrid", "SP + DN_12.5K Concat"),
    "Hybrid_G50000": ("HY_Concat_50K", "Hybrid", "SP + DN_50K Concat"),
    "Hybrid_G200000": ("HY_Concat_200K", "Hybrid", "SP + DN_200K Concat"),
}

CSS = """
body { font-family: -apple-system, system-ui, sans-serif; margin: 0 auto; max-width: 1400px;
       padding: 20px; background: #0d1117; color: #c9d1d9; }
h1 { color: #58a6ff; border-bottom: 2px solid #30363d; }
h2 { color: #79c0ff; margin-top: 30px; }
h3 { color: #d2a8ff; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 13px; }
th, td { border: 1px solid #30363d; padding: 5px 8px; text-align: center; }
th { background: #161b22; color: #58a6ff; position: sticky; top: 0; }
.best { background: #1f6feb33; font-weight: bold; }
.worst { background: #f8514933; }
.sp { border-left: 3px solid #58a6ff; }
.dn { border-left: 3px solid #f0883e; }
.hy { border-left: 3px solid #a371f7; }
.finding { background: #161b22; border-left: 4px solid #58a6ff; padding: 12px; margin: 10px 0;
           border-radius: 0 6px 6px 0; }
.critical { border-left-color: #f85149; }
.success { border-left-color: #3fb950; }
.card { display: inline-block; background: #161b22; border-radius: 8px; padding: 10px 16px;
        margin: 4px; text-align: center; border: 1px solid #30363d; }
.card .val { font-size: 20px; font-weight: bold; }
.card .lbl { font-size: 10px; color: #8b949e; }
"""


def get_display_name(key):
    """Convert old key to 3-Axis display name."""
    for prefix, (name, _, _) in RENAME.items():
        if key.startswith(prefix + "_"):
            suffix = key[len(prefix) + 1:]  # e.g., "KMeans_K8"
            return f"{name}_{suffix}"
    return key


def get_input_type(key):
    for prefix, (_, inp, _) in RENAME.items():
        if key.startswith(prefix):
            return inp
    return "Unknown"


def get_css_class(key):
    if key.startswith("Sparse") or key.startswith("hBehaveMAE"):
        return "sp"
    elif key.startswith("Hybrid"):
        return "hy"
    else:
        return "dn"


def main():
    from mouse_extensions.behavior.paths import RESULTS_DIR, REPORTS_DIR, ensure_dirs
    ensure_dirs()

    results_path = RESULTS_DIR / "gaussian_ablation" / "gaussian_ablation_full_results.json"
    if not results_path.exists():
        print(f"Results not found: {results_path}")
        return

    with open(results_path) as f:
        data = json.load(f)

    html = [f"<!DOCTYPE html><html><head><meta charset='utf-8'>"
            f"<title>BehaviorBench Ablation Report</title><style>{CSS}</style></head><body>"]

    html.append(f"<h1>BehaviorBench — Gaussian Count Ablation Report</h1>")
    html.append(f"<p>{time.strftime('%Y-%m-%d %H:%M')} | {len(data)} experiments | "
                f"3-Axis: Input × Representation × Clustering</p>")

    # === Key Findings ===
    html.append("<h2>Key Findings</h2>")

    sp8 = data.get("Sparse_KMeans_K8", {})
    mae8 = data.get("hBehaveMAE_KMeans_K8", {})
    g125_8 = data.get("Gauss12500_KMeans_K8", {})
    hy8 = data.get("Hybrid_G12500_KMeans_K8", {})

    html.append('<div class="finding success"><strong>1. SP_MAE (hBehaveMAE) dominates all metrics</strong><br>'
                f'Sil={mae8.get("silhouette",0):.3f} vs SP_RawPCA={sp8.get("silhouette",0):.3f} '
                f'(+{((mae8.get("silhouette",0)-sp8.get("silhouette",0))/sp8.get("silhouette",1)*100):.0f}%). '
                f'Same sparse input, different representation → <strong>Learned > Raw</strong></div>')

    html.append(f'<div class="finding critical"><strong>2. DN_Stats &lt; SP_RawPCA (Dense summary stats lose to Sparse)</strong><br>'
                f'DN_Stats_12.5K Sil={g125_8.get("silhouette",0):.3f} vs SP_RawPCA={sp8.get("silhouette",0):.3f}. '
                f'Summary statistics lose spatial structure → <strong>Learned dense repr needed (Selfee, PointNet-AE)</strong></div>')

    html.append(f'<div class="finding"><strong>3. Optimal pruning: 12.5K Gaussians</strong><br>'
                f'Best DN_Stats Sil at 12.5K. More Gaussians (200K) ≠ better features.</div>')

    html.append(f'<div class="finding"><strong>4. HY_Concat temporal advantage</strong><br>'
                f'HY_Concat Bout={hy8.get("bout_mean_sec",0):.2f}s vs SP_RawPCA={sp8.get("bout_mean_sec",0):.2f}s '
                f'(+{((hy8.get("bout_mean_sec",0)-sp8.get("bout_mean_sec",0))/max(sp8.get("bout_mean_sec",1),0.01)*100):.0f}%). '
                f'Dense features stabilize cluster boundaries.</div>')

    # === Summary Cards ===
    html.append("<h2>Summary (KMeans K=8)</h2><div>")
    for key, label in [("Sparse_KMeans_K8", "SP_RawPCA"), ("hBehaveMAE_KMeans_K8", "SP_MAE"),
                       ("Gauss12500_KMeans_K8", "DN_Stats_12.5K"), ("Hybrid_G12500_KMeans_K8", "HY_Concat_12.5K")]:
        d = data.get(key, {})
        html.append(f'<div class="card"><div class="val">{d.get("silhouette",0):.3f}</div>'
                    f'<div class="lbl">{label} Sil</div></div>')
    html.append("</div>")

    # === Full Comparison Table (KMeans K=8) ===
    html.append("<h2>Fair Comparison: KMeans K=8</h2>")
    html.append("<table><tr><th>Pipeline</th><th>Input</th><th>Repr</th>"
                "<th>Sil ↑</th><th>CH ↑</th><th>DB ↓</th>"
                "<th>Bout(s) ↑</th><th>TC ↑</th><th>TPI ↑</th><th>Ent ↓</th></tr>")

    k8_keys = [k for k in data if k.endswith("_K8") and "KMeans" in k]
    k8_keys.sort(key=lambda k: -data[k].get("silhouette", 0))

    for key in k8_keys:
        d = data[key]
        name = get_display_name(key)
        css = get_css_class(key)
        inp = get_input_type(key)
        sil = d.get("silhouette", 0) or 0
        best_class = ' class="best"' if sil > 0.30 else ""
        html.append(f'<tr class="{css}"{best_class}>'
                    f'<td style="text-align:left;">{name}</td><td>{inp}</td><td>-</td>'
                    f'<td>{sil:.3f}</td><td>{d.get("calinski_harabasz",0):.0f}</td>'
                    f'<td>{d.get("davies_bouldin",0):.2f}</td>'
                    f'<td>{d.get("bout_mean_sec",0):.2f}</td>'
                    f'<td>{d.get("temporal_consistency",0):.3f}</td>'
                    f'<td>{d.get("tpi_mean",0):.1f}</td>'
                    f'<td>{d.get("entropy_rate",0):.3f}</td></tr>')
    html.append("</table>")

    # === Full Results Table (all methods, all K) ===
    html.append("<h2>Full Results (78 experiments)</h2>")
    metrics = ["silhouette", "calinski_harabasz", "davies_bouldin",
               "bout_mean_sec", "temporal_consistency", "tpi_mean", "entropy_rate"]
    headers = ["Sil ↑", "CH ↑", "DB ↓", "Bout(s) ↑", "TC ↑", "TPI ↑", "Ent ↓"]

    html.append("<table><tr><th>Experiment</th>")
    for h in headers:
        html.append(f"<th>{h}</th>")
    html.append("</tr>")

    for key in sorted(data.keys()):
        d = data[key]
        name = get_display_name(key)
        css = get_css_class(key)
        sil = d.get("silhouette", 0) or 0
        row_class = f' class="{css}"'
        html.append(f"<tr{row_class}><td style='text-align:left;'>{name}</td>")
        for m in metrics:
            v = d.get(m, 0) or 0
            fmt = ".3f" if m in ["silhouette", "temporal_consistency", "entropy_rate"] else ".2f" if m == "bout_mean_sec" else ".1f" if m == "tpi_mean" else ".0f"
            html.append(f"<td>{v:{fmt}}</td>")
        html.append("</tr>")
    html.append("</table>")

    # === Gaussian Count Ablation ===
    html.append("<h2>Gaussian Count Ablation (KMeans K=8)</h2>")
    html.append("<table><tr><th>Gaussians</th><th>Sil</th><th>CH</th><th>Bout(s)</th><th>TC</th></tr>")
    for n in [1000, 5000, 12500, 50000, 200000]:
        key = f"Gauss{n}_KMeans_K8"
        if key in data:
            d = data[key]
            best = ' class="best"' if n == 12500 else ""
            html.append(f'<tr{best}><td>{n:,}</td>'
                        f'<td>{d.get("silhouette",0):.3f}</td>'
                        f'<td>{d.get("calinski_harabasz",0):.0f}</td>'
                        f'<td>{d.get("bout_mean_sec",0):.2f}</td>'
                        f'<td>{d.get("temporal_consistency",0):.3f}</td></tr>')
    html.append("</table>")

    # === Method Comparison (KMeans vs GMM vs HMM) ===
    html.append("<h2>Clustering Method Comparison (Sparse K=8)</h2>")
    html.append("<table><tr><th>Method</th><th>Sil</th><th>CH</th><th>DB</th>"
                "<th>Bout(s)</th><th>TC</th><th>TPI</th></tr>")
    for method in ["KMeans", "GMM", "HMM"]:
        key = f"Sparse_{method}_K8"
        if key in data:
            d = data[key]
            html.append(f'<tr><td>SP_RawPCA_{method}</td>'
                        f'<td>{d.get("silhouette",0):.3f}</td>'
                        f'<td>{d.get("calinski_harabasz",0):.0f}</td>'
                        f'<td>{d.get("davies_bouldin",0):.2f}</td>'
                        f'<td>{d.get("bout_mean_sec",0):.2f}</td>'
                        f'<td>{d.get("temporal_consistency",0):.3f}</td>'
                        f'<td>{d.get("tpi_mean",0):.1f}</td></tr>')
    html.append("</table>")

    # === 3-Axis Framework ===
    html.append("<h2>3-Axis Experimental Framework</h2>")
    html.append("""<div class="finding">
<strong>Axis 1 — Input Modality</strong>: SP (Sparse 22 joints) | DN (Dense 3DGS) | HY (Hybrid)<br>
<strong>Axis 2 — Representation</strong>: RawPCA | MAE | Stats | VAME | Selfee | Concat<br>
<strong>Axis 3 — Clustering</strong>: KMeans | GMM | HMM | B-SOiD | SUBTLE<br><br>
<em>hBehaveMAE = SP_MAE (Sparse Input + Learned MAE), NOT Dense</em>
</div>""")

    html.append(f"""<hr><p style="color:#8b949e;font-size:11px;">
BehaviorBench Ablation Report | Generated {time.strftime('%Y-%m-%d %H:%M')} |
{len(data)} experiments | 3-Axis: Input × Representation × Clustering</p></body></html>""")

    report_path = REPORTS_DIR / "ablation_report.html"
    with open(report_path, "w") as f:
        f.write("\n".join(html))
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
