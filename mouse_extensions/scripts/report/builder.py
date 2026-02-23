"""HTML report builder using Jinja2 templates."""

from pathlib import Path

from jinja2 import Template

from .schema import ReportConfig, ProtocolConfig
from .loaders import load_fair_eval, get_metrics, load_render, load_gt
from .visualizers import (
    image_to_base64,
    create_comparison_grid,
    create_multi_frame_grid,
    create_metric_bars,
)


# Metric display names
METRIC_LABELS = {
    "psnr_gt_masked": "PSNR (FG)",
    "psnr_intersection": "PSNR (Inter.)",
    "psnr_whole": "PSNR (Whole)",
    "iou": "IoU",
    "coverage": "Coverage",
    "ssim_gt_masked": "SSIM (FG)",
    "color_bias": "Color Bias",
}

HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{{ title }}</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #fafafa; color: #333; line-height: 1.5; padding: 24px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 1.6em; margin-bottom: 4px; color: #1a1a1a; }
h2 { font-size: 1.2em; margin: 28px 0 12px; color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 4px; }
h3 { font-size: 1.05em; margin: 16px 0 8px; color: #34495e; }
.subtitle { color: #666; font-size: 0.9em; margin-bottom: 20px; }
.section { background: #fff; border-radius: 8px; padding: 20px; margin: 16px 0; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 0.88em; }
th, td { padding: 8px 12px; text-align: center; border: 1px solid #e0e0e0; }
th { background: #f5f7fa; font-weight: 600; color: #2c3e50; }
td.name { text-align: left; font-weight: 500; }
.best { font-weight: 700; color: #27ae60; }
.worst { color: #999; }
.pending { color: #aaa; font-style: italic; }
.grid-img { width: 100%; max-width: 100%; border-radius: 4px; margin: 8px 0; }
.bar-img { max-width: 600px; margin: 8px 0; }
.metric-grid { display: flex; flex-wrap: wrap; gap: 12px; }
.metric-card { flex: 1; min-width: 280px; }
.note { background: #fff3cd; padding: 10px 14px; border-radius: 4px; border-left: 4px solid #ffc107;
        font-size: 0.85em; margin: 10px 0; }
.info { background: #d1ecf1; padding: 10px 14px; border-radius: 4px; border-left: 4px solid #17a2b8;
        font-size: 0.85em; margin: 10px 0; }
.footer { margin-top: 30px; padding-top: 12px; border-top: 1px solid #ddd; font-size: 0.8em; color: #999; text-align: center; }
@media print { body { padding: 10px; } .section { box-shadow: none; border: 1px solid #ddd; } }
</style>
</head>
<body>

<h1>{{ title }}</h1>
<p class="subtitle">Generated: {{ date }} | Dataset: {{ dataset.name }} ({{ dataset.split }}) | Resolution: {{ dataset.resolution }}px</p>

<!-- Setup -->
<div class="section">
<h2>Experiment Setup</h2>
<table>
<tr><th>Experiment</th><th>Method</th><th>Pipeline</th><th>Input Views</th></tr>
{% for exp in experiments %}
<tr>
  <td class="name" style="color:{{ exp.color }}">{{ exp.name }}</td>
  <td>{{ exp.method }}</td>
  <td>{{ exp.pipeline }}</td>
  <td>{{ exp.input_views }}</td>
</tr>
{% endfor %}
</table>

{% if dataset_info %}
<div class="info">
{{ dataset_info }}
</div>
{% endif %}
</div>

<!-- Quantitative Results -->
{% for proto_key, proto in protocols.items() %}
<div class="section">
<h2>{{ proto.name }}</h2>
<p style="font-size:0.88em; color:#666; margin-bottom:10px;">{{ proto.description }}</p>

{% set table_data = metrics_tables[proto_key] %}
{% if table_data %}
<table>
<tr>
  <th>Experiment</th>
  {% for m in metric_names %}
  <th>{{ metric_labels[m] }}</th>
  {% endfor %}
</tr>
{% for row in table_data %}
<tr>
  <td class="name" style="color:{{ row.color }}">{{ row.name }}</td>
  {% for m in metric_names %}
  {% set cell = row.metrics.get(m, {}) %}
  {% if cell.get("mean") is not none %}
    {% if cell.get("is_best") %}
    <td class="best">{{ "%.2f"|format(cell.mean) }}{% if cell.std %} <small>&plusmn;{{ "%.2f"|format(cell.std) }}</small>{% endif %}</td>
    {% else %}
    <td>{{ "%.2f"|format(cell.mean) }}{% if cell.std %} <small>&plusmn;{{ "%.2f"|format(cell.std) }}</small>{% endif %}</td>
    {% endif %}
  {% else %}
  <td class="pending">---</td>
  {% endif %}
  {% endfor %}
</tr>
{% endfor %}
</table>
{% endif %}

<!-- Bar charts for this protocol -->
{% if bar_charts.get(proto_key) %}
<h3>Metric Comparison</h3>
<div class="metric-grid">
{% for chart_b64 in bar_charts[proto_key] %}
<div class="metric-card">
<img src="{{ chart_b64 }}" class="bar-img" alt="metric chart">
</div>
{% endfor %}
</div>
{% endif %}
</div>
{% endfor %}

<!-- Qualitative Results -->
{% if comparison_grids %}
<div class="section">
<h2>Qualitative Comparison</h2>
<p style="font-size:0.88em; color:#666; margin-bottom:10px;">
Sample frames from test set. All views (0-{{ total_views - 1 }}) shown.
</p>

{% if multi_frame_grid %}
<img src="{{ multi_frame_grid }}" class="grid-img" alt="multi-frame comparison">
{% endif %}

{% for fid, grid_b64 in comparison_grids.items() %}
<h3>Frame {{ fid }}</h3>
<img src="{{ grid_b64 }}" class="grid-img" alt="frame {{ fid }} comparison">
{% endfor %}
</div>
{% endif %}

<!-- Key Findings -->
{% if findings %}
<div class="section">
<h2>Key Findings</h2>
<ul>
{% for f in findings %}
<li>{{ f }}</li>
{% endfor %}
</ul>
</div>
{% endif %}

<div class="footer">
Modular Report System v1.0 | {{ title }}
</div>

</body>
</html>
"""


def _mark_best(rows: list[dict], metric_names: list[str],
               higher_better: set[str]):
    """Mark the best value per metric across rows."""
    for m in metric_names:
        vals = []
        for row in rows:
            v = row["metrics"].get(m, {}).get("mean")
            vals.append(v)

        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if not valid:
            continue

        if m in higher_better:
            best_i = max(valid, key=lambda x: x[1])[0]
        else:
            best_i = min(valid, key=lambda x: x[1])[0]

        rows[best_i]["metrics"][m]["is_best"] = True


def build_report(config: ReportConfig, output_path: str):
    """Build HTML report from config.

    Args:
        config: ReportConfig loaded from YAML.
        output_path: Path to write HTML file.
    """
    from datetime import datetime

    higher_better = {"psnr_gt_masked", "psnr_intersection", "psnr_whole",
                     "iou", "coverage", "ssim_gt_masked"}

    all_metric_names = config.metrics.get("primary", []) + \
        config.metrics.get("secondary", [])

    # --- Load metrics ---
    exp_data = {}
    for exp in config.experiments:
        try:
            data = load_fair_eval(exp.metrics_json)
            exp_data[exp.name] = data
        except (FileNotFoundError, Exception) as e:
            print(f"WARNING: Cannot load {exp.metrics_json}: {e}")
            exp_data[exp.name] = None

    # --- Build metrics tables per protocol ---
    metrics_tables = {}
    bar_charts = {}

    for proto_key, proto in config.protocols.items():
        table_rows = []
        for exp in config.experiments:
            data = exp_data.get(exp.name)
            if data is None:
                table_rows.append({
                    "name": exp.name, "color": exp.color,
                    "metrics": {m: {"mean": None, "std": None}
                                for m in all_metric_names},
                })
                continue

            view_key = proto.holdout_key if proto.source == "per_view" else None
            raw = get_metrics(data, view_key)

            row_metrics = {}
            for m in all_metric_names:
                if m in raw:
                    row_metrics[m] = {
                        "mean": raw[m]["mean"],
                        "std": raw[m]["std"],
                    }
                else:
                    row_metrics[m] = {"mean": None, "std": None}

            table_rows.append({
                "name": exp.name,
                "color": exp.color,
                "metrics": row_metrics,
            })

        _mark_best(table_rows, all_metric_names, higher_better)
        metrics_tables[proto_key] = table_rows

        # Bar charts for primary metrics
        charts = []
        for m in config.metrics.get("primary", []):
            bar_data = []
            for row in table_rows:
                bar_data.append({
                    "name": row["name"],
                    "color": row["color"],
                    "value": row["metrics"].get(m, {}).get("mean"),
                    "std": row["metrics"].get(m, {}).get("std"),
                })
            chart_img = create_metric_bars(
                bar_data, METRIC_LABELS.get(m, m))
            charts.append(image_to_base64(chart_img))
        bar_charts[proto_key] = charts

    # --- Load images for qualitative comparison ---
    comparison_grids = {}
    multi_frame_grid_b64 = None
    sample_frames = config.visualization.get("sample_frames", [])
    img_size = config.visualization.get("img_size", 200)
    total_views = config.dataset.get("total_views", 6)
    gt_dir = config.dataset.get("gt_dir", "")
    gt_pattern = config.dataset.get("gt_pattern",
                                     "{fid}/images/cam_{vid:03d}.png")

    if sample_frames and gt_dir:
        rows_per_frame = {}

        for fid in sample_frames:
            # GT row
            gt_imgs = [load_gt(gt_dir, gt_pattern, fid, v)
                       for v in range(total_views)]
            frame_rows = [{"name": "GT", "color": "#333333",
                           "images": gt_imgs}]

            # Experiment rows
            for exp in config.experiments:
                if not exp.render_dir:
                    exp_imgs = [None] * total_views
                else:
                    exp_imgs = [load_render(exp.render_dir, exp.render_pattern,
                                            fid, v)
                                for v in range(total_views)]
                frame_rows.append({
                    "name": exp.name,
                    "color": exp.color,
                    "images": exp_imgs,
                })

            rows_per_frame[fid] = frame_rows

            # Per-frame grid
            grid_img = create_comparison_grid(
                frame_rows, fid, total_views, img_size)
            comparison_grids[fid] = image_to_base64(grid_img)

        # Multi-frame grid
        if len(sample_frames) > 1:
            mf_img = create_multi_frame_grid(
                rows_per_frame, sample_frames, total_views,
                min(img_size, 160))
            multi_frame_grid_b64 = image_to_base64(mf_img)

    # --- Render HTML ---
    template = Template(HTML_TEMPLATE)
    html = template.render(
        title=config.title,
        date=datetime.now().strftime("%Y-%m-%d %H:%M"),
        dataset=config.dataset,
        experiments=[{
            "name": e.name, "method": e.method,
            "pipeline": e.pipeline, "input_views": e.input_views,
            "color": e.color,
        } for e in config.experiments],
        dataset_info=config.dataset.get("info", ""),
        protocols=config.protocols,
        metrics_tables=metrics_tables,
        metric_names=all_metric_names,
        metric_labels=METRIC_LABELS,
        bar_charts=bar_charts,
        comparison_grids=comparison_grids,
        multi_frame_grid=multi_frame_grid_b64,
        total_views=total_views,
        findings=config.visualization.get("findings", []),
    )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        f.write(html)
    print(f"Report saved: {out}")
    print(f"  Size: {out.stat().st_size / 1024:.1f} KB")
