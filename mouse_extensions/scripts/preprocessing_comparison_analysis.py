#!/usr/bin/env python3
"""
Preprocessing Methods Comparison Analysis
==========================================

Compares D1, D3, D7, D8 preprocessing methods:
1. PP shift x-y distribution per camera view
2. Ray direction error before/after preprocessing
3. Camera parameter comparison across 6 views

Usage:
    python preprocessing_comparison_analysis.py --output_dir ./reports/preprocessing_comparison

Created: 2026-01-22
"""

import argparse
import json
import pickle
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from typing import Dict, List, Optional, Tuple


# Dataset configurations
DATASET_CONFIGS = {
    "D1": {
        "path": "/home/joon/data/preprocessed/FaceLift_mouse/D1_pp_centered",
        "sample_format": "sample_{:06d}",
        "name": "D1 (PP-Crop)",
        "color": "#e74c3c",
        "description": "Per-view PP crop to 256,256",
        "method": "object_centered_crop",
    },
    "D3": {
        "path": "/home/joon/data/preprocessed/FaceLift_mouse/D3",
        "sample_format": "{:06d}",
        "name": "D3 (Triangulation)",
        "color": "#f39c12",
        "description": "3D triangulation center, not normalized",
        "method": "triangulation_center",
    },
    "D7_1": {
        "path": "/home/joon/data/preprocessed/FaceLift_mouse/D7_1",
        "sample_format": "{:06d}",
        "name": "D7_1 (PP-Shift)",
        "color": "#27ae60",
        "description": "PP-centered shift, individual scale",
        "method": "pp_shift_individual",
    },
    "D8": {
        "path": "/home/joon/data/preprocessed/FaceLift_mouse/D8",
        "sample_format": "{:06d}",
        "name": "D8 (Homography)",
        "color": "#3498db",
        "description": "Precision homography transform",
        "method": "homography",
    },
}

ORIGINAL_DATA_PATH = "/home/joon/data/markerless_mouse_1_nerf/new_cam.pkl"
NUM_VIEWS = 6


class PreprocessingComparison:
    """Compare preprocessing methods across datasets."""

    def __init__(self, output_dir: str, datasets: Optional[List[str]] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = datasets or ["D1", "D3", "D7_1", "D8"]
        self.original = None
        self.results = {}

    def load_original_cameras(self) -> Dict:
        """Load original camera parameters."""
        with open(ORIGINAL_DATA_PATH, "rb") as f:
            cameras = pickle.load(f)

        cam_data = []
        for i, cam in enumerate(cameras):
            K = cam["K"]
            R = np.array(cam["R"])
            T = np.array(cam["T"])
            C = -R.T @ T  # Camera position in world

            cam_data.append({
                "view_id": i,
                "fx": K[0, 0],
                "fy": K[1, 1],
                "cx": K[0, 2],
                "cy": K[1, 2],
                "w": 1152,  # Original resolution
                "h": 1024,
                "R": R,
                "T": T,
                "C": C,
                "distance": np.linalg.norm(C),
            })

        self.original = cam_data
        return cam_data

    def load_dataset_cameras(self, dataset: str, sample_idx: int = 0) -> Optional[List[Dict]]:
        """Load camera parameters from preprocessed dataset."""
        config = DATASET_CONFIGS.get(dataset)
        if not config:
            print(f"Unknown dataset: {dataset}")
            return None

        sample_name = config["sample_format"].format(sample_idx)
        json_path = Path(config["path"]) / "train" / sample_name / "opencv_cameras.json"

        if not json_path.exists():
            print(f"File not found: {json_path}")
            return None

        with open(json_path) as f:
            data = json.load(f)

        frames = data.get("frames", data)
        return frames[:NUM_VIEWS]

    def compute_ray_direction(self, fx: float, fy: float, cx: float, cy: float,
                               pixel: Tuple[float, float]) -> np.ndarray:
        """Compute ray direction from camera parameters and pixel location."""
        u, v = pixel
        # Ray in camera coordinates (normalized)
        ray = np.array([
            (u - cx) / fx,
            (v - cy) / fy,
            1.0
        ])
        return ray / np.linalg.norm(ray)

    def compute_ray_error(self, orig_cam: Dict, proc_cam: Dict,
                          test_pixels: Optional[List[Tuple]] = None) -> Dict:
        """
        Compute ray direction error between original and processed camera.

        Returns error in degrees for center and corner pixels.
        """
        if test_pixels is None:
            # Test at center and corners
            h_orig, w_orig = 1024, 1152
            h_proc, w_proc = proc_cam["h"], proc_cam["w"]

            test_pixels = [
                ((w_orig/2, h_orig/2), (w_proc/2, h_proc/2)),  # Center
                ((0, 0), (0, 0)),  # Top-left
                ((w_orig, 0), (w_proc, 0)),  # Top-right
                ((0, h_orig), (0, h_proc)),  # Bottom-left
                ((w_orig, h_orig), (w_proc, h_proc)),  # Bottom-right
            ]

        errors = []
        for (orig_pix, proc_pix) in test_pixels:
            # Original ray
            ray_orig = self.compute_ray_direction(
                orig_cam["fx"], orig_cam["fy"],
                orig_cam["cx"], orig_cam["cy"],
                orig_pix
            )
            # Processed ray
            ray_proc = self.compute_ray_direction(
                proc_cam["fx"], proc_cam["fy"],
                proc_cam["cx"], proc_cam["cy"],
                proc_pix
            )

            # Angular error in degrees
            cos_angle = np.clip(np.dot(ray_orig, ray_proc), -1, 1)
            angle_deg = np.degrees(np.arccos(cos_angle))
            errors.append(angle_deg)

        return {
            "center": errors[0],
            "corners": errors[1:],
            "mean": np.mean(errors),
            "max": np.max(errors),
        }

    def analyze_pp_distribution(self) -> Dict:
        """Analyze PP shift distribution for each dataset and view."""
        results = {}

        # Load multiple samples for statistical analysis
        num_samples = 100

        for ds_name in self.datasets:
            config = DATASET_CONFIGS[ds_name]
            results[ds_name] = {
                "config": config,
                "views": {}
            }

            for view_id in range(NUM_VIEWS):
                cx_vals, cy_vals = [], []
                shift_x_vals, shift_y_vals = [], []

                for sample_idx in range(num_samples):
                    cams = self.load_dataset_cameras(ds_name, sample_idx)
                    if cams and view_id < len(cams):
                        cam = cams[view_id]
                        cx_vals.append(cam["cx"])
                        cy_vals.append(cam["cy"])

                        # Get shift info if available
                        if "_transform" in cam:
                            t = cam["_transform"]
                            if "shift_x" in t:
                                shift_x_vals.append(t["shift_x"])
                                shift_y_vals.append(t["shift_y"])
                            elif "applied_shift_x" in t:
                                shift_x_vals.append(t["applied_shift_x"])
                                shift_y_vals.append(t["applied_shift_y"])

                results[ds_name]["views"][view_id] = {
                    "cx": {"mean": np.mean(cx_vals) if cx_vals else 0,
                           "std": np.std(cx_vals) if cx_vals else 0,
                           "values": cx_vals},
                    "cy": {"mean": np.mean(cy_vals) if cy_vals else 0,
                           "std": np.std(cy_vals) if cy_vals else 0,
                           "values": cy_vals},
                    "shift_x": {"mean": np.mean(shift_x_vals) if shift_x_vals else 0,
                                "std": np.std(shift_x_vals) if shift_x_vals else 0,
                                "values": shift_x_vals},
                    "shift_y": {"mean": np.mean(shift_y_vals) if shift_y_vals else 0,
                                "std": np.std(shift_y_vals) if shift_y_vals else 0,
                                "values": shift_y_vals},
                    "pp_forced": np.std(cx_vals) < 1 if cx_vals else False,
                }

        return results

    def generate_pp_distribution_plot(self, pp_data: Dict) -> str:
        """Generate PP shift x-y distribution plot for all views and datasets."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("PP Shift Distribution by Camera View\n(100 samples each)",
                     fontsize=14, fontweight="bold")

        for view_id in range(NUM_VIEWS):
            ax = axes[view_id // 3, view_id % 3]

            for ds_name in self.datasets:
                config = DATASET_CONFIGS[ds_name]
                view_data = pp_data[ds_name]["views"].get(view_id, {})

                shift_x = view_data.get("shift_x", {}).get("values", [])
                shift_y = view_data.get("shift_y", {}).get("values", [])

                if shift_x and shift_y:
                    ax.scatter(shift_x, shift_y, alpha=0.5, label=config["name"],
                              color=config["color"], s=20)
                    # Add mean marker
                    mean_x = np.mean(shift_x)
                    mean_y = np.mean(shift_y)
                    ax.scatter([mean_x], [mean_y], marker="x", s=100,
                              color=config["color"], linewidths=2)

            ax.set_xlabel("Shift X (pixels)")
            ax.set_ylabel("Shift Y (pixels)")
            ax.set_title(f"View {view_id}")
            ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax.axvline(x=0, color="gray", linestyle="--", alpha=0.5)
            ax.grid(True, alpha=0.3)
            ax.set_aspect("equal")

            if view_id == 0:
                ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        out_path = self.output_dir / "fig1_pp_shift_distribution.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        return str(out_path)

    def generate_ray_error_plot(self) -> str:
        """Generate ray direction error comparison plot."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("Ray Direction Error by Camera View\n(Original vs Preprocessed)",
                     fontsize=14, fontweight="bold")

        # Load original cameras
        if self.original is None:
            self.load_original_cameras()

        # Compute ray errors for each dataset
        ray_errors = {}
        for ds_name in self.datasets:
            ray_errors[ds_name] = []
            cams = self.load_dataset_cameras(ds_name, 0)
            if cams:
                for view_id in range(NUM_VIEWS):
                    error = self.compute_ray_error(self.original[view_id], cams[view_id])
                    ray_errors[ds_name].append(error)

        # Plot
        x = np.arange(NUM_VIEWS)
        width = 0.2

        for ax_idx, metric in enumerate(["center", "mean", "max"]):
            ax = axes[ax_idx // 3, ax_idx % 3]

            for i, ds_name in enumerate(self.datasets):
                config = DATASET_CONFIGS[ds_name]
                if ray_errors[ds_name]:
                    values = [e[metric] if isinstance(e[metric], (int, float)) else e[metric]
                              for e in ray_errors[ds_name]]
                    offset = width * (i - len(self.datasets)/2 + 0.5)
                    ax.bar(x + offset, values, width, label=config["name"],
                           color=config["color"], alpha=0.8)

            ax.set_xlabel("Camera View")
            ax.set_ylabel("Error (degrees)")
            ax.set_title(f"Ray Error - {metric.capitalize()}")
            ax.set_xticks(x)
            ax.set_xticklabels([f"View {i}" for i in range(NUM_VIEWS)])
            ax.legend(loc="upper right", fontsize=8)
            ax.grid(True, alpha=0.3, axis="y")

        # Summary table
        ax = axes[1, 0]
        ax.axis("off")

        table_data = []
        for ds_name in self.datasets:
            config = DATASET_CONFIGS[ds_name]
            if ray_errors[ds_name]:
                mean_center = np.mean([e["center"] for e in ray_errors[ds_name]])
                mean_all = np.mean([e["mean"] for e in ray_errors[ds_name]])
                max_all = np.max([e["max"] for e in ray_errors[ds_name]])
                table_data.append([config["name"], f"{mean_center:.2f} deg",
                                   f"{mean_all:.2f} deg", f"{max_all:.2f} deg"])

        table = ax.table(cellText=table_data,
                        colLabels=["Dataset", "Center Error", "Mean Error", "Max Error"],
                        cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax.set_title("Ray Error Summary", fontsize=12, fontweight="bold", pad=20)

        # Camera parameter comparison
        ax = axes[1, 1]
        ax.axis("off")

        param_data = []
        for ds_name in self.datasets:
            cams = self.load_dataset_cameras(ds_name, 0)
            if cams:
                config = DATASET_CONFIGS[ds_name]
                cam0 = cams[0]
                fx = cam0["fx"]
                cx, cy = cam0["cx"], cam0["cy"]
                param_data.append([config["name"], f"{fx:.1f}", f"({cx:.1f}, {cy:.1f})"])

        table2 = ax.table(cellText=param_data,
                         colLabels=["Dataset", "fx", "(cx, cy)"],
                         cellLoc="center", loc="center")
        table2.auto_set_font_size(False)
        table2.set_fontsize(10)
        table2.scale(1.2, 1.5)
        ax.set_title("Camera Parameters (View 0)", fontsize=12, fontweight="bold", pad=20)

        # Method description
        ax = axes[1, 2]
        ax.axis("off")

        desc_text = "\n".join([
            "D1: Per-view crop to center PP at (256,256)",
            "    - Different shift per view -> inconsistency",
            "",
            "D3: 3D triangulation center, no normalization",
            "    - fx~885, translation~237mm",
            "",
            "D7_1: PP-centered shift with individual scale",
            "    - fx=549, cx=cy=256, trans~2.7",
            "",
            "D8: Homography transform with skew correction",
            "    - fx=548.99, cx=cy=256, trans~2.7"
        ])
        ax.text(0.1, 0.5, desc_text, transform=ax.transAxes, fontsize=9,
               verticalalignment="center", fontfamily="monospace")
        ax.set_title("Method Descriptions", fontsize=12, fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / "fig2_ray_error_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        return str(out_path)

    def generate_camera_comparison_plot(self) -> str:
        """Generate 6-view camera parameter comparison."""
        fig, axes = plt.subplots(3, 2, figsize=(14, 16))
        fig.suptitle("Camera Parameter Comparison Across 6 Views",
                     fontsize=14, fontweight="bold")

        params = ["fx", "fy", "cx", "cy"]

        # Load original
        if self.original is None:
            self.load_original_cameras()

        # Collect data
        data = {"Original": {}}
        for p in params:
            data["Original"][p] = [self.original[v][p] for v in range(NUM_VIEWS)]

        for ds_name in self.datasets:
            cams = self.load_dataset_cameras(ds_name, 0)
            if cams:
                data[ds_name] = {}
                for p in params:
                    data[ds_name][p] = [cams[v][p] for v in range(NUM_VIEWS)]

        # Plot each parameter
        x = np.arange(NUM_VIEWS)

        for i, param in enumerate(params):
            ax = axes[i // 2, i % 2]

            # Original
            ax.plot(x, data["Original"][param], "k--", marker="o", label="Original", linewidth=2)

            # Preprocessed
            for ds_name in self.datasets:
                if ds_name in data:
                    config = DATASET_CONFIGS[ds_name]
                    ax.plot(x, data[ds_name][param], marker="s", label=config["name"],
                           color=config["color"], linewidth=1.5)

            ax.set_xlabel("Camera View")
            ax.set_ylabel(f"{param} value")
            ax.set_title(f"Parameter: {param}")
            ax.set_xticks(x)
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Translation comparison
        ax = axes[2, 0]

        # Get translation norms
        trans_orig = [self.original[v]["distance"] for v in range(NUM_VIEWS)]
        ax.plot(x, trans_orig, "k--", marker="o", label="Original", linewidth=2)

        for ds_name in self.datasets:
            cams = self.load_dataset_cameras(ds_name, 0)
            if cams:
                config = DATASET_CONFIGS[ds_name]
                trans = []
                for v in range(NUM_VIEWS):
                    w2c = np.array(cams[v]["w2c"])
                    R = w2c[:3, :3]
                    t = w2c[:3, 3]
                    C = -R.T @ t
                    trans.append(np.linalg.norm(C))
                ax.plot(x, trans, marker="s", label=config["name"],
                       color=config["color"], linewidth=1.5)

        ax.set_xlabel("Camera View")
        ax.set_ylabel("Distance from origin")
        ax.set_title("Camera Distance (||C||)")
        ax.set_xticks(x)
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Summary stats
        ax = axes[2, 1]
        ax.axis("off")

        summary = []
        for ds_name in ["Original"] + self.datasets:
            if ds_name == "Original":
                fx_mean = np.mean(data["Original"]["fx"])
                cx_mean = np.mean(data["Original"]["cx"])
                cy_mean = np.mean(data["Original"]["cy"])
                dist_mean = np.mean(trans_orig)
                summary.append([ds_name, f"{fx_mean:.1f}", f"{cx_mean:.1f}",
                               f"{cy_mean:.1f}", f"{dist_mean:.1f}"])
            elif ds_name in data:
                config = DATASET_CONFIGS[ds_name]
                fx_mean = np.mean(data[ds_name]["fx"])
                cx_mean = np.mean(data[ds_name]["cx"])
                cy_mean = np.mean(data[ds_name]["cy"])
                cams = self.load_dataset_cameras(ds_name, 0)
                if cams:
                    dists = []
                    for v in range(NUM_VIEWS):
                        w2c = np.array(cams[v]["w2c"])
                        R = w2c[:3, :3]
                        t = w2c[:3, 3]
                        C = -R.T @ t
                        dists.append(np.linalg.norm(C))
                    dist_mean = np.mean(dists)
                else:
                    dist_mean = 0
                summary.append([config["name"], f"{fx_mean:.1f}", f"{cx_mean:.1f}",
                               f"{cy_mean:.1f}", f"{dist_mean:.2f}"])

        table = ax.table(cellText=summary,
                        colLabels=["Dataset", "fx (mean)", "cx (mean)", "cy (mean)", "dist (mean)"],
                        cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.5)
        ax.set_title("Parameter Summary", fontsize=12, fontweight="bold", pad=20)

        plt.tight_layout()
        out_path = self.output_dir / "fig3_camera_comparison.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        return str(out_path)

    def generate_html_report(self, figures: List[str]) -> str:
        """Generate HTML report."""
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Preprocessing Comparison Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; }}
        h1 {{ color: #2c3e50; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 5px; }}
        img {{ max-width: 100%; border: 1px solid #ddd; margin: 10px 0; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .highlight {{ background-color: #e8f6ff; }}
        .warning {{ background-color: #fff3e0; }}
        .error {{ background-color: #ffebee; }}
    </style>
</head>
<body>
    <h1>Preprocessing Methods Comparison Report</h1>
    <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

    <h2>1. Overview</h2>
    <p>This report compares preprocessing methods D1, D3, D7_1, and D8 for the FaceLift mouse dataset.</p>

    <table>
        <tr>
            <th>Dataset</th>
            <th>Method</th>
            <th>Description</th>
        </tr>
"""
        for ds_name in self.datasets:
            config = DATASET_CONFIGS[ds_name]
            html += f"""        <tr>
            <td>{config["name"]}</td>
            <td>{config["method"]}</td>
            <td>{config["description"]}</td>
        </tr>
"""
        html += """    </table>

    <h2>2. PP Shift Distribution</h2>
    <p>Shows the principal point shift applied during preprocessing for each camera view.</p>
"""
        if figures[0]:
            html += f'    <img src="{Path(figures[0]).name}" alt="PP Shift Distribution">\n'

        html += """
    <h2>3. Ray Direction Error</h2>
    <p>Compares ray direction errors between original and preprocessed cameras.</p>
"""
        if figures[1]:
            html += f'    <img src="{Path(figures[1]).name}" alt="Ray Error Comparison">\n'

        html += """
    <h2>4. Camera Parameter Comparison</h2>
    <p>Detailed comparison of camera intrinsics across all 6 views.</p>
"""
        if figures[2]:
            html += f'    <img src="{Path(figures[2]).name}" alt="Camera Comparison">\n'

        html += """
    <h2>5. Recommendations</h2>
    <ul>
        <li><strong>D1 (PP-Crop):</strong> Not recommended - per-view shifts cause cross-view inconsistency</li>
        <li><strong>D3 (Triangulation):</strong> Not normalized - requires additional normalization step</li>
        <li><strong>D7_1 (PP-Shift):</strong> Recommended - proper normalization with PP centering</li>
        <li><strong>D8 (Homography):</strong> Most precise - includes skew correction</li>
    </ul>

</body>
</html>
"""
        out_path = self.output_dir / "report.html"
        with open(out_path, "w") as f:
            f.write(html)
        print(f"Saved: {out_path}")
        return str(out_path)

    def run(self) -> Dict:
        """Run full comparison analysis."""
        print("=" * 70)
        print("Preprocessing Methods Comparison Analysis")
        print("=" * 70)

        # Load original data
        print("\n[1] Loading original camera data...")
        self.load_original_cameras()

        # Analyze PP distribution
        print("\n[2] Analyzing PP distribution...")
        pp_data = self.analyze_pp_distribution()

        # Generate figures
        print("\n[3] Generating figures...")
        fig1 = self.generate_pp_distribution_plot(pp_data)
        fig2 = self.generate_ray_error_plot()
        fig3 = self.generate_camera_comparison_plot()

        # Generate HTML report
        print("\n[4] Generating HTML report...")
        report = self.generate_html_report([fig1, fig2, fig3])

        # Summary
        print("\n" + "=" * 70)
        print("Analysis Complete!")
        print(f"Output directory: {self.output_dir}")
        print("=" * 70)

        return {
            "figures": [fig1, fig2, fig3],
            "report": report,
            "pp_data": pp_data,
        }


def main():
    parser = argparse.ArgumentParser(description="Preprocessing Methods Comparison")
    parser.add_argument("--output_dir", type=str,
                       default="/home/joon/dev/FaceLift/mouse_extensions/reports/preprocessing_comparison",
                       help="Output directory for reports")
    parser.add_argument("--datasets", type=str, default="D1,D3,D7_1,D8",
                       help="Comma-separated list of datasets to compare")
    args = parser.parse_args()

    datasets = args.datasets.split(",")
    comparison = PreprocessingComparison(args.output_dir, datasets)
    comparison.run()


if __name__ == "__main__":
    main()


class RGBImageComparison:
    """Generate RGB image comparison across datasets."""

    def __init__(self, output_dir: str, datasets: List[str] = None):
        self.output_dir = Path(output_dir)
        self.datasets = datasets or ["D1", "D3", "D7_1", "D8"]
        self.original_video_dir = Path("/home/joon/data/markerless_mouse_1_nerf/videos_undist")

    def load_original_frame(self, view_id: int, frame_idx: int = 0) -> Optional[np.ndarray]:
        """Load frame from original video."""
        import cv2
        video_path = self.original_video_dir / f"{view_id}.mp4"
        if not video_path.exists():
            return None
        cap = cv2.VideoCapture(str(video_path))
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()
        if ret:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None

    def load_preprocessed_frame(self, dataset: str, view_id: int, sample_idx: int = 0) -> Optional[np.ndarray]:
        """Load preprocessed image."""
        from PIL import Image
        config = DATASET_CONFIGS.get(dataset)
        if not config:
            return None
        sample_name = config["sample_format"].format(sample_idx)
        img_path = Path(config["path"]) / "train" / sample_name / "images" / f"cam_{view_id:03d}.png"
        if img_path.exists():
            return np.array(Image.open(img_path))
        return None

    def generate_6view_comparison(self, sample_idx: int = 0) -> str:
        """Generate 6-view RGB comparison across all datasets."""
        num_datasets = len(self.datasets) + 1  # +1 for original
        fig, axes = plt.subplots(num_datasets, 6, figsize=(24, 4 * num_datasets))
        fig.suptitle(f"6-View RGB Comparison (Sample {sample_idx})", fontsize=16, fontweight="bold")

        row_labels = ["Original (1152x1024)"] + [DATASET_CONFIGS[d]["name"] for d in self.datasets]

        for view_id in range(6):
            # Row 0: Original
            ax = axes[0, view_id]
            orig_img = self.load_original_frame(view_id, sample_idx * 5)  # Assuming frame_step=5
            if orig_img is not None:
                ax.imshow(orig_img)
                ax.set_title(f"View {view_id}\n{orig_img.shape[1]}x{orig_img.shape[0]}", fontsize=10)
            ax.axis("off")
            if view_id == 0:
                ax.text(-0.1, 0.5, row_labels[0], transform=ax.transAxes, fontsize=11,
                       verticalalignment="center", horizontalalignment="right", fontweight="bold")

            # Rows 1+: Preprocessed datasets
            for ds_idx, ds_name in enumerate(self.datasets):
                ax = axes[ds_idx + 1, view_id]
                img = self.load_preprocessed_frame(ds_name, view_id, sample_idx)
                if img is not None:
                    # Handle RGBA
                    if img.shape[-1] == 4:
                        img = img[:, :, :3]
                    ax.imshow(img)
                    ax.set_title(f"{img.shape[1]}x{img.shape[0]}", fontsize=9)
                ax.axis("off")
                if view_id == 0:
                    ax.text(-0.1, 0.5, row_labels[ds_idx + 1], transform=ax.transAxes, fontsize=11,
                           verticalalignment="center", horizontalalignment="right",
                           color=DATASET_CONFIGS[ds_name]["color"], fontweight="bold")

        plt.tight_layout()
        out_path = self.output_dir / f"fig_rgb_comparison_sample{sample_idx:04d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        return str(out_path)

    def generate_detail_crop_comparison(self, sample_idx: int = 0, view_id: int = 0) -> str:
        """Generate detail crop comparison showing PP and mouse position."""
        fig, axes = plt.subplots(2, len(self.datasets) + 1, figsize=(20, 8))
        fig.suptitle(f"Detail Comparison: View {view_id}, Sample {sample_idx}", fontsize=14, fontweight="bold")

        # Row 0: Full images with PP markers
        # Row 1: Center crop (256x256)

        # Original
        orig_img = self.load_original_frame(view_id, sample_idx * 5)
        if orig_img is not None:
            # Full image
            axes[0, 0].imshow(orig_img)
            axes[0, 0].set_title("Original", fontsize=11)
            h, w = orig_img.shape[:2]
            axes[0, 0].plot(w/2, h/2, "+", color="lime", markersize=15, markeredgewidth=2)
            axes[0, 0].axis("off")

            # Center crop
            ch, cw = h//2, w//2
            crop_size = min(h, w) // 4
            crop = orig_img[ch-crop_size:ch+crop_size, cw-crop_size:cw+crop_size]
            axes[1, 0].imshow(crop)
            axes[1, 0].set_title("Center Crop", fontsize=10)
            axes[1, 0].axis("off")

        for ds_idx, ds_name in enumerate(self.datasets):
            config = DATASET_CONFIGS[ds_name]
            img = self.load_preprocessed_frame(ds_name, view_id, sample_idx)
            if img is not None:
                if img.shape[-1] == 4:
                    img = img[:, :, :3]

                # Full image with PP
                axes[0, ds_idx + 1].imshow(img)
                axes[0, ds_idx + 1].set_title(config["name"], fontsize=11, color=config["color"])
                h, w = img.shape[:2]
                # Mark center
                axes[0, ds_idx + 1].plot(w/2, h/2, "+", color="lime", markersize=15, markeredgewidth=2)
                axes[0, ds_idx + 1].axis("off")

                # Center crop
                ch, cw = h//2, w//2
                crop_size = h // 4
                crop = img[ch-crop_size:ch+crop_size, cw-crop_size:cw+crop_size]
                axes[1, ds_idx + 1].imshow(crop)
                axes[1, ds_idx + 1].set_title("Center Crop", fontsize=10)
                axes[1, ds_idx + 1].axis("off")

        plt.tight_layout()
        out_path = self.output_dir / f"fig_detail_crop_v{view_id}_s{sample_idx:04d}.png"
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"Saved: {out_path}")
        return str(out_path)


def run_full_comparison(output_dir: str, datasets: List[str] = None):
    """Run comprehensive comparison with RGB images."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    datasets = datasets or ["D1", "D3", "D7_1", "D8"]

    # 1. Camera parameter comparison
    print("\n[1] Running camera parameter comparison...")
    comparison = PreprocessingComparison(output_dir, datasets)
    comparison.run()

    # 2. RGB image comparison
    print("\n[2] Generating RGB image comparisons...")
    rgb_comp = RGBImageComparison(output_dir, datasets)
    rgb_comp.generate_6view_comparison(sample_idx=0)
    rgb_comp.generate_6view_comparison(sample_idx=100)
    rgb_comp.generate_detail_crop_comparison(sample_idx=0, view_id=0)
    rgb_comp.generate_detail_crop_comparison(sample_idx=0, view_id=3)

    print(f"\nFull comparison complete! Output: {output_dir}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--full":
        datasets = sys.argv[2].split(",") if len(sys.argv) > 2 else None
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "/home/joon/dev/FaceLift/mouse_extensions/reports/preprocessing_comparison_full"
        run_full_comparison(output_dir, datasets)
    else:
        main()
