"""
Visualization module for FaceLift evaluation reports.

Generates GT vs Prediction comparison images and grids.

Usage:
    from mouse_extensions.evaluation.visualization import VisualizationGenerator

    vis = VisualizationGenerator(output_dir="outputs/reports/images")

    # Single sample comparison
    vis.create_gt_vs_pred_grid(
        gt_images=[view0, view1, ...],
        pred_images=[pred0, pred1, ...],
        sample_id="000001",
        view_labels=["cam_000", "cam_001", ...]
    )

    # Experiment comparison grid
    vis.create_experiment_comparison(
        experiments={
            "3-view": {"pred_dir": "...", "gt_dir": "..."},
            "5-view": {"pred_dir": "...", "gt_dir": "..."},
        },
        sample_ids=["000001", "000002", ...]
    )
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
from PIL import Image, ImageDraw, ImageFont
import io


def get_font(size: int = 16) -> ImageFont.FreeTypeFont:
    """Get a font, falling back to default if custom font not available."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except (OSError, IOError):
        try:
            return ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", size)
        except (OSError, IOError):
            return ImageFont.load_default()


class VisualizationGenerator:
    """
    Generate visualization images for experiment evaluation.

    Features:
    - GT vs Prediction side-by-side comparison
    - Multi-view grid layout
    - Experiment comparison across multiple conditions
    - Error heatmaps (optional)
    """

    def __init__(
        self,
        output_dir: Union[str, Path],
        default_image_size: Tuple[int, int] = (256, 256),
        font_size: int = 16,
        border_width: int = 2,
        label_height: int = 24,
    ):
        """
        Initialize visualization generator.

        Args:
            output_dir: Output directory for generated images
            default_image_size: Default size for images (width, height)
            font_size: Font size for labels
            border_width: Border width around images
            label_height: Height for label rows
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.default_image_size = default_image_size
        self.font_size = font_size
        self.border_width = border_width
        self.label_height = label_height
        self.font = get_font(font_size)

    def load_image(
        self,
        path: Union[str, Path],
        size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """Load and optionally resize an image."""
        img = Image.open(path)
        if img.mode == "RGBA":
            # Convert RGBA to RGB with white background
            background = Image.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif img.mode != "RGB":
            img = img.convert("RGB")

        if size is not None:
            img = img.resize(size, Image.LANCZOS)

        return img

    def add_label(
        self,
        image: Image.Image,
        label: str,
        position: str = "top",
        bg_color: Tuple[int, int, int] = (40, 40, 40),
        text_color: Tuple[int, int, int] = (255, 255, 255),
    ) -> Image.Image:
        """Add a label bar to an image."""
        w, h = image.size
        label_h = self.label_height

        if position == "top":
            new_img = Image.new("RGB", (w, h + label_h), bg_color)
            new_img.paste(image, (0, label_h))
            text_y = (label_h - self.font_size) // 2
        else:  # bottom
            new_img = Image.new("RGB", (w, h + label_h), bg_color)
            new_img.paste(image, (0, 0))
            text_y = h + (label_h - self.font_size) // 2

        draw = ImageDraw.Draw(new_img)
        # Center text
        try:
            text_bbox = draw.textbbox((0, 0), label, font=self.font)
            text_w = text_bbox[2] - text_bbox[0]
        except AttributeError:
            text_w = len(label) * 8
        text_x = (w - text_w) // 2
        draw.text((text_x, text_y), label, fill=text_color, font=self.font)

        return new_img

    def create_gt_vs_pred_row(
        self,
        gt_image: Union[str, Path, Image.Image],
        pred_image: Union[str, Path, Image.Image],
        view_label: str = "",
        add_labels: bool = True,
        image_size: Optional[Tuple[int, int]] = None,
    ) -> Image.Image:
        """
        Create a single row with GT and Prediction side by side.

        Args:
            gt_image: Ground truth image (path or PIL Image)
            pred_image: Prediction image (path or PIL Image)
            view_label: Label for this view (e.g., "cam_000")
            add_labels: Whether to add GT/Pred labels
            image_size: Size to resize images to

        Returns:
            Combined image with GT | Pred layout
        """
        size = image_size or self.default_image_size

        # Load images
        if isinstance(gt_image, (str, Path)):
            gt = self.load_image(gt_image, size)
        else:
            gt = gt_image.resize(size, Image.LANCZOS) if gt_image.size != size else gt_image

        if isinstance(pred_image, (str, Path)):
            pred = self.load_image(pred_image, size)
        else:
            pred = pred_image.resize(size, Image.LANCZOS) if pred_image.size != size else pred_image

        # Add labels if requested
        if add_labels:
            gt = self.add_label(gt, f"GT ({view_label})" if view_label else "GT")
            pred = self.add_label(pred, f"Pred ({view_label})" if view_label else "Pred")

        # Combine horizontally
        w, h = gt.size
        combined = Image.new("RGB", (w * 2 + self.border_width, h), (128, 128, 128))
        combined.paste(gt, (0, 0))
        combined.paste(pred, (w + self.border_width, 0))

        return combined

    def create_gt_vs_pred_grid(
        self,
        gt_images: List[Union[str, Path, Image.Image]],
        pred_images: List[Union[str, Path, Image.Image]],
        sample_id: str = "",
        view_labels: Optional[List[str]] = None,
        image_size: Optional[Tuple[int, int]] = None,
        max_cols: int = 3,
        save: bool = True,
        filename: Optional[str] = None,
    ) -> Image.Image:
        """
        Create a grid comparing GT and Predictions for multiple views.

        Layout:
            View 0: [GT] [Pred]
            View 1: [GT] [Pred]
            ...

        Args:
            gt_images: List of GT images (paths or PIL Images)
            pred_images: List of prediction images
            sample_id: Sample identifier for filename
            view_labels: Labels for each view
            image_size: Size for each image
            max_cols: Maximum columns (each col is GT+Pred pair)
            save: Whether to save to file
            filename: Custom filename (default: gt_vs_pred_{sample_id}.png)

        Returns:
            Combined grid image
        """
        assert len(gt_images) == len(pred_images), "GT and Pred must have same length"

        n_views = len(gt_images)
        if view_labels is None:
            view_labels = [f"view_{i}" for i in range(n_views)]

        size = image_size or self.default_image_size

        # Create rows
        rows = []
        for i, (gt, pred, label) in enumerate(zip(gt_images, pred_images, view_labels)):
            row = self.create_gt_vs_pred_row(gt, pred, label, image_size=size)
            rows.append(row)

        # Stack vertically
        if rows:
            row_w, row_h = rows[0].size
            grid = Image.new("RGB", (row_w, row_h * n_views + self.border_width * (n_views - 1)), (64, 64, 64))

            y = 0
            for row in rows:
                grid.paste(row, (0, y))
                y += row_h + self.border_width
        else:
            grid = Image.new("RGB", (256, 256), (64, 64, 64))

        # Add sample ID header
        if sample_id:
            grid = self.add_label(grid, f"Sample: {sample_id}", position="top")

        # Save if requested
        if save:
            fname = filename or f"gt_vs_pred_{sample_id}.png"
            save_path = self.output_dir / fname
            grid.save(save_path, quality=95)

        return grid

    def create_experiment_comparison(
        self,
        experiments: Dict[str, Dict[str, Union[str, Path]]],
        sample_ids: List[str],
        view_idx: int = 0,
        image_size: Optional[Tuple[int, int]] = None,
        save: bool = True,
        filename: str = "experiment_comparison.png",
    ) -> Image.Image:
        """
        Create a grid comparing multiple experiments.

        Layout (for view 0):
                    Exp1      Exp2      Exp3
            GT     [img]     [img]     [img]
            Pred   [img]     [img]     [img]
            Sample1
            ----
            GT     [img]     [img]     [img]
            Pred   [img]     [img]     [img]
            Sample2

        Args:
            experiments: Dict mapping experiment name to {"gt_dir": ..., "pred_dir": ...}
            sample_ids: List of sample IDs to include
            view_idx: Which view to compare
            image_size: Size for each image
            save: Whether to save
            filename: Output filename

        Returns:
            Comparison grid image
        """
        size = image_size or self.default_image_size
        exp_names = list(experiments.keys())
        n_exps = len(exp_names)
        n_samples = len(sample_ids)

        # Calculate dimensions
        img_w, img_h = size
        label_w = 60  # Width for row labels
        header_h = self.label_height

        # Total size
        total_w = label_w + (img_w + self.border_width) * n_exps
        rows_per_sample = 2  # GT and Pred
        sample_h = header_h + (img_h + self.label_height) * rows_per_sample + self.border_width * 2
        total_h = header_h + sample_h * n_samples

        # Create canvas
        canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
        draw = ImageDraw.Draw(canvas)

        # Draw header row (experiment names)
        x = label_w
        for exp_name in exp_names:
            # Center text in column
            try:
                text_bbox = draw.textbbox((0, 0), exp_name, font=self.font)
                text_w = text_bbox[2] - text_bbox[0]
            except AttributeError:
                text_w = len(exp_name) * 8
            text_x = x + (img_w - text_w) // 2
            draw.text((text_x, 4), exp_name, fill=(0, 0, 0), font=self.font)
            x += img_w + self.border_width

        # Draw samples
        y = header_h
        for sample_id in sample_ids:
            # Sample header
            draw.rectangle([(0, y), (total_w, y + header_h)], fill=(220, 220, 220))
            draw.text((4, y + 4), sample_id, fill=(0, 0, 0), font=self.font)
            y += header_h

            # GT row
            draw.text((4, y + img_h // 2 - 8), "GT", fill=(0, 0, 0), font=self.font)
            x = label_w
            for exp_name in exp_names:
                exp_info = experiments[exp_name]
                gt_dir = Path(exp_info.get("gt_dir", ""))
                gt_path = gt_dir / sample_id / "images" / f"cam_{view_idx:03d}.png"

                if gt_path.exists():
                    img = self.load_image(gt_path, size)
                    canvas.paste(img, (x, y))
                else:
                    # Placeholder
                    draw.rectangle([(x, y), (x + img_w, y + img_h)], fill=(200, 200, 200))
                    draw.text((x + 10, y + img_h // 2), "N/A", fill=(100, 100, 100), font=self.font)

                x += img_w + self.border_width

            y += img_h + self.border_width

            # Pred row
            draw.text((4, y + img_h // 2 - 8), "Pred", fill=(0, 0, 0), font=self.font)
            x = label_w
            for exp_name in exp_names:
                exp_info = experiments[exp_name]
                pred_dir = Path(exp_info.get("pred_dir", ""))

                # Try different prediction path formats
                pred_paths = [
                    pred_dir / sample_id / f"render_view_{view_idx:02d}.png",
                    pred_dir / "samples" / sample_id / f"render_view_{view_idx:02d}.png",
                ]

                img_loaded = False
                for pred_path in pred_paths:
                    if pred_path.exists():
                        img = self.load_image(pred_path, size)
                        canvas.paste(img, (x, y))
                        img_loaded = True
                        break

                if not img_loaded:
                    draw.rectangle([(x, y), (x + img_w, y + img_h)], fill=(200, 200, 200))
                    draw.text((x + 10, y + img_h // 2), "N/A", fill=(100, 100, 100), font=self.font)

                x += img_w + self.border_width

            y += img_h + self.border_width

        if save:
            save_path = self.output_dir / filename
            canvas.save(save_path, quality=95)

        return canvas

    def create_view_ablation_grid(
        self,
        checkpoint_dirs: Dict[str, Path],
        dataset_root: Path,
        sample_ids: List[str],
        save: bool = True,
        filename: str = "view_ablation_grid.png",
    ) -> Image.Image:
        """
        Create a grid for view ablation experiments.

        Specialized for comparing different input view counts.

        Args:
            checkpoint_dirs: Dict mapping "N-view" to checkpoint directory
            dataset_root: Dataset root for GT images
            sample_ids: Sample IDs to include
            save: Whether to save
            filename: Output filename

        Returns:
            Comparison grid
        """
        experiments = {}
        for name, ckpt_dir in checkpoint_dirs.items():
            # Find render directory (usually in iter_XXXXX folders)
            render_dirs = sorted(ckpt_dir.glob("iter_*/"))
            if render_dirs:
                latest_iter = render_dirs[-1]
                experiments[name] = {
                    "gt_dir": dataset_root,
                    "pred_dir": latest_iter,
                }

        return self.create_experiment_comparison(
            experiments, sample_ids, save=save, filename=filename
        )

    def create_error_heatmap(
        self,
        gt_image: Union[str, Path, Image.Image],
        pred_image: Union[str, Path, Image.Image],
        save_path: Optional[Path] = None,
    ) -> Image.Image:
        """
        Create an error heatmap showing per-pixel differences.

        Args:
            gt_image: Ground truth image
            pred_image: Prediction image
            save_path: Optional path to save

        Returns:
            Error heatmap image
        """
        # Load images
        if isinstance(gt_image, (str, Path)):
            gt = np.array(self.load_image(gt_image))
        else:
            gt = np.array(gt_image)

        if isinstance(pred_image, (str, Path)):
            pred = np.array(self.load_image(pred_image))
        else:
            pred = np.array(pred_image)

        # Ensure same size
        if gt.shape != pred.shape:
            pred = np.array(Image.fromarray(pred).resize((gt.shape[1], gt.shape[0]), Image.LANCZOS))

        # Compute error
        error = np.abs(gt.astype(float) - pred.astype(float)).mean(axis=-1)
        error = (error / error.max() * 255).astype(np.uint8)

        # Apply colormap (simple red gradient)
        heatmap = np.zeros((*error.shape, 3), dtype=np.uint8)
        heatmap[..., 0] = error  # Red channel
        heatmap[..., 2] = 255 - error  # Blue channel (inverse)

        result = Image.fromarray(heatmap)

        if save_path:
            result.save(save_path)

        return result


def generate_report_visualizations(
    experiments_dir: Path,
    dataset_root: Path,
    output_dir: Path,
    sample_ids: Optional[List[str]] = None,
    max_samples: int = 5,
) -> Dict[str, Path]:
    """
    Generate all visualization images for a set of experiments.

    Args:
        experiments_dir: Directory containing experiment subdirectories
        dataset_root: Dataset root for GT images
        output_dir: Output directory for visualizations
        sample_ids: Specific samples to visualize (default: first N from dataset)
        max_samples: Maximum number of samples if sample_ids not provided

    Returns:
        Dict mapping visualization type to output path
    """
    vis = VisualizationGenerator(output_dir)

    # Find experiment directories
    exp_dirs = {}
    for d in experiments_dir.iterdir():
        if d.is_dir() and (d / "config.yaml").exists():
            exp_dirs[d.name] = d

    # Get sample IDs if not provided
    if sample_ids is None:
        sample_dirs = sorted((dataset_root).glob("*/"))[:max_samples]
        sample_ids = [d.name for d in sample_dirs if d.is_dir()]

    outputs = {}

    # Generate experiment comparison
    if exp_dirs:
        experiments = {
            name: {
                "gt_dir": dataset_root,
                "pred_dir": d,
            }
            for name, d in exp_dirs.items()
        }

        grid = vis.create_experiment_comparison(
            experiments,
            sample_ids,
            filename="experiment_comparison.png"
        )
        outputs["comparison"] = output_dir / "experiment_comparison.png"

    # Generate per-sample GT vs Pred grids
    for sample_id in sample_ids[:3]:  # Limit to 3 samples
        gt_dir = dataset_root / sample_id / "images"
        if not gt_dir.exists():
            continue

        gt_images = sorted(gt_dir.glob("cam_*.png"))[:6]

        for exp_name, exp_dir in exp_dirs.items():
            # Find predictions
            pred_base = exp_dir
            for iter_dir in sorted(exp_dir.glob("iter_*/")):
                sample_pred_dir = iter_dir / sample_id
                if sample_pred_dir.exists():
                    pred_base = iter_dir
                    break

            pred_images = [
                pred_base / sample_id / f"render_view_{i:02d}.png"
                for i in range(len(gt_images))
            ]

            # Check if predictions exist
            existing_preds = [p for p in pred_images if p.exists()]
            if existing_preds:
                view_labels = [f"cam_{i:03d}" for i in range(len(gt_images))]
                grid = vis.create_gt_vs_pred_grid(
                    gt_images,
                    pred_images,
                    sample_id=f"{exp_name}_{sample_id}",
                    view_labels=view_labels,
                )
                outputs[f"gt_vs_pred_{exp_name}_{sample_id}"] = (
                    output_dir / f"gt_vs_pred_{exp_name}_{sample_id}.png"
                )

    return outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate evaluation visualizations")
    parser.add_argument(
        "--experiments_dir",
        type=str,
        help="Directory containing experiment subdirectories",
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/home/joon/data/preprocessed/FaceLift_mouse/M5",
        help="Dataset root for GT images",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/reports/images",
        help="Output directory for visualizations",
    )
    parser.add_argument(
        "--samples",
        type=str,
        nargs="+",
        help="Sample IDs to visualize",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=5,
        help="Maximum number of samples",
    )

    args = parser.parse_args()

    if args.experiments_dir:
        outputs = generate_report_visualizations(
            Path(args.experiments_dir),
            Path(args.dataset_root),
            Path(args.output_dir),
            sample_ids=args.samples,
            max_samples=args.max_samples,
        )
        print(f"Generated {len(outputs)} visualizations")
        for name, path in outputs.items():
            print(f"  {name}: {path}")
    else:
        print("Usage: python -m mouse_extensions.evaluation.visualization --experiments_dir /path/to/experiments")
