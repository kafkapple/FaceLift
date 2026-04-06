# no-split: single MetricsComputer class — _compute_psnr/_ssim/_lpips share instance state and numpy helpers
"""
Metrics computation module for FaceLift evaluation.

Computes PSNR, SSIM, LPIPS between rendered and ground truth images.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
from dataclasses import dataclass, field
import json
from PIL import Image
import torchvision.transforms as T

try:
    import lpips
    LPIPS_AVAILABLE = True
except ImportError:
    LPIPS_AVAILABLE = False

try:
    from skimage.metrics import structural_similarity as ssim
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


@dataclass
class MetricResult:
    """Single sample metric result."""
    sample_id: str
    psnr: float
    ssim: float
    lpips: Optional[float] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class AggregatedMetrics:
    """Aggregated metrics over multiple samples."""
    psnr_mean: float
    psnr_std: float
    ssim_mean: float
    ssim_std: float
    lpips_mean: Optional[float] = None
    lpips_std: Optional[float] = None
    n_samples: int = 0

    def to_dict(self) -> Dict:
        result = {
            "psnr": {"mean": self.psnr_mean, "std": self.psnr_std},
            "ssim": {"mean": self.ssim_mean, "std": self.ssim_std},
            "n_samples": self.n_samples,
        }
        if self.lpips_mean is not None:
            result["lpips"] = {"mean": self.lpips_mean, "std": self.lpips_std}
        return result


class MetricsComputer:
    """
    Compute image quality metrics (PSNR, SSIM, LPIPS).

    Usage:
        computer = MetricsComputer(device='cuda')

        # Single pair
        result = computer.compute(rendered_img, gt_img, sample_id='sample_001')

        # Batch from directories
        results = computer.compute_from_dirs(rendered_dir, gt_dir)
        aggregated = computer.aggregate(results)
    """

    def __init__(
        self,
        device: str = 'cuda',
        lpips_net: str = 'alex',
        compute_lpips: bool = True,
    ):
        """
        Args:
            device: Computation device
            lpips_net: LPIPS network type ('alex', 'vgg', 'squeeze')
            compute_lpips: Whether to compute LPIPS (slower but more perceptual)
        """
        self.device = device
        self.compute_lpips_flag = compute_lpips and LPIPS_AVAILABLE

        if self.compute_lpips_flag:
            self.lpips_fn = lpips.LPIPS(net=lpips_net).to(device)
            self.lpips_fn.eval()
        else:
            self.lpips_fn = None
            if compute_lpips and not LPIPS_AVAILABLE:
                print("Warning: lpips not available, skipping LPIPS computation")

        self.to_tensor = T.ToTensor()

    def compute(
        self,
        rendered: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        gt: Union[torch.Tensor, np.ndarray, Image.Image, str, Path],
        sample_id: str = "unknown",
        mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
        use_gt_alpha_as_mask: bool = True,
    ) -> MetricResult:
        """
        Compute metrics for a single image pair.

        Args:
            rendered: Rendered image (H,W,C) or (C,H,W) tensor, or path
            gt: Ground truth image (can be RGBA, alpha used as mask)
            sample_id: Identifier for this sample
            mask: Optional mask for masked metrics (H,W) or (1,H,W)
            use_gt_alpha_as_mask: If GT has alpha channel, use it as mask

        Returns:
            MetricResult with PSNR, SSIM, and optionally LPIPS
        """
        # Load images (preserving alpha if present)
        rendered_np, rendered_alpha = self._to_numpy_with_alpha(rendered)
        gt_np, gt_alpha = self._to_numpy_with_alpha(gt)

        # Ensure same shape
        if rendered_np.shape[:2] != gt_np.shape[:2]:
            # Resize rendered to match GT
            h, w = gt_np.shape[:2]
            rendered_pil = Image.fromarray((rendered_np * 255).astype(np.uint8))
            rendered_pil = rendered_pil.resize((w, h), Image.BILINEAR)
            rendered_np = np.array(rendered_pil).astype(np.float32) / 255.0
            if rendered_alpha is not None:
                alpha_pil = Image.fromarray((rendered_alpha * 255).astype(np.uint8))
                alpha_pil = alpha_pil.resize((w, h), Image.BILINEAR)
                rendered_alpha = np.array(alpha_pil).astype(np.float32) / 255.0

        # Determine mask for comparison
        if mask is None and use_gt_alpha_as_mask and gt_alpha is not None:
            # Use GT alpha as foreground mask (alpha > 0.5)
            mask = (gt_alpha > 0.5).astype(np.float32)

        # Compute PSNR
        psnr = self._compute_psnr(rendered_np, gt_np, mask)

        # Compute SSIM (on full image for structural comparison)
        ssim_val = self._compute_ssim(rendered_np, gt_np, mask)

        # Compute LPIPS (on full image)
        lpips_val = None
        if self.compute_lpips_flag:
            lpips_val = self._compute_lpips(rendered_np, gt_np)

        return MetricResult(
            sample_id=sample_id,
            psnr=psnr,
            ssim=ssim_val,
            lpips=lpips_val,
        )

    def compute_from_dirs(
        self,
        rendered_dir: Union[str, Path],
        gt_dir: Union[str, Path],
        pattern: str = "*.png",
        gt_pattern: Optional[str] = None,
        sample_list: Optional[List[str]] = None,
    ) -> List[MetricResult]:
        """
        Compute metrics for all matching images in directories.

        Args:
            rendered_dir: Directory containing rendered images
            gt_dir: Directory containing GT images
            pattern: Glob pattern for rendered images
            gt_pattern: Pattern for GT images (if different from rendered naming)
            sample_list: Optional list of sample IDs to process

        Returns:
            List of MetricResult for each matched pair
        """
        rendered_dir = Path(rendered_dir)
        gt_dir = Path(gt_dir)

        results = []

        # Find rendered images
        rendered_files = sorted(rendered_dir.glob(pattern))

        for rendered_path in rendered_files:
            sample_id = rendered_path.stem

            # Skip if not in sample list
            if sample_list and sample_id not in sample_list:
                continue

            # Find corresponding GT
            gt_path = self._find_gt_path(gt_dir, sample_id, gt_pattern)

            if gt_path is None or not gt_path.exists():
                print(f"Warning: No GT found for {sample_id}")
                continue

            result = self.compute(rendered_path, gt_path, sample_id)
            results.append(result)

        return results

    def compute_for_h1_experiment(
        self,
        experiment_dir: Union[str, Path],
        dataset_root: Union[str, Path],
        split: str = "test",
        views_to_compare: List[int] = None,
    ) -> List[MetricResult]:
        """
        Compute metrics for H1 diagnosis experiment output.

        H1 output structure (auto-detected):

        GS-LRM only mode:
            experiment_dir/
                samples/
                    000000/
                        render_view_00.png
                        ...

        E2E mode (with MVDiffusion):
            experiment_dir/
                samples/
                    000000/
                        cam_000/  (reference view used as input)
                            render_view_00.png
                            ...

        Dataset structure:
            dataset_root/
                000000/
                    images/
                        cam_000.png
                        ...

        Args:
            experiment_dir: H1 experiment output directory
            dataset_root: Dataset root
            split: 'train' or 'test' (for logging purposes)
            views_to_compare: List of view indices to compare (default: [0])

        Returns:
            List of MetricResult (one per sample, averaged across views)
        """
        experiment_dir = Path(experiment_dir)
        dataset_root = Path(dataset_root)
        samples_dir = experiment_dir / "samples"

        if not samples_dir.exists():
            print(f"Warning: samples directory not found: {samples_dir}")
            return []

        if views_to_compare is None:
            views_to_compare = [0]  # Default: only compare view 0 (reference)

        results = []

        # Find sample directories
        sample_dirs = sorted([d for d in samples_dir.iterdir() if d.is_dir()])

        for sample_dir in sample_dirs:
            sample_id = sample_dir.name

            # Auto-detect output structure
            # Check if this is E2E mode (has cam_XXX subdirectory)
            render_base_dir = sample_dir
            cam_subdirs = list(sample_dir.glob("cam_*"))
            if cam_subdirs:
                # E2E mode: use first cam subdirectory (typically cam_000)
                render_base_dir = sorted(cam_subdirs)[0]

            # Collect metrics for all views
            view_psnrs = []
            view_ssims = []
            view_lpips = []

            for view_idx in views_to_compare:
                # Find rendered view
                rendered_path = render_base_dir / f"render_view_{view_idx:02d}.png"
                if not rendered_path.exists():
                    continue

                # Find GT view
                gt_path = dataset_root / sample_id / "images" / f"cam_{view_idx:03d}.png"
                if not gt_path.exists():
                    # Try alternative naming
                    gt_path = dataset_root / sample_id / "images" / f"cam_{view_idx:02d}.png"
                if not gt_path.exists():
                    continue

                # Compute metrics for this view
                view_result = self.compute(rendered_path, gt_path, f"{sample_id}_v{view_idx}")
                view_psnrs.append(view_result.psnr)
                view_ssims.append(view_result.ssim)
                if view_result.lpips is not None:
                    view_lpips.append(view_result.lpips)

            # Aggregate across views for this sample
            if view_psnrs:
                avg_psnr = float(np.mean(view_psnrs))
                avg_ssim = float(np.mean(view_ssims))
                avg_lpips = float(np.mean(view_lpips)) if view_lpips else None

                results.append(MetricResult(
                    sample_id=sample_id,
                    psnr=avg_psnr,
                    ssim=avg_ssim,
                    lpips=avg_lpips,
                    metadata={"n_views": len(view_psnrs), "split": split},
                ))

        return results

    def aggregate(self, results: List[MetricResult]) -> AggregatedMetrics:
        """Aggregate metrics over multiple samples."""
        if not results:
            return AggregatedMetrics(
                psnr_mean=0, psnr_std=0,
                ssim_mean=0, ssim_std=0,
                n_samples=0
            )

        psnrs = [r.psnr for r in results]
        ssims = [r.ssim for r in results]

        agg = AggregatedMetrics(
            psnr_mean=float(np.mean(psnrs)),
            psnr_std=float(np.std(psnrs)),
            ssim_mean=float(np.mean(ssims)),
            ssim_std=float(np.std(ssims)),
            n_samples=len(results),
        )

        lpips_vals = [r.lpips for r in results if r.lpips is not None]
        if lpips_vals:
            agg.lpips_mean = float(np.mean(lpips_vals))
            agg.lpips_std = float(np.std(lpips_vals))

        return agg

    def _to_numpy_with_alpha(
        self, img: Union[torch.Tensor, np.ndarray, Image.Image, str, Path]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Convert image to numpy array, preserving alpha channel separately.

        Returns:
            (rgb, alpha) where rgb is (H,W,3) in [0,1], alpha is (H,W) in [0,1] or None
        """
        alpha = None

        if isinstance(img, (str, Path)):
            img = Image.open(img)
            # Don't convert to RGB yet - preserve RGBA if present

        if isinstance(img, Image.Image):
            img_np = np.array(img).astype(np.float32) / 255.0
            if img_np.ndim == 2:
                img_np = np.stack([img_np] * 3, axis=-1)
            elif img_np.shape[-1] == 4:
                alpha = img_np[..., 3]
                img_np = img_np[..., :3]
            elif img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            return img_np.clip(0, 1), alpha

        elif isinstance(img, torch.Tensor):
            img_np = img.detach().cpu().numpy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.shape[0] in [1, 3, 4]:  # CHW format
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.max() > 1.0:
                img_np = img_np / 255.0
            if img_np.shape[-1] == 4:
                alpha = img_np[..., 3]
                img_np = img_np[..., :3]
            elif img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            return img_np.clip(0, 1).astype(np.float32), alpha

        elif isinstance(img, np.ndarray):
            img_np = img.copy()
            if img_np.ndim == 4:
                img_np = img_np[0]
            if img_np.shape[0] in [1, 3, 4] and img_np.ndim == 3:
                img_np = np.transpose(img_np, (1, 2, 0))
            if img_np.max() > 1.0:
                img_np = img_np.astype(np.float32) / 255.0
            else:
                img_np = img_np.astype(np.float32)
            if img_np.shape[-1] == 4:
                alpha = img_np[..., 3]
                img_np = img_np[..., :3]
            elif img_np.shape[-1] == 1:
                img_np = np.repeat(img_np, 3, axis=-1)
            return img_np.clip(0, 1), alpha

        return img.clip(0, 1), None

    def _to_numpy(self, img: Union[torch.Tensor, np.ndarray, Image.Image, str, Path]) -> np.ndarray:
        """Convert various image formats to numpy array (H,W,C) in [0,1]."""
        rgb, _ = self._to_numpy_with_alpha(img)
        return rgb

    def _compute_psnr(
        self,
        rendered: np.ndarray,
        gt: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute PSNR between two images.

        Expects inputs already clipped to [0, 1] (done in compute_per_view_metrics).
        """
        if mask is not None:
            mask = mask.squeeze()
            if mask.ndim == 2:
                mask = mask[..., np.newaxis]
            # Binarize soft alpha masks to avoid denominator underestimation
            binary_mask = (mask > 0.5).astype(np.float32)
            mse = np.sum((rendered - gt) ** 2 * binary_mask) / (np.sum(binary_mask) * 3 + 1e-8)
        else:
            mse = np.mean((rendered - gt) ** 2)

        if mse < 1e-10:
            return 100.0

        return float(10 * np.log10(1.0 / mse))

    def _compute_ssim(
        self,
        rendered: np.ndarray,
        gt: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> float:
        """Compute SSIM between two images."""
        if not SKIMAGE_AVAILABLE:
            return 0.0

        # Convert to uint8 for skimage
        rendered_uint8 = (rendered * 255).astype(np.uint8)
        gt_uint8 = (gt * 255).astype(np.uint8)

        try:
            ssim_val = ssim(
                rendered_uint8,
                gt_uint8,
                channel_axis=-1,
                data_range=255,
            )
            return float(ssim_val)
        except Exception as e:
            print(f"SSIM computation failed: {e}")
            return 0.0

    def _compute_lpips(
        self,
        rendered: np.ndarray,
        gt: np.ndarray,
    ) -> float:
        """Compute LPIPS perceptual distance."""
        if self.lpips_fn is None:
            return 0.0

        # Convert to tensor (B,C,H,W) in [-1, 1]
        rendered_t = torch.from_numpy(rendered).permute(2, 0, 1).unsqueeze(0)
        gt_t = torch.from_numpy(gt).permute(2, 0, 1).unsqueeze(0)

        rendered_t = rendered_t.to(self.device) * 2 - 1
        gt_t = gt_t.to(self.device) * 2 - 1

        with torch.no_grad():
            lpips_val = self.lpips_fn(rendered_t, gt_t)

        return float(lpips_val.item())

    def _find_gt_path(
        self,
        gt_dir: Path,
        sample_id: str,
        gt_pattern: Optional[str] = None,
    ) -> Optional[Path]:
        """Find GT image path for a given sample ID."""
        # Try exact match first
        for ext in ['.png', '.jpg', '.jpeg']:
            gt_path = gt_dir / f"{sample_id}{ext}"
            if gt_path.exists():
                return gt_path

        # Try pattern match
        if gt_pattern:
            pattern = gt_pattern.replace("{sample_id}", sample_id)
            matches = list(gt_dir.glob(pattern))
            if matches:
                return matches[0]

        return None


    def compute_per_view_metrics(
        self,
        target_images: torch.Tensor,
        rendered_images: torch.Tensor,
        gt_mask: Optional[torch.Tensor] = None,
        rendered_alpha: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Compute metrics for each view and aggregate.
        
        Args:
            target_images: (V, 3, H, W) ground truth RGB images
            rendered_images: (V, C, H, W) rendered images (C can be 3 or 4)
            gt_mask: Optional (V, 1, H, W) mask
            rendered_alpha: Optional (V, 1, H, W) rendered alpha from model
        """
        num_views = target_images.size(0)
        
        per_view_psnr = []
        per_view_ssim = []
        per_view_lpips = []
        per_view_l1 = []
        mask_ious = []
        
        for v in range(num_views):
            gt_v = target_images[v]
            rendered_v = rendered_images[v]
            
            if rendered_v.size(0) == 4:
                rendered_rgb = rendered_v[:3]
                alpha_v = rendered_v[3:4]
            else:
                rendered_rgb = rendered_v
                alpha_v = None
            
            # Use externally provided rendered_alpha if available
            if alpha_v is None and rendered_alpha is not None:
                alpha_v = rendered_alpha[v]
            
            mask_v = None
            if gt_mask is not None:
                mask_v = gt_mask[v]
            
            gt_np = gt_v.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            rendered_np = rendered_rgb.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
            
            mask_np = None
            if mask_v is not None:
                mask_np = mask_v.squeeze(0).cpu().numpy()
            
            psnr = self._compute_psnr(rendered_np, gt_np, mask_np)
            ssim_val = self._compute_ssim(rendered_np, gt_np, mask_np)
            
            lpips_val = 0.0
            if self.compute_lpips_flag:
                lpips_val = self._compute_lpips(rendered_np, gt_np)
            
            # L1 error
            if mask_np is not None:
                mask_3d = mask_np[..., np.newaxis] if mask_np.ndim == 2 else mask_np
                l1_val = float(np.sum(np.abs(rendered_np - gt_np) * mask_3d) / (np.sum(mask_3d) * 3 + 1e-8))
            else:
                l1_val = float(np.mean(np.abs(rendered_np - gt_np)))
            
            per_view_psnr.append(psnr)
            per_view_ssim.append(ssim_val)
            per_view_lpips.append(lpips_val)
            per_view_l1.append(l1_val)
            
            if alpha_v is not None and mask_v is not None:
                pred_mask = (alpha_v > 0.5).float()
                gt_mask_v = (mask_v > 0.5).float()
                intersection = (pred_mask * gt_mask_v).sum()
                union = ((pred_mask + gt_mask_v) > 0).float().sum()
                iou = (intersection / (union + 1e-6)).item()
                mask_ious.append(iou)
        
        return {
            "psnr": sum(per_view_psnr) / len(per_view_psnr) if per_view_psnr else 0.0,
            "ssim": sum(per_view_ssim) / len(per_view_ssim) if per_view_ssim else 0.0,
            "lpips": sum(per_view_lpips) / len(per_view_lpips) if per_view_lpips else 0.0,
            "mask_iou": sum(mask_ious) / len(mask_ious) if mask_ious else 0.0,
            "l1": sum(per_view_l1) / len(per_view_l1) if per_view_l1 else 0.0,
            "per_view_psnr": per_view_psnr,
            "per_view_ssim": per_view_ssim,
            "per_view_lpips": per_view_lpips,
            "per_view_l1": per_view_l1,
        }

    def compute_all_views(
        self,
        experiment_dir: Union[str, Path],
        dataset_root: Union[str, Path],
        split: str = "test",
    ) -> List[MetricResult]:
        """
        Compute metrics comparing all 6 views.

        Returns per-sample metrics averaged across all 6 views.
        """
        return self.compute_for_h1_experiment(
            experiment_dir, dataset_root, split,
            views_to_compare=[0, 1, 2, 3, 4, 5]
        )


def compute_metrics_for_experiment(
    experiment_name: str,
    output_root: Path,
    dataset_root: Path,
    split: str = "test",
) -> Dict:
    """
    Convenience function to compute metrics for a named H1 experiment.

    Args:
        experiment_name: e.g., "h1a_gslrm_train", "h1d_e2e_test"
        output_root: Root of eval outputs
        dataset_root: Dataset root
        split: 'train' or 'test'

    Returns:
        Dict with aggregated metrics
    """
    computer = MetricsComputer()

    experiment_dir = output_root / experiment_name
    if not experiment_dir.exists():
        return {"error": f"Experiment dir not found: {experiment_dir}"}

    results = computer.compute_for_h1_experiment(
        experiment_dir, dataset_root, split
    )

    if not results:
        return {"error": "No metrics computed"}

    aggregated = computer.aggregate(results)

    return {
        "experiment": experiment_name,
        "split": split,
        "metrics": aggregated.to_dict(),
        "samples": [
            {"id": r.sample_id, "psnr": r.psnr, "ssim": r.ssim, "lpips": r.lpips}
            for r in results
        ]
    }


