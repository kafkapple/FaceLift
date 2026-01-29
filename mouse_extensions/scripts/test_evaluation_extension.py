#!/usr/bin/env python3
"""
Test Evaluation Extension for GSLRM Training
=============================================

Adds test set evaluation capability to the training script.
Test evaluation runs ONCE at the end of training (not during training).

This module provides:
1. Test dataloader creation
2. run_test() function (similar to run_validation)
3. Test metrics logging to wandb (test/ prefix)

Usage:
    # In train_gslrm.py, add:
    from mouse_extensions.scripts.test_evaluation_extension import (
        create_test_dataloader,
        run_test_evaluation
    )

Created: 2026-01-19
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path

import torch
import wandb


def create_test_dataloader(
    config,
    dataset_class,
    batch_size: int = 1,
    num_workers: int = 4
) -> Optional[torch.utils.data.DataLoader]:
    """
    Create test dataloader if test dataset exists.

    Args:
        config: Training configuration
        dataset_class: Dataset class to use
        batch_size: Batch size for test
        num_workers: Number of workers

    Returns:
        DataLoader or None if test set doesn't exist
    """
    # Get test dataset path
    val_config = config.get("validation", {})
    val_dataset_path = val_config.get("dataset_path", "")

    # Derive test path from validation path
    # e.g., data_mouse_val.txt -> data_mouse_test.txt
    if val_dataset_path:
        test_dataset_path = val_dataset_path.replace("_val.txt", "_test.txt")
    else:
        # Try to derive from training path
        train_path = config.training.dataset.dataset_path
        test_dataset_path = train_path.replace("_train.txt", "_test.txt")

    # Check if test file exists
    if not Path(test_dataset_path).exists():
        print(f"[Test Evaluation] No test set found at {test_dataset_path}")
        return None

    print(f"[Test Evaluation] Found test set at {test_dataset_path}")

    # Create test config
    test_config = config.copy()
    test_config.training.dataset.dataset_path = test_dataset_path

    # Create dataset
    test_dataset = dataset_class(test_config, split="test")

    # Create dataloader
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    print(f"[Test Evaluation] Test dataloader created with {len(test_dataset)} samples")
    return test_dataloader


def run_test_evaluation(
    trainer,  # GSLRMTrainer instance
    test_dataloader: torch.utils.data.DataLoader,
    output_dir: str = "experiments/test",
    log_to_wandb: bool = True
) -> Dict[str, float]:
    """
    Run test evaluation at the end of training.

    This should be called ONCE at the end of training, not during training.
    It evaluates on the held-out test set and logs results to wandb.

    Args:
        trainer: GSLRMTrainer instance
        test_dataloader: Test dataloader
        output_dir: Directory to save test results
        log_to_wandb: Whether to log to wandb

    Returns:
        Dictionary of test metrics
    """
    print(f"\n{'='*60}")
    print("Running FINAL TEST Evaluation")
    print(f"{'='*60}")
    print(f"Test samples: {len(test_dataloader.dataset)}")
    print(f"Output dir: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Get device and model
    device = trainer.device
    model = trainer.model
    model.eval()

    # Metrics collection
    test_metrics = {
        "psnr": [],
        "ssim": [],
        "lpips": [],
        "mask_iou": [],
        "l1": [],
        "psnr_train_mask": [],
        "per_view_psnr": [],
        "per_view_lpips": [],
        "per_view_ssim": []
    }

    # Evaluation loop
    with torch.no_grad():
        amp_dtype = getattr(trainer, 'amp_dtype_mapping', {}).get(
            trainer.config.training.runtime.amp_dtype, torch.bfloat16
        )

        with torch.autocast(
            enabled=trainer.config.training.runtime.use_amp,
            device_type="cuda",
            dtype=amp_dtype
        ):
            for idx, batch in enumerate(test_dataloader):
                batch = {k: v.to(device) for k, v in batch.items()}
                result = model(batch, create_visual=False)

                try:
                    # Save validation results
                    metrics = trainer.model_module.save_validations(
                        output_dir,
                        result,
                        batch,
                        trainer.dataset,
                        save_img=(idx < 5),  # Save images for first 5 samples
                        prefix=f"test_{idx:04d}"
                    )

                    # Collect metrics
                    test_metrics["psnr"].append(metrics["psnr"])
                    test_metrics["ssim"].append(metrics["ssim"])
                    test_metrics["lpips"].append(metrics["lpips"])
                    test_metrics["mask_iou"].append(metrics.get("mask_iou", 0.0))
                    test_metrics["l1"].append(metrics.get("l1", 0.0))
                    test_metrics["psnr_train_mask"].append(
                        metrics.get("psnr_train_mask", metrics["psnr"])
                    )

                    # Collect per-view metrics
                    if "per_view_psnr" in metrics:
                        test_metrics["per_view_psnr"].append(metrics["per_view_psnr"])
                        test_metrics["per_view_lpips"].append(metrics["per_view_lpips"])
                        test_metrics["per_view_ssim"].append(metrics["per_view_ssim"])

                    # Progress
                    if (idx + 1) % 100 == 0:
                        print(f"  Processed {idx + 1}/{len(test_dataloader)} samples")

                except Exception as e:
                    print(f"Error processing test sample {idx}: {e}")

    # Compute averages
    avg_metrics = {
        "psnr": sum(test_metrics["psnr"]) / max(len(test_metrics["psnr"]), 1),
        "ssim": sum(test_metrics["ssim"]) / max(len(test_metrics["ssim"]), 1),
        "lpips": sum(test_metrics["lpips"]) / max(len(test_metrics["lpips"]), 1),
        "mask_iou": sum(test_metrics["mask_iou"]) / max(len(test_metrics["mask_iou"]), 1),
        "l1": sum(test_metrics.get("l1", [0.0])) / max(len(test_metrics.get("l1", [1.0])), 1),
        "psnr_train_mask": sum(test_metrics["psnr_train_mask"]) / max(len(test_metrics["psnr_train_mask"]), 1),
    }

    # Compute per-view averages
    if test_metrics["per_view_psnr"]:
        n_views = len(test_metrics["per_view_psnr"][0])
        for view_idx in range(n_views):
            view_psnr = [m[view_idx] for m in test_metrics["per_view_psnr"]]
            view_lpips = [m[view_idx] for m in test_metrics["per_view_lpips"]]
            view_ssim = [m[view_idx] for m in test_metrics["per_view_ssim"]]

            avg_metrics[f"view{view_idx}_psnr"] = sum(view_psnr) / len(view_psnr)
            avg_metrics[f"view{view_idx}_lpips"] = sum(view_lpips) / len(view_lpips)
            avg_metrics[f"view{view_idx}_ssim"] = sum(view_ssim) / len(view_ssim)

    # Print results
    print(f"\n{'='*60}")
    print("TEST RESULTS (Final Generalization Metrics)")
    print(f"{'='*60}")
    print(f"  PSNR:     {avg_metrics['psnr']:.4f}")
    print(f"  SSIM:     {avg_metrics['ssim']:.4f}")
    print(f"  LPIPS:    {avg_metrics['lpips']:.4f}")
    print(f"  L1:       {avg_metrics.get('l1', 0.0):.4f}")
    print(f"  Mask IoU: {avg_metrics['mask_iou']:.4f}")
    print(f"{'='*60}")
    
    # Pose Splatter comparison table (NeurIPS 2025)
    print()
    print("Comparison with Pose Splatter (NeurIPS 2025):")
    print("-" * 60)
    ps_psnr, ps_ssim, ps_iou, ps_l1 = 29.0, 0.982, 0.760, 0.632
    our_l1 = avg_metrics.get('l1', 0.0)
    header = f"{'Metric':<12} | {'Ours':>10} | {'PoseSplatter':>12} | {'Diff':>10}"
    print(header)
    print("-" * 60)
    print(f"{'PSNR':<12} | {avg_metrics['psnr']:>10.2f} | {ps_psnr:>12.2f} | {avg_metrics['psnr']-ps_psnr:>+10.2f}")
    print(f"{'SSIM':<12} | {avg_metrics['ssim']:>10.4f} | {ps_ssim:>12.4f} | {avg_metrics['ssim']-ps_ssim:>+10.4f}")
    print(f"{'IoU':<12} | {avg_metrics['mask_iou']:>10.4f} | {ps_iou:>12.4f} | {avg_metrics['mask_iou']-ps_iou:>+10.4f}")
    print(f"{'L1':<12} | {our_l1:>10.4f} | {ps_l1:>12.4f} | {our_l1-ps_l1:>+10.4f}")
    print("-" * 60)
    print("Note: Pose Splatter uses 5cam train, 1 holdout view NVS eval")

    # Log to wandb
    if log_to_wandb and trainer.ddp_rank == 0:
        # Derive loss values
        test_l2_loss = 10 ** (-avg_metrics['psnr'] / 10) if avg_metrics['psnr'] > 0 else 1.0
        test_ssim_loss = 1.0 - avg_metrics['ssim']

        # Compute total loss
        loss_weights = trainer.config.training.losses
        test_total_loss = (
            loss_weights.l2_loss_weight * test_l2_loss
            + loss_weights.lpips_loss_weight * avg_metrics['lpips']
            + loss_weights.ssim_loss_weight * test_ssim_loss
        )

        wandb_test_metrics = {
            # Primary test metrics
            "test/loss": test_total_loss,
            "test/l2_loss": test_l2_loss,
            "test/psnr": avg_metrics['psnr'],
            "test/ssim": avg_metrics['ssim'],
            "test/ssim_loss": test_ssim_loss,
            "test/lpips": avg_metrics['lpips'],
            "test/mask_iou": avg_metrics['mask_iou'],
            "test/psnr_train_mask": avg_metrics['psnr_train_mask'],
            "test/num_samples": len(test_metrics['psnr']),
            # Final summary (for easy comparison)
            "final/test_psnr": avg_metrics['psnr'],
            "final/test_lpips": avg_metrics['lpips'],
            "final/test_ssim": avg_metrics['ssim'],
        }

        # Add per-view metrics
        for key, value in avg_metrics.items():
            if key.startswith("view"):
                wandb_test_metrics[f"test_view/{key}"] = value

        # Log at final step
        wandb.log(wandb_test_metrics, step=trainer.fwdbwd_pass_step)

        # Also log as summary for easy access
        for key, value in wandb_test_metrics.items():
            if key.startswith("final/") or key.startswith("test/"):
                wandb.run.summary[key] = value

        # Log test images to wandb (turntable, gt_vs_pred, etc.)
        try:
            import glob
            test_images = {}
            
            # Find and log turntable images
            turntable_files = glob.glob(os.path.join(output_dir, "turntable_*.jpg"))
            for i, tt_file in enumerate(sorted(turntable_files)[:5]):  # Max 5
                test_images[f"test/turntable_{i}"] = wandb.Image(
                    tt_file, caption=f"Test Sample {i} Turntable"
                )
            
            # Find and log gt_vs_pred images
            gt_pred_files = glob.glob(os.path.join(output_dir, "*_gt_vs_pred.jpg")) + \
                           glob.glob(os.path.join(output_dir, "*gt_pred*.jpg"))
            for i, gp_file in enumerate(sorted(gt_pred_files)[:5]):  # Max 5
                test_images[f"test/gt_vs_pred_{i}"] = wandb.Image(
                    gp_file, caption=f"Test Sample {i} GT vs Pred"
                )
            
            if test_images:
                wandb.log(test_images, step=trainer.fwdbwd_pass_step)
                print(f"[Test Evaluation] Logged {len(test_images)} test images to wandb")
        except Exception as e:
            print(f"[Test Evaluation] Warning: Could not log test images: {e}")

        print(f"[Test Evaluation] Logged test metrics to wandb")

    # Save metrics to file
    import json
    metrics_file = os.path.join(output_dir, "test_metrics.json")
    with open(metrics_file, "w") as f:
        json.dump({
            "avg_metrics": avg_metrics,
            "num_samples": len(test_metrics['psnr']),
            "step": trainer.fwdbwd_pass_step
        }, f, indent=2)
    print(f"[Test Evaluation] Saved metrics to {metrics_file}")

    # Return to training mode
    model.train()

    return avg_metrics


def add_test_evaluation_to_trainer(trainer_class):
    """
    Decorator/mixin to add test evaluation capability to GSLRMTrainer.

    Usage:
        GSLRMTrainer = add_test_evaluation_to_trainer(GSLRMTrainer)
    """
    original_init = trainer_class.__init__
    original_train = trainer_class.train

    def new_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.test_dataloader = None

    def setup_test_dataloader(self):
        """Setup test dataloader if available."""
        if hasattr(self, 'dataset') and self.dataset is not None:
            self.test_dataloader = create_test_dataloader(
                self.config,
                type(self.dataset),
                batch_size=1,
                num_workers=self.config.training.dataloader.num_workers
            )

    def run_final_test(self):
        """Run final test evaluation."""
        if self.test_dataloader is None:
            print("[Test Evaluation] No test dataloader available, skipping test evaluation")
            return None

        test_output_dir = self.config.get("validation", {}).get(
            "output_dir", "experiments/validation"
        ).replace("validation", "test")

        return run_test_evaluation(
            self,
            self.test_dataloader,
            output_dir=test_output_dir,
            log_to_wandb=True
        )

    def new_train(self):
        """Modified train method that runs test at the end."""
        result = original_train(self)

        # Run test evaluation at the very end
        if hasattr(self, 'test_dataloader') and self.test_dataloader is not None:
            self.run_final_test()

        return result

    trainer_class.__init__ = new_init
    trainer_class.setup_test_dataloader = setup_test_dataloader
    trainer_class.run_final_test = run_final_test
    trainer_class.train = new_train

    return trainer_class


if __name__ == "__main__":
    # Demo/test code
    print("Test Evaluation Extension Module")
    print("=" * 60)
    print("""
Usage in train_gslrm.py:

1. Import the module:
   from mouse_extensions.scripts.test_evaluation_extension import (
       create_test_dataloader,
       run_test_evaluation
   )

2. After training, call:
   test_dataloader = create_test_dataloader(config, dataset_class)
   if test_dataloader:
       run_test_evaluation(trainer, test_dataloader)

Or use the decorator:
   GSLRMTrainer = add_test_evaluation_to_trainer(GSLRMTrainer)
   # Then test will run automatically at the end of training
""")
