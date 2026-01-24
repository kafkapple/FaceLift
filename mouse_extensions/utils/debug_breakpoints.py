#!/usr/bin/env python3
"""
Debug Breakpoints Utility

코드에 조건부 브레이크포인트를 삽입하는 유틸리티.
환경변수 DEBUG_MASK=1 설정 시에만 활성화됩니다.

Usage:
    # 환경변수로 디버그 모드 활성화
    DEBUG_MASK=1 python train_gslrm.py -d D7_1 -e E1_2_gt_alpha

    # 또는 launch.json에서
    "env": {"DEBUG_MASK": "1"}
"""

import os
from functools import wraps

# Debug mode check
DEBUG_MASK = os.environ.get("DEBUG_MASK", "0") == "1"
DEBUG_LOSS = os.environ.get("DEBUG_LOSS", "0") == "1"
DEBUG_ALL = os.environ.get("DEBUG_ALL", "0") == "1"


def debug_break(category: str = "all"):
    """
    Conditional breakpoint based on environment variable.
    
    Args:
        category: "mask", "loss", or "all"
    
    Usage:
        from mouse_extensions.utils.debug_breakpoints import debug_break
        debug_break("mask")  # Only breaks if DEBUG_MASK=1
    """
    should_break = False
    
    if DEBUG_ALL:
        should_break = True
    elif category == "mask" and DEBUG_MASK:
        should_break = True
    elif category == "loss" and DEBUG_LOSS:
        should_break = True
    
    if should_break:
        breakpoint()


def debug_on_condition(condition_fn, message: str = ""):
    """
    Break only when condition is met.
    
    Usage:
        debug_on_condition(lambda: loss > 1.0, "Loss too high!")
    """
    if DEBUG_ALL or DEBUG_MASK or DEBUG_LOSS:
        if condition_fn():
            print(f"🔴 DEBUG BREAK: {message}")
            breakpoint()


def debug_inspect(name: str, tensor, save_path: str = None):
    """
    Print tensor info and optionally save as image.
    
    Usage:
        debug_inspect("mask", mask_tensor, "/tmp/mask.png")
    """
    if not (DEBUG_ALL or DEBUG_MASK or DEBUG_LOSS):
        return
    
    import torch
    
    print(f"\n{=*50}")
    print(f"🔍 DEBUG INSPECT: {name}")
    print(f"{=*50}")
    print(f"  Shape: {tensor.shape}")
    print(f"  Dtype: {tensor.dtype}")
    print(f"  Device: {tensor.device}")
    
    if tensor.numel() > 0:
        print(f"  Min: {tensor.min().item():.6f}")
        print(f"  Max: {tensor.max().item():.6f}")
        print(f"  Mean: {tensor.mean().item():.6f}")
        
        if tensor.dtype in [torch.float16, torch.float32, torch.float64]:
            nan_count = torch.isnan(tensor).sum().item()
            inf_count = torch.isinf(tensor).sum().item()
            if nan_count > 0:
                print(f"  ⚠️ NaN count: {nan_count}")
            if inf_count > 0:
                print(f"  ⚠️ Inf count: {inf_count}")
    
    if save_path and len(tensor.shape) >= 3:
        try:
            import torchvision
            # Normalize to [0, 1] if needed
            t = tensor.detach().float()
            if t.min() < 0 or t.max() > 1:
                t = (t - t.min()) / (t.max() - t.min() + 1e-8)
            
            # Handle batch dimension
            if len(t.shape) == 4:
                t = t[0]  # Take first batch
            if len(t.shape) == 4:
                t = t[0]  # Take first view
                
            torchvision.utils.save_image(t, save_path)
            print(f"  💾 Saved to: {save_path}")
        except Exception as e:
            print(f"  ❌ Save failed: {e}")
    
    print(f"{=*50}\n")


# Decorator for function entry/exit debugging
def debug_trace(func):
    """
    Decorator to trace function entry/exit.
    
    Usage:
        @debug_trace
        def compute_loss(...):
            ...
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if DEBUG_ALL:
            print(f">>> ENTER: {func.__name__}")
            breakpoint()
        result = func(*args, **kwargs)
        if DEBUG_ALL:
            print(f"<<< EXIT: {func.__name__}")
        return result
    return wrapper


# Quick reference for common debug points
BREAKPOINT_LOCATIONS = """
╔══════════════════════════════════════════════════════════════════╗
║                    권장 브레이크포인트 위치                        ║
╠══════════════════════════════════════════════════════════════════╣
║ 파일                        │ 라인  │ 용도                       ║
╠════════════════════════════╪═══════╪════════════════════════════╣
║ gslrm/model/gslrm.py       │ ~347  │ images, masks 로딩         ║
║ gslrm/model/gslrm.py       │ ~427  │ compute_loss() 호출        ║
║ loss_extensions.py         │ ~89   │ mask_mode 분기             ║
║ loss_extensions.py         │ ~120  │ compute_mask_from_config() ║
║ loss_extensions.py         │ ~180  │ masked_l2_loss 계산        ║
║ mask_losses.py             │ ~45   │ alpha_loss 계산            ║
╚══════════════════════════════════════════════════════════════════╝

사용법:
    DEBUG_MASK=1 python train_gslrm.py -d D7_1 -e E1_2_gt_alpha
"""

if __name__ == "__main__":
    print(BREAKPOINT_LOCATIONS)
