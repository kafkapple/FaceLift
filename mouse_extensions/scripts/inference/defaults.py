"""Default paths and settings for mouse inference scripts."""

import os
from pathlib import Path

# Base paths
FACELIFT_ROOT = Path("/home/joon/dev/FaceLift")
DATA_ROOT = Path("/home/joon/data/preprocessed/FaceLift_mouse")
CHECKPOINT_ROOT = Path("/node_data/joon/checkpoints/FaceLift")

# Default checkpoints
DEFAULT_CHECKPOINTS = {
    "M5t": {
        "gslrm": CHECKPOINT_ROOT / "gslrm/M5t_E0_1_facelift/best_psnr.pt",
        "mvdiffusion": CHECKPOINT_ROOT / "mvdiffusion/mouse_M5t/checkpoint-8000",
    },
    "M5t2": {
        "gslrm": CHECKPOINT_ROOT / "gslrm/M5t2_E0_1_facelift/best_psnr.pt",
        "mvdiffusion": CHECKPOINT_ROOT / "mvdiffusion/mouse_M5t2/checkpoint-5000",
    },
}

# Default splits
DEFAULT_SPLITS = {
    "M5t": DATA_ROOT / "M5/data_mouse_1to1_test.txt",
    "M5t2": DATA_ROOT / "M5/data_mouse_t2_test.txt",
}

# Default data directory
DEFAULT_DATA_DIR = DATA_ROOT / "M5"

# Slow playback defaults (recommended for analysis)
SLOW_DEFAULTS = {
    "fps": 10,
    "rotation_speed": 0.3,
    "num_views": 60,
    "end_frame": 200,
}

# Fast playback defaults
FAST_DEFAULTS = {
    "fps": 24,
    "rotation_speed": 0.5,
    "num_views": 36,
}

def get_checkpoint(model: str = "M5t", type: str = "gslrm") -> Path:
    """Get default checkpoint path."""
    return DEFAULT_CHECKPOINTS.get(model, DEFAULT_CHECKPOINTS["M5t"])[type]

def get_split(model: str = "M5t") -> Path:
    """Get default split file path."""
    return DEFAULT_SPLITS.get(model, DEFAULT_SPLITS["M5t"])

def expand_path(path: str) -> str:
    """Expand ~ and environment variables in path."""
    return os.path.expanduser(os.path.expandvars(path))

# Prompt embed paths
PROMPT_EMBED_PATH = Path("/home/joon/dev/FaceLift/mvdiffusion/data/mouse_prompt_embeds_6view_1024")
