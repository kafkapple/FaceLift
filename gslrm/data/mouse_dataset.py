"""
DEPRECATED: This module has moved to mouse_extensions.data.mouse_dataset

This file provides backwards compatibility. 
Please update your imports to:
    from mouse_extensions.data import MouseViewDataset
"""

import warnings

warnings.warn(
    "Importing from gslrm.data.mouse_dataset is deprecated. "
    "Please use: from mouse_extensions.data import MouseViewDataset",
    DeprecationWarning,
    stacklevel=2
)

# Re-export for backwards compatibility
from mouse_extensions.data.mouse_dataset import *
from mouse_extensions.data.mouse_dataset import MouseViewDataset

# Also export any utility functions that might be used
try:
    from mouse_extensions.data.mouse_dataset import (
        normalize_cameras_to_y_up,
        normalize_cameras_to_z_up,
        normalize_camera_distance,
    )
except ImportError:
    pass
