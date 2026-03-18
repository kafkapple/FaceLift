"""Re-export from mouse_extensions.coordinate_utils for backward compatibility.

SSOT: mouse_extensions/coordinate_utils.py
"""

from mouse_extensions.coordinate_utils import (  # noqa: F401
    M5_SCENE_CENTER,
    M5_DISTANCE_SCALE,
    GSLRM_MAX_ABS,
    MAMMAL_MIN_EXTENT,
    mammal_to_gslrm,
    gslrm_to_mammal,
    assert_gslrm_space,
    assert_mammal_space,
    assert_matching_space,
)
