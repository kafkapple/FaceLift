"""Unit tests for scene_centered_normalizer."""

import numpy as np

try:
    import pytest
except ImportError:
    pytest = None

from mouse_extensions.preprocessing.scene_centered_normalizer import (
    normalize_cameras_scene_centric,
    compute_scene_center_from_com3d,
)


def make_camera_params(positions):
    """Helper: build cam_params_list from camera positions (assume identity rotation)."""
    params_list = []
    for pos in positions:
        c2w = np.eye(4)
        c2w[:3, 3] = pos
        w2c = np.linalg.inv(c2w)
        params_list.append({"w2c": w2c.tolist(), "fx": 549.0})
    return params_list


def get_positions(params_list):
    positions = []
    for p in params_list:
        w2c = np.asarray(p["w2c"])
        c2w = np.linalg.inv(w2c)
        positions.append(c2w[:3, 3])
    return np.array(positions)


def test_scene_at_origin_after_translation():
    """If scene_center matches camera centroid, result equals mouse-style recenter."""
    # 6 cameras around origin at distance 2.7 (mouse-like)
    positions = np.array([
        [2.7, 0, 0], [-2.7, 0, 0],
        [0, 2.7, 0], [0, -2.7, 0],
        [0, 0, 2.7], [0, 0, -2.7],
    ])
    params = make_camera_params(positions)
    result = normalize_cameras_scene_centric(params, scene_center=[0, 0, 0], target_distance=2.7)
    new_pos = get_positions(result)

    # Should be unchanged (already centered, already at distance 2.7)
    np.testing.assert_allclose(new_pos, positions, atol=1e-9)


def test_offcenter_scene_recenters():
    """Scene at (5, 0, 0) → cameras shift by -5 in x."""
    positions = np.array([
        [5+2.7, 0, 0], [5-2.7, 0, 0],
        [5, 2.7, 0], [5, -2.7, 0],
    ])
    params = make_camera_params(positions)
    result = normalize_cameras_scene_centric(params, scene_center=[5, 0, 0], target_distance=2.7)
    new_pos = get_positions(result)

    expected = positions - np.array([5, 0, 0])
    np.testing.assert_allclose(new_pos, expected, atol=1e-9)


def test_uniform_scaling():
    """Cameras at distance 5 → scaled to distance 2.7."""
    positions = np.array([
        [5, 0, 0], [-5, 0, 0],
        [0, 5, 0], [0, -5, 0],
    ])
    params = make_camera_params(positions)
    result = normalize_cameras_scene_centric(params, scene_center=[0, 0, 0], target_distance=2.7)
    new_pos = get_positions(result)

    # Mean distance after scaling = 2.7
    mean_dist = np.linalg.norm(new_pos, axis=1).mean()
    assert abs(mean_dist - 2.7) < 1e-9


def test_pairwise_angles_preserved():
    """Translation + uniform scale should NOT change pairwise camera angles."""
    # 6 narrow-baseline cameras (rat-like)
    positions = np.array([
        [-0.02, 2.44, 0.94], [0.56, 2.76, 0.66], [-0.33, 2.76, 0.77],
        [-0.02, 2.44, 0.24], [0.34, 2.73, 0.25], [-0.29, 2.54, 0.40],
    ])
    params = make_camera_params(positions)

    # Compute angles before
    def pairwise_angles_around(positions, center):
        vecs = positions - center
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        vn = vecs / norms
        angles = []
        for i in range(len(vn)):
            for j in range(i+1, len(vn)):
                cos_a = np.clip(np.dot(vn[i], vn[j]), -1, 1)
                angles.append(np.degrees(np.arccos(cos_a)))
        return np.array(angles)

    angles_before = pairwise_angles_around(positions, np.array([0.04, 2.61, 0.54]))

    result = normalize_cameras_scene_centric(params, scene_center=[0.04, 2.61, 0.54], target_distance=2.7)
    new_pos = get_positions(result)

    angles_after = pairwise_angles_around(new_pos, np.array([0, 0, 0]))

    np.testing.assert_allclose(angles_before, angles_after, atol=1e-6)


def test_rat_realistic_case():
    """Simulate actual rat sdannce setup."""
    # Camera positions from real sdannce data (in scaled FaceLift units)
    positions = np.array([
        [-0.02, 2.44, 0.94],
        [0.56, 2.76, 0.66],
        [-0.33, 2.76, 0.77],
        [-0.02, 2.44, 0.24],
        [0.34, 2.73, 0.25],
        [-0.29, 2.54, 0.40],
    ])
    # Rat actual mean position (from com3d)
    rat_center = np.array([-0.37, 0.31, 0.12])

    params = make_camera_params(positions)
    result = normalize_cameras_scene_centric(params, scene_center=rat_center, target_distance=2.7)
    new_pos = get_positions(result)

    # Verify rat is now at origin (cameras have shifted away from it)
    # Cam-to-origin distance should equal cam-to-rat distance in original (after scale)
    new_distances = np.linalg.norm(new_pos, axis=1)
    assert abs(new_distances.mean() - 2.7) < 1e-6, f"Mean distance {new_distances.mean()}"

    # Pairwise angles should remain ~6° (narrow baseline preserved)
    def pairwise_mean_angle(positions, center):
        vecs = positions - center
        vn = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        angles = []
        for i in range(len(vn)):
            for j in range(i+1, len(vn)):
                angles.append(np.degrees(np.arccos(np.clip(np.dot(vn[i], vn[j]), -1, 1))))
        return np.mean(angles)

    angle_before = pairwise_mean_angle(positions, rat_center)
    angle_after = pairwise_mean_angle(new_pos, [0, 0, 0])
    assert abs(angle_before - angle_after) < 1e-3


def test_compute_scene_center_median():
    com3d = np.array([
        [1, 2, 3], [1.1, 2.1, 3.1], [0.9, 1.9, 2.9], [10, 20, 30],  # outlier
    ])
    center = compute_scene_center_from_com3d(com3d, method="median")
    np.testing.assert_allclose(center, [1.05, 2.05, 3.05], atol=1e-9)


def test_compute_scene_center_mean():
    com3d = np.array([[0, 0, 0], [2, 4, 6]])
    center = compute_scene_center_from_com3d(com3d, method="mean")
    np.testing.assert_allclose(center, [1, 2, 3])


def test_invalid_scene_center_shape():
    params = make_camera_params([[1, 0, 0], [-1, 0, 0]])
    try:
        normalize_cameras_scene_centric(params, scene_center=[0, 0], target_distance=2.7)
        raise AssertionError("Expected ValueError for invalid shape")
    except ValueError as e:
        assert "must be" in str(e)


def test_empty_input():
    result = normalize_cameras_scene_centric([], scene_center=[0, 0, 0])
    assert result == []


if __name__ == "__main__":
    # Quick run without pytest
    test_scene_at_origin_after_translation()
    print("✓ test_scene_at_origin_after_translation")
    test_offcenter_scene_recenters()
    print("✓ test_offcenter_scene_recenters")
    test_uniform_scaling()
    print("✓ test_uniform_scaling")
    test_pairwise_angles_preserved()
    print("✓ test_pairwise_angles_preserved")
    test_rat_realistic_case()
    print("✓ test_rat_realistic_case")
    test_compute_scene_center_median()
    print("✓ test_compute_scene_center_median")
    test_compute_scene_center_mean()
    print("✓ test_compute_scene_center_mean")
    test_invalid_scene_center_shape()
    print("✓ test_invalid_scene_center_shape")
    test_empty_input()
    print("✓ test_empty_input")
    print("\nAll tests passed.")
