"""Quick check of M5 sample camera positions and scene centering."""
import json
import os
import numpy as np

sample_dir = "/home/joon/data/preprocessed/FaceLift_mouse/M5"
samples = sorted(os.listdir(sample_dir))
print(f"Total samples: {len(samples)}")

for s_name in ["000000", "000100", "000200", "001000"]:
    cam_path = os.path.join(sample_dir, s_name, "opencv_cameras.json")
    if not os.path.exists(cam_path):
        print(f"\nSample {s_name}: NOT FOUND")
        continue

    with open(cam_path) as f:
        data = json.load(f)
    frames = data["frames"]
    print(f"\nSample {s_name}: {len(frames)} cameras")

    cam_positions = []
    for i, fr in enumerate(frames):
        w2c = np.array(fr["w2c"])
        c2w = np.linalg.inv(w2c)
        cam_pos = c2w[:3, 3]
        cam_positions.append(cam_pos)
        foc = fr.get("fx", fr.get("focal_length", "?"))
        print(f"  cam_{i}: pos=[{cam_pos[0]:.3f}, {cam_pos[1]:.3f}, {cam_pos[2]:.3f}]  fx={foc}")

    cam_positions = np.array(cam_positions)
    center = cam_positions.mean(axis=0)
    dists = np.linalg.norm(cam_positions - center, axis=1)
    print(f"  Camera center: [{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]")
    print(f"  Camera distances from center: mean={dists.mean():.3f}, min={dists.min():.3f}, max={dists.max():.3f}")

    # Check if cameras are looking at origin
    origin_dists = np.linalg.norm(cam_positions, axis=1)
    print(f"  Camera distances from origin: mean={origin_dists.mean():.3f}")
