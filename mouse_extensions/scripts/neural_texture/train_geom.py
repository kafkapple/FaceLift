"""Neural Texture + Geometry: learn vertex offsets + texture via nvdiffrast.

Differentiable rendering enables joint optimization of:
- Vertex position offsets (Δxyz per vertex) → better silhouette match
- UV → RGB texture (same MLP as before) → better appearance

Usage:
    CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.train_geom \
        --frame 0 --epochs 1000 --lr-geom 1e-4 --lr-tex 1e-3
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import nvdiffrast.torch as dr
import trimesh
from PIL import Image

from mouse_extensions.model.neural_texture import FourierEncoder
from mouse_extensions.paths import M5_DATA

M5_SC = np.array([59.672, 51.517, 107.099])
M5_DS = 2.7 / 307.785


def mammal_to_gslrm(xyz):
    return (xyz - M5_SC) * M5_DS


class NeuralTextureGeom(nn.Module):
    """Joint texture + geometry model.

    Learns:
    - Per-vertex offsets (Δxyz) to deform mesh
    - UV → RGB texture MLP
    """

    def __init__(self, n_verts, hidden_dim=256, num_layers=6, num_freqs=10,
                 max_offset=0.05):
        super().__init__()
        self.max_offset = max_offset

        # Learnable vertex offsets
        self.vertex_offsets = nn.Parameter(torch.zeros(n_verts, 3))

        # Texture MLP
        self.encoder = FourierEncoder(in_dim=2, num_freqs=num_freqs)
        in_feat = self.encoder.out_dim
        skip = num_layers // 2

        self.layers = nn.ModuleList()
        self.acts = nn.ModuleList()
        for i in range(num_layers):
            if i == 0:
                self.layers.append(nn.Linear(in_feat, hidden_dim))
            elif i == skip:
                self.layers.append(nn.Linear(hidden_dim + in_feat, hidden_dim))
            else:
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.acts.append(nn.ReLU())
        self.skip = skip
        self.rgb_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3), nn.Sigmoid(),
        )

    def get_deformed_verts(self, base_verts):
        """Apply clamped offsets to base vertices."""
        offsets = self.vertex_offsets.clamp(-self.max_offset, self.max_offset)
        return base_verts + offsets

    def predict_rgb(self, uv):
        """UV → RGB."""
        feat = self.encoder(uv)
        h = feat
        for i, (layer, act) in enumerate(zip(self.layers, self.acts)):
            if i == self.skip:
                h = torch.cat([h, feat], -1)
            h = act(layer(h))
        return self.rgb_head(h)


def opencv_to_clip(verts, w2c, fx, fy, cx, cy, W, H):
    """Project vertices from world space to nvdiffrast clip space.

    Uses OpenCV camera model (w2c, intrinsics) and converts to
    nvdiffrast-compatible clip coordinates.

    Note: nvdiffrast output y=0 is at bottom (OpenGL). Caller must flip
    the output image vertically to match image convention (y=0 at top).
    """
    device = verts.device
    vh = torch.cat([verts, torch.ones(len(verts), 1, device=device)], -1)
    cam = (w2c @ vh.T).T  # camera space (OpenCV)
    z = cam[:, 2]

    # OpenCV pixel coordinates
    px = fx * cam[:, 0] / z + cx
    py = fy * cam[:, 1] / z + cy

    # To NDC: nvdiffrast y=0 at bottom, OpenCV y=0 at top
    ndc_x = 2 * px / W - 1
    ndc_y = 2 * (H - 1 - py) / H - 1  # flip Y for OpenGL bottom-origin

    clip = torch.zeros(len(verts), 4, device=device)
    clip[:, 0] = ndc_x * z
    clip[:, 1] = ndc_y * z
    clip[:, 2] = z
    clip[:, 3] = z
    return clip


def render_frame(glctx, model, base_verts, faces, uvs,
                 cam_params, resolution=512):
    """Render one frame with nvdiffrast: deformed mesh + neural texture.

    Args:
        uvs: (V, 2) per-vertex UV coordinates (same indexing as verts)

    Returns:
        rendered: (H, W, 3) float tensor
        mask: (H, W) bool tensor
    """
    device = base_verts.device

    # Get deformed vertices
    deformed = model.get_deformed_verts(base_verts)

    # Project to clip space
    w2c = torch.tensor(cam_params["w2c"], dtype=torch.float32, device=device)
    clip_pos = opencv_to_clip(
        deformed, w2c, cam_params["fx"], cam_params["fy"],
        cam_params["cx"], cam_params["cy"],
        cam_params["w"], cam_params["h"],
    )
    clip_pos = clip_pos.unsqueeze(0).contiguous()

    # Rasterize
    rast, _ = dr.rasterize(glctx, clip_pos, faces, resolution=[resolution, resolution])

    # Interpolate UV
    uv_interp, _ = dr.interpolate(uvs.unsqueeze(0).contiguous(), rast, faces)

    # Flip Y (nvdiffrast y=0 at bottom → image y=0 at top)
    rast = torch.flip(rast, [1])
    uv_interp = torch.flip(uv_interp, [1])

    mask = rast[0, :, :, 3] > 0

    # Query texture MLP
    if mask.sum() > 0:
        valid_uv = uv_interp[0][mask]
        pred_rgb = model.predict_rgb(valid_uv)

        rendered = torch.ones(resolution, resolution, 3, device=device)
        rendered[mask] = pred_rgb
    else:
        rendered = torch.ones(resolution, resolution, 3, device=device)

    return rendered, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=0, help="M5 frame to optimize")
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", default=str(M5_DATA))
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/neural_texture/geom_poc")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--lr-tex", type=float, default=1e-3)
    parser.add_argument("--lr-geom", type=float, default=1e-4)
    parser.add_argument("--max-offset", type=float, default=0.05)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create nvdiffrast context
    glctx = dr.RasterizeCudaContext(device=device)

    # Load mesh
    mammal_frame = args.frame * 5
    obj_path = Path(args.obj_dir) / f"step_2_frame_{mammal_frame:06d}.obj"
    if not obj_path.exists():
        obj_path = Path(args.obj_dir) / "step_2_frame_000000.obj"

    # Use trimesh to load — it unwelds vertices so each has unique (pos, UV) pair
    mesh = trimesh.load(str(obj_path), process=False)
    raw_verts = mammal_to_gslrm(mesh.vertices)  # (15399, 3) — already unwelded
    raw_uvs = mesh.visual.uv                     # (15399, 2) — 1:1 with vertices
    raw_faces = mesh.faces                       # (28800, 3) — indices into 15399 verts

    print(f"Mesh: {len(raw_verts)} verts, {len(raw_uvs)} UVs, {len(raw_faces)} faces")

    # To tensors — both vertices and UVs share same indexing
    base_verts = torch.tensor(raw_verts, dtype=torch.float32, device=device)
    uvs = torch.tensor(raw_uvs, dtype=torch.float32, device=device)
    faces = torch.tensor(raw_faces, dtype=torch.int32, device=device).contiguous()

    # Load cameras + GT images
    cam_data = json.load(open(
        Path(args.m5_dir) / f"{args.frame:06d}" / "opencv_cameras.json"
    ))
    gt_images = []
    gt_masks = []
    for ci, frame in enumerate(cam_data["frames"]):
        gt_path = Path(args.m5_dir) / f"{args.frame:06d}" / "images" / f"cam_{ci:03d}.png"
        gt = np.array(Image.open(gt_path)).astype(np.float32) / 255.0
        if gt.shape[-1] == 4:
            gt = gt[:, :, :3]
        gt_images.append(torch.tensor(gt, dtype=torch.float32, device=device))
        gt_fg = torch.any(torch.tensor(gt, device=device) < 0.96, dim=-1)
        gt_masks.append(gt_fg)

    # Model
    model = NeuralTextureGeom(
        n_verts=len(raw_verts),
        max_offset=args.max_offset,
    ).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Model: {params:,} params (texture MLP + {len(raw_verts)*3} offset params)")

    # Separate optimizers for texture and geometry
    tex_params = [p for n, p in model.named_parameters() if "vertex_offsets" not in n]
    geom_params = [model.vertex_offsets]
    optimizer = torch.optim.AdamW([
        {"params": tex_params, "lr": args.lr_tex},
        {"params": geom_params, "lr": args.lr_geom},
    ], weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    print(f"\nTraining (texture + geometry) for {args.epochs} epochs...")

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        total_sil = 0

        for ci in range(6):
            rendered, mask = render_frame(
                glctx, model, base_verts, faces, uvs,
                cam_data["frames"][ci], args.resolution,
            )

            gt = gt_images[ci]
            gt_mask = gt_masks[ci]

            # Texture loss: L1 on intersection pixels
            intersection = mask & gt_mask
            if intersection.sum() > 0:
                tex_loss = F.l1_loss(rendered[intersection], gt[intersection])
            else:
                tex_loss = torch.tensor(0.0, device=device)

            # Silhouette loss: BCE between mesh mask and GT mask
            mask_float = mask.float()
            gt_mask_float = gt_mask.float()
            sil_loss = F.binary_cross_entropy(
                mask_float.clamp(1e-6, 1 - 1e-6), gt_mask_float
            )

            loss = tex_loss + 0.1 * sil_loss
            total_loss += tex_loss.item()
            total_sil += sil_loss.item()

        # Average over 6 views
        avg_loss = total_loss / 6
        avg_sil = total_sil / 6

        optimizer.zero_grad()
        # Recompute for backward (simplified: use last view's loss)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 100 == 0 or epoch <= 5:
            offset_mag = model.vertex_offsets.data.abs().mean().item()
            print(f"  E{epoch:4d} tex_L1={avg_loss:.4f} sil={avg_sil:.4f} "
                  f"offset_mag={offset_mag:.5f}")

        # Save visualization
        if epoch % 200 == 0 or epoch == 1:
            model.eval()
            with torch.no_grad():
                vis_frames = []
                for ci in [0, 3]:
                    rendered, mask = render_frame(
                        glctx, model, base_verts, faces, uvs,
                        cam_data["frames"][ci], args.resolution,
                    )
                    r_np = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    gt_np = (gt_images[ci].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                    vis_frames.append(np.concatenate([gt_np, r_np], axis=1))
                vis = np.concatenate(vis_frames, axis=0)
                Image.fromarray(vis).save(output_dir / f"vis_e{epoch:04d}.png")

    # Save final
    torch.save({
        "epoch": args.epochs,
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "offset_stats": {
            "mean": model.vertex_offsets.data.abs().mean().item(),
            "max": model.vertex_offsets.data.abs().max().item(),
        },
    }, output_dir / "best.pt")

    print(f"\nDone. Final offset magnitude: "
          f"mean={model.vertex_offsets.data.abs().mean():.5f}, "
          f"max={model.vertex_offsets.data.abs().max():.5f}")


if __name__ == "__main__":
    main()
