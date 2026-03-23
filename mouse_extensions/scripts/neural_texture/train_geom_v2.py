"""Neural Texture + Geometry v2: soft silhouette loss for stronger geometry learning.

Key improvement over v1: replaces MSE silhouette loss with differentiable
IoU-based loss that focuses on boundary disagreement regions.

Staged training:
  Stage 1 (texture only): Learn appearance with frozen geometry
  Stage 2 (joint): Fine-tune both with soft silhouette loss

Usage:
    PYTHONPATH=/home/joon/dev/FaceLift CUDA_VISIBLE_DEVICES=4 python -m \
        mouse_extensions.scripts.neural_texture.train_geom_v2 \
        --frame 0 --epochs-tex 500 --epochs-joint 500
"""

import argparse
import json
import os
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
from mouse_extensions.scripts.neural_texture.train_geom import (
    NeuralTextureGeom, opencv_to_clip,
)

M5_SC = np.array([59.672, 51.517, 107.099])
M5_DS = 2.7 / 307.785


def soft_iou_loss(pred_mask, gt_mask, eps=1e-6):
    """Differentiable IoU loss using soft masks.

    Focuses gradient on disagreement regions (boundary), much stronger
    signal than MSE on binary masks.
    """
    pred = pred_mask.float()
    gt = gt_mask.float()
    intersection = (pred * gt).sum()
    union = pred.sum() + gt.sum() - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou


def boundary_loss(pred_mask, gt_mask, dilate_k=5):
    """Loss focused on boundary region only.

    Computes MSE only on pixels within dilate_k of the mask boundary,
    ignoring the vast interior/exterior agreement.
    """
    pred = pred_mask.float().unsqueeze(0).unsqueeze(0)
    gt = gt_mask.float().unsqueeze(0).unsqueeze(0)

    # Dilate both masks
    k = dilate_k
    kernel = torch.ones(1, 1, k, k, device=pred.device)
    pred_dilated = F.conv2d(pred, kernel, padding=k // 2) > 0
    gt_dilated = F.conv2d(gt, kernel, padding=k // 2) > 0

    # Boundary = dilated XOR original
    pred_boundary = pred_dilated.squeeze() & ~pred_mask
    gt_boundary = gt_dilated.squeeze() & ~gt_mask

    # Focus region = union of boundaries
    focus = pred_boundary | gt_boundary | (pred_mask != gt_mask)

    if focus.sum() < 10:
        return torch.tensor(0.0, device=pred.device)

    return F.mse_loss(pred_mask[focus].float(), gt_mask[focus].float())


def render_frame_v2(glctx, model, base_verts, faces, uvs,
                    cam_params, resolution=512):
    """Render with nvdiffrast. Returns rendered image, mesh mask, rast output."""
    device = base_verts.device
    deformed = model.get_deformed_verts(base_verts)

    w2c = torch.tensor(
        np.array(cam_params["w2c"]), dtype=torch.float32, device=device
    )
    clip = opencv_to_clip(
        deformed, w2c, cam_params["fx"], cam_params["fy"],
        cam_params["cx"], cam_params["cy"],
        cam_params["w"], cam_params["h"],
    )
    clip = clip.unsqueeze(0).contiguous()

    rast, _ = dr.rasterize(glctx, clip, faces, resolution=[resolution, resolution])
    uv_i, _ = dr.interpolate(uvs.unsqueeze(0).contiguous(), rast, faces)

    # Flip Y (nvdiffrast bottom-origin → image top-origin)
    rast = torch.flip(rast, [1])
    uv_i = torch.flip(uv_i, [1])

    mask = rast[0, :, :, 3] > 0

    rendered = torch.ones(resolution, resolution, 3, device=device)
    if mask.sum() > 0:
        rendered[mask] = model.predict_rgb(uv_i[0][mask])

    return rendered, mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--obj-dir", default="/home/joon/data/synthetic/textured_obj")
    parser.add_argument("--m5-dir", default="/home/joon/data/preprocessed/FaceLift_mouse/M5_4")
    parser.add_argument("--output-dir", default="outputs/analysis/mouse/neural_texture/geom_v2")
    parser.add_argument("--epochs-tex", type=int, default=500)
    parser.add_argument("--epochs-joint", type=int, default=500)
    parser.add_argument("--lr-tex", type=float, default=1e-3)
    parser.add_argument("--lr-geom", type=float, default=5e-5)
    parser.add_argument("--max-offset", type=float, default=0.05)
    parser.add_argument("--sil-weight", type=float, default=2.0)
    parser.add_argument("--sil-type", choices=["iou", "boundary", "mse"], default="iou")
    args = parser.parse_args()

    device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    glctx = dr.RasterizeCudaContext(device=device)

    # Load mesh
    mammal_frame = args.frame * 5
    obj_path = Path(args.obj_dir) / f"step_2_frame_{mammal_frame:06d}.obj"
    if not obj_path.exists():
        obj_path = Path(args.obj_dir) / "step_2_frame_000000.obj"

    mesh = trimesh.load(str(obj_path), process=False)
    mesh.vertices = (mesh.vertices - M5_SC) * M5_DS
    verts = torch.tensor(mesh.vertices, dtype=torch.float32, device=device)
    faces = torch.tensor(mesh.faces, dtype=torch.int32, device=device).contiguous()
    uvs = torch.tensor(mesh.visual.uv, dtype=torch.float32, device=device)
    print(f"Mesh: {len(verts)} verts, {len(faces)} faces")

    # Load GT
    cam_data = json.load(open(
        Path(args.m5_dir) / f"{args.frame:06d}" / "opencv_cameras.json"
    ))
    gt_imgs, gt_masks = [], []
    for ci in range(6):
        g = np.array(Image.open(
            Path(args.m5_dir) / f"{args.frame:06d}" / "images" / f"cam_{ci:03d}.png"
        ))
        gt_imgs.append(torch.tensor(g[:, :, :3].astype(np.float32) / 255, device=device))
        gt_masks.append(torch.tensor(g[:, :, 3] > 128, device=device))

    # Model
    model = NeuralTextureGeom(len(verts), max_offset=args.max_offset).to(device)
    tex_params = [p for n, p in model.named_parameters() if "vertex_offsets" not in n]
    print(f"Model: {sum(p.numel() for p in model.parameters()):,} params")

    # Select silhouette loss
    sil_fn = {"iou": soft_iou_loss, "boundary": boundary_loss,
              "mse": lambda p, g: F.mse_loss(p.float(), g.float())}[args.sil_type]
    print(f"Silhouette loss: {args.sil_type} (weight={args.sil_weight})")

    # ===== STAGE 1: Texture only =====
    model.vertex_offsets.requires_grad_(False)
    opt1 = torch.optim.AdamW(tex_params, lr=args.lr_tex, weight_decay=1e-5)
    sch1 = torch.optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.epochs_tex)

    print(f"\nStage 1: Texture only ({args.epochs_tex} ep)...")
    for ep in range(1, args.epochs_tex + 1):
        model.train()
        total_tex = 0
        for ci in range(6):
            rendered, mask = render_frame_v2(
                glctx, model, verts, faces, uvs, cam_data["frames"][ci])
            inter = mask & gt_masks[ci]
            if inter.sum() > 0:
                total_tex += F.l1_loss(rendered[inter], gt_imgs[ci][inter])
        opt1.zero_grad()
        (total_tex / 6).backward()
        opt1.step()
        sch1.step()
        if ep % 100 == 0 or ep <= 5:
            print(f"  E{ep:4d} L1={total_tex.item() / 6:.4f}")

    tex_final = total_tex.item() / 6

    # ===== STAGE 2: Joint =====
    model.vertex_offsets.requires_grad_(True)
    opt2 = torch.optim.AdamW([
        {"params": tex_params, "lr": args.lr_tex * 0.1},
        {"params": [model.vertex_offsets], "lr": args.lr_geom},
    ], weight_decay=1e-5)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=args.epochs_joint)

    print(f"\nStage 2: Joint ({args.epochs_joint} ep, sil={args.sil_type})...")
    for ep in range(1, args.epochs_joint + 1):
        model.train()
        total_tex = 0
        total_sil = 0
        for ci in range(6):
            rendered, mask = render_frame_v2(
                glctx, model, verts, faces, uvs, cam_data["frames"][ci])
            inter = mask & gt_masks[ci]
            if inter.sum() > 0:
                total_tex += F.l1_loss(rendered[inter], gt_imgs[ci][inter])
            total_sil += sil_fn(mask, gt_masks[ci])

        loss = (total_tex + args.sil_weight * total_sil) / 6
        opt2.zero_grad()
        loss.backward()
        opt2.step()
        sch2.step()

        if ep % 100 == 0 or ep <= 5:
            off = model.vertex_offsets.data.abs()
            print(f"  E{ep:4d} L1={total_tex.item()/6:.4f} "
                  f"sil={total_sil.item()/6:.4f} "
                  f"off={off.mean():.5f}/{off.max():.5f}")

    # ===== Save =====
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": vars(args),
        "tex_final_l1": tex_final,
        "joint_final_l1": total_tex.item() / 6,
    }, output_dir / "best.pt")

    # ===== Visualize =====
    model.eval()
    vis_frames = []
    with torch.no_grad():
        for ci in [0, 3]:
            rendered, mask = render_frame_v2(
                glctx, model, verts, faces, uvs, cam_data["frames"][ci])
            r = (rendered.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            g = (gt_imgs[ci].cpu().numpy() * 255).astype(np.uint8)
            vis_frames.append(np.concatenate([g, r], axis=1))
    Image.fromarray(np.concatenate(vis_frames, axis=0)).save(output_dir / "vis_final.png")

    off = model.vertex_offsets.data.abs()
    print(f"\nDone! tex_L1={total_tex.item()/6:.4f} "
          f"offset: mean={off.mean():.5f} max={off.max():.5f}")


if __name__ == "__main__":
    main()
