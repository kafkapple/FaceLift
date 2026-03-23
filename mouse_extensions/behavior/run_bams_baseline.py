"""BAMS baseline training on s-DANNCE sparse features.

Trains BAMS (NeurIPS 2023) on raw 3D keypoints from s-DANNCE predictions.
Uses KeypointsDataset which auto-computes states + actions (diff).
After training, extracts embeddings for clustering comparison.

Usage on gpu03:
    source /home/joon/anaconda3/etc/profile.d/conda.sh && conda activate facelift
    cd /home/joon/dev/FaceLift

    # Train BAMS
    CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.behavior.run_bams_baseline \
        --features outputs/analysis/mouse/bams/features/sparse_features.npz \
        --output outputs/analysis/mouse/bams/ \
        --epochs 100

    # Extract embeddings only (from trained model)
    CUDA_VISIBLE_DEVICES=7 python -m mouse_extensions.behavior.run_bams_baseline \
        --features outputs/analysis/mouse/bams/features/sparse_features.npz \
        --output outputs/analysis/mouse/bams/ \
        --extract_only --checkpoint outputs/analysis/mouse/bams/best_model.pt
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, "/home/joon/dev/bams")
from bams.data import KeypointsDataset
from bams.models import BAMS
from bams import HoALoss


def prepare_sequences(
    features_path: str,
    seq_len: int = 1000,
    overlap: int = 500,
    mode: str = "single",  # "single" (rat1 only) or "dyadic"
) -> np.ndarray:
    """Split continuous keypoint data into overlapping sequences.

    Args:
        features_path: Path to sparse_features.npz
        seq_len: Length of each sequence
        overlap: Overlap between consecutive sequences
        mode: "single" (23kp × 3 = 69d) or "dyadic" (46kp × 3 = 138d)

    Returns:
        keypoints: (n_sequences, seq_len, num_feats)
    """
    data = np.load(features_path, allow_pickle=True)

    if mode == "single":
        # Use rat1 raw keypoints: (T, 23, 3) -> (T, 69)
        kp = data["kp1_raw"]  # (90000, 23, 3)
        T, J, D = kp.shape
        kp_flat = kp.reshape(T, J * D)  # (90000, 69)
    elif mode == "dyadic":
        # Both rats: (T, 138)
        kp1 = data["kp1_raw"].reshape(data["kp1_raw"].shape[0], -1)
        kp2 = data["kp2_raw"].reshape(data["kp2_raw"].shape[0], -1)
        kp_flat = np.concatenate([kp1, kp2], axis=1)  # (90000, 138)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    # COM-center per sequence
    # (center each sequence independently for position-invariance)
    T = kp_flat.shape[0]
    step = seq_len - overlap
    sequences = []

    for start in range(0, T - seq_len + 1, step):
        seq = kp_flat[start : start + seq_len].copy()
        # COM-center: subtract mean position per frame
        # For 3D keypoints, center by mean of all joints
        n_joints = seq.shape[1] // 3
        seq_3d = seq.reshape(seq_len, n_joints, 3)
        com = seq_3d.mean(axis=1, keepdims=True)  # (seq_len, 1, 3)
        seq_3d = seq_3d - com
        sequences.append(seq_3d.reshape(seq_len, -1))

    keypoints = np.array(sequences, dtype=np.float32)
    print(f"Prepared {len(sequences)} sequences of length {seq_len} "
          f"({mode}, {kp_flat.shape[1]}d, overlap={overlap})")
    return keypoints


def train_epoch(model, device, loader, optimizer, criterion, epoch):
    """Train one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0

    for data in loader:
        input_data = data["input"].float().to(device)
        target = data["target_hist"].float().to(device)
        ignore_weights = data["ignore_weights"].to(device)

        optimizer.zero_grad()
        embs, hoa_pred, byol_preds = model(input_data)

        # HoA prediction loss
        hoa_loss = criterion(target, hoa_pred, ignore_weights)

        # BYOL short-term loss
        batch_size, sequence_length, emb_dim = embs["short_term"].size()
        skip_frames, delta = 60, 5
        if sequence_length > skip_frames + delta + 1:
            view_1_id = torch.randint(
                sequence_length - skip_frames - delta, (batch_size,)
            ) + skip_frames
            view_2_id = torch.randint(delta + 1, (batch_size,)) + view_1_id
            view_2_id = torch.clip(view_2_id, 0, sequence_length - 1)

            view_1 = byol_preds["short_term"][torch.arange(batch_size), view_1_id]
            view_2 = embs["short_term"][torch.arange(batch_size), view_2_id]
            byol_loss_short = 1 - F.cosine_similarity(
                view_1, view_2.clone().detach(), dim=-1
            ).mean()
        else:
            byol_loss_short = torch.tensor(0.0, device=device)

        # BYOL long-term loss
        batch_size, sequence_length, emb_dim = embs["long_term"].size()
        skip_frames = 100
        if sequence_length > skip_frames + 1:
            view_1_id = torch.randint(
                sequence_length - skip_frames, (batch_size,)
            ) + skip_frames
            view_2_id = torch.randint(
                sequence_length - skip_frames, (batch_size,)
            ) + skip_frames

            view_1 = byol_preds["long_term"][torch.arange(batch_size), view_1_id]
            view_2 = embs["long_term"][torch.arange(batch_size), view_2_id]
            byol_loss_long = 1 - F.cosine_similarity(
                view_1, view_2.clone().detach(), dim=-1
            ).mean()
        else:
            byol_loss_long = torch.tensor(0.0, device=device)

        loss = 5e2 * hoa_loss + 0.5 * byol_loss_short + 0.5 * byol_loss_long

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def extract_embeddings(model, device, keypoints: np.ndarray, seq_len: int = 1000) -> dict:
    """Extract BAMS embeddings for all frames.

    Applies same preprocessing as KeypointsDataset: states + diff(actions).
    Returns dict with short_term and long_term embeddings.
    """
    model.eval()

    if keypoints.ndim == 2:
        keypoints = keypoints[np.newaxis]

    all_short = []
    all_long = []

    for seq in keypoints:
        # Replicate KeypointsDataset preprocessing: states + actions
        states = np.nan_to_num(seq)
        actions = np.diff(states, axis=0, prepend=states[:1])
        input_feats = np.concatenate([states, actions], axis=-1)  # (seq_len, 2*n_feats)

        # Normalize (same as Dataset.process)
        for i in range(input_feats.shape[1]):
            max_val = np.nanmax(np.abs(input_feats[:, i]))
            if max_val > 0:
                input_feats[:, i] /= max_val

        input_tensor = torch.from_numpy(input_feats).float().unsqueeze(0).to(device)
        embs, _, _ = model(input_tensor)
        all_short.append(embs["short_term"].cpu().numpy()[0])
        all_long.append(embs["long_term"].cpu().numpy()[0])

    return {
        "short_term": np.concatenate(all_short, axis=0),
        "long_term": np.concatenate(all_long, axis=0),
        "combined": np.concatenate(
            [np.concatenate(all_short, axis=0), np.concatenate(all_long, axis=0)],
            axis=1,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="BAMS baseline on s-DANNCE")
    parser.add_argument("--features", required=True, help="Path to sparse_features.npz")
    parser.add_argument("--output", default="outputs/analysis/mouse/bams/")
    parser.add_argument("--mode", default="single", choices=["single", "dyadic"])
    parser.add_argument("--seq_len", type=int, default=1000)
    parser.add_argument("--overlap", type=int, default=500)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hoa_bins", type=int, default=32)
    parser.add_argument("--extract_only", action="store_true")
    parser.add_argument("--checkpoint", type=str, default=None)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Prepare data
    print("Preparing sequences...")
    keypoints = prepare_sequences(
        args.features,
        seq_len=args.seq_len,
        overlap=args.overlap,
        mode=args.mode,
    )

    # Create dataset
    dataset = KeypointsDataset(
        keypoints=keypoints,
        hoa_bins=args.hoa_bins,
        cache_path=str(output_dir / "cache"),
        cache=False,
    )
    print(f"Dataset: {len(dataset)} sequences, input_size={dataset.input_size}, "
          f"target_size={dataset.target_size}")

    # Build model
    model = BAMS(
        input_size=dataset.input_size,
        short_term=dict(num_channels=(64, 64, 64, 64), kernel_size=3),
        long_term=dict(num_channels=(64, 64, 64, 64, 64), kernel_size=3, dilation=4),
        predictor=dict(
            hidden_layers=(-1, 256, 512, 512, dataset.target_size * args.hoa_bins)
        ),
    )

    # Initialize lazy modules with a dummy forward pass before moving to device
    dummy_input = torch.randn(1, args.seq_len, dataset.input_size)
    model(dummy_input)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    if args.extract_only:
        if args.checkpoint:
            model.load_state_dict(torch.load(args.checkpoint, map_location=device))
            print(f"Loaded checkpoint: {args.checkpoint}")
        else:
            print("ERROR: --checkpoint required for --extract_only")
            return

        print("Extracting embeddings...")
        embeddings = extract_embeddings(model, device, keypoints)
        emb_path = output_dir / "bams_embeddings.npz"
        np.savez_compressed(
            str(emb_path),
            **embeddings,
            mode=args.mode,
            seq_len=args.seq_len,
        )
        print(f"Saved: {emb_path}")
        for k, v in embeddings.items():
            if isinstance(v, np.ndarray):
                print(f"  {k}: shape={v.shape}")
        return

    # Training
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
    )

    main_params = [p for name, p in model.named_parameters() if "byol" not in name]
    byol_params = list(model.byol_predictors.parameters())

    optimizer = optim.AdamW(
        [{"params": main_params}, {"params": byol_params, "lr": args.lr * 10}],
        lr=args.lr,
        weight_decay=4e-5,
    )
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60], gamma=0.1)
    criterion = HoALoss(hoa_bins=args.hoa_bins, skip_frames=100)

    print(f"\nTraining BAMS for {args.epochs} epochs...")
    best_loss = float("inf")
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, device, loader, optimizer, criterion, epoch)
        scheduler.step()

        if loss < best_loss:
            best_loss = loss
            torch.save(model.state_dict(), str(output_dir / "best_model.pt"))

        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - t0
            print(f"  Epoch {epoch:4d}/{args.epochs} | loss={loss:.4f} | "
                  f"best={best_loss:.4f} | {elapsed:.0f}s")

    elapsed = time.time() - t0
    print(f"\nTraining complete in {elapsed:.0f}s. Best loss: {best_loss:.4f}")
    torch.save(model.state_dict(), str(output_dir / "final_model.pt"))

    # Extract embeddings with best model
    print("\nExtracting embeddings with best model...")
    model.load_state_dict(torch.load(str(output_dir / "best_model.pt"), map_location=device))
    embeddings = extract_embeddings(model, device, keypoints)

    emb_path = output_dir / "bams_embeddings.npz"
    np.savez_compressed(
        str(emb_path),
        **embeddings,
        mode=args.mode,
        seq_len=args.seq_len,
    )
    print(f"Saved embeddings: {emb_path}")
    for k, v in embeddings.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: shape={v.shape}")


if __name__ == "__main__":
    main()
