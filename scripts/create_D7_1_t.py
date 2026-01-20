#!/usr/bin/env python3
"""Create D7_1_t dataset with temporal split from D7_1"""
import os
import shutil
from pathlib import Path

# Paths
D7_1_DIR = Path('/home/joon/data/preprocessed/FaceLift_mouse/D7_1')
D7_1_T_DIR = Path('/home/joon/data/preprocessed/FaceLift_mouse/D7_1_t')

# Get all samples from D7_1 (both train and val)
all_samples = []

# Collect train samples
train_dir = D7_1_DIR / 'train'
if train_dir.exists():
    for d in sorted(train_dir.iterdir()):
        if d.is_dir():
            all_samples.append(('train', d.name))

# Collect val samples  
val_dir = D7_1_DIR / 'val'
if val_dir.exists():
    for d in sorted(val_dir.iterdir()):
        if d.is_dir():
            all_samples.append(('val', d.name))

print(f'Total samples: {len(all_samples)}')

# Sort by sample number (temporal order)
all_samples.sort(key=lambda x: int(x[1]))

# Temporal split: train/val/test = 1:1:1
n = len(all_samples)
n_train = n // 3
n_val = n // 3
n_test = n - n_train - n_val

train_samples = all_samples[:n_train]
val_samples = all_samples[n_train:n_train+n_val]
test_samples = all_samples[n_train+n_val:]

print(f'Split: train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}')

# Create output directory
D7_1_T_DIR.mkdir(exist_ok=True)
(D7_1_T_DIR / 'train').mkdir(exist_ok=True)
(D7_1_T_DIR / 'val').mkdir(exist_ok=True)
(D7_1_T_DIR / 'test').mkdir(exist_ok=True)

# Create symlinks
def create_symlinks(samples, split_name):
    split_dir = D7_1_T_DIR / split_name
    paths = []
    for i, (orig_split, sample_name) in enumerate(samples):
        src = D7_1_DIR / orig_split / sample_name
        dst = split_dir / f'{i:06d}'
        if not dst.exists():
            dst.symlink_to(src)
        paths.append(str(dst) + '/')
    return paths

train_paths = create_symlinks(train_samples, 'train')
val_paths = create_symlinks(val_samples, 'val')
test_paths = create_symlinks(test_samples, 'test')

# Write txt files
with open(D7_1_T_DIR / 'data_mouse_train.txt', 'w') as f:
    f.write('\n'.join(train_paths) + '\n')
with open(D7_1_T_DIR / 'data_mouse_val.txt', 'w') as f:
    f.write('\n'.join(val_paths) + '\n')
with open(D7_1_T_DIR / 'data_mouse_test.txt', 'w') as f:
    f.write('\n'.join(test_paths) + '\n')

print(f'Created D7_1_t at {D7_1_T_DIR}')
print(f'Train txt: {len(train_paths)} entries')
print(f'Val txt: {len(val_paths)} entries')
print(f'Test txt: {len(test_paths)} entries')
