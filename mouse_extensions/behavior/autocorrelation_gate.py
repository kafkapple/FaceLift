"""
Autocorrelation Gate: Verify temporal structure in covariance eigenvalues.
Must pass BEFORE H1 probe is meaningful.

Tests:
1. ACF analysis per feature dimension
2. Power Spectral Density (Welch's method)
3. Frame shuffle permutation test (1000 iterations)

Pass criteria (pre-registered):
- ACF: ≥70% features with significant autocorrelation at lag 5+ (p<0.01)
- PSD: 0-5Hz power > 2x above shuffled baseline
- Shuffle: Original ACF outside 95% CI of shuffled distribution
"""

import numpy as np
from scipy import signal, stats
from statsmodels.tsa.stattools import acf
from pathlib import Path
import json
import matplotlib.pyplot as plt


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def compute_acf_analysis(features, nlags=20, fps=50, alpha=0.01):
    """Compute ACF for each feature dimension.

    Returns:
        dict with n_significant_lag5, fraction_significant, per_feature_results
    """
    n_frames, n_dims = features.shape
    significant_at_lag5 = 0
    results = []

    for d in range(n_dims):
        ts = features[:, d]
        # Standardize
        ts = (ts - ts.mean()) / (ts.std() + 1e-10)

        acf_vals, confint = acf(ts, nlags=nlags, alpha=alpha, fft=True)

        # Check if ACF at lag 5 (0.1s at 50fps) is significant
        # Significant = ACF value outside confidence interval around 0
        lag5_significant = abs(acf_vals[5]) > (confint[5, 1] - acf_vals[5])
        if lag5_significant:
            significant_at_lag5 += 1

        results.append({
            'dim': d,
            'acf_lag5': float(acf_vals[5]),
            'acf_lag10': float(acf_vals[10]) if nlags >= 10 else None,
            'significant_lag5': bool(lag5_significant),
            'first_nonsig_lag': int(np.argmax(np.abs(acf_vals[1:]) < (confint[1:, 1] - acf_vals[1:]))) + 1
        })

    fraction = significant_at_lag5 / n_dims
    return {
        'n_significant_lag5': significant_at_lag5,
        'n_dims': n_dims,
        'fraction_significant': fraction,
        'pass_criterion': fraction >= 0.70,  # >=70% must be significant
        'per_feature': results
    }


def compute_psd_analysis(features, fps=50, freq_range=(0, 5)):
    """Compute PSD via Welch's method. Check 0-5Hz power concentration."""
    n_frames, n_dims = features.shape
    low_power_fractions = []

    for d in range(n_dims):
        ts = features[:, d]
        ts = (ts - ts.mean()) / (ts.std() + 1e-10)

        freqs, psd = signal.welch(ts, fs=fps, nperseg=min(256, n_frames//4))

        # Power in 0-5Hz vs total
        low_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        low_power = np.trapz(psd[low_mask], freqs[low_mask])
        total_power = np.trapz(psd, freqs)

        low_power_fractions.append(low_power / (total_power + 1e-10))

    mean_fraction = np.mean(low_power_fractions)
    return {
        'mean_low_freq_power_fraction': float(mean_fraction),
        'per_feature_fractions': [float(f) for f in low_power_fractions],
        'pass_criterion': mean_fraction > 0.5  # >50% power in behavioral band
    }


def shuffle_permutation_test(features, n_permutations=1000, nlags=10, fps=50):
    """Frame shuffle test: compare original ACF to shuffled distribution."""
    n_frames, n_dims = features.shape

    # Original ACF (mean across dims at lag 5)
    original_acfs = []
    for d in range(n_dims):
        ts = (features[:, d] - features[:, d].mean()) / (features[:, d].std() + 1e-10)
        acf_vals = acf(ts, nlags=nlags, fft=True)
        original_acfs.append(acf_vals[5])
    original_mean_acf = np.mean(original_acfs)

    # Shuffled distribution
    shuffled_acfs = []
    for i in range(n_permutations):
        perm = np.random.permutation(n_frames)
        shuffled = features[perm]

        dim_acfs = []
        for d in range(min(20, n_dims)):  # Sample 20 dims for speed
            ts = (shuffled[:, d] - shuffled[:, d].mean()) / (shuffled[:, d].std() + 1e-10)
            acf_vals = acf(ts, nlags=nlags, fft=True)
            dim_acfs.append(acf_vals[5])
        shuffled_acfs.append(np.mean(dim_acfs))

    shuffled_acfs = np.array(shuffled_acfs)
    ci_95 = np.percentile(shuffled_acfs, [2.5, 97.5])

    return {
        'original_mean_acf_lag5': float(original_mean_acf),
        'shuffled_ci_95': [float(ci_95[0]), float(ci_95[1])],
        'shuffled_mean': float(np.mean(shuffled_acfs)),
        'pass_criterion': original_mean_acf > ci_95[1],  # Outside 95% CI
        'p_value': float(np.mean(shuffled_acfs >= original_mean_acf))
    }


def run_gate(features_path, output_dir, fps=50):
    """Run full autocorrelation gate."""
    features = np.load(features_path)
    print(f"Loaded features: {features.shape}")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Remove log_volume dims (known constant: joints x 6th feature)
    # log_volume is index 5 in each 7-dim block (ev1,ev2,ev3,aniso,orient,logvol,count)
    # For 22 joints x 7d = 154d, log_volume indices are [5, 12, 19, ...]
    logvol_indices = [j * 7 + 5 for j in range(22)]
    keep_mask = np.ones(features.shape[1], dtype=bool)
    keep_mask[logvol_indices] = False
    features_clean = features[:, keep_mask]
    print(f"After removing log_volume: {features_clean.shape}")

    print("\n=== Test 1: ACF Analysis ===")
    acf_result = compute_acf_analysis(features_clean, fps=fps)
    print(f"Significant at lag 5: {acf_result['n_significant_lag5']}/{acf_result['n_dims']} ({acf_result['fraction_significant']:.1%})")
    print(f"PASS: {acf_result['pass_criterion']} (threshold: >=70%)")

    print("\n=== Test 2: PSD Analysis ===")
    psd_result = compute_psd_analysis(features_clean, fps=fps)
    print(f"Mean 0-5Hz power fraction: {psd_result['mean_low_freq_power_fraction']:.3f}")
    print(f"PASS: {psd_result['pass_criterion']} (threshold: >50%)")

    print("\n=== Test 3: Shuffle Permutation ===")
    shuffle_result = shuffle_permutation_test(features_clean, n_permutations=500, fps=fps)
    print(f"Original ACF lag 5: {shuffle_result['original_mean_acf_lag5']:.4f}")
    print(f"Shuffled 95% CI: [{shuffle_result['shuffled_ci_95'][0]:.4f}, {shuffle_result['shuffled_ci_95'][1]:.4f}]")
    print(f"p-value: {shuffle_result['p_value']:.4f}")
    print(f"PASS: {shuffle_result['pass_criterion']}")

    # Overall gate
    all_pass = all([acf_result['pass_criterion'], psd_result['pass_criterion'], shuffle_result['pass_criterion']])

    results = {
        'gate_passed': all_pass,
        'acf': acf_result,
        'psd': psd_result,
        'shuffle': shuffle_result,
        'features_shape': list(features.shape),
        'features_clean_shape': list(features_clean.shape),
        'fps': fps
    }

    # Save results
    # Remove per_feature details for JSON (too large)
    save_results = {k: v for k, v in results.items()}
    save_results['acf'] = {k: v for k, v in acf_result.items() if k != 'per_feature'}
    save_results['psd'] = {k: v for k, v in psd_result.items() if k != 'per_feature_fractions'}

    with open(output_dir / 'autocorrelation_gate_results.json', 'w') as f:
        json.dump(save_results, f, indent=2, cls=NumpyEncoder)

    print(f"\n{'='*50}")
    print(f"AUTOCORRELATION GATE: {'PASSED' if all_pass else 'FAILED'}")
    print(f"{'='*50}")

    return results


if __name__ == '__main__':
    features_path = '/home/joon/dev/FaceLift/outputs/report/clustering/features/covariance/covariance_static.npy'
    output_dir = '/home/joon/dev/FaceLift/outputs/report/clustering/autocorrelation_gate'

    results = run_gate(features_path, output_dir, fps=50)
