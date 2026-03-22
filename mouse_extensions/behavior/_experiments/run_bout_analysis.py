"""Bout duration distribution + autocorrelation analysis.
Compares Sparse vs Hybrid to validate temporal stability claim."""
import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import sys
sys.path.insert(0, ".")

# Load data
from plyfile import PlyData
from mouse_extensions.paths import KP_22
kp_all = np.load(str(KP_22),
                 allow_pickle=True)["keypoints"]

plys = sorted(Path("outputs/experiments/phase3_e2e/H3_resume_pose").rglob("gaussians.ply"))
frame_ply = {}
for p in plys:
    for part in p.parts:
        if part.isdigit() and len(part) == 6:
            fi = int(part)
            if fi not in frame_ply:
                frame_ply[fi] = str(p)
            break
frames = np.array(sorted(frame_ply.keys()))

kp = kp_all[frames]
kp_c = kp - kp.mean(1, keepdims=True)
kp_flat = StandardScaler().fit_transform(kp_c.reshape(len(kp_c), -1))
kp_pca = PCA(n_components=20).fit_transform(kp_flat)

def load_gauss(p):
    ply = PlyData.read(p)
    v = ply["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], axis=1)
    sc = np.stack([v["scale_0"], v["scale_1"], v["scale_2"]], axis=1)
    op = np.array(v["opacity"])
    f = []
    f.extend(xyz.mean(0)); f.extend(xyz.std(0))
    f.extend(np.percentile(xyz, [25,75], axis=0).flatten())
    f.extend([op.mean(), op.std(), np.median(op)])
    f.extend(sc.mean(0)); f.extend(sc.std(0))
    f.append(sc[:,0].mean() - sc[:,2].mean())
    xyz_c = xyz - xyz.mean(0)
    eigvals = np.linalg.eigvalsh(np.cov(xyz_c.T))
    f.extend(sorted(eigvals, reverse=True))
    f.extend(xyz.max(0) - xyz.min(0))
    return np.array(f, dtype=np.float32)

gauss_feats = np.stack([load_gauss(frame_ply[f]) for f in frames])
gauss_s = StandardScaler().fit_transform(gauss_feats)
gauss_pca = PCA(n_components=min(15, gauss_s.shape[1])).fit_transform(gauss_s)
hybrid = np.concatenate([kp_pca, gauss_pca], axis=1)

# Cluster at K=8
lab_sp = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(kp_pca)
lab_hy = KMeans(n_clusters=8, random_state=42, n_init=10).fit_predict(hybrid)

fps = 20.0

def bout_durations(labels):
    changes = np.where(np.diff(labels) != 0)[0]
    lengths = np.diff(np.concatenate([[0], changes + 1, [len(labels)]]))
    return lengths / fps

def autocorrelation(labels, max_lag=20):
    """Label autocorrelation: probability that label at t+lag == label at t."""
    result = []
    for lag in range(max_lag + 1):
        if lag >= len(labels):
            break
        matches = (labels[:-lag or None] == labels[lag:]).mean() if lag > 0 else 1.0
        result.append(float(matches))
    return result

bouts_sp = bout_durations(lab_sp)
bouts_hy = bout_durations(lab_hy)

print("=" * 60)
print("BOUT DURATION DISTRIBUTION ANALYSIS (K=8)")
print("=" * 60)

# Distribution statistics
thresholds = [0.10, 0.15, 0.20, 0.30, 0.50, 1.00]
print("\nBout duration distribution:")
print("%-20s %8s %8s" % ("Metric", "Sparse", "Hybrid"))
print("-" * 40)
print("%-20s %8.3f %8.3f" % ("Mean (s)", bouts_sp.mean(), bouts_hy.mean()))
print("%-20s %8.3f %8.3f" % ("Median (s)", np.median(bouts_sp), np.median(bouts_hy)))
print("%-20s %8.3f %8.3f" % ("Std (s)", bouts_sp.std(), bouts_hy.std()))
print("%-20s %8d %8d" % ("N bouts", len(bouts_sp), len(bouts_hy)))
print("%-20s %8.3f %8.3f" % ("Min (s)", bouts_sp.min(), bouts_hy.min()))
print("%-20s %8.3f %8.3f" % ("Max (s)", bouts_sp.max(), bouts_hy.max()))

print("\nProportion of SHORT bouts (flickering indicator):")
for t in thresholds:
    pct_sp = (bouts_sp < t).mean() * 100
    pct_hy = (bouts_hy < t).mean() * 100
    better = "HYBRID" if pct_hy < pct_sp else "SPARSE" if pct_sp < pct_hy else "SAME"
    print("  < %.2fs: Sparse=%5.1f%% Hybrid=%5.1f%%  (%s has fewer short bouts)" % (t, pct_sp, pct_hy, better))

# Autocorrelation
print("\n" + "=" * 60)
print("LABEL AUTOCORRELATION (persistence measure)")
print("=" * 60)

ac_sp = autocorrelation(lab_sp, max_lag=15)
ac_hy = autocorrelation(lab_hy, max_lag=15)

print("\n%-6s %12s %12s %12s" % ("Lag", "Sparse", "Hybrid", "Delta"))
print("-" * 45)
for lag in range(len(ac_sp)):
    t_sec = lag / fps
    delta = ac_hy[lag] - ac_sp[lag]
    print("%-6s %12.4f %12.4f %+12.4f" % ("%.2fs" % t_sec, ac_sp[lag], ac_hy[lag], delta))

print("\nInterpretation:")
print("  Higher autocorrelation at short lags = more persistent/stable clusters")
print("  Hybrid > Sparse at short lags = Hybrid has less noise-induced flickering")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
short_sp = (bouts_sp < 0.15).mean() * 100
short_hy = (bouts_hy < 0.15).mean() * 100
print("Short bout (<150ms, likely flickering):")
print("  Sparse: %.1f%%" % short_sp)
print("  Hybrid: %.1f%%" % short_hy)
print("  Delta: %+.1f pp (%s)" % (short_hy - short_sp,
    "Hybrid has FEWER short bouts = LESS flickering" if short_hy < short_sp
    else "Sparse has fewer short bouts" if short_sp < short_hy
    else "Same"))

ac1_sp = ac_sp[1] if len(ac_sp) > 1 else 0
ac1_hy = ac_hy[1] if len(ac_hy) > 1 else 0
print("\nLag-1 autocorrelation (frame-to-frame persistence):")
print("  Sparse: %.4f" % ac1_sp)
print("  Hybrid: %.4f" % ac1_hy)
print("  Delta: %+.4f (%s)" % (ac1_hy - ac1_sp,
    "Hybrid MORE persistent" if ac1_hy > ac1_sp else "Sparse more persistent"))
