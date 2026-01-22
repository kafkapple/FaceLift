# Floater Mitigation in 3D Gaussian Splatting

## 1. Problem: Floater Artifacts

### What are Floaters?
Floaters are spurious Gaussian primitives that appear at incorrect depths, causing visual artifacts like:
- **Ghosting**: Double/overlapping objects
- **Haze**: Semi-transparent artifacts in empty space
- **Depth inconsistency**: Objects appearing at wrong distances

### Why Do Floaters Occur?

```
Training View → Gaussians optimize to minimize 2D loss
                ↓
         Ambiguity in 3D position (depth)
                ↓
         Gaussians with intermediate opacity (0.3-0.7)
         get stuck in local minima
                ↓
         These "ghost" Gaussians persist
```

**Root Cause**: 2D supervision alone cannot fully constrain 3D geometry.

---

## 2. Solution: Opacity Regularization

### Theoretical Basis

The key insight: **Real surfaces should have binary opacity** (fully opaque or fully transparent).

Intermediate opacity values (e.g., 0.5) indicate:
- Uncertain Gaussians
- Potential floaters
- Local minima in optimization

### Mathematical Formulation

#### 2.1 Binary Entropy Regularization (Recommended)

$$L_{entropy} = -\frac{1}{N} \sum_{i=1}^{N} \left[ \alpha_i \log(\alpha_i) + (1-\alpha_i) \log(1-\alpha_i) \right]$$

Where $\alpha_i$ is the opacity of Gaussian $i$.

**Properties**:
- Minimum at $\alpha = 0$ or $\alpha = 1$
- Maximum at $\alpha = 0.5$
- Smooth gradient for optimization

```python
entropy = -opacity * log(opacity) - (1 - opacity) * log(1 - opacity)
loss = entropy.mean()
```

#### 2.2 L1 Sparsity

$$L_{sparse} = \left| \frac{1}{N} \sum_{i=1}^{N} \alpha_i - \tau \right|$$

Where $\tau$ is target sparsity (e.g., 0.5).

**Use case**: When you want to control the overall number of opaque Gaussians.

#### 2.3 L2 Binary Distance

$$L_{binary} = \frac{1}{N} \sum_{i=1}^{N} \min(\alpha_i, 1-\alpha_i)^2$$

**Use case**: Direct penalty on intermediate values.

---

## 3. Implementation

### Config Options

```yaml
training:
  losses:
    opacity_reg_weight: 0.01     # 0.0 = disabled
    opacity_reg_type: "entropy"  # entropy | l1_sparse | l2_binary
    opacity_target_sparsity: 0.5 # for l1_sparse only
```

### Recommended Values

| Scenario | Weight | Type |
|----------|--------|------|
| Mild floaters | 0.01 | entropy |
| Severe floaters | 0.05 | entropy |
| Very sparse scenes | 0.01 | l1_sparse |

---

## 4. References

1. **3D Gaussian Splatting** (Kerbl et al., 2023)
   - Original 3DGS paper
   - https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/

2. **StableGS: A Floater-Free Framework** (2025)
   - Systematic analysis of floater causes
   - https://arxiv.org/html/2503.18458

3. **Mip-Splatting** (Yu et al., 2024)
   - Anti-aliasing and regularization
   - https://niujinshuchong.github.io/mip-splatting/

4. **GaussianPro** (Cheng et al., 2024)
   - Progressive training with opacity control
   - https://kcheng1021.github.io/gaussianpro.github.io/

---

## 5. Experimental Validation

### Experiments in This Project

| Version | Config | Purpose |
|---------|--------|---------|
| v67 | opacity_reg: 0.01 | Baseline floater mitigation |
| v68 | opacity_reg: 0.05 | Stronger regularization |
| v69 | v13 + opacity_reg | Dataset comparison |

### Expected Outcomes
- Reduced ghosting in turntable visualization
- Cleaner depth maps
- Potentially slight decrease in PSNR (trade-off)

---

*Created: 2026-01-16 | Author: Claude Code*
