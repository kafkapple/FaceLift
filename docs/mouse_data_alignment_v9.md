# FaceLift Mouse Data Alignment Strategy (Experiment v9)

This document details the systematic alignment of Mouse Data to FaceLift's default camera & geometric assumptions.

## 1. FaceLift Default Camera Parameters
The pretrained FaceLift/GS-LRM model makes specific assumptions about the camera intrinsics and coordinate system.

| Parameter | Value | Description |
| :--- | :--- | :--- |
| **Image Size** | 512 x 512 | Fixed input resolution |
| **Focal Length (`fx`, `fy`)** | **548.99** | approx. FOV 50° |
| **Principal Point (`cx`, `cy`)** | **256.0, 256.0** | Exact image center |
| **Coordinate System** | OpenCV / Right-handed | +X Right, +Y Down, +Z Forward (from camera) |
| **World Up Vector** | +Z | The model assumes objects are upright along +Z |

> [!IMPORTANT]
> The pretrained model assumes these intrinsics are **fixed**. Providing data with varying `cx, cy` (e.g., from `pixel_based` centering) breaks the geometric consistency of the pretrained transformer.

## 2. Mouse Data (Input)
The raw mouse data (`data_mouse`) already adheres to the target intrinsics but has issues with object placement.

- **Intrinsics**: `fx=548.99`, `cx=256`, `cy=256` (Matches Default ✅)
- **Object Position**: The mouse is often NOT at the 3D origin corresponding to these cameras.
- **Problem**: In previous experiments, we moved the *camera* (changed `cx, cy`) to center the object. This violated the "Fixed Intrinsics" assumption.

## 3. Alignment Strategy (v9: Global Uniform Preprocessing)

Instead of moving the cameras (changing intrinsics), we move the **image content** to align with the fixed cameras.

### Geometric Transformation
We apply a single **Global Affine Transformation** to all views in a sample to preserve 3D Ray Consistency.

For a pixel $p = (x, y)$ in view $i$, the new position $p'$ is:

$$ p' = s \cdot (p - \text{GlobalCoM}) + \text{Center}_{target} $$

Where:
- $\text{GlobalCoM} = \frac{1}{N} \sum_{i=1}^{N} \text{CoM}_i$ (Average Center of Mass across all views)
- $\text{Center}_{target} = (256, 256)$
- $s$ is the Global Scale Factor.

### Scale Calculation with Safety Clamp
To prevent cropping (image cut-off), the scale $s$ is calculated systematically:

1.  **Target Scale** ($s_{target}$): Based on pixel area ratio.
    $$ s_{target} = \frac{\text{TargetRatio}}{\text{GlobalSizeRatio}} $$

2.  **Safety Limit** ($s_{max}$): The maximum scale that keeps the object within bounds $[0, 512]$.
    For every bounding box corner $c$ of every view, relative to $\text{GlobalCoM}$:
    $$ s_{max} = \min_{views} \left( \min \left( \frac{512 - 256}{x_{rel}}, \frac{256}{-x_{rel}}, \dots \right) \right) $$

3.  **Final Scale**:
    $$ s = \min(s_{target}, s_{max}) \times 0.95 \text{ (safety margin)} $$

## 4. Summary of Settings for v9
| Component | Setting | Notes |
| :--- | :--- | :--- |
| **Input Data** | `data_mouse` | Raw images |
| **Intrinsics** | **UNCHANGED** | `fx=549`, `cx=256` kept fixed |
| **Centering** | **Global Average CoM** | Moves content, not camera |
| **Scaling** | **Global Safe Scale** | Prevents cropping |
| **Background** | White (255, 255, 255) | Standard for FaceLift |
| **Constraint** | **Ray Consistency** | Transformation is uniform across views |
