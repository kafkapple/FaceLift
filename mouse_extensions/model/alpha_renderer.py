"""
Alpha Rendering Extensions

Extends original render functions to return alpha channel from Gaussian splatting.
Requires: pip install git+https://github.com/slothfulxtx/diff-gaussian-rasterization.git
"""

import torch
from typing import Dict, Tuple, Optional

# Try to import diff_gauss, fall back to original if not available
try:
    from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
    DIFF_GAUSS_AVAILABLE = True
except ImportError:
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    DIFF_GAUSS_AVAILABLE = False
    print("[mouse_extensions] Warning: diff_gauss not available, alpha rendering disabled")


def render_opencv_cam_with_alpha(
    pc,  # GaussianModel
    height: int,
    width: int,
    C2W: torch.Tensor,
    fxfycxcy: torch.Tensor,
    bg_color: torch.Tensor = None,
    scaling_modifier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Render Gaussians with alpha channel support.

    Returns:
        dict with keys: render, alpha, depth, norm, viewspace_points, visibility_filter, radii
    """
    if bg_color is None:
        bg_color = torch.ones(3, device=C2W.device)

    # Camera setup (same as original)
    fx, fy, cx, cy = fxfycxcy[0], fxfycxcy[1], fxfycxcy[2], fxfycxcy[3]
    W2C = torch.inverse(C2W)
    R = W2C[:3, :3].T
    T = W2C[:3, 3]

    FoVx = 2 * torch.atan(width / (2 * fx))
    FoVy = 2 * torch.atan(height / (2 * fy))
    tanfovx = (width / (2 * fx)).item()
    tanfovy = (height / (2 * fy)).item()

    # Rasterizer settings
    raster_settings = GaussianRasterizationSettings(
        image_height=height,
        image_width=width,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=W2C.T,
        projmatrix=(W2C.T @ _get_projection_matrix(fx, fy, cx, cy, width, height, 0.01, 100.0, C2W.device)).T,
        sh_degree=pc.active_sh_degree,
        campos=C2W[:3, 3],
        prefiltered=False,
        debug=False,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    # Gaussian properties
    means3D = pc.get_xyz
    screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device=means3D.device)
    means2D = screenspace_points
    opacity = pc.get_opacity
    scales = pc.get_scaling
    rotations = pc.get_rotation
    shs = pc.get_features

    if DIFF_GAUSS_AVAILABLE:
        # diff_gauss returns 6 values
        color, depth, norm, alpha, radii, extra = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3Ds_precomp=None,
            norm3Ds_precomp=None,
            extra_attrs=None,
        )
        return {
            "render": color,
            "alpha": alpha,
            "depth": depth,
            "norm": norm,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
    else:
        # Original rasterizer returns 2 values
        result = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=None,
        )
        rendered_image, radii = result[0], result[1]
        return {
            "render": rendered_image,
            "alpha": None,  # Not available
            "depth": None,
            "norm": None,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }


def _get_projection_matrix(fx, fy, cx, cy, width, height, znear, zfar, device):
    """Get OpenGL-style projection matrix."""
    P = torch.zeros(4, 4, device=device)
    P[0, 0] = 2 * fx / width
    P[1, 1] = 2 * fy / height
    P[0, 2] = (width - 2 * cx) / width
    P[1, 2] = (height - 2 * cy) / height
    P[2, 2] = -(zfar + znear) / (zfar - znear)
    P[2, 3] = -2 * zfar * znear / (zfar - znear)
    P[3, 2] = -1
    return P


class DeferredGaussianRenderWithAlpha(torch.autograd.Function):
    """
    Deferred Gaussian rendering with alpha support.

    Returns (renders, alphas) tuple instead of just renders.
    """

    @staticmethod
    def forward(ctx, xyz, features, scaling, rotation, opacity,
                height, width, C2W, fxfycxcy, gaussians_model, scaling_modifier=None):
        """
        Forward pass.

        Returns:
            tuple: (renders [b,v,3,h,w], alphas [b,v,1,h,w])
        """
        ctx.scaling_modifier = scaling_modifier
        ctx.rendering_size = (height, width)

        with torch.no_grad():
            b, v = C2W.size(0), C2W.size(1)
            renders = []
            alphas = []

            for i in range(b):
                pc = gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                for j in range(v):
                    result = render_opencv_cam_with_alpha(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )
                    renders.append(result["render"])
                    if result["alpha"] is not None:
                        alphas.append(result["alpha"])
                    else:
                        # Create dummy alpha if not available
                        alphas.append(torch.ones(1, height, width, device=xyz.device))

            renders = torch.stack(renders, dim=0).reshape(b, v, 3, height, width)
            alphas = torch.stack(alphas, dim=0).reshape(b, v, 1, height, width)

        renders = renders.requires_grad_()
        alphas = alphas.requires_grad_()

        ctx.save_for_backward(xyz, features, scaling, rotation, opacity, C2W, fxfycxcy)
        ctx.gaussians_model_class = type(gaussians_model)

        return renders, alphas

    @staticmethod
    def backward(ctx, grad_renders, grad_alphas):
        """Backward pass with dual gradient support."""
        xyz, features, scaling, rotation, opacity, C2W, fxfycxcy = ctx.saved_tensors
        height, width = ctx.rendering_size
        scaling_modifier = ctx.scaling_modifier

        b, v = C2W.size(0), C2W.size(1)
        gaussians_model = ctx.gaussians_model_class()

        with torch.enable_grad():
            xyz = xyz.detach().requires_grad_(True)
            features = features.detach().requires_grad_(True)
            scaling = scaling.detach().requires_grad_(True)
            rotation = rotation.detach().requires_grad_(True)
            opacity = opacity.detach().requires_grad_(True)

            for i in range(b):
                pc = gaussians_model.set_data(
                    xyz[i], features[i], scaling[i], rotation[i], opacity[i]
                )
                for j in range(v):
                    result = render_opencv_cam_with_alpha(
                        pc, height, width, C2W[i, j], fxfycxcy[i, j]
                    )
                    render = result["render"]
                    alpha = result["alpha"]

                    if grad_renders is not None:
                        render.backward(
                            grad_renders[i, j],
                            retain_graph=(grad_alphas is not None and alpha is not None)
                        )

                    if grad_alphas is not None and alpha is not None:
                        alpha.backward(grad_alphas[i, j])

        return (
            xyz.grad, features.grad, scaling.grad, rotation.grad, opacity.grad,
            None, None, None, None, None, None
        )
