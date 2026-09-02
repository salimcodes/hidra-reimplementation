"""Core rotation math for Angular Steering.

Given an activation x and a unit behavior direction d_hat, x decomposes
uniquely as

    x = (x . d_hat) d_hat + x_perp,   x_perp = x - (x . d_hat) d_hat

d_hat and x_perp/||x_perp|| span a 2D plane that x lies in *entirely* (by
construction, x is a linear combination of exactly these two vectors). This
module rotates x within that plane by angle theta:

    e2_hat = x_perp / ||x_perp||
    a = x . d_hat          (x's coordinate along d_hat)
    b = ||x_perp||          (x's coordinate along e2_hat; always >= 0)

    x_rot = (a cos(theta) + b sin(theta)) d_hat
          + (b cos(theta) - a sin(theta)) e2_hat

This is a proper rotation matrix ([[cos, sin], [-sin, cos]] acting on (a, b)
in the (d_hat, e2_hat) basis), so ||x_rot|| == ||x|| exactly, theta=0 is the
identity, and d/dtheta (x_rot . d_hat) at theta=0 equals b = ||x_perp|| >= 0
-- i.e. increasing theta from 0 always increases alignment with d_hat (never
decreases it), which fixes the sign convention: positive theta rotates
*toward* d_hat.
"""

import torch

Tensor = torch.Tensor


def rotate_toward_direction(x: Tensor, d_hat: Tensor, theta: float, eps: float = 1e-8) -> Tensor:
    """Rotate x toward d_hat by angle theta (radians), within the 2D plane
    spanned by {d_hat, x}. x: (..., dim), d_hat: (dim,) unit vector.
    """
    a = (x * d_hat).sum(dim=-1, keepdim=True)  # (..., 1)
    x_parallel = a * d_hat  # (..., dim)
    x_perp = x - x_parallel
    b = x_perp.norm(dim=-1, keepdim=True)  # (..., 1)

    # If x is (numerically) already parallel to d_hat, there's no unique
    # orthogonal direction to rotate into -- leave x unchanged in that case.
    degenerate = b < eps
    e2_hat = x_perp / b.clamp_min(eps)

    cos_t, sin_t = torch.cos(torch.tensor(theta, dtype=x.dtype, device=x.device)), torch.sin(
        torch.tensor(theta, dtype=x.dtype, device=x.device)
    )
    new_a = a * cos_t + b * sin_t
    new_b = b * cos_t - a * sin_t
    rotated = new_a * d_hat + new_b * e2_hat
    return torch.where(degenerate, x, rotated)
