"""Steering-vector extraction and the Fisher-ratio diagnostic (paper Sec. 2.1, 3.1).

Difference-in-means (Eq. 1):

    d = mean_{x in target}[x] - mean_{x in source}[x]

Regularized Fisher discriminant ratio (Eq. 2, App. A):

    R(gamma) = v^T (S_W + gamma I)^-1 v,  v = mu_A - mu_B
    S_W = sum_{x in A} (x-mu_A)(x-mu_A)^T + sum_{x in B} (x-mu_B)(x-mu_B)^T

Both work identically on raw activations (linear baseline, R_lin) or on
lifted activations sigma(Ax) (HiDRA's R_phi) -- callers just pass in whichever
representation they extracted, which is what lets Proposition 3.1's
comparison (R_phi(gamma) >= R_lin(gamma)) be checked empirically.

S_W has rank <= n (number of contrastive samples) and is typically far
smaller than the ambient dimension m (e.g. n=256 vs m=8192 in the paper's
runs), so forming and inverting the m x m matrix directly is wasteful. We
instead use the Woodbury identity to reduce the (S_W + gamma I) solve to an
n x n linear system, which is exact and cheap regardless of m.
"""

from typing import Tuple

import torch

Tensor = torch.Tensor


def difference_in_means(target: Tensor, source: Tensor) -> Tensor:
    """Eq. 1: d = mean(target) - mean(source), over the sample dimension (dim 0)."""
    return target.mean(dim=0) - source.mean(dim=0)


def _centered_and_diff(class_a: Tensor, class_b: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    mu_a = class_a.mean(dim=0)
    mu_b = class_b.mean(dim=0)
    v = mu_a - mu_b
    centered = torch.cat([class_a - mu_a, class_b - mu_b], dim=0)  # (n, dim)
    return v, centered, mu_a


def _woodbury_solve(v: Tensor, centered: Tensor, gamma: float) -> Tensor:
    """Solve (S_W + gamma I)^-1 v, where S_W = C^T C (C = `centered`, shape
    (n, dim)) has rank <= n. Uses the Woodbury identity to invert whichever
    of the n x n Gram matrix or the dim x dim scatter matrix is smaller --
    exact either way, but this keeps it cheap in both of HiDRA's regimes
    (few contrastive samples in a very high-dimensional lifted space, or
    vice versa). The solve itself runs in float64 since the smaller matrix
    can still be ill-conditioned when gamma is small relative to its
    eigenvalues.
    """
    n, dim = centered.shape
    orig_dtype = v.dtype
    v64 = v.to(torch.float64)
    c64 = centered.to(torch.float64)
    eye_n = torch.eye(n, dtype=torch.float64, device=centered.device)
    eye_dim = torch.eye(dim, dtype=torch.float64, device=centered.device)

    if n <= dim:
        cv = c64 @ v64  # (n,)
        gram = c64 @ c64.T  # (n, n) = C C^T
        rhs = torch.linalg.solve(gram + gamma * eye_n, cv)
        # (S_W + gamma I)^-1 v = (1/gamma) [ v - C^T (gamma I_n + C C^T)^-1 (C v) ]
        result = (v64 - c64.T @ rhs) / gamma
    else:
        s_w = c64.T @ c64  # (dim, dim), formed directly since dim < n here
        result = torch.linalg.solve(s_w + gamma * eye_dim, v64)
    return result.to(orig_dtype)


def fisher_ratio(class_a: Tensor, class_b: Tensor, gamma: float) -> float:
    """R(gamma) = v^T (S_W + gamma I)^-1 v (Eq. 2)."""
    v, centered, _ = _centered_and_diff(class_a, class_b)
    solved = _woodbury_solve(v, centered, gamma)
    return (v @ solved).item()


def lda_direction(class_a: Tensor, class_b: Tensor, gamma: float) -> Tensor:
    """Regularized Fisher-LDA direction w* ~ (S_W + gamma I)^-1 v (App. B.1),
    used as the alternative steering-vector extractor in the paper's Sec. 6.1
    ablation (LDA vs. DiM).
    """
    v, centered, _ = _centered_and_diff(class_a, class_b)
    return _woodbury_solve(v, centered, gamma)
