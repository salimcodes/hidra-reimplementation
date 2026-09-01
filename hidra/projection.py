"""Random high-dimensional projection at the heart of HiDRA (paper Sec. 4.1).

Given activation x in R^d, HiDRA maps it into a higher-dimensional space via a
fixed random Gaussian matrix A in R^(m x d) (m > d) and an invertible
elementwise nonlinearity sigma:

    x' = sigma(A x)                         (lift, Eq. before Sec 4.2)
    y  = A_pinv sigma^-1(y')                 (project back down)

A is drawn once at construction time (seeded) and reused across all layers,
together with its precomputed pseudo-inverse, exactly as described in the
paper ("We fix A after initialization and use the same A and its precomputed
pseudo-inverse A_pinv across all layers.").
"""

from typing import Optional

import torch

from .nonlinearities import Nonlinearity, get_nonlinearity

Tensor = torch.Tensor


class RandomProjector:
    def __init__(
        self,
        d: int,
        m: int,
        nonlinearity: str = "leaky_relu",
        seed: Optional[int] = 0,
        device: str = "cpu",
        dtype: torch.dtype = torch.float32,
        **nonlinearity_kwargs,
    ):
        if m <= d:
            raise ValueError(f"HiDRA requires a higher-dimensional lift (m > d); got m={m}, d={d}")
        self.d = d
        self.m = m
        self.device = device
        self.dtype = dtype

        gen = torch.Generator(device="cpu")
        if seed is not None:
            gen.manual_seed(seed)
        # A_ij ~ N(0, 1) i.i.d., as specified in Eq. (x' = sigma(Ax)).
        A = torch.randn(m, d, generator=gen, dtype=torch.float64)
        # Precompute the pseudo-inverse once in double precision for a
        # numerically clean round trip, then cast to the working dtype.
        A_pinv = torch.linalg.pinv(A)

        self.A = A.to(device=device, dtype=dtype)
        self.A_pinv = A_pinv.to(device=device, dtype=dtype)
        self.nonlinearity: Nonlinearity = get_nonlinearity(nonlinearity, **nonlinearity_kwargs)

    def lift(self, x: Tensor) -> Tensor:
        """x' = sigma(A x). x: (..., d) -> (..., m)."""
        return self.nonlinearity.forward(x @ self.A.T)

    def unlift(self, y: Tensor) -> Tensor:
        """A_pinv sigma^-1(y'). y: (..., m) -> (..., d)."""
        return self.nonlinearity.inverse(y) @ self.A_pinv.T

    def steer(self, x: Tensor, d_vec: Tensor, alpha: float) -> Tensor:
        """Full HiDRA pipeline (Eq. 9): x <- A_pinv sigma^-1(sigma(Ax) + alpha d)."""
        lifted = self.lift(x)
        return self.unlift(lifted + alpha * d_vec)
