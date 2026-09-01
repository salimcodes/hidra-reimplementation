"""Invertible elementwise nonlinear feature maps used by HiDRA (paper Sec. 4.1, App. D.1).

Each map is monotonic and globally invertible, so that the HiDRA round trip
``A_pinv @ sigma_inv(sigma(A @ x))`` recovers ``x`` exactly when no steering
term is added. LeakyReLU is the paper's default (Lemma 3.4's kernel
decomposition is derived for it); the others are the ablation set from
Table 5 / Appendix D.1.
"""

from dataclasses import dataclass
from typing import Callable, Dict

import torch

Tensor = torch.Tensor


def _newton_inverse(
    forward: Callable[[Tensor], Tensor],
    dforward: Callable[[Tensor], Tensor],
    y: Tensor,
    iters: int = 40,
) -> Tensor:
    """Invert a smooth, strictly monotonic scalar function via Newton's method.

    Used for maps (Cubic, Skip-softplus) with no closed-form inverse. All
    candidate nonlinearities here have derivatives bounded away from zero, so
    Newton's method converges quadratically from the ``t0 = y`` starting
    point (valid since f(t) ~ t or ~2t for large |t| in every case used).
    """
    t = y.clone().to(torch.float64)
    y64 = y.to(torch.float64)
    for _ in range(iters):
        t = t - (forward(t) - y64) / dforward(t)
    return t.to(y.dtype)


@dataclass(frozen=True)
class Nonlinearity:
    name: str
    forward: Callable[[Tensor], Tensor]
    inverse: Callable[[Tensor], Tensor]


def make_leaky_relu(slope: float = 0.5) -> Nonlinearity:
    """sigma_s(t) = (1+s)/2 * t + (1-s)/2 * |t|, equivalent to LeakyReLU(slope=s)."""

    def fwd(t: Tensor) -> Tensor:
        return torch.where(t >= 0, t, slope * t)

    def inv(y: Tensor) -> Tensor:
        return torch.where(y >= 0, y, y / slope)

    return Nonlinearity(f"leaky_relu(slope={slope})", fwd, inv)


def make_cube() -> Nonlinearity:
    """f(t) = t^3, invertible in closed form."""

    def fwd(t: Tensor) -> Tensor:
        return t**3

    def inv(y: Tensor) -> Tensor:
        return torch.sign(y) * torch.abs(y) ** (1.0 / 3.0)

    return Nonlinearity("cube", fwd, inv)


def make_cubic() -> Nonlinearity:
    """f(t) = t + t^3; f' = 1 + 3t^2 > 0 everywhere, inverted via Newton."""

    def fwd(t: Tensor) -> Tensor:
        return t + t**3

    def dfwd(t: Tensor) -> Tensor:
        return 1.0 + 3.0 * t**2

    def inv(y: Tensor) -> Tensor:
        return _newton_inverse(fwd, dfwd, y)

    return Nonlinearity("cubic", fwd, inv)


def make_skip_softplus(normalized: bool = False) -> Nonlinearity:
    """f(t) = t + softplus(t) [- log 2]; f' = 1 + sigmoid(t) in (1, 2)."""

    shift = torch.log(torch.tensor(2.0)).item() if normalized else 0.0

    def fwd(t: Tensor) -> Tensor:
        return t + torch.nn.functional.softplus(t) - shift

    def dfwd(t: Tensor) -> Tensor:
        return 1.0 + torch.sigmoid(t)

    def inv(y: Tensor) -> Tensor:
        return _newton_inverse(fwd, dfwd, y)

    name = "norm_skip_softplus" if normalized else "skip_softplus"
    return Nonlinearity(name, fwd, inv)


NONLINEARITIES: Dict[str, Callable[..., Nonlinearity]] = {
    "leaky_relu": make_leaky_relu,
    "cube": lambda **kw: make_cube(),
    "cubic": lambda **kw: make_cubic(),
    "skip_softplus": lambda **kw: make_skip_softplus(normalized=False),
    "norm_skip_softplus": lambda **kw: make_skip_softplus(normalized=True),
}


def get_nonlinearity(name: str, **kwargs) -> Nonlinearity:
    if name not in NONLINEARITIES:
        raise ValueError(f"Unknown nonlinearity '{name}'. Options: {list(NONLINEARITIES)}")
    return NONLINEARITIES[name](**kwargs)
