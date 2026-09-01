"""HiDRA: High-Dimensional Random-projection for Activation Steering.

Reimplementation of Pham, Do, Abdullaev, Nguyen, Than (2026),
"High-Dimensional Random Projection for Activation Steering in Language Models".

Core pipeline (paper Eq. 9):

    x <- A_pinv( sigma_inv( sigma(A @ x) + alpha * d ) )

where A in R^(m x d) is a fixed random Gaussian projection (m > d), sigma is an
invertible elementwise nonlinearity (LeakyReLU by default), and d is a
difference-in-means steering vector estimated in the projected space sigma(A @ x).
"""

from .nonlinearities import get_nonlinearity, NONLINEARITIES
from .projection import RandomProjector
from .steering import difference_in_means, fisher_ratio, lda_direction
from .pipeline import HiDRASteerer

__all__ = [
    "get_nonlinearity",
    "NONLINEARITIES",
    "RandomProjector",
    "difference_in_means",
    "fisher_ratio",
    "lda_direction",
    "HiDRASteerer",
]
