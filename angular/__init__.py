"""Angular Steering: rotation-based activation steering.

A companion method to hidra/, built from the project brief rather than a
source paper: instead of adding alpha*d to an activation (unbounded, changes
the activation's norm, and can push it off-manifold at high strength -- the
mechanism behind the "degenerate generations" failure mode observed with
additive steering), Angular Steering rotates the activation within the 2D
plane spanned by the activation itself and a learned behavior direction d_hat,
by an angle theta. This is an isometry (exactly norm-preserving) with a
bounded, periodic effect in theta, which is the structural reason it can
give smooth control without the blow-up additive steering is prone to.
"""

from .rotation import rotate_toward_direction
from .pipeline import AngularSteerer

__all__ = ["rotate_toward_direction", "AngularSteerer"]
