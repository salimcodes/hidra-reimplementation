import math

import torch

from angular.rotation import rotate_toward_direction


def _random_unit(dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g, dtype=torch.float64)
    return v / v.norm()


def test_rotation_preserves_norm():
    torch.manual_seed(0)
    x = torch.randn(10, 16, dtype=torch.float64)
    d_hat = _random_unit(16, seed=1)
    for theta in [0.1, 0.5, 1.3, math.pi / 2, math.pi, 2.7]:
        rotated = rotate_toward_direction(x, d_hat, theta)
        assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-9)


def test_theta_zero_is_identity():
    torch.manual_seed(0)
    x = torch.randn(5, 12, dtype=torch.float64)
    d_hat = _random_unit(12, seed=2)
    rotated = rotate_toward_direction(x, d_hat, 0.0)
    assert torch.allclose(rotated, x, atol=1e-10)


def test_positive_theta_increases_alignment_with_direction():
    # d/dtheta (x_rot . d_hat) at theta=0 equals ||x_perp|| >= 0, so a small
    # positive theta should never *decrease* cosine similarity with d_hat.
    torch.manual_seed(0)
    d_hat = _random_unit(20, seed=3)
    x = torch.randn(20, dtype=torch.float64)
    # ensure x isn't (numerically) already parallel to d_hat
    x = x - (x @ d_hat) * d_hat + 0.3 * d_hat

    def cos_sim(v):
        return (v @ d_hat / v.norm()).item()

    base = cos_sim(x)
    for theta in [0.05, 0.2, 0.5, 1.0]:
        rotated = rotate_toward_direction(x, d_hat, theta)
        assert cos_sim(rotated) > base - 1e-9


def test_full_pi_over_2_rotation_aligns_with_direction_when_orthogonal():
    # If x is purely orthogonal to d_hat (a=0), rotating by +pi/2 should
    # land exactly on b*d_hat (full alignment, same magnitude).
    torch.manual_seed(0)
    d_hat = _random_unit(15, seed=4)
    raw = torch.randn(15, dtype=torch.float64)
    x_perp = raw - (raw @ d_hat) * d_hat  # purely orthogonal to d_hat
    rotated = rotate_toward_direction(x_perp, d_hat, math.pi / 2)
    expected = x_perp.norm() * d_hat
    assert torch.allclose(rotated, expected, atol=1e-8)


def test_rotation_is_periodic():
    torch.manual_seed(0)
    x = torch.randn(8, dtype=torch.float64)
    d_hat = _random_unit(8, seed=5)
    theta = 0.7
    r1 = rotate_toward_direction(x, d_hat, theta)
    r2 = rotate_toward_direction(x, d_hat, theta + 2 * math.pi)
    assert torch.allclose(r1, r2, atol=1e-8)


def test_degenerate_parallel_input_is_left_unchanged():
    d_hat = _random_unit(10, seed=6)
    x = 3.0 * d_hat  # exactly parallel to d_hat -> no unique orthogonal axis
    rotated = rotate_toward_direction(x, d_hat, 0.9)
    assert torch.allclose(rotated, x, atol=1e-8)


def test_batched_shapes():
    d_hat = _random_unit(24, seed=7)
    x = torch.randn(3, 5, 24, dtype=torch.float64)
    rotated = rotate_toward_direction(x, d_hat, 0.4)
    assert rotated.shape == x.shape
    assert torch.allclose(rotated.norm(dim=-1), x.norm(dim=-1), atol=1e-8)
