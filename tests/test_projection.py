import torch

from hidra.projection import RandomProjector


def test_unlift_lift_reconstructs_input_when_unsteered():
    # A_pinv @ sigma^-1(sigma(A x)) should recover x exactly (up to numerical
    # precision) when no steering term is added, since m > d gives A full
    # column rank almost surely, so A_pinv @ A = I_d.
    torch.manual_seed(0)
    d, m = 12, 300
    proj = RandomProjector(d=d, m=m, nonlinearity="leaky_relu", seed=1, dtype=torch.float64)
    x = torch.randn(5, d, dtype=torch.float64)
    recovered = proj.unlift(proj.lift(x))
    assert torch.allclose(recovered, x, atol=1e-6)


def test_steer_with_zero_alpha_is_a_no_op():
    torch.manual_seed(0)
    d, m = 8, 200
    proj = RandomProjector(d=d, m=m, seed=2, dtype=torch.float64)
    x = torch.randn(3, d, dtype=torch.float64)
    d_vec = torch.randn(m, dtype=torch.float64)
    steered = proj.steer(x, d_vec, alpha=0.0)
    assert torch.allclose(steered, x, atol=1e-6)


def test_rejects_low_dimensional_lift():
    import pytest

    with pytest.raises(ValueError):
        RandomProjector(d=64, m=32)  # m must exceed d


def test_lift_shape():
    proj = RandomProjector(d=16, m=128, seed=0)
    x = torch.randn(4, 16)
    lifted = proj.lift(x)
    assert lifted.shape == (4, 128)
    back = proj.unlift(lifted)
    assert back.shape == (4, 16)
