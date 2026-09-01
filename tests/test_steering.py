import torch

from hidra.projection import RandomProjector
from hidra.steering import difference_in_means, fisher_ratio, lda_direction


def test_difference_in_means_basic():
    target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    source = torch.tensor([[0.0, 0.0], [0.0, 0.0]])
    d = difference_in_means(target, source)
    assert torch.allclose(d, torch.tensor([2.0, 3.0]))


def test_lda_matches_dim_under_isotropic_scatter():
    # Lemma B.1(1): the DiM direction coincides with the LDA direction
    # whenever the within-class scatter is (close to) isotropic.
    torch.manual_seed(0)
    n, dim = 500, 10
    mu_a = torch.zeros(dim)
    mu_a[0] = 2.0
    class_a = mu_a + torch.randn(n, dim)
    class_b = torch.randn(n, dim)

    dim_vec = difference_in_means(class_a, class_b)
    lda_vec = lda_direction(class_a, class_b, gamma=1e-2)

    cos_sim = torch.dot(dim_vec, lda_vec) / (dim_vec.norm() * lda_vec.norm())
    assert cos_sim.item() > 0.99


def test_feature_space_fisher_ratio_captures_variance_only_signal():
    """Sanity check of Proposition 3.1 / Theorem 3.5: when two classes share
    the same mean but differ in per-coordinate variance (a purely
    second-order / superposition-style signal), the raw-space Fisher ratio
    is ~0 (no linear mean-shift to exploit) while HiDRA's lifted-space
    Fisher ratio picks up the variance gap through the nonlinearity.
    """
    torch.manual_seed(0)
    n, d = 400, 20
    # Definition 3.2 with v_k = e_k (standard basis), mu^A_k = mu^B_k = 0,
    # and a variance gap concentrated on the last coordinate.
    class_a = torch.randn(n, d)
    class_a[:, -1] *= 3.0  # sigma_A,last = 3 -> variance 9
    class_b = torch.randn(n, d)  # sigma_B,k = 1 for all k

    gamma = 1e-2
    r_lin = fisher_ratio(class_a, class_b, gamma)

    proj = RandomProjector(d=d, m=8 * d, nonlinearity="leaky_relu", seed=0)
    lifted_a = proj.lift(class_a)
    lifted_b = proj.lift(class_b)
    r_hidra = fisher_ratio(lifted_a, lifted_b, gamma)

    assert r_lin < 0.01  # near-zero baseline: no mean-difference signal to find
    assert r_hidra > 10.0 * max(r_lin, 1e-6)  # HiDRA recovers a much stronger discriminative signal
