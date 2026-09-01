import torch

from hidra.nonlinearities import NONLINEARITIES, get_nonlinearity


def test_all_nonlinearities_invert():
    torch.manual_seed(0)
    t = torch.randn(2000, dtype=torch.float64) * 3.0  # wide range, incl. values near 0
    for name in NONLINEARITIES:
        nl = get_nonlinearity(name)
        y = nl.forward(t)
        recovered = nl.inverse(y)
        assert torch.allclose(recovered, t, atol=1e-4), f"{name} failed to invert"


def test_leaky_relu_matches_standard_definition():
    # sigma_s(t) = (1+s)/2 t + (1-s)/2 |t| should equal standard LeakyReLU(negative_slope=s)
    nl = get_nonlinearity("leaky_relu", slope=0.3)
    t = torch.linspace(-5, 5, steps=101)
    expected = torch.nn.functional.leaky_relu(t, negative_slope=0.3)
    assert torch.allclose(nl.forward(t), expected, atol=1e-6)
