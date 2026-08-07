import numpy as np
import pytest
import torch

from cpic.CPIC import CPIC
from cpic.utils.data import InputOutputPastFutureDataset


# Run these tests from the repo root with pytest -q
W = 4          # window size
N_IN = 6       # input feature dim
N_OUT = 1      # output feature dim (MISO: differs from N_IN)
YDIM = 2       # latent dim


def make_trial(T: int, n_in: int = N_IN, n_out: int = N_OUT, seed: int = 0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((T, n_in)), rng.standard_normal((T, n_out))


def make_miso_model(predictive_space: str = "observation", mi_params=None) -> CPIC:
    return CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        predictive_space=predictive_space,
        critic_params={"x_dim": W * YDIM, "y_dim": W * N_OUT, "hidden_dim": 16},
        mi_params=mi_params,
        hidden_dim=16,
        beta2=0,
        device="cpu",
        encoder_params={"encoder_type": "mlp", "n_layers": 1, "activation": "relu"},
    )


def test_window_pairing_and_length():
    T = 20
    inp = np.arange(T * N_IN, dtype=float).reshape(T, N_IN)
    out = np.arange(T * N_OUT, dtype=float).reshape(T, N_OUT)
    ds = InputOutputPastFutureDataset([inp], [out], window_size=W)

    assert len(ds) == T - 2 * W

    past, future = ds[0]
    np.testing.assert_array_equal(past, inp[0:W])
    np.testing.assert_array_equal(future, out[W : 2 * W])

    past, future = ds[3]
    np.testing.assert_array_equal(past, inp[3 : 3 + W])
    np.testing.assert_array_equal(future, out[3 + W : 3 + 2 * W])


def test_multi_trial_indexing_crosses_segments():
    T1, T2 = 20, 15
    inp1, out1 = make_trial(T1, seed=1)
    inp2, out2 = make_trial(T2, seed=2)
    ds = InputOutputPastFutureDataset([inp1, inp2], [out1, out2], window_size=W)

    nw1, nw2 = T1 - 2 * W, T2 - 2 * W
    assert len(ds) == nw1 + nw2

    # First index of the second trial maps to that trial's first window.
    past, future = ds[nw1]
    np.testing.assert_array_equal(past, inp2[0:W])
    np.testing.assert_array_equal(future, out2[W : 2 * W])

    with pytest.raises(IndexError):
        ds[len(ds)]


def test_dataset_validation_errors():
    inp, out = make_trial(20)

    # Trial-count mismatch.
    with pytest.raises(ValueError, match="same number of trials"):
        InputOutputPastFutureDataset([inp, inp], [out], window_size=W)

    # Per-trial length mismatch.
    with pytest.raises(ValueError, match="same number of time steps"):
        InputOutputPastFutureDataset([inp], [out[:-1]], window_size=W)

    # Series too short: needs length > 2 * window_size.
    short_inp, short_out = make_trial(2 * W)
    with pytest.raises(ValueError, match="2\\*window_size"):
        InputOutputPastFutureDataset([short_inp], [short_out], window_size=W)

    with pytest.raises(ValueError, match="positive"):
        InputOutputPastFutureDataset([inp], [out], window_size=0)


def test_fit_miso_observation_space():
    inp, out = make_trial(60)
    ds = InputOutputPastFutureDataset([inp], [out], window_size=W)
    model = make_miso_model("observation")

    model.fit(ds, epochs=2, batch_size=8, lr=1e-3, verbose=False)

    x = torch.randn(3, W, N_IN)
    mean, vars_ = model.encoder(x)
    assert mean.shape == (3, W, YDIM)
    assert torch.isfinite(mean).all()

    past = torch.randn(5, W, N_IN)
    future = torch.randn(5, W, N_OUT)
    loss, i_compress, i_predictive = model(past, future)
    assert torch.isfinite(loss)


def test_latent_space_dim_mismatch_raises():
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        predictive_space="latent",
        hidden_dim=16,
        beta2=0,
        device="cpu",
        encoder_params={"encoder_type": "mlp", "n_layers": 1, "activation": "relu"},
    )
    past = torch.randn(5, W, N_IN)
    future = torch.randn(5, W, N_OUT)
    with pytest.raises(ValueError, match="observation"):
        model(past, future)


def test_baseline_sized_from_critic_y_dim():
    model = make_miso_model(
        "observation", mi_params={"baseline": "unnormalized"}
    )
    assert model.baseline.input_dim == W * N_OUT
