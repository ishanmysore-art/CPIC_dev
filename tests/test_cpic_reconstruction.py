import numpy as np
import pytest
import torch

from cpic.CPIC import CPIC
from cpic.utils.data import InputOutputPastFutureDataset, PastFutureDataset


W = 4
N_IN = 8
N_OUT = 3
YDIM = 2


def _make_synthetic_dataset(n_samples=80, n_in=N_IN, seed=0):
    rng = np.random.default_rng(seed)
    T = n_samples
    data = rng.standard_normal((T, n_in)).astype(np.float32)
    return PastFutureDataset([data], window_size=W)


def test_mi_mode_backward_compat():
    ds = _make_synthetic_dataset()
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        hidden_dim=16,
        device="cpu",
        encoder_params={"encoder_type": "mlp", "n_layers": 1},
    )
    model.fit(ds, epochs=2, batch_size=8, lr=1e-3, verbose=False)
    past = torch.randn(4, W, N_IN)
    future = torch.randn(4, W, N_IN)
    loss, i_compress, metric = model(past, future)
    assert torch.isfinite(loss)
    assert torch.isfinite(i_compress)
    assert torch.isfinite(metric)


def test_reconstruction_past_loss_finite_and_decreases():
    ds = _make_synthetic_dataset(n_samples=200)
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        hidden_dim=16,
        device="cpu",
        predictive_loss="reconstruction",
        reconstruction_targets=("past",),
        encoder_params={"encoder_type": "mlp", "n_layers": 1},
        mi_params={"estimator_compress": "vub"},
    )
    assert model.decoder_past is not None
    assert model.decoder_past.encoder_type == model.encoder.encoder_type

    _, _, loss_before = model.score(
        torch.from_numpy(ds[0][0]).unsqueeze(0).float(),
        torch.from_numpy(ds[0][1]).unsqueeze(0).float(),
    )
    model.fit(ds, epochs=5, batch_size=16, lr=1e-2, verbose=False)
    _, _, loss_after = model.score(
        torch.from_numpy(ds[0][0]).unsqueeze(0).float(),
        torch.from_numpy(ds[0][1]).unsqueeze(0).float(),
    )
    assert loss_after < loss_before


def test_predictive_loss_none_skips_critic():
    ds = _make_synthetic_dataset()
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        hidden_dim=16,
        device="cpu",
        predictive_loss="none",
        encoder_params={"encoder_type": "mlp", "n_layers": 1},
        mi_params={"estimator_compress": "vub"},
    )
    assert model.critic is None
    past = torch.randn(3, W, N_IN)
    future = torch.randn(3, W, N_IN)
    loss, i_compress, metric = model(past, future)
    assert torch.isfinite(loss)
    assert metric.item() == 0.0
    model.fit(ds, epochs=1, batch_size=8, verbose=False)


def test_reconstruction_future_miso():
    rng = np.random.default_rng(1)
    inp = rng.standard_normal((60, N_IN)).astype(np.float32)
    out = rng.standard_normal((60, N_OUT)).astype(np.float32)
    ds = InputOutputPastFutureDataset([inp], [out], window_size=W)
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        hidden_dim=16,
        device="cpu",
        predictive_loss="reconstruction",
        reconstruction_targets=("past", "future"),
        future_xdim=N_OUT,
        predictive_space="observation",
        encoder_params={"encoder_type": "mlp", "n_layers": 1},
        mi_params={"estimator_compress": "vub"},
    )
    model.fit(ds, epochs=2, batch_size=8, verbose=False)
    past, future = ds[0]
    z = model.encode(torch.from_numpy(past).unsqueeze(0).float())
    assert z.shape == (1, W, YDIM)
    decoded_future = model.decoder_future(z)
    assert decoded_future.shape == (1, W, N_OUT)


def test_reconstruction_decoder_output_shape():
    model = CPIC(
        ydim=YDIM,
        T=W,
        xdim=N_IN,
        hidden_dim=16,
        device="cpu",
        predictive_loss="reconstruction",
        reconstruction_targets=("past",),
        encoder_params={"encoder_type": "linear", "deterministic": True},
        mi_params={"estimator_compress": "vub"},
    )
    z = torch.randn(2, W, YDIM)
    out = model.decoder_past(z)
    assert out.shape == (2, W, N_IN)


def test_resolve_device_falls_back_to_cpu(monkeypatch):
    from cpic.utils.helpers import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("cuda:0") == "cpu"
