"""Held-out evaluation must not change the model, must be reproducible, and uses full batches."""

import unittest

import numpy as np
import torch

from cpic.CPIC import CPIC
from cpic.evaluation import (
    block_bootstrap_r2, fit_linear_decoder, heldout_bounds, probe_suite, score_linear_decoder,
)


def small_model(encoder_type):
    torch.manual_seed(0)
    return CPIC(ydim=2, T=2, xdim=6, hidden_dim=8, beta=1e-3, device="cpu",
                encoder_params={"encoder_type": encoder_type, "n_layers": 0})


def model_state(model):
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


class HeldoutBoundTests(unittest.TestCase):
    def setUp(self):
        self.X = np.random.default_rng(0).normal(size=(300, 6)).astype(np.float32)

    def test_parameters_buffers_and_mode_are_unchanged(self):
        for encoder_type in ("linear", "mlp"):  # the MLP log-variance head has BatchNorm buffers
            model = small_model(encoder_type)
            model.train()
            before = model_state(model)
            heldout_bounds(model, self.X, batch_size=64, seed=3)
            heldout_bounds(model, self.X, batch_size=64, seed=3, sample_latents=False)
            after = model_state(model)
            self.assertEqual(before.keys(), after.keys())
            for key in before:
                self.assertTrue(torch.equal(before[key], after[key]), f"{encoder_type}: {key} changed")
            self.assertTrue(model.training, "training mode restored")

    def test_reproducible_and_leaves_global_rng_alone(self):
        model = small_model("linear")
        rng_before = torch.get_rng_state().clone()
        a = heldout_bounds(model, self.X, batch_size=64, seed=5)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        b = heldout_bounds(model, self.X, batch_size=64, seed=5)
        np.testing.assert_array_equal(a["I_predictive_per_batch"], b["I_predictive_per_batch"])

    def test_only_full_batches_are_used(self):
        model = small_model("linear")
        result = heldout_bounds(model, self.X, batch_size=64)
        n_windows = len(self.X) - 2 * model.T
        self.assertEqual(result["n_windows"], n_windows)
        self.assertEqual(result["n_batches"], n_windows // 64)
        self.assertEqual(len(result["I_predictive_per_batch"]), n_windows // 64)
        self.assertTrue(np.all(result["I_predictive_per_batch"] <= np.log(64) + 1e-6))
        with self.assertRaises(ValueError):
            heldout_bounds(model, self.X[:50], batch_size=64)


class ProbeTests(unittest.TestCase):
    def test_frozen_decoder_is_not_refit(self):
        rng = np.random.default_rng(1)
        Z = rng.normal(size=(400, 3))
        Y = Z @ rng.normal(size=(3, 2)) + 0.1 * rng.normal(size=(400, 2))
        decoder = fit_linear_decoder(Z[:300], Y[:300], decoding_window=3, offset=2)
        coef = decoder.coef_.copy()
        score_linear_decoder(decoder, Z[300:], Y[300:], decoding_window=3, offset=2)
        np.testing.assert_array_equal(decoder.coef_, coef)

    def test_probe_suite_and_bootstrap(self):
        rng = np.random.default_rng(2)
        Z = rng.normal(size=(3000, 3))
        Y = np.roll(Z[:, :2], 5, axis=0) + 0.1 * rng.normal(size=(3000, 2))
        suite = probe_suite(Z[:2000], Y[:2000], Z[2000:], Y[2000:], offsets=(5,), decoding_window=1)
        self.assertGreater(suite["r2_by_offset"][5], 0.9)
        ci = block_bootstrap_r2(suite["decoders"][5], Z[2000:], Y[2000:], 1, 5,
                                block_bins=100, n_boot=200)
        self.assertLessEqual(ci["ci_low"], ci["r2"])
        self.assertLessEqual(ci["r2"], ci["ci_high"])


if __name__ == "__main__":
    unittest.main()
