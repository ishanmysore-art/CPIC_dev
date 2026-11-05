"""CPU smoke test for Gate 3 electrode-space transfer on two synthetic sessions."""

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cpic.multisession import BENCHMARK

SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "cross_session"
sys.path.insert(0, str(SCRIPTS))  # transfer_sessions imports transfer_common by name
spec = importlib.util.spec_from_file_location("transfer_sessions", SCRIPTS / "transfer_sessions.py")
transfer_sessions = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_sessions)

TINY = dict(BENCHMARK, ydim=2, T=2, hidden_dim=8, batch_size=64, epochs=1)


def fake_electrode_session(seed, n_bins=1600, n_channels=6):
    rng = np.random.default_rng(seed)
    latent = np.cumsum(rng.normal(size=(n_bins, 2)), axis=0) * 0.05
    rates = rng.uniform(0.05, 0.5, size=n_channels) * np.exp(0.3 * np.tanh(latent @ rng.normal(size=(2, n_channels))))
    counts = rng.poisson(rates)
    counts[:, -1] = 0 if seed == 1 else counts[:, -1]  # one electrode silent in the target
    return {"counts": counts, "channel": np.arange(n_channels), "unit": -np.ones(n_channels, dtype=int),
            "chan_names": [f"M1 {i + 1:03d}" for i in range(n_channels)],
            "cursor_pos": 10 * latent + 5 * seed, "duration_s": n_bins * 0.05}


class TransferSessionsSmokeTest(unittest.TestCase):
    def test_every_arm_is_reported(self):
        result = transfer_sessions.run_pair(fake_electrode_session(0), fake_electrode_session(1),
                                            "cpu", 0, TINY, use_dca=False)
        expected = {"zero_shot_source_norm", "zero_shot_target_norm", "zero_shot_permuted",
                    "refine_pretrained", "refine_random", "refine_fresh",
                    "refine_pretrained_shuffled", "permuted_init_pretrained",
                    "random_init_pretrained", "target_specific_cpic"}
        self.assertEqual(set(result["arms"]), expected)
        for name, arm in result["arms"].items():
            self.assertEqual(set(arm["frozen_source_decoder"]),
                             {"velocity", "position_raw", "position_restandardized"})
            self.assertEqual("relative_change_from_init" in arm,
                             name.startswith("refine") or name.startswith("permuted_init"))
        self.assertEqual(result["n_electrodes"]["silent_in_target_train"], 1)

    def test_dropout_trained_source(self):
        result = transfer_sessions.run_pair(fake_electrode_session(0), fake_electrode_session(1),
                                            "cpu", 0, TINY, use_dca=False, source_dropout=0.25)
        self.assertEqual(result["source_dropout"], 0.25)
        self.assertIn("refine_pretrained", result["arms"])

    def test_mismatched_channels_raise(self):
        b = fake_electrode_session(1)
        b["chan_names"] = list(reversed(b["chan_names"]))
        with self.assertRaises(ValueError):
            transfer_sessions.run_pair(fake_electrode_session(0), b, "cpu", 0, TINY, use_dca=False)

    def test_cross_session_table_and_predictor_tests(self):
        base = transfer_sessions.run_pair(fake_electrode_session(0), fake_electrode_session(1),
                                          "cpu", 0, TINY, use_dca=False)
        runs = []
        for k, target in enumerate(["indy_20160624_03", "indy_20160630_01", "indy_20160915_01",
                                    "indy_20161006_02", "indy_20161212_02"]):
            run = json.loads(json.dumps(base))
            run.update({"source_session": "indy_20160627_01", "target_session": target})
            run["arms"]["zero_shot_target_norm"]["bound_source_critic_test_mean_latent"] = 2.0 - k
            run["arms"]["zero_shot_target_norm"]["frozen_source_decoder"]["position_restandardized"]["r2_mean"] = 0.3 - 0.1 * k
            runs.append(run)
        with tempfile.TemporaryDirectory() as tmp:
            for k, name in enumerate(["indy_20160627_01"] + [r["target_session"] for r in runs]):
                rates = [1.0, 2.0, 3.0, 4.0, 5.0 + k, 0.1]  # last electrode below the 0.5 Hz cut
                Path(tmp, f"electrode_rates_{name}.json").write_text(json.dumps(
                    {"chan_names": [f"M1 {i + 1:03d}" for i in range(6)], "train_rate_hz": rates}))
            rows = transfer_sessions.cross_session_table(runs, tmp)
        self.assertEqual([r["days_apart"] for r in rows], [-3, 3, 80, 101, 168])
        self.assertTrue(all(r["rate_correlation"] is not None for r in rows))
        tests, verdict = transfer_sessions.predictor_tests(rows)
        self.assertAlmostEqual(tests["zero_shot_r2 ~ critic_bound_zero_shot"]["rho"], 1.0)
        self.assertIn(verdict, (True, False))

    def test_permutation_spearman(self):
        rho, p = transfer_sessions.permutation_spearman([1, 2, 3, 4, 5, 6, 7, 8], [2, 4, 6, 8, 10, 12, 14, 16])
        self.assertAlmostEqual(rho, 1.0)
        self.assertLess(p, 0.001)  # exact two-sided p for n=8 is 2/40320
        rho, p = transfer_sessions.permutation_spearman([1, 2, 3, 4, 5], [3, 1, 4, 1, 5], seed=1)
        self.assertGreater(p, 0.05)
        rho, p = transfer_sessions.permutation_spearman([1, 2, 3, 4, 5], [0.2] * 5)
        self.assertTrue(np.isnan(rho) and np.isnan(p), "constant input must not look significant")

    def test_circular_shuffle_keeps_marginals(self):
        X = np.random.default_rng(0).normal(size=(500, 3))
        S = transfer_sessions.circular_shuffle(X, np.random.default_rng(1))
        np.testing.assert_allclose(np.sort(S, axis=0), np.sort(X, axis=0))
        self.assertFalse(np.allclose(S, X))


if __name__ == "__main__":
    unittest.main()
