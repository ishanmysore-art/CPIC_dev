"""Leakage, causality and centering checks for train-only session preprocessing."""

import os
import unittest
from pathlib import Path

import numpy as np

from cpic.exp_utils.sessions import (
    SessionPreprocessor, causal_moving_center, chronological_splits, legacy_unit_mask,
    prepare_session,
)

REAL_SESSION = Path(__file__).resolve().parents[1] / "data" / "real_data" / "M1" / "indy_20160627_01.mat"


def fake_session(n_bins=2000, n_units=6, seed=0):
    rng = np.random.default_rng(seed)
    rates = np.linspace(0.05, 2.0, n_units)
    counts = rng.poisson(rates, size=(n_bins, n_units)).astype(np.int64)
    return {"counts": counts, "channel": np.arange(n_units), "unit": np.ones(n_units, dtype=int),
            "cursor_pos": rng.normal(size=(n_bins, 2))}


class SplitTests(unittest.TestCase):
    def test_segments_are_ordered_disjoint_and_gapped(self):
        s = chronological_splits(1000, fractions=(0.7, 0.1, 0.2), gap=30)
        self.assertEqual(s.train[0], 0)
        self.assertEqual(s.val[0] - s.train[1], 30)
        self.assertEqual(s.test[0] - s.val[1], 30)
        self.assertEqual(s.test[1], 1000)
        self.assertEqual(s.train[1] - s.train[0], int(np.floor(940 * 0.7)))

    def test_bad_fractions_raise(self):
        with self.assertRaises(ValueError):
            chronological_splits(1000, fractions=(0.5, 0.1, 0.1))


class CausalDriftTests(unittest.TestCase):
    def test_output_depends_only_on_past_bins(self):
        X = np.random.default_rng(1).normal(size=(200, 3))
        Y = X.copy()
        Y[120:] += 5.0
        np.testing.assert_array_equal(causal_moving_center(X, 50)[:120],
                                      causal_moving_center(Y, 50)[:120])

    def test_matches_explicit_trailing_mean(self):
        X = np.random.default_rng(2).normal(size=(30, 2))
        expected = np.stack([X[t] - X[max(0, t - 9):t + 1].mean(axis=0) for t in range(30)])
        np.testing.assert_allclose(causal_moving_center(X, 10), expected, atol=1e-12)


class LeakageTests(unittest.TestCase):
    def test_val_and_test_values_cannot_change_train_outputs_or_statistics(self):
        session = fake_session()
        splits = chronological_splits(len(session["counts"]))
        reference = prepare_session(session, splits, SessionPreprocessor())

        perturbed = dict(session, counts=session["counts"].copy())
        for name in ("val", "test"):
            perturbed["counts"][splits.segment(name)] += 7
        result = prepare_session(perturbed, splits, SessionPreprocessor())

        np.testing.assert_array_equal(result["X"]["train"], reference["X"]["train"])
        np.testing.assert_array_equal(result["preprocessor"]["keep"], reference["preprocessor"]["keep"])
        np.testing.assert_array_equal(result["preprocessor"]["mean"], reference["preprocessor"]["mean"])

    def test_drift_window_restarts_at_each_segment(self):
        session = fake_session()
        splits = chronological_splits(len(session["counts"]))
        pre = SessionPreprocessor().fit(session["counts"][splits.segment("train")])
        X = pre._transform_unfitted(session["counts"][splits.segment("test")])
        # The first test bin is centered on itself only, never on preceding train/val bins.
        np.testing.assert_allclose(X[0], 0.0, atol=1e-12)

    def test_rate_filter_uses_train_only(self):
        session = fake_session()
        splits = chronological_splits(len(session["counts"]))
        session["counts"][:, 0] = 0
        session["counts"][splits.segment("test"), 0] = 50  # active only in test
        result = prepare_session(session, splits, SessionPreprocessor(min_rate_hz=0.5))
        self.assertFalse(result["preprocessor"]["keep"][0])

    def test_centering_is_per_feature_on_train(self):
        session = fake_session()
        splits = chronological_splits(len(session["counts"]))
        result = prepare_session(session, splits, SessionPreprocessor(min_rate_hz=None))
        np.testing.assert_allclose(result["X"]["train"].mean(axis=0), 0.0, atol=1e-5)

    def test_legacy_centering_was_a_scalar(self):
        # Audit finding: the legacy script's X_mean flattens a 2-D array before averaging.
        X_train = np.arange(12.0).reshape(4, 3)
        legacy_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
        self.assertEqual(legacy_mean.shape, (1,))
        self.assertNotEqual(legacy_mean[0], X_train.mean(axis=0)[0])

    def test_fixed_keep_overrides_rate_filter(self):
        session = fake_session()
        splits = chronological_splits(len(session["counts"]))
        keep = legacy_unit_mask(session["counts"], count_threshold=1000)
        result = prepare_session(session, splits, SessionPreprocessor(), keep=keep)
        np.testing.assert_array_equal(result["preprocessor"]["keep"], keep)


@unittest.skipUnless(REAL_SESSION.exists() and os.environ.get("CPIC_SLOW_TESTS"),
                     "set CPIC_SLOW_TESTS=1 with the Indy session present")
class RealSessionTests(unittest.TestCase):
    def test_legacy_indexing_reproduces_original_loader_columns(self):
        from cpic.exp_utils.data_util import load_sabes_data, load_sabes_session
        legacy = load_sabes_data(str(REAL_SESSION))["M1"]
        session = load_sabes_session(str(REAL_SESSION), legacy_indexing=True)
        self.assertEqual(legacy.shape[1], 109)
        self.assertEqual(int(legacy_unit_mask(session["counts"]).sum()), legacy.shape[1])
        corrected = load_sabes_session(str(REAL_SESSION))
        print("corrected units above 5000 spikes:", int(legacy_unit_mask(corrected["counts"]).sum()))

    def test_where_legacy_bins_spikes_recorded_before_t0(self):
        # Legacy floors (spike - t0) / bin with no lower bound; negative indices wrap to
        # the END of the array, i.e. into the test split. Measure that directly.
        from cpic.exp_utils.data_util import load_sabes_data, load_sabes_session
        legacy = load_sabes_data(str(REAL_SESSION), preprocess=False)["M1"]
        session = load_sabes_session(str(REAL_SESSION), legacy_indexing=True)
        new = session["counts"][:, session["counts"].sum(axis=0) > 0]
        self.assertEqual(legacy.shape, new.shape)
        diff = legacy - new
        self.assertTrue(np.all(diff >= 0), "legacy only ever adds spikes")
        affected = np.flatnonzero(np.abs(diff).sum(axis=1))
        n_bins = len(legacy)
        print(f"extra legacy spikes: {int(diff.sum())} in {len(affected)} bins; "
              f"affected bins span [{affected.min() if len(affected) else '-'}, "
              f"{affected.max() if len(affected) else '-'}] of {n_bins}; "
              f"80/20 test split starts at bin {int(n_bins * 0.8)}")


if __name__ == "__main__":
    unittest.main()
