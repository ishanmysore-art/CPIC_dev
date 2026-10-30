"""Read-in arms around a frozen, random or fresh critic (tiny CPU models, no DCA)."""

import importlib.util
import unittest
from pathlib import Path

import numpy as np
import torch

from cpic.multisession import (
    BENCHMARK, fit_readin, fit_standard, population_partitions, random_critic_like,
)

TINY = dict(BENCHMARK, ydim=2, T=2, hidden_dim=8, batch_size=64, epochs=2)

COMMON = Path(__file__).resolve().parents[1] / "experiments" / "cross_session" / "transfer_common.py"
spec = importlib.util.spec_from_file_location("transfer_common", COMMON)
transfer_common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transfer_common)


def population(n_bins, n_units, seed):
    rng = np.random.default_rng(seed)
    latent = np.cumsum(rng.normal(size=(n_bins, 2)), axis=0) * 0.1
    return latent @ rng.normal(size=(2, n_units)) + 0.3 * rng.normal(size=(n_bins, n_units)), latent


def state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def same(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


class ReadinArmTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.X1, _ = population(800, 5, 0)
        cls.X2, _ = population(800, 7, 1)  # a different number of units
        cls.source = fit_standard(cls.X1, "cpu", 0, TINY, use_dca=False)

    def test_pretrained_critic_stays_frozen_and_identical(self):
        before = state(self.source.critic)
        model, history = fit_readin(self.X2, "cpu", 1, "pretrained", self.source.critic, TINY)
        self.assertTrue(same(before, state(self.source.critic)), "source critic untouched")
        self.assertTrue(same(before, state(model.critic)), "arm critic is the source critic")
        self.assertFalse(any(p.requires_grad for p in model.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in model.critic.parameters()))
        self.assertEqual(tuple(model.encoder._mean.weight.shape), (2, 7))
        self.assertEqual(len(history), TINY["epochs"])

    def test_readin_trains_and_starts_from_the_same_init_in_every_arm(self):
        inits = [state(fit_readin(self.X2, "cpu", 3, arm, self.source.critic, TINY, epochs=0)[0].encoder)
                 for arm in ("pretrained", "random", "fresh")]
        self.assertTrue(same(inits[0], inits[1]) and same(inits[0], inits[2]))
        trained = state(fit_readin(self.X2, "cpu", 3, "pretrained", self.source.critic, TINY)[0].encoder)
        self.assertFalse(same(inits[0], trained))

    def test_random_arm_is_frozen_and_differs_from_source(self):
        model, _ = fit_readin(self.X2, "cpu", 1, "random", self.source.critic, TINY)
        self.assertFalse(any(p.requires_grad for p in model.critic.parameters()))
        self.assertFalse(same(state(model.critic), state(self.source.critic)))

    def test_fresh_arm_trains_its_critic(self):
        initial = state(fit_readin(self.X2, "cpu", 1, "fresh", cfg=TINY, epochs=0)[0].critic)
        trained = state(fit_readin(self.X2, "cpu", 1, "fresh", cfg=TINY)[0].critic)
        self.assertFalse(same(initial, trained))

    def test_random_critic_like_is_seeded_and_leaves_global_rng(self):
        rng_before = torch.get_rng_state().clone()
        a = random_critic_like(self.source.critic, 5)
        b = random_critic_like(self.source.critic, 5)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        self.assertTrue(same(state(a), state(b)))
        self.assertFalse(same(state(a), state(random_critic_like(self.source.critic, 6))))

    def test_bad_arm_arguments(self):
        with self.assertRaises(ValueError):
            fit_readin(self.X2, "cpu", 1, "pretrained", None, TINY)
        with self.assertRaises(ValueError):
            fit_readin(self.X2, "cpu", 1, "sideways", self.source.critic, TINY)


class PartitionTests(unittest.TestCase):
    def test_overlap_levels(self):
        rng = np.random.default_rng(0)
        p1, p2 = population_partitions(100, 0.0, rng)
        self.assertEqual((len(p1), len(p2), len(np.intersect1d(p1, p2))), (50, 50, 0))
        p1, p2 = population_partitions(100, 0.5, rng)
        self.assertEqual((len(p1), len(p2), len(np.intersect1d(p1, p2))), (66, 66, 33))
        p1, p2 = population_partitions(100, 1.0, rng)
        np.testing.assert_array_equal(p1, np.arange(100))
        np.testing.assert_array_equal(p2, np.arange(100))


class TransferSmokeTest(unittest.TestCase):
    def test_run_transfer_produces_every_readout(self):
        X_all, latent = population(1500, 8, 2)
        splits = {"train": slice(0, 1000), "test": slice(1100, 1500)}
        X1 = {s: X_all[sl, :4] for s, sl in splits.items()}
        X2 = {s: X_all[sl, 4:] for s, sl in splits.items()}
        Y = {s: latent[sl] for s, sl in splits.items()}
        source = fit_standard(X1["train"], "cpu", 0, TINY, use_dca=False)
        result = transfer_common.run_transfer(source, X1, X2, Y, Y, "cpu", 0, TINY, 0.05,
                                              simultaneous=True, use_dca=False)
        self.assertEqual(set(result["arms"]), {"pretrained", "random", "fresh"})
        for r in list(result["arms"].values()) + [result["target_specific_cpic"]]:
            for key in ("frozen_source_decoder", "refit_probe", "agreement",
                        "oracle_procrustes_frozen_source_decoder", "bound_own_critic_test",
                        "bound_source_critic_test_mean_latent"):
                self.assertIn(key, r)
        gaps = transfer_common.paired_gaps(
            [result, result], lambda a: a["agreement"]["direct_r2"], "pretrained", "random",
            lambda r: 0)
        self.assertEqual(gaps["n"], 2)


if __name__ == "__main__":
    unittest.main()
