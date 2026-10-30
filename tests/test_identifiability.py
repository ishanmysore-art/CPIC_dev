"""Frozen-critic identifiability tools and latent-agreement metrics."""

import unittest

import numpy as np
import torch
from torch import nn

from cpic.evaluation import encode, heldout_bounds, latent_agreement
from cpic.identifiability import (
    critic_bound_under_transform, haar_orthogonal, invariance_profile, objective_sensitivity,
    plane_rotation, recover_linear_map, transformation_suite,
)
from cpic.multisession import BENCHMARK, make_cpic

TINY = dict(BENCHMARK, ydim=4, T=2, hidden_dim=8, batch_size=64)


class NormCritic(nn.Module):
    """Scores depend only on window norms, so it is blind to any orthogonal latent map."""

    def forward(self, x, y):
        return -(x.norm(dim=1)[:, None] - y.norm(dim=1)[None, :]) ** 2


class IdentifiabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        rng = np.random.default_rng(0)
        cls.X = np.cumsum(rng.normal(size=(800, 6)), axis=0).astype(np.float32) * 0.05
        cls.model = make_cpic(6, "cpu", TINY)
        cls.Z = encode(cls.model, cls.X)

    def test_identity_matches_heldout_mean_latent_bound(self):
        a = critic_bound_under_transform(self.model, self.X, batch_size=64, seed=3)
        b = critic_bound_under_transform(self.model, self.X, G=np.eye(4), batch_size=64, seed=3)
        c = heldout_bounds(self.model, self.X, batch_size=64, seed=3, sample_latents=False)
        self.assertAlmostEqual(a, b, places=6)
        self.assertAlmostEqual(a, c["I_predictive_mean"], places=5)

    def test_suite_maps_are_valid(self):
        for t in transformation_suite(self.Z, np.random.default_rng(1)):
            if t["family"] in ("identity", "sign_flip", "permutation", "orthogonal_haar") \
                    or t["family"].startswith("rotation"):
                np.testing.assert_allclose(t["G"] @ t["G"].T, np.eye(4), atol=1e-10)
        U = haar_orthogonal(4, np.random.default_rng(2))[:, :2]
        np.testing.assert_allclose(plane_rotation(U, 0.0), np.eye(4), atol=1e-12)

    def test_rotation_blind_critic_shows_orthogonal_invariance(self):
        rows = invariance_profile(self.model, self.X, self.Z, critic=NormCritic(), batch_size=64)
        for row in rows:
            if row["family"] in ("sign_flip", "permutation", "orthogonal_haar", "rotation_global"):
                self.assertLess(abs(row["delta"]), 1e-4, row)
        scaled = [r for r in rows if r["family"] == "scale_isotropic" and r["param"] == 2.0]
        self.assertGreater(abs(scaled[0]["delta"]), 1e-4)

    def test_sensitivity_and_recovery_run(self):
        sens = objective_sensitivity(self.model, self.X, self.Z, batch_size=64, max_batches=2)
        self.assertEqual(sens["energy_by_principal_axis"].shape, (4,))
        before = [p.requires_grad for p in self.model.critic.parameters()]
        for parametrization in ("orthogonal", "linear"):
            rec = recover_linear_map(self.model, self.Z, self.Z, parametrization=parametrization,
                                     steps=5, batch_size=64)
            self.assertTrue(np.isfinite(rec["distance_from_identity"]))
            if parametrization == "orthogonal":
                np.testing.assert_allclose(rec["G"] @ rec["G"].T, np.eye(4), atol=1e-4)
        self.assertEqual(before, [p.requires_grad for p in self.model.critic.parameters()])


class LatentAgreementTests(unittest.TestCase):
    def test_three_senses_of_agreement(self):
        rng = np.random.default_rng(0)
        Z = rng.normal(size=(500, 4)) * np.array([3.0, 2.0, 1.0, 0.5])
        same = latent_agreement(Z[:300], Z[:300], Z[300:], Z[300:])
        self.assertAlmostEqual(same["direct_r2"], 1.0, places=10)
        Q = haar_orthogonal(4, rng)
        rotated = latent_agreement(Z[:300] @ Q, Z[:300], Z[300:] @ Q, Z[300:])
        self.assertLess(rotated["direct_r2"], 0.9)
        self.assertAlmostEqual(rotated["procrustes_r2"], 1.0, places=6)
        self.assertAlmostEqual(rotated["cka"], 1.0, places=6)


if __name__ == "__main__":
    unittest.main()
