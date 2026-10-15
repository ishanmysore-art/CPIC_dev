"""Small CPU checks for the opt-in MLP diagnostic path.

Run with: PYTHONPATH=src python -m unittest discover -s tests -p test_mlp_diagnostics.py
"""

import json
import unittest
from unittest.mock import patch

import torch
from torch.utils.data import TensorDataset

from cpic.CPIC import CPIC
from cpic.mi import infonce_lower_bound
from cpic.models import ConcatCritic


def _model(xdim=3):
    return CPIC(
        ydim=2, T=2, xdim=xdim, hidden_dim=8, beta=1e-3, beta1=1,
        device="cpu", encoder_params={"encoder_type": "mlp", "n_layers": 0},
    )


def _data():
    past = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 20
    return TensorDataset(past, past + 0.2)


class MLPDiagnosticTests(unittest.TestCase):
    def test_gradient_probe_does_not_populate_parameter_grads(self):
        torch.manual_seed(5)
        model = _model()
        past, future = next(iter(torch.utils.data.DataLoader(_data(), batch_size=8)))
        loss, compression, predictive = model(past, future)
        params = tuple(model.encoder.parameters())
        compression_grads = torch.autograd.grad(model.beta * compression, params,
                                                retain_graph=True, allow_unused=True)
        predictive_grads = torch.autograd.grad(-model.beta1 * predictive, params,
                                               retain_graph=True, allow_unused=True)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in compression_grads))
        self.assertTrue(any(g is not None and g.abs().sum() > 0 for g in predictive_grads))
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.encoder.parameters()))

    def test_critic_score_diagonal_matches_positive_pairs(self):
        class NegativeSquaredDistance(torch.nn.Module):
            def forward(self, pairs):
                x, y = pairs[:, :2], pairs[:, 2:]
                return -(x - y).square().sum(dim=1, keepdim=True)

        critic = ConcatCritic(x_dim=2, y_dim=2, hidden_dim=4)
        critic._f = NegativeSquaredDistance()
        x = torch.tensor([[0., 0.], [4., 0.], [0., 4.]])
        scores = critic(x, x)
        expected = -(x[:, None, :] - x[None, :, :]).square().sum(dim=-1)
        self.assertTrue(torch.equal(scores, expected))
        self.assertGreater(infonce_lower_bound(scores).item(), 0)
        self.assertAlmostEqual(infonce_lower_bound(torch.zeros_like(scores)).item(), 0)

    def test_diagnostics_preserve_training_and_rng(self):
        original_step = torch.optim.Adam.step
        step_calls = []

        def count_step(optimizer, *args, **kwargs):
            step_calls[-1].append(id(optimizer))
            return original_step(optimizer, *args, **kwargs)

        runs = []
        with patch.object(torch.optim.Adam, "step", count_step):
            for diagnose in (False, True):
                torch.manual_seed(123)
                model = _model()
                step_calls.append([])
                model.fit(_data(), epochs=9, batch_size=8, lr=1e-3,
                          early_stop=20, verbose=False, diagnose_mlp=diagnose)
                runs.append((
                    {name: value.detach().clone() for name, value in model.state_dict().items()},
                    torch.get_rng_state().clone(), model,
                ))

        self.assertTrue(all(torch.equal(runs[0][0][key], runs[1][0][key])
                            for key in runs[0][0]))
        self.assertTrue(torch.equal(runs[0][1], runs[1][1]))
        for calls in step_calls:
            self.assertEqual(len(calls), 9)
            self.assertEqual(sorted(calls.count(opt) for opt in set(calls)), [3, 6])

        model = runs[1][2]
        rows = model.mlp_diagnostics
        self.assertEqual([row["epoch"] for row in rows], [0, 2, 3, 4, 8])
        self.assertEqual([row["optimizer_phase"] for row in rows],
                         ["critic_only", "critic_only", "joint", "joint", "joint"])
        self.assertFalse(model.encoder._forward_hooks)
        self.assertFalse(model.critic._forward_hooks)
        for row in rows:
            json.dumps(row)
            self.assertEqual(row["batch_index"], 0)
            for group in ("encoder_mean", "encoder_logvar", "critic"):
                self.assertGreater(row[f"{group}_connected_params"], 0)
            for group in ("predictive_mean", "predictive_logvar"):
                self.assertGreater(row[f"{group}_connected_params"], 0)
            self.assertGreater(row["critic_grad_norm"], 0)
            self.assertGreater(row["past_encoded_variance_median"], 0)
            self.assertGreater(row["future_encoded_variance_median"], 0)

    def test_availability_mask_training_input(self):
        for dropout_p in (0.0, 0.5):
            with self.subTest(dropout_p=dropout_p):
                torch.manual_seed(11)
                model = _model(xdim=6)
                original_forward = model.forward
                seen = []

                def capture_inputs(past, future, *args, **kwargs):
                    seen.append((past.detach().clone(), future.detach().clone()))
                    return original_forward(past, future, *args, **kwargs)

                model.forward = capture_inputs
                model.fit(_data(), epochs=1, batch_size=8, lr=1e-3,
                          early_stop=2, verbose=False, neuron_dropout_p=dropout_p,
                          neuron_dropout_seed=27, availability_mask=True)
                self.assertEqual(len(seen), 1)
                past, future = seen[0]
                past_activity, past_mask = past[..., :3], past[..., 3:]
                future_activity, future_mask = future[..., :3], future[..., 3:]
                self.assertTrue(torch.equal(past_mask, future_mask))
                self.assertTrue(torch.equal(past_mask, past_mask[:1, :1].expand_as(past_mask)))
                self.assertTrue(torch.all((past_mask == 0) | (past_mask == 1)))
                self.assertTrue(torch.allclose(future_activity - past_activity,
                                               0.2 * past_mask, atol=1e-7))
                self.assertTrue(torch.equal(past_activity * (1 - past_mask),
                                            torch.zeros_like(past_activity)))
                original_past = _data().tensors[0]
                for neuron in range(3):
                    if torch.all(past_mask[..., neuron] == 1):
                        self.assertTrue(torch.equal(
                            torch.sort(past_activity[..., neuron].flatten()).values,
                            torch.sort(original_past[..., neuron].flatten()).values,
                        ))
                if dropout_p == 0:
                    self.assertTrue(torch.all(past_mask == 1))
                else:
                    self.assertTrue(torch.any(past_mask == 0))


if __name__ == "__main__":
    unittest.main()
