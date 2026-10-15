"""Small CPU checks for controlled availability-feature ablations."""

import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset

HELPERS = Path(__file__).resolve().parents[1] / "src" / "cpic" / "masking_diagnostics.py"
helper_spec = importlib.util.spec_from_file_location("masking_diagnostics", HELPERS)
helpers = importlib.util.module_from_spec(helper_spec)
helper_spec.loader.exec_module(helpers)
append_features_torch = helpers.append_features_torch
build_encoder_input = helpers.build_encoder_input
clean_decoder_stats = helpers.clean_decoder_stats
first_layer_stats = helpers.first_layer_stats
latent_pair_stats = helpers.latent_pair_stats
paired_first_layer_stats = helpers.paired_first_layer_stats
prediction_pair_stats = helpers.prediction_pair_stats

class MaskAblationTests(unittest.TestCase):
    def test_training_and_evaluation_input_construction(self):
        activity = torch.tensor([[[2., 3., 4.]], [[5., 6., 7.]]])
        future = activity + 1
        mask = torch.tensor([[[1., 0., 1.]]])
        for mode, expected in (("actual", [1., 0., 1.]),
                               ("zeros", [0., 0., 0.]),
                               ("ones", [1., 1., 1.]),
                               ("centered", [0.5, -1.5, 0.5])):
            with self.subTest(mode=mode):
                past_input = append_features_torch(activity * mask, mask, mode)
                future_input = append_features_torch(future * mask, mask, mode)
                np_input = build_encoder_input(activity.numpy(), mask.numpy(), True, mode)
                self.assertTrue(torch.equal(past_input, torch.from_numpy(np_input)))
                self.assertTrue(torch.equal(past_input[..., :3], activity * mask))
                self.assertTrue(torch.equal(future_input[..., :3], future * mask))
                self.assertTrue(torch.equal(past_input[..., 3:], future_input[..., 3:]))
                np.testing.assert_array_equal(past_input[0, 0, 3:].numpy(), expected)
                clean = build_encoder_input(activity.numpy(), np.ones(3), True, mode)
                if mode in ("zeros", "ones"):
                    np.testing.assert_array_equal(np_input[..., 3:], clean[..., 3:])
                if mode == "centered":
                    self.assertTrue(np.all(clean[..., 3:] == 0.5))
                    np.testing.assert_array_equal(np_input[..., 3:] / 2 + 0.75,
                                                  np.broadcast_to(mask.numpy(), activity.shape))

    def test_centered_features_have_expected_training_moments(self):
        present, missing = 2 * (1 - 0.75), 2 * (0 - 0.75)
        self.assertEqual(0.75 * present + 0.25 * missing, 0)
        self.assertEqual(0.75 * present ** 2 + 0.25 * missing ** 2, 0.75)
        self.assertEqual(0.75 * 1 ** 2 + 0.25 * 0 ** 2, 0.75)

    def test_fit_preserves_neural_dropout_across_modes(self):
        from cpic.CPIC import CPIC
        past = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 10
        dataset = TensorDataset(past, past + 0.25)
        observed = {}
        rng_states = {}
        for mode in ("actual", "zeros", "ones", "centered"):
            torch.manual_seed(71)
            model = CPIC(ydim=2, T=2, xdim=6, hidden_dim=8, beta=1e-4,
                         device="cpu", encoder_params={"encoder_type": "mlp", "n_layers": 0})
            original_forward = model.forward
            seen = []

            def capture(x_past, x_future):
                seen.append((x_past.detach().clone(), x_future.detach().clone()))
                return original_forward(x_past, x_future)

            model.forward = capture
            model.fit(dataset, epochs=1, batch_size=8, lr=1e-3, early_stop=20,
                      verbose=False, neuron_dropout_p=0.5, neuron_dropout_seed=17,
                      availability_mask=True, availability_features=mode)
            observed[mode] = seen[0]
            rng_states[mode] = torch.get_rng_state().clone()
        for mode in ("zeros", "ones", "centered"):
            self.assertTrue(torch.equal(rng_states[mode], rng_states["actual"]))
            self.assertTrue(torch.equal(observed[mode][0][..., :3], observed["actual"][0][..., :3]))
            self.assertTrue(torch.equal(observed[mode][1][..., :3], observed["actual"][1][..., :3]))
            if mode == "centered":
                self.assertTrue(torch.equal(observed[mode][0][..., 3:] / 2 + 0.75,
                                            observed["actual"][0][..., 3:]))
            else:
                self.assertTrue(torch.all(observed[mode][0][..., 3:] == (0 if mode == "zeros" else 1)))

    def test_first_layer_decomposition_and_gate_changes(self):
        layer = torch.nn.Linear(4, 2)
        with torch.no_grad():
            layer.weight.copy_(torch.tensor([[1., 0., -2., 0.], [0., 1., 0., 2.]]))
            layer.bias.copy_(torch.tensor([0.5, -0.5]))
        clean = torch.tensor([[1., 1., 1., 1.], [2., -1., 1., 1.]])
        masked = torch.tensor([[1., 0., 1., 0.], [2., 0., 1., 0.]])
        rng_state = torch.get_rng_state().clone()
        weights = layer.weight.detach().clone()
        stats = first_layer_stats(masked, layer, 2)
        neural = torch.nn.functional.linear(masked[:, :2], layer.weight[:, :2])
        extra = torch.nn.functional.linear(masked[:, 2:], layer.weight[:, 2:])
        self.assertAlmostEqual(stats["neural_contribution_rms"], neural.square().mean().sqrt().item())
        self.assertAlmostEqual(stats["appended_contribution_rms"], extra.square().mean().sqrt().item())
        paired = paired_first_layer_stats(clean, masked, layer, 2)
        expected_flips = ((layer(clean) > 0) != (layer(masked) > 0)).float().mean().item()
        self.assertAlmostEqual(paired["relu_gate_change_clean_to_masked_fraction"], expected_flips)
        self.assertTrue(torch.equal(rng_state, torch.get_rng_state()))
        self.assertTrue(torch.equal(weights, layer.weight))

    def test_first_layer_logging_preserves_training_state(self):
        from cpic.CPIC import CPIC
        past = torch.arange(48, dtype=torch.float32).reshape(8, 2, 3) / 10
        dataset = TensorDataset(past, past + 0.25)
        for mode in ("ones", "centered"):
            with self.subTest(mode=mode):
                runs = []
                for enabled in (False, True):
                    torch.manual_seed(81)
                    model = CPIC(ydim=2, T=2, xdim=6, hidden_dim=8, beta=1e-4,
                                 device="cpu", encoder_params={"encoder_type": "mlp", "n_layers": 0})
                    model.fit(dataset, epochs=4, batch_size=8, lr=1e-3, early_stop=20,
                              verbose=False, neuron_dropout_p=0.25, neuron_dropout_seed=19,
                              availability_mask=True, availability_features=mode,
                              diagnose_mlp=enabled)
                    runs.append((model, {k: v.detach().clone() for k, v in model.state_dict().items()},
                                 torch.get_rng_state().clone()))
                self.assertTrue(torch.equal(runs[0][2], runs[1][2]))
                for key in runs[0][1]:
                    self.assertTrue(torch.equal(runs[0][1][key], runs[1][1][key]), key)
                rows = runs[1][0].mlp_diagnostics
                self.assertEqual([row["epoch"] for row in rows], [0, 2, 3])
                for row in rows:
                    self.assertEqual(row["availability_features"], mode)
                    self.assertTrue(row["shared_neuron_mask_across_batch_and_time"])
                    self.assertIn("neural_contribution_rms", row["first_layer"])

    def test_decoder_diagnostics_are_read_only_and_match_score(self):
        rng = np.random.default_rng(4)
        z_train = rng.normal(size=(40, 3))
        y_train = z_train @ np.array([[2., 0.], [0., -1.], [0.5, 0.5]])
        z_test = rng.normal(size=(12, 3))
        z_masked = z_test + np.array([0.2, -0.3, 0.4])
        y_test = z_test @ np.array([[2., 0.], [0., -1.], [0.5, 0.5]])
        class LeastSquares:
            def __init__(self, x, y):
                weights = np.linalg.lstsq(
                    np.column_stack((x, np.ones(len(x)))), y, rcond=None)[0]
                self.coef_ = weights[:-1].T
                self.intercept_ = weights[-1]

            def predict(self, x):
                return x @ self.coef_.T + self.intercept_

            def score(self, x, y):
                return prediction_pair_stats(self.predict(x), self.predict(x), y)[
                    "clean_r2_per_output"].mean()

        decoder = LeastSquares(z_train, y_train)
        coefficients = decoder.coef_.copy()
        intercept = decoder.intercept_.copy()
        state = torch.get_rng_state().clone()
        latent = latent_pair_stats(z_test, z_masked)
        design = clean_decoder_stats(decoder, z_train)
        predictions = prediction_pair_stats(decoder.predict(z_test),
                                            decoder.predict(z_masked), y_test)
        zero_missing = prediction_pair_stats(decoder.predict(z_test),
                                             decoder.predict(z_test), y_test)
        self.assertEqual(latent["clean_variance"].shape, (3,))
        self.assertEqual(len(design["singular_values"]), 3)
        self.assertAlmostEqual(predictions["masked_r2_per_output"].mean(),
                               decoder.score(z_masked, y_test))
        np.testing.assert_array_equal(zero_missing["masked_r2_per_output"],
                                      zero_missing["clean_r2_per_output"])
        np.testing.assert_array_equal(decoder.coef_, coefficients)
        np.testing.assert_array_equal(decoder.intercept_, intercept)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_result_names_are_unique_and_exclusive(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = [helpers.result_filename(tmp, "MLP", True, mode, 1e-4,
                                             32, 0, 0.25, True, True, True)
                     for mode in ("actual", "zeros", "ones", "centered")]
            self.assertEqual(len(set(paths)), 4)
            self.assertIn("_availfeatcentered_", paths[-1])
            for path in paths:
                with open(path, "xb") as output:
                    output.write(b"test")
                with self.assertRaises(FileExistsError):
                    open(path, "xb")
            for mode in ("zeros", "centered"):
                with self.assertRaises(ValueError):
                    helpers.result_filename(tmp, "MLP", False, mode, 1e-4, 32, 0, 0.25)


if __name__ == "__main__":
    unittest.main()
