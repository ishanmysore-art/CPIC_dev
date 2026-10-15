"""Plotting smoke tests using synthetic result dictionaries only."""

import importlib.util
import os
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np


os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cpic_mpl_tests"))
SCRIPT = (Path(__file__).resolve().parents[1] / "experiments" /
          "real_data_experiments" / "plot_mlp_comparison.py")
spec = importlib.util.spec_from_file_location("plot_mlp_comparison", SCRIPT)
plot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plot)


class MLPPlotTests(unittest.TestCase):
    def test_modern_results_with_replicate_uncertainty_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            frozen = np.array([[-0.1, -0.2], [-2.0, -4.0], [-100.0, -200.0]])
            refit = np.array([[0.1, 0.1], [0.0, -0.1], [-0.5, -0.6]])
            payload = {
                "ydim": 32,
                "masking": {"eval_missing_ps": [0.0, 0.25, 0.5]},
                "frozen_mask_eval_r2": frozen[None, :, :, None, None],
                "mask_eval_r2": refit[None, :, :, None, None],
                "mi": np.array([[[1.4, 0.07]]]),
                "config": {"T": [4]},
                "mlp_diagnostics": [
                    {"T": 4, "ydim": 32, "epoch": 0, "I_predictive": 0.01},
                    {"T": 4, "ydim": 32, "epoch": 3, "I_predictive": 0.03},
                ],
            }
            source = root / "masked.pkl"
            source.write_bytes(pickle.dumps(payload))
            original = source.read_bytes()
            outputs = root / "figures"
            run = plot.load_run(f"Mask-aware={source}")
            x, selected = plot.masked_values(run, "frozen_mask_eval_r2", 0, 0, 0)
            np.testing.assert_array_equal(x, [0.0, 0.25, 0.5])
            np.testing.assert_array_equal(selected, frozen)
            means, sds = plot.mean_sd(selected)
            self.assertAlmostEqual(means[1], -3.0)
            self.assertAlmostEqual(sds[1], np.sqrt(2.0))
            stems = plot.make_plots([run], outputs, symlog=True)
            self.assertEqual(set(stems), {"frozen_r2_vs_missingness",
                                           "refit_vs_frozen_r2", "predictive_information"})
            for stem in stems:
                for extension in ("pdf", "png"):
                    self.assertGreater((outputs / f"{stem}.{extension}").stat().st_size, 0)
            self.assertEqual(source.read_bytes(), original)
            with self.assertRaises(FileExistsError):
                plot.make_plots([run], outputs, symlog=True)

    def test_legacy_result_only_produces_available_mi_plot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "legacy.pkl"
            source.write_bytes(pickle.dumps([
                np.array([[[0.3]]]), np.array([[[2.0, 0.4]]]),
            ]))
            stems = plot.make_plots([plot.load_run(f"Linear={source}")], root / "figures")
            self.assertEqual(stems, ["predictive_information"])


if __name__ == "__main__":
    unittest.main()
