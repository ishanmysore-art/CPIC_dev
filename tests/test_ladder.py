"""CPU smoke test for the Experiment A ladder on a synthetic session (no DCA, tiny model)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

LADDER = Path(__file__).resolve().parents[1] / "experiments" / "cross_session" / "ladder.py"
spec = importlib.util.spec_from_file_location("ladder", LADDER)
ladder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ladder)


def fake_sources(n_bins=6000, seed=0):
    rng = np.random.default_rng(seed)
    # 4 channels x rows 1..4; row 4 is the row the legacy loader never reads.
    channel = np.repeat(np.arange(4), 4)
    unit = np.tile(np.arange(1, 5), 4)
    rates = rng.uniform(0.3, 1.5, size=len(unit))  # spikes/bin; some units clear the legacy 5000 cut
    latent = np.cumsum(rng.normal(size=(n_bins, 2)), axis=0) * 0.05
    counts = rng.poisson(rates * np.exp(0.5 * np.tanh(latent @ rng.normal(size=(2, len(unit))))))
    cursor = latent + 0.1 * rng.normal(size=latent.shape)
    legacy_cols = unit != 4
    legacy_counts = counts[:, legacy_cols]
    legacy_X = np.sqrt(legacy_counts[:, legacy_counts.sum(axis=0) > 300]).astype(float)
    return {
        "legacy": {"M1": legacy_X - legacy_X.mean(axis=0), "cursor": cursor},
        "legacy_raw": {"M1": legacy_counts, "cursor": cursor},
        "session": {"counts": counts, "channel": channel, "unit": unit, "n_spike_rows": 5,
                    "cursor_pos": cursor},
    }


class LadderSmokeTests(unittest.TestCase):
    def setUp(self):
        self.saved = dict(ladder.MODEL)
        ladder.MODEL.update({"ydim": 2, "T": 2, "hidden_dim": 8, "batch_size": 64, "epochs": 1})
        self.sources = fake_sources()

    def tearDown(self):
        ladder.MODEL.clear()
        ladder.MODEL.update(self.saved)

    def test_builders_change_one_thing_at_a_time(self):
        data = {rung: ladder.BUILDERS[rung](self.sources) for rung in ladder.RUNGS}
        self.assertNotIn("val", data["L0_legacy"]["X"])
        n_legacy_units = data["L4_train_only_prep"]["X"]["train"].shape[1]
        self.assertTrue(0 < n_legacy_units < 12, "legacy 5000-spike cut keeps some, not all")
        for rung in ladder.RUNGS[1:]:
            self.assertEqual(set(data[rung]["X"]), {"train", "val", "test"})
            self.assertEqual(data[rung]["split"], data["L3_split"]["split"])
        # Without any pre-t0 spikes the wraparound fix must not change the unit set or data.
        np.testing.assert_array_equal(data["L4_train_only_prep"]["X"]["train"],
                                      data["L5_no_wraparound"]["X"]["train"])
        n_legacy_rows = data["L6_train_rate_filter"]["X"]["train"].shape[1]
        self.assertGreater(data["L7_all_sorted_rows"]["X"]["train"].shape[1], n_legacy_rows)
        for rung in ladder.RUNGS:
            self.assertEqual(data[rung]["X"]["train"].dtype, np.float64)

    def test_run_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmp:
            for seed in (0, 1):
                for rung in ("L3_split", "L4_train_only_prep"):
                    result = ladder.run_rung(rung, ladder.BUILDERS[rung](self.sources), seed,
                                             "cpu", use_dca=False)
                    Path(tmp, f"{rung}_seed{seed}.json").write_text(json.dumps(result))
            result = json.loads(Path(tmp, "L4_train_only_prep_seed0.json").read_text())
            self.assertEqual(result["epochs"], 1)
            self.assertIn("val_sampled", result["bounds"])
            self.assertEqual(set(result["probes"]), {"position", "velocity"})
            rows = ladder.summarize(tmp)
            self.assertEqual([r["rung"] for r in rows], ["L3_split", "L4_train_only_prep"])
            self.assertIn("delta_test_bound_mean", rows[1])
            self.assertTrue(Path(tmp, "ladder_summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
