"""Sorted-unit indexing checks for the Sabes loaders on small synthetic .mat (HDF5) files."""

import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

from cpic.exp_utils.data_util import load_sabes_data, load_sabes_session

BIN = 0.05
T0 = 10.0
N_BINS = 40


def spike_train(bins):
    """Spike times placed safely inside the given bins (one spike per listed bin)."""
    return T0 + BIN * np.asarray(bins, dtype=float) + 0.01


def write_fake_sabes(path, cells, chan_names):
    """
    Write a file laid out like the Sabes MATLAB v7.3 files.

    cells[row][chan] is an array of spike times or None for an empty cell; row 0 is hash.
    """
    n_rows, n_chan = len(cells), len(chan_names)
    # Run slightly past N_BINS bins so floor(duration / BIN) == N_BINS without float edge cases.
    t = T0 + np.arange(0, N_BINS * BIN + 0.02, 0.004)
    with h5py.File(path, "w") as f:
        refs = f.create_group("#refs#")
        names = np.empty((1, n_chan), dtype=h5py.ref_dtype)
        for i, name in enumerate(chan_names):
            chars = np.array([ord(c) for c in name], dtype=np.uint16)[:, None]
            names[0, i] = refs.create_dataset(f"name{i}", data=chars).ref
        spikes = np.empty((n_rows, n_chan), dtype=h5py.ref_dtype)
        for row in range(n_rows):
            for chan in range(n_chan):
                times = cells[row][chan]
                data = np.zeros(2, dtype=np.uint64) if times is None else np.asarray(times)[None, :]
                spikes[row, chan] = refs.create_dataset(f"s{row}_{chan}", data=data).ref
        f.create_dataset("chan_names", data=names)
        f.create_dataset("spikes", data=spikes)
        f.create_dataset("t", data=t[None, :])
        f.create_dataset("cursor_pos", data=np.stack([t - T0, 2 * (t - T0)]))
        f.create_dataset("target_pos", data=np.zeros((2, len(t))))
        f.create_dataset("finger_pos", data=np.zeros((3, len(t))))


class SortedUnitIndexingTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = str(Path(self.tmp.name) / "fake.mat")
        # 5 rows (hash + 4 sorted units) x 3 M1 channels. Each listed cell has a unique
        # spike count so every column can be identified from its total.
        self.cells = [
            [spike_train([0, 1]), spike_train([2]), None],             # hash
            [spike_train([3] * 3), spike_train([4] * 4), None],         # unit 1
            [spike_train([5] * 5), None, None],                         # unit 2
            [None, spike_train([6] * 6), None],                         # unit 3
            [spike_train([7] * 7), spike_train([8] * 8), None],         # unit 4 (last row)
        ]
        write_fake_sabes(self.path, self.cells, ["M1 001", "M1 002", "M1 003"])

    def tearDown(self):
        self.tmp.cleanup()

    def test_every_sorted_row_is_binned_exactly_once(self):
        s = load_sabes_session(self.path, bin_width_s=BIN)
        self.assertEqual(s["counts"].shape, (N_BINS, 6))
        self.assertEqual(sorted(s["counts"].sum(axis=0).tolist()), [3, 4, 5, 6, 7, 8])
        labels = sorted(zip(s["channel"].tolist(), s["unit"].tolist()))
        self.assertEqual(labels, [(0, 1), (0, 2), (0, 4), (1, 1), (1, 3), (1, 4)])
        for col, (chan, row) in enumerate(zip(s["channel"], s["unit"])):
            n = len(self.cells[row][chan])
            # Each sorted cell with n spikes puts all of them in bin n (see setUp).
            self.assertEqual(s["counts"][n, col], n, "spikes land in the expected bin")
            self.assertEqual(s["counts"][:, col].sum(), n)

    def test_legacy_indexing_skips_last_sorted_row(self):
        s = load_sabes_session(self.path, bin_width_s=BIN, legacy_indexing=True)
        self.assertNotIn(4, s["unit"].tolist())
        self.assertEqual(sorted(s["counts"].sum(axis=0).tolist()), [3, 4, 5, 6])

    def test_original_loader_matches_legacy_indexing(self):
        legacy = load_sabes_data(self.path, bin_width_s=BIN, preprocess=False)["M1"]
        self.assertEqual(sorted(legacy.sum(axis=0).tolist()), [3, 4, 5, 6])

    def test_three_row_files(self):
        cells = [self.cells[0], self.cells[1], self.cells[2]]
        write_fake_sabes(self.path, cells, ["M1 001", "M1 002", "M1 003"])
        self.assertEqual(sorted(load_sabes_session(self.path).get("unit").tolist()), [1, 1, 2])
        legacy = load_sabes_session(self.path, legacy_indexing=True)
        self.assertEqual(sorted(legacy["unit"].tolist()), [1, 1])

    def test_electrode_mode_sums_all_rows_including_hash(self):
        s = load_sabes_session(self.path, bin_width_s=BIN, representation="electrode")
        self.assertEqual(s["channel"].tolist(), [0, 1, 2])
        self.assertEqual(s["counts"].sum(axis=0).tolist(), [2 + 3 + 5 + 7, 1 + 4 + 6 + 8, 0])

    def test_region_selection_and_outside_spikes(self):
        cells = [row[:] for row in self.cells]
        cells[1][0] = np.concatenate([spike_train([3] * 3), [T0 - 1.0, T0 + N_BINS * BIN + 1.0]])
        write_fake_sabes(self.path, cells, ["M1 001", "M1 002", "S1 001"])
        s = load_sabes_session(self.path, bin_width_s=BIN, representation="electrode")
        self.assertEqual(s["channel"].tolist(), [0, 1])
        self.assertEqual(s["n_spikes_outside"], 2)
        self.assertEqual(s["counts"][:, 0].sum(), 2 + 3 + 5 + 7)

    def test_kinematics_are_raw_and_at_bin_centers(self):
        s = load_sabes_session(self.path, bin_width_s=BIN)
        np.testing.assert_allclose(s["cursor_pos"][:, 0], s["t_mid"], atol=1e-9)
        np.testing.assert_allclose(s["cursor_pos"][:, 1], 2 * s["t_mid"], atol=1e-9)
        self.assertEqual(s["finger_pos"].shape, (N_BINS, 3))


if __name__ == "__main__":
    unittest.main()
