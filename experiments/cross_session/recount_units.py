"""
Independent recount of sorted units in Sabes .mat sessions.

Deliberately shares no code with ``cpic.exp_utils.data_util``: it reads the raw HDF5
cells with h5py and counts spikes per cell directly, so it can confirm or refute the
loader's unit indexing. For each sorted-unit row it reports how many cells are
non-empty, how many pass spike-count / rate thresholds, whether waveform snippets
exist, and the fraction of inter-spike intervals shorter than a refractory period
(a crude check that a row holds genuine units rather than placeholders).

Usage:
    python experiments/cross_session/recount_units.py data/real_data/M1/indy_20160627_01.mat \
        [--out inventory/recount_indy_20160627_01.json]
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def channel_names(f):
    return [f[f["chan_names"][0, i]][()].tobytes()[::2].decode()
            for i in range(f["chan_names"].shape[1])]


def cell(f, name, row, chan):
    data = f[f[name][row, chan]][()]
    return None if data.shape == (2,) else data


def recount(path, count_threshold=5000, rate_threshold_hz=0.5, refractory_s=0.0015):
    with h5py.File(path, "r") as f:
        names = channel_names(f)
        t = f["t"][0, :]
        t0, t1 = float(t[0]), float(t[-1])
        duration = t1 - t0
        n_rows, n_chan = f["spikes"].shape
        rows = []
        for row in range(n_rows):
            counts, isi_violation, n_wf_missing = [], [], 0
            n_before, n_after = 0, 0
            earliest = t0
            regions = {}
            for chan in range(n_chan):
                spikes = cell(f, "spikes", row, chan)
                if spikes is None:
                    continue
                spikes = np.sort(spikes.reshape(-1))
                counts.append(len(spikes))
                n_before += int((spikes < t0).sum())
                n_after += int((spikes > t1).sum())
                if len(spikes):
                    earliest = min(earliest, float(spikes[0]))
                region = names[chan].split(" ")[0]
                regions[region] = regions.get(region, 0) + 1
                if len(spikes) > 1:
                    isi_violation.append(float((np.diff(spikes) < refractory_s).mean()))
                # Check the waveform cell's shape only; reading the snippets would load ~1 GB.
                if "wf" in f and f[f["wf"][row, chan]].shape == (2,):
                    n_wf_missing += 1
            counts = np.asarray(counts)
            rates = counts / duration if len(counts) else counts
            rows.append({
                "row": row,
                "role": "hash" if row == 0 else "sorted",
                "n_nonempty_cells": int(len(counts)),
                "cells_by_region": regions,
                "n_with_spikes": int((counts > 0).sum()),
                f"n_above_{count_threshold}_spikes": int((counts > count_threshold).sum()),
                f"n_at_least_{rate_threshold_hz}_hz": int((rates >= rate_threshold_hz).sum()),
                "rate_hz_percentiles_0_25_50_75_100": (
                    np.percentile(rates, [0, 25, 50, 75, 100]).round(3).tolist() if len(rates) else []),
                "median_isi_violation_fraction": (
                    float(np.median(isi_violation)) if isi_violation else None),
                "n_nonempty_without_waveforms": n_wf_missing,
                "n_spikes_before_t0": n_before,
                "n_spikes_after_t_end": n_after,
                "earliest_spike_s_before_t0": t0 - earliest,
            })
        sorted_rows = [r for r in rows if r["role"] == "sorted"]
        above = f"n_above_{count_threshold}_spikes"
        summary = {
            "file": str(path),
            "duration_s": duration,
            "n_bins_50ms": int(np.floor(duration / 0.05)),
            "n_channels": n_chan,
            "channels_by_region": {r: sum(n.startswith(r + " ") for n in names) for r in ("M1", "S1")},
            "spikes_rows": n_rows,
            "kinematics": {k: list(f[k].shape) for k in ("cursor_pos", "finger_pos", "target_pos") if k in f},
            "sorted_units_with_spikes": sum(r["n_with_spikes"] for r in sorted_rows),
            f"sorted_units_above_{count_threshold}": sum(r[above] for r in sorted_rows),
            f"sorted_units_above_{count_threshold}_excluding_last_row": sum(r[above] for r in sorted_rows[:-1]),
            f"count_threshold_{count_threshold}_as_rate_hz": count_threshold / duration,
            "rows": rows,
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--out", default=None, help="Write JSON (one object per file) here")
    parser.add_argument("--count-threshold", type=int, default=5000)
    parser.add_argument("--rate-threshold-hz", type=float, default=0.5)
    args = parser.parse_args()
    results = [recount(p, args.count_threshold, args.rate_threshold_hz) for p in args.paths]
    text = json.dumps(results if len(results) > 1 else results[0], indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
