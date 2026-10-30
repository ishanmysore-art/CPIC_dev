"""
Experiment A: corrected single-session CPIC baseline, one change at a time.

Each training rung changes exactly one thing relative to the rung before it, with the
model fixed (linear DCA-initialized CPIC, ydim 32, T 4, beta 1e-3, batch 512, lr 1e-3,
10 epochs without early stopping). Every rung records the same measurements, so the
measurement corrections (L1, L2) are columns rather than separate trainings:

    reported_train_bound    what the legacy pipeline reported: the mean minibatch
                            I_pred of the best-training-loss epoch, taken while the
                            weights were still changing                  (L0)
    final_train_bound       the returned model, frozen, scored on training windows
                            with full batches                            (L2)
    val_bound, test_bound   the returned model, frozen, on held-out windows (L1)

Training rungs (cumulative):

    L0_legacy                  load_sabes_data (109 units, whole-session filtering,
                               sqrt, centered 30 s drift filter), 80/20 split, scalar
                               "mean" centering; reproduces the legacy script
    L3_split                   + chronological 70/10/20 split with gap bins
    L4_train_only_prep         + raw counts, same legacy unit set, sqrt + causal
                               per-segment drift removal + per-feature train centering
    L5_no_wraparound           + new loader: spikes recorded before t0 are dropped
                               instead of wrapping into the last bins (same unit rule)
    L6_train_rate_filter       + units chosen by >= 0.5 Hz on the training split
    L7_all_sorted_rows         + every sorted-unit row (fixes the loader off-by-one)

Behavior is used only by the linear probes after training. Usage:

    python experiments/cross_session/ladder.py --seed 0 --out results/ladder
    python experiments/cross_session/ladder.py --summarize results/ladder
"""

import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from cpic.CPIC import CPIC
from cpic.evaluation import (
    block_bootstrap_r2, cursor_velocity, encode, heldout_bounds, probe_suite,
)
from cpic.exp_utils.data_util import load_sabes_data, load_sabes_session
from cpic.exp_utils.sessions import SessionPreprocessor, chronological_splits, legacy_unit_mask
from cpic.utils.data import PastFutureDataset
from cpic.utils.helpers import DCA_init, resolve_device

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "real_data" / "M1" / "indy_20160627_01.mat"
SOURCE_FILES = ["src/cpic/CPIC.py", "src/cpic/models.py", "src/cpic/mi.py", "src/cpic/evaluation.py",
                "src/cpic/exp_utils/data_util.py", "src/cpic/exp_utils/sessions.py",
                "src/cpic/utils/data.py", "src/cpic/utils/helpers.py",
                "experiments/cross_session/ladder.py"]

# Frozen, pre-registered benchmark settings (config_m1_scaling_benchmark.ini).
MODEL = {"ydim": 32, "T": 4, "beta": 1e-3, "hidden_dim": 256, "batch_size": 512,
         "lr": 1e-3, "epochs": 10, "n_init_dca": 5}
OFFSETS = (5, 10, 15)
DECODING_WINDOW = 3
SPLIT = {"fractions": (0.7, 0.1, 0.2), "gap": 30}
MIN_RATE_HZ = 0.5
BIN_WIDTH_S = 0.05

RUNGS = ["L0_legacy", "L3_split", "L4_train_only_prep", "L5_no_wraparound",
         "L6_train_rate_filter", "L7_all_sorted_rows"]


def source_hash():
    digest = hashlib.sha256()
    for rel in SOURCE_FILES:
        digest.update((REPO_ROOT / rel).read_bytes())
    return digest.hexdigest()[:16]


def legacy_center(X_train, X_others):
    """The legacy script's centering: one scalar (np.concatenate flattens the 2-D array)."""
    X_mean = np.concatenate(X_train).mean(axis=0, keepdims=True)
    return X_train - X_mean, [X - X_mean for X in X_others]


# ---------------------------------------------------------------------------
# Data for each rung. Each builder returns float64 X segments and cursor positions.
# ---------------------------------------------------------------------------

def load_sources(path):
    timings = {}
    t0 = time.perf_counter()
    legacy = load_sabes_data(str(path))
    timings["load_legacy_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    legacy_raw = load_sabes_data(str(path), preprocess=False)
    timings["load_legacy_raw_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    session = load_sabes_session(str(path), bin_width_s=BIN_WIDTH_S)
    timings["load_session_s"] = time.perf_counter() - t0
    return {"legacy": legacy, "legacy_raw": legacy_raw, "session": session}, timings


def legacy_preprocessed(sources):
    X = sources["legacy"]["M1"]
    X = X[:, X.std(axis=0) > 1e-6]  # legacy good_cols, computed on the whole session
    return X, sources["legacy"]["cursor"]


def build_legacy(sources):
    X, Y = legacy_preprocessed(sources)
    n_train = int(len(X) * 0.8)
    X_train, (X_test,) = legacy_center(X[:n_train], [X[n_train:]])
    return {"X": {"train": X_train, "test": X_test},
            "Y": {"train": Y[:n_train], "test": Y[n_train:]},
            "split": {"train": [0, n_train], "test": [n_train, len(X)], "gap": 0}}


def _chronological(X, Y):
    splits = chronological_splits(len(X), **SPLIT)
    return splits, {name: X[splits.segment(name)] for name in ("train", "val", "test")}, \
        {name: Y[splits.segment(name)] for name in ("train", "val", "test")}


def build_split(sources):
    X, Y = legacy_preprocessed(sources)
    splits, Xs, Ys = _chronological(X, Y)
    X_train, (X_val, X_test) = legacy_center(Xs["train"], [Xs["val"], Xs["test"]])
    return {"X": {"train": X_train, "val": X_val, "test": X_test}, "Y": Ys,
            "split": splits.as_dict()}


def _train_only(counts, cursor, keep=None, min_rate_hz=None):
    splits = chronological_splits(len(counts), **SPLIT)
    pre = SessionPreprocessor(bin_width_s=BIN_WIDTH_S, min_rate_hz=min_rate_hz)
    pre.fit(counts[splits.segment("train")], keep=keep)
    X = {name: pre.transform(counts[splits.segment(name)], dtype=np.float64)
         for name in ("train", "val", "test")}
    Y = {name: cursor[splits.segment(name)] for name in ("train", "val", "test")}
    return {"X": X, "Y": Y, "split": splits.as_dict(), "keep": pre.keep}


def _legacy_rows(session):
    """Columns the legacy loader would have read (it never visits the last spikes row)."""
    rows = session["unit"] != session["n_spike_rows"] - 1
    return session["counts"][:, rows], session["channel"][rows], session["unit"][rows]


def build_train_only_prep(sources):
    counts = sources["legacy_raw"]["M1"]
    return _train_only(counts, sources["legacy_raw"]["cursor"], keep=legacy_unit_mask(counts))


def build_no_wraparound(sources):
    counts, _, _ = _legacy_rows(sources["session"])
    return _train_only(counts, sources["session"]["cursor_pos"], keep=legacy_unit_mask(counts))


def build_train_rate_filter(sources):
    counts, _, _ = _legacy_rows(sources["session"])
    return _train_only(counts, sources["session"]["cursor_pos"], min_rate_hz=MIN_RATE_HZ)


def build_all_sorted_rows(sources):
    session = sources["session"]
    return _train_only(session["counts"], session["cursor_pos"], min_rate_hz=MIN_RATE_HZ)


BUILDERS = {
    "L0_legacy": build_legacy,
    "L3_split": build_split,
    "L4_train_only_prep": build_train_only_prep,
    "L5_no_wraparound": build_no_wraparound,
    "L6_train_rate_filter": build_train_rate_filter,
    "L7_all_sorted_rows": build_all_sorted_rows,
}


# ---------------------------------------------------------------------------
# Training and evaluation (identical for every rung)
# ---------------------------------------------------------------------------

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(xdim, device):
    T, ydim, hidden = MODEL["T"], MODEL["ydim"], MODEL["hidden_dim"]
    # Same kwargs and construction order as real_data_experiment_standard.py, so the
    # torch RNG stream (and hence L0) matches the legacy script for the same seed.
    return CPIC(
        ydim=ydim, xdim=xdim, T=T,
        mi_params={"estimator_compress": "infonce_upper", "estimator_predictive": "infonce_lower",
                   "critic": "concat", "baseline": "constant"},
        baseline_params={"hidden_dim": hidden},
        encoder_params={"deterministic": False, "n_layers": 1, "encoder_type": "linear"},
        hidden_dim=hidden, beta=MODEL["beta"], beta1=1.0, beta2=0.0,
        beta_warmup_epochs=0, beta_ramp_epochs=0, device=device,
        predictive_space="latent", predictive_loss="mi", reconstruction_targets=("past",),
        regularization_weight=0,
        critic_params={"x_dim": T * ydim, "y_dim": T * ydim, "hidden_dim": hidden},
    ).to(device)


def probe_metrics(Z, Y, bin_width_s):
    out = {}
    for target, transform in (("position", lambda y: y),
                              ("velocity", lambda y: cursor_velocity(y, bin_width_s))):
        Y_train, Y_test = transform(Y["train"]), transform(Y["test"])
        suite = probe_suite(Z["train"], Y_train, Z["test"], Y_test, OFFSETS, DECODING_WINDOW)
        out[target] = {
            "r2_by_offset": {str(k): v for k, v in suite["r2_by_offset"].items()},
            "r2_mean": suite["r2_mean"],
            "bootstrap_by_offset": {
                str(o): block_bootstrap_r2(d, Z["test"], Y_test, DECODING_WINDOW, o,
                                           block_bins=int(round(60 / bin_width_s)), n_boot=500)
                for o, d in suite["decoders"].items()},
        }
    return out


def run_rung(rung, data, seed, device, use_dca=True, epochs=None):
    epochs = MODEL["epochs"] if epochs is None else epochs
    X, T = data["X"], MODEL["T"]
    cuda = str(device).startswith("cuda")
    seed_everything(seed)
    timings = {}

    t0 = time.perf_counter()
    init_weights = (DCA_init(X["train"], T=T, d=MODEL["ydim"], n_init=MODEL["n_init_dca"])
                    if use_dca else None)
    timings["dca_init_s"] = time.perf_counter() - t0

    model = build_model(X["train"].shape[1], device)
    if cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    _, reported_compress, reported_predictive = model.fit(
        PastFutureDataset([X["train"]], window_size=T), init_weights=init_weights,
        epochs=epochs, batch_size=MODEL["batch_size"], lr=MODEL["lr"],
        early_stop=epochs + 1, verbose=True)
    if cuda:
        torch.cuda.synchronize()
    timings["train_s"] = time.perf_counter() - t0
    peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2 if cuda else 0.0

    t0 = time.perf_counter()
    bounds = {}
    for split in X:
        bounds[f"{split}_sampled"] = heldout_bounds(model, X[split], MODEL["batch_size"], seed=seed)
        bounds[f"{split}_mean_latent"] = heldout_bounds(model, X[split], MODEL["batch_size"],
                                                        seed=seed, sample_latents=False)
    Z = {split: encode(model, X[split]) for split in X}
    probes = probe_metrics(Z, data["Y"], BIN_WIDTH_S)
    timings["eval_s"] = time.perf_counter() - t0

    for b in bounds.values():
        b["I_predictive_per_batch"] = b["I_predictive_per_batch"].tolist()
    return {
        "rung": rung, "seed": int(seed), "device": str(device), "epochs": int(epochs),
        "dca_init": bool(use_dca),
        "n_units": int(X["train"].shape[1]),
        "n_bins": {split: int(len(X[split])) for split in X},
        "split": data["split"],
        "reported_train_bound": float(reported_predictive),
        "reported_train_compress": float(reported_compress),
        "final_epoch_running_mean_bound": float(model.final_mean_I_predictive),
        "bounds": bounds,
        "probes": probes,
        "timings": timings,
        "peak_gpu_memory_mb": float(peak_mb),
    }


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

SUMMARY_METRICS = {
    "n_units": lambda r: r["n_units"],
    "reported_train_bound": lambda r: r["reported_train_bound"],
    "final_train_bound": lambda r: r["bounds"]["train_sampled"]["I_predictive_mean"],
    "val_bound": lambda r: r["bounds"].get("val_sampled", {}).get("I_predictive_mean", np.nan),
    "test_bound": lambda r: r["bounds"]["test_sampled"]["I_predictive_mean"],
    "test_bound_mean_latent": lambda r: r["bounds"]["test_mean_latent"]["I_predictive_mean"],
    "test_r2_position": lambda r: r["probes"]["position"]["r2_mean"],
    "test_r2_velocity": lambda r: r["probes"]["velocity"]["r2_mean"],
    "train_s": lambda r: r["timings"]["train_s"],
    "dca_init_s": lambda r: r["timings"]["dca_init_s"],
    "peak_gpu_memory_mb": lambda r: r["peak_gpu_memory_mb"],
}


def summarize(results_dir):
    """Mean ± sd over seeds per rung, plus deltas vs the previous rung paired by seed."""
    runs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("*.json"))]
    if not runs:
        print(f"No ladder results in {results_dir}; check the job logs (logs/cpic-ladder-*).")
        return []
    by_rung = {rung: {r["seed"]: r for r in runs if r["rung"] == rung} for rung in RUNGS}
    rows, previous = [], None
    for rung in RUNGS:
        seeds = by_rung[rung]
        if not seeds:
            continue
        row = {"rung": rung, "n_seeds": len(seeds)}
        for name, get in SUMMARY_METRICS.items():
            values = np.array([get(r) for r in seeds.values()], dtype=float)
            row[f"{name}_mean"] = float(np.mean(values))
            row[f"{name}_sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
        if previous is not None:
            shared = sorted(set(seeds) & set(by_rung[previous]))
            for name in ("test_bound", "test_r2_position", "test_r2_velocity"):
                get = SUMMARY_METRICS[name]
                deltas = np.array([get(seeds[s]) - get(by_rung[previous][s]) for s in shared])
                row[f"delta_{name}_mean"] = float(deltas.mean()) if len(deltas) else float("nan")
                row[f"delta_{name}_sd"] = (float(deltas.std(ddof=1)) if len(deltas) > 1
                                           else float("nan"))
        rows.append(row)
        previous = rung

    out_csv = Path(results_dir) / "ladder_summary.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row},
                                                     key=lambda k: (k != "rung", k)))
        writer.writeheader()
        writer.writerows(rows)

    def fmt(row, name, digits=4):
        m, s = row.get(f"{name}_mean", np.nan), row.get(f"{name}_sd", np.nan)
        return f"{m:.{digits}f} ± {s:.{digits}f}" if np.isfinite(s) else f"{m:.{digits}f}"

    print("| rung | seeds | units | reported train | final train | val | test | test R² pos | "
          "test R² vel | Δ test bound | Δ R² pos | train s |")
    print("|---" * 12 + "|")
    for row in rows:
        print(f"| {row['rung']} | {row['n_seeds']} | {row['n_units_mean']:.0f} | "
              f"{fmt(row, 'reported_train_bound')} | {fmt(row, 'final_train_bound')} | "
              f"{fmt(row, 'val_bound')} | {fmt(row, 'test_bound')} | "
              f"{fmt(row, 'test_r2_position')} | {fmt(row, 'test_r2_velocity')} | "
              f"{fmt(row, 'delta_test_bound') if 'delta_test_bound_mean' in row else '—'} | "
              f"{fmt(row, 'delta_test_r2_position') if 'delta_test_r2_position_mean' in row else '—'} | "
              f"{row['train_s_mean']:.1f} |")
    print(f"\nWrote {out_csv}")
    return rows


def main():
    parser = argparse.ArgumentParser(description="Experiment A attribution ladder")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rungs", default="all", help="Comma-separated rung names, or 'all'")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "ladder"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", metavar="RESULTS_DIR", default=None)
    args = parser.parse_args()

    if args.summarize:
        summarize(args.summarize)
        return

    rungs = RUNGS if args.rungs == "all" else [r.strip() for r in args.rungs.split(",")]
    unknown = set(rungs) - set(RUNGS)
    if unknown:
        parser.error(f"Unknown rungs: {sorted(unknown)}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for rung in rungs:
        if (out_dir / f"{rung}_seed{args.seed}.json").exists():
            raise FileExistsError(f"Result already exists: {out_dir / f'{rung}_seed{args.seed}.json'}")

    device = resolve_device(args.device)
    sources, load_timings = load_sources(args.data)
    provenance = {"source_hash": source_hash(), "data": str(args.data), "model": MODEL,
                  "offsets": list(OFFSETS), "decoding_window": DECODING_WINDOW,
                  "split_protocol": {"fractions": list(SPLIT["fractions"]), "gap": SPLIT["gap"]},
                  "min_rate_hz": MIN_RATE_HZ, "load_timings": load_timings}
    for rung in rungs:
        print(f"=== {rung} | seed {args.seed} | device {device} ===", flush=True)
        t0 = time.perf_counter()
        data = BUILDERS[rung](sources)
        build_s = time.perf_counter() - t0
        result = run_rung(rung, data, args.seed, device)
        result["timings"]["build_data_s"] = build_s
        result["provenance"] = provenance
        path = out_dir / f"{rung}_seed{args.seed}.json"
        with open(path, "x") as f:
            json.dump(result, f, indent=1)
        print(f"{rung}: units={result['n_units']} reported={result['reported_train_bound']:.4f} "
              f"final_train={result['bounds']['train_sampled']['I_predictive_mean']:.4f} "
              f"test={result['bounds']['test_sampled']['I_predictive_mean']:.4f} "
              f"R2_pos={result['probes']['position']['r2_mean']:.4f} -> {path}", flush=True)


if __name__ == "__main__":
    main()
