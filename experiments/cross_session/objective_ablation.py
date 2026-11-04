"""
What does CPIC's compression term buy over plain latent prediction?

Single session, corrected pipeline (ladder L7: all sorted rows, >= 0.5 Hz on train,
train-only preprocessing), linear DCA-initialized CPIC with everything fixed except the
compression weight beta. beta = 0 leaves only the InfoNCE prediction of future latents
from past latents: a no-compression, contrastive latent-prediction model (CPC-like in
objective, though not CPC's autoregressive architecture). Neural-only training; behavior
only in post-hoc linear probes.

Readouts on held-out data: InfoNCE predictive bound (sampled and mean latents),
compression bound, refit-probe R² (position, velocity), and the participation ratio of
test latents (effective dimensionality).

    python experiments/cross_session/objective_ablation.py --seed 0
    python experiments/cross_session/objective_ablation.py --summarize results/objective_ablation
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cpic.evaluation import encode, heldout_bounds
from cpic.multisession import BENCHMARK, fit_standard
from cpic.utils.helpers import resolve_device
from pseudo_pop import BIN_WIDTH_S, DEFAULT_DATA, load_population
from transfer_common import behavior_targets, paired_gaps, refit_probes

REPO_ROOT = Path(__file__).resolve().parents[2]
BETAS = (0.0, 1e-4, 1e-3)
REFERENCE_BETA = 1e-3


def participation_ratio(Z):
    eig = np.clip(np.linalg.eigvalsh(np.cov(Z, rowvar=False)), 0, None)
    return float(eig.sum() ** 2 / (eig ** 2).sum())


def run(args):
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    X, Y, splits = load_population(args.data, "corrected")
    targets = behavior_targets({"train": Y["train"], "test": Y["test"]}, BIN_WIDTH_S)
    for beta in BETAS:
        path = out_dir / f"ablation_beta{beta:g}_seed{args.seed}.json"
        if path.exists():
            raise FileExistsError(f"Result already exists: {path}")
        cfg = dict(BENCHMARK, beta=beta)
        t0 = time.perf_counter()
        model = fit_standard(X["train"], device, args.seed, cfg)
        train_s = time.perf_counter() - t0
        sampled = heldout_bounds(model, X["test"], cfg["batch_size"], seed=args.seed)
        mean_latent = heldout_bounds(model, X["test"], cfg["batch_size"], seed=args.seed,
                                     sample_latents=False)
        Z = {s: encode(model, X[s]) for s in ("train", "test")}
        result = {
            "beta": beta, "seed": args.seed, "cfg": cfg, "n_units": int(X["train"].shape[1]),
            "split": splits.as_dict(),
            "test_bound_sampled": sampled["I_predictive_mean"],
            "test_bound_mean_latent": mean_latent["I_predictive_mean"],
            "test_compress_bound": sampled.get("I_compress_mean"),
            "refit_probe": refit_probes(Z, targets),
            "participation_ratio_test": participation_ratio(Z["test"]),
            "train_s": train_s,
        }
        with open(path, "x") as f:
            json.dump(result, f, indent=1)
        print(f"beta={beta:g} seed={args.seed}: test bound {result['test_bound_sampled']:.4f} | "
              f"compress {result['test_compress_bound']:.2f} | R² pos "
              f"{result['refit_probe']['position']['r2_mean']:.4f} | PR "
              f"{result['participation_ratio_test']:.2f} -> {path}", flush=True)


def summarize(results_dir):
    runs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("ablation_*.json"))]
    if not runs:
        print(f"No ablation results in {results_dir}")
        return
    metrics = {
        "test bound (sampled)": lambda r: r["test_bound_sampled"],
        "test bound (mean latent)": lambda r: r["test_bound_mean_latent"],
        "test compression bound": lambda r: r["test_compress_bound"],
        "refit R² pos": lambda r: r["refit_probe"]["position"]["r2_mean"],
        "refit R² vel": lambda r: r["refit_probe"]["velocity"]["r2_mean"],
        "participation ratio": lambda r: r["participation_ratio_test"],
    }
    betas = sorted({r["beta"] for r in runs})
    print("| beta | seeds | " + " | ".join(metrics) + " |")
    print("|---" * (len(metrics) + 2) + "|")
    for beta in betas:
        group = [r for r in runs if r["beta"] == beta]
        cells = []
        for fn in metrics.values():
            v = np.array([fn(r) for r in group], dtype=float)
            cells.append(f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}")
        print(f"| {beta:g} | {len(group)} | " + " | ".join(cells) + " |")
    reference = {r["seed"]: r for r in runs if r["beta"] == REFERENCE_BETA}
    for beta in betas:
        if beta == REFERENCE_BETA:
            continue
        paired = [{"seed": s, "arms": {"a": r, "b": reference[s]}}
                  for s, r in ((r["seed"], r) for r in runs if r["beta"] == beta) if s in reference]
        for name in ("test bound (sampled)", "refit R² pos"):
            g = paired_gaps(paired, metrics[name], "a", "b", lambda r: r["seed"])
            ci = f"95% t-CI {g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}" if g["ci95"] else "n=1"
            print(f"Δ {name}: beta={beta:g} − beta={REFERENCE_BETA:g}: {g['mean']:+.4f} ({ci}; n={g['n']})")


def main():
    parser = argparse.ArgumentParser(description="Compression-weight ablation on the corrected baseline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "objective_ablation"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", metavar="RESULTS_DIR", default=None)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":
    main()
