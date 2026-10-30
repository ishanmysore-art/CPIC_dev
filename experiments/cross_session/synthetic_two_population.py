"""
Synthetic sanity check for critic-anchored alignment (run before trusting real data).

Two populations observe the SAME 3-D Lorenz trajectory through different random linear
loadings plus independent noise. CPIC is trained on population 1; population 2 gets a
new read-in against (i) the frozen pretrained critic, (ii) a frozen random critic, and
(iii) a fresh jointly trained critic. Expected if the method works: arm (i) places
population-2 latents in population-1 coordinates (direct latent R² high, e.g. > 0.8)
and arm (ii) does not. The Lorenz state stands in for "behavior" in the probes.

    python experiments/cross_session/synthetic_two_population.py --seed 0
"""

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import odeint
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from cpic.evaluation import encode
from cpic.exp_utils.sessions import chronological_splits
from cpic.multisession import BENCHMARK, fit_standard
from cpic.utils.helpers import resolve_device
from transfer_common import run_transfer

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DT = 0.025


def lorenz(n_samples, integration_dt=0.005, skip=1000):
    def dx_dt(state, _t, sigma=10.0, rho=28.0, beta=8 / 3.0):
        x, y, z = state
        return sigma * (y - x), x * (rho - z) - y, x * y - beta * z

    stride = int(round(DATA_DT / integration_dt))
    t = np.arange(0, (n_samples + skip) * stride) * integration_dt
    X = odeint(dx_dt, np.ones(3), t)[::stride][skip:skip + n_samples]
    return (X - X.mean(axis=0)) / X.std(axis=0)


def observe(latent, n_units, noise_std, rng):
    loadings = rng.normal(size=(n_units, latent.shape[1])) / np.sqrt(latent.shape[1])
    return latent @ loadings.T + noise_std * rng.normal(size=(len(latent), n_units))


def split_and_center(X, splits):
    seg = {s: X[splits.segment(s)] for s in ("train", "val", "test")}
    mean = seg["train"].mean(axis=0)
    return {s: x - mean for s, x in seg.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-samples", type=int, default=30000)
    parser.add_argument("--n-units", type=int, default=30)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--ydim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--no-dca", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "synthetic_two_population"))
    args = parser.parse_args()

    device = resolve_device(args.device)
    cfg = dict(BENCHMARK, ydim=args.ydim, epochs=args.epochs)
    rng = np.random.default_rng(args.seed)
    latent = lorenz(args.n_samples)
    splits = chronological_splits(len(latent), fractions=(0.7, 0.1, 0.2), gap=30)
    X1 = split_and_center(observe(latent, args.n_units, args.noise_std, rng), splits)
    X2 = split_and_center(observe(latent, args.n_units, args.noise_std, rng), splits)
    Y = {s: latent[splits.segment(s)] for s in ("train", "val", "test")}

    source = fit_standard(X1["train"], device, args.seed, cfg, use_dca=not args.no_dca)
    result = run_transfer(source, X1, X2, Y, Y, device, args.seed, cfg, DATA_DT,
                          simultaneous=True, use_dca=not args.no_dca)

    def true_latent_r2(Z_train, Z_test):
        fit = LinearRegression().fit(Z_train, Y["train"])
        return float(r2_score(Y["test"], fit.predict(Z_test), multioutput="variance_weighted"))

    Z_src = {s: encode(source, X1[s]) for s in ("train", "test")}
    result["source"]["true_latent_linear_r2"] = true_latent_r2(Z_src["train"], Z_src["test"])
    result.update({"seed": args.seed, "cfg": cfg, "n_units": args.n_units,
                   "noise_std": args.noise_std, "n_samples": args.n_samples})

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"synthetic_seed{args.seed}.json"
    with open(path, "x") as f:
        json.dump(result, f, indent=1)

    print(f"source: true-latent linear R² {result['source']['true_latent_linear_r2']:.4f}")
    for arm, r in list(result["arms"].items()) + [("target_specific_cpic", result["target_specific_cpic"])]:
        print(f"{arm:>22}: direct latent R² {r['agreement']['direct_r2']:+.4f} | procrustes "
              f"{r['agreement']['procrustes_r2']:+.4f} | frozen source decoder R² "
              f"{r['frozen_source_decoder']['position']['r2_mean']:+.4f} | refit "
              f"{r['refit_probe']['position']['r2_mean']:+.4f}")
    pretrained = result["arms"]["pretrained"]["agreement"]["direct_r2"]
    random_arm = result["arms"]["random"]["agreement"]["direct_r2"]
    verdict = "PASS" if pretrained > 0.8 and pretrained - random_arm > 0.3 else "FAIL"
    print(f"sanity check (pretrained direct R² > 0.8 and > random + 0.3): {verdict} -> {path}")


if __name__ == "__main__":
    main()
