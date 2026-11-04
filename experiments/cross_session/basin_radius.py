"""
Basin-radius test: is the frozen CPIC critic a LOCAL refiner of latent orientation?

Gate 2 found that a frozen pretrained critic strongly penalizes rotated latents (a
Haar-random rotation costs ~2 nats; a random critic ~0), yet read-ins and latent maps
started from random rotations do not find the source coordinates. One explanation is a
narrow basin around the correct orientation with a flat plateau outside it. This test
starts at controlled distances from the correct answer, on the source model's OWN units:

    latent level  maximize frozen-critic InfoNCE over a map z -> z G^T (+ b), starting
                  from a rotation by theta in every plane of a random basis (theta in
                  ANGLES, plus Haar-random), for the pretrained critic (orthogonal and
                  linear maps) and a random critic (orthogonal)
    input level   initialize a read-in as the source encoder rotated by theta and train it
                  against the pretrained / random / fresh critic (same steps as Gate 2)

Pre-registered reading (orthogonal map, pretrained critic): "local refiner" if the median
run recovers (direct R² >= 0.9 vs the source latents) from at least one start >= 20 deg
while the random critic does not recover from that start; "no refinement" if the
pretrained critic fails to recover even from 10 deg. Anything else is "inconclusive".
Neural-only optimization; behavior is used only in post-hoc frozen-decoder readouts.

    python experiments/cross_session/basin_radius.py --seed 0
    python experiments/cross_session/basin_radius.py --summarize results/basin_radius
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cpic.evaluation import encode
from cpic.identifiability import recover_linear_map, rotation_by_angle
from cpic.multisession import BENCHMARK, fit_standard, random_critic_like, transformed_encoder_state
from cpic.utils.helpers import resolve_device
from pseudo_pop import BIN_WIDTH_S, DEFAULT_DATA, load_population
from transfer_common import behavior_targets, fit_source_decoders, run_transfer, score_decoders

REPO_ROOT = Path(__file__).resolve().parents[2]
ANGLES = (0, 5, 10, 20, 30, 45, 60, 90)
INPUT_ANGLES = (0, 5, 20, 45, 90)
LATENT_CONDITIONS = (("pretrained", "orthogonal"), ("pretrained", "linear"), ("random", "orthogonal"))
RECOVERED_R2 = 0.9


def run(args):
    device = resolve_device(args.device)
    cfg = dict(BENCHMARK)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"basin_seed{args.seed}.json"
    if path.exists():
        raise FileExistsError(f"Result already exists: {path}")

    t0 = time.perf_counter()
    X, Y, splits = load_population(args.data, "corrected")
    source = fit_standard(X["train"], device, args.seed, cfg)
    Z = {s: encode(source, X[s]) for s in ("train", "test")}
    targets = behavior_targets({"train": Y["train"], "test": Y["test"]}, BIN_WIDTH_S)
    decoders = fit_source_decoders(Z["train"], {k: v["train"] for k, v in targets.items()})
    test_targets = {k: v["test"] for k, v in targets.items()}
    critics = {"pretrained": source.critic,
               "random": random_critic_like(source.critic, args.seed + 10_000).to(device)}
    d = cfg["ydim"]
    timings = {"setup_s": time.perf_counter() - t0}

    t0 = time.perf_counter()
    latent_rows = []
    for a, angle in enumerate(list(ANGLES) + ["haar"]):
        for init in range(args.n_inits):
            for critic_name, parametrization in LATENT_CONDITIONS:
                rec = recover_linear_map(
                    source, Z["train"], Z["test"], critic=critics[critic_name],
                    parametrization=parametrization, seed=1000 * args.seed + 100 * a + init,
                    steps=args.steps, init_angle_deg=None if angle == "haar" else angle)
                mapped = Z["test"] @ rec.pop("G").T + rec["b"]
                rec["b"] = rec["b"].tolist()
                rec["frozen_decoder_after_map"] = score_decoders(decoders, mapped, test_targets)
                latent_rows.append({"start": str(angle), "init": init, "critic": critic_name, **rec})
        print(f"latent level: start {angle} done", flush=True)
    timings["latent_level_s"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    input_rows = []
    rng = np.random.default_rng(args.seed)
    Z_test = Z["test"].astype(np.float64)
    for angle in INPUT_ANGLES:
        G = rotation_by_angle(d, angle, rng)
        initial_direct = 1.0 - (((Z_test @ G.T - Z_test) ** 2).sum()
                                / ((Z_test - Z_test.mean(axis=0)) ** 2).sum())
        result = run_transfer(source, X, X, Y, Y, device, args.seed, cfg, BIN_WIDTH_S,
                              simultaneous=True, upper_reference=False,
                              init_encoder_state=transformed_encoder_state(source.encoder, rotation=G))
        input_rows.append({"start_angle_deg": angle, "initial_direct_r2": float(initial_direct),
                           "arms": result["arms"], "source": result["source"]})
        print(f"input level: start {angle} deg done", flush=True)
    timings["input_level_s"] = time.perf_counter() - t0

    out = {"seed": args.seed, "device": str(device), "cfg": cfg, "split": splits.as_dict(),
           "steps": args.steps, "n_inits": args.n_inits, "n_units": int(X["train"].shape[1]),
           "identity_frozen_decoder": score_decoders(decoders, Z["test"], test_targets),
           "latent_level": latent_rows, "input_level": input_rows, "timings": timings}
    with open(path, "x") as f:
        json.dump(out, f, indent=1)
    print(f"-> {path}")


def summarize(results_dir):
    runs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("basin_seed*.json"))]
    if not runs:
        print(f"No basin-radius results in {results_dir}")
        return
    identity_r2 = np.mean([r["identity_frozen_decoder"]["position"]["r2_mean"] for r in runs])
    print(f"## Latent level ({len(runs)} source models; identity frozen R² pos = {identity_r2:.4f})")
    print("| start | critic | map | ‖G−I‖/√d init → final (median) | recovered (direct R² ≥ 0.9) "
          "| median direct R² | median frozen R² pos |\n|---|---|---|---|---|---|---|")
    medians = {}
    for start in [str(a) for a in ANGLES] + ["haar"]:
        for critic, parametrization in LATENT_CONDITIONS:
            rows = [x for r in runs for x in r["latent_level"]
                    if x["start"] == start and x["critic"] == critic
                    and x["parametrization"] == parametrization]
            if not rows:
                continue
            med = lambda key: float(np.median([key(x) for x in rows]))
            direct = med(lambda x: x["direct_r2_vs_identity"])
            medians[(start, critic, parametrization)] = direct
            recovered = np.mean([x["direct_r2_vs_identity"] >= RECOVERED_R2 for x in rows])
            print(f"| {start} | {critic} | {parametrization} | "
                  f"{med(lambda x: x['initial_distance_from_identity']):.3f} → "
                  f"{med(lambda x: x['distance_from_identity']):.3f} | {recovered:.0%} of {len(rows)} | "
                  f"{direct:.4f} | {med(lambda x: x['frozen_decoder_after_map']['position']['r2_mean']):.4f} |")

    print("\n## Input level (read-in initialized as the rotated source encoder; mean ± sd over seeds)")
    print("| start | arm | initial direct R² | final direct R² | frozen source decoder R² pos "
          "| refit R² pos |\n|---|---|---|---|---|---|")
    for angle in INPUT_ANGLES:
        rows = [x for r in runs for x in r["input_level"] if x["start_angle_deg"] == angle]
        if not rows:
            continue
        initial = np.mean([x["initial_direct_r2"] for x in rows])
        for arm in rows[0]["arms"]:
            def ms(fn):
                v = np.array([fn(x["arms"][arm]) for x in rows])
                return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
            print(f"| {angle} | {arm} | {initial:.4f} | {ms(lambda a: a['agreement']['direct_r2'])} | "
                  f"{ms(lambda a: a['frozen_source_decoder']['position']['r2_mean'])} | "
                  f"{ms(lambda a: a['refit_probe']['position']['r2_mean'])} |")

    recovered_from = [a for a in ANGLES if a >= 20
                      and medians.get((str(a), "pretrained", "orthogonal"), -np.inf) >= RECOVERED_R2
                      and medians.get((str(a), "random", "orthogonal"), np.inf) < RECOVERED_R2]
    fails_small = all(medians.get((str(a), "pretrained", "orthogonal"), -np.inf) < RECOVERED_R2
                      for a in (5, 10))
    verdict = ("local refiner (recovers from " + ", ".join(f"{a}°" for a in recovered_from) + ")"
               if recovered_from else "no refinement" if fails_small else "inconclusive")
    print(f"\nPre-registered reading (orthogonal map, pretrained vs random critic): {verdict}")


def main():
    parser = argparse.ArgumentParser(description="Basin-radius test for critic-anchored alignment")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-inits", type=int, default=3)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "basin_radius"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", metavar="RESULTS_DIR", default=None)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":
    main()
