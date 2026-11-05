"""
Gate 2: what can a frozen CPIC critic align? (pseudo-populations from one session)

I3  Split the session's units into populations P1 and P2 (disjoint, 50% or 100% overlap).
    Train standard CPIC on P1, then a new P2 read-in against (i) P1's frozen critic,
    (ii) a matched frozen random critic, (iii) a fresh critic trained jointly. Because P1
    and P2 are recorded simultaneously, the correct alignment is known: P2 latents should
    match P1 latents at the same time points. P2 is treated as having no correspondence
    with P1 (its own read-in); this is NOT cross-session transfer.
I1/I2 (--identifiability) On the P1 source model: frozen-critic bound under controlled
    latent maps, InfoNCE gradient energy per latent direction, and self-recovery of the
    identity map from random rotations, each for the pretrained and a random critic.

Neural-only training throughout; behavior enters only through post-hoc linear probes.

    python experiments/cross_session/pseudo_pop.py --overlap 0 --partition-seed 0 --seed 0
    python experiments/cross_session/pseudo_pop.py --overlap 1 --partition-seed 0 --seed 0 --identifiability
    python experiments/cross_session/pseudo_pop.py --summarize results/pseudo_pop
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from cpic.evaluation import encode
from cpic.exp_utils.data_util import load_sabes_session
from cpic.exp_utils.sessions import SessionPreprocessor, chronological_splits, legacy_unit_mask
from cpic.identifiability import invariance_profile, objective_sensitivity, recover_linear_map
from cpic.multisession import BENCHMARK, fit_standard, population_partitions, random_critic_like
from cpic.utils.helpers import resolve_device
from transfer_common import (
    behavior_targets, fit_source_decoders, paired_gaps, run_transfer, score_decoders,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = REPO_ROOT / "data" / "real_data" / "M1" / "indy_20160627_01.mat"
BIN_WIDTH_S = 0.05
SPLIT = {"fractions": (0.7, 0.1, 0.2), "gap": 30}
MIN_RATE_HZ = 0.5


def load_population(path, pipeline):
    """Train-only preprocessed session: "corrected" = all sorted rows, >= 0.5 Hz on train
    (ladder L7); "legacy_units" = the legacy 109-unit set (ladder L5)."""
    session = load_sabes_session(str(path), bin_width_s=BIN_WIDTH_S)
    counts = session["counts"]
    keep = None
    pre = SessionPreprocessor(bin_width_s=BIN_WIDTH_S, min_rate_hz=MIN_RATE_HZ)
    if pipeline == "legacy_units":
        counts = counts[:, session["unit"] != session["n_spike_rows"] - 1]
        keep = legacy_unit_mask(counts)
    splits = chronological_splits(len(counts), **SPLIT)
    pre.fit(counts[splits.segment("train")], keep=keep)
    X = {s: pre.transform(counts[splits.segment(s)], dtype=np.float64) for s in ("train", "val", "test")}
    Y = {s: session["cursor_pos"][splits.segment(s)] for s in ("train", "val", "test")}
    return X, Y, splits


def identifiability(source, X1, Y, device, seed, n_inits, steps):
    """I1 and I2 on the source model, for its own critic and a matched random critic."""
    Z = {s: encode(source, X1[s]) for s in ("train", "test")}
    critics = {"pretrained": source.critic,
               "random": random_critic_like(source.critic, seed + 10_000).to(device)}
    targets = behavior_targets({"train": Y["train"], "test": Y["test"]}, BIN_WIDTH_S)
    decoders = fit_source_decoders(Z["train"], {k: v["train"] for k, v in targets.items()})
    test_targets = {k: v["test"] for k, v in targets.items()}
    identity_frozen = score_decoders(decoders, Z["test"], test_targets)
    out = {"identity_frozen_decoder": identity_frozen, "profile": {}, "sensitivity": {},
           "recovery": []}
    for name, critic in critics.items():
        out["profile"][name] = invariance_profile(source, X1["test"], Z["train"], critic=critic,
                                                  seed=seed)
        sens = objective_sensitivity(source, X1["test"], Z["train"], critic=critic, seed=seed)
        out["sensitivity"][name] = {k: v.tolist() for k, v in sens.items()}
        for parametrization in ("orthogonal", "linear"):
            for init in range(n_inits):
                rec = recover_linear_map(source, Z["train"], Z["test"], critic=critic,
                                         parametrization=parametrization,
                                         seed=1000 * seed + init, steps=steps)
                mapped = Z["test"] @ rec["G"].T + rec["b"]
                rec["frozen_decoder_after_map"] = score_decoders(decoders, mapped, test_targets)
                rec.pop("G")
                rec["b"] = rec["b"].tolist()
                out["recovery"].append({"critic": name, "init": init, **rec})
    return out


def run(args):
    device = resolve_device(args.device)
    cfg = dict(BENCHMARK)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.pipeline}_ov{args.overlap:g}_p{args.partition_seed}_s{args.seed}"
    path = out_dir / f"pseudo_{tag}.json"
    id_path = out_dir / f"identifiability_{tag}.json"
    for p in [path] + ([id_path] if args.identifiability else []):
        if p.exists():
            raise FileExistsError(f"Result already exists: {p}")

    t0 = time.perf_counter()
    X, Y, splits = load_population(args.data, args.pipeline)
    idx1, idx2 = population_partitions(X["train"].shape[1], args.overlap,
                                       np.random.default_rng(args.partition_seed))
    X1 = {s: x[:, idx1] for s, x in X.items()}
    X2 = {s: x[:, idx2] for s, x in X.items()}
    load_s = time.perf_counter() - t0
    print(f"{tag}: {X['train'].shape[1]} units -> P1 {len(idx1)}, P2 {len(idx2)}, "
          f"shared {len(np.intersect1d(idx1, idx2))}", flush=True)

    t0 = time.perf_counter()
    source = fit_standard(X1["train"], device, args.seed, cfg, use_dca=not args.no_dca)
    source_s = time.perf_counter() - t0

    t0 = time.perf_counter()
    result = run_transfer(source, X1, X2, Y, Y, device, args.seed, cfg, BIN_WIDTH_S,
                          simultaneous=True, use_dca=not args.no_dca)
    result.update({
        "pipeline": args.pipeline, "overlap": args.overlap, "partition_seed": args.partition_seed,
        "seed": args.seed, "device": str(device), "cfg": cfg, "split": splits.as_dict(),
        "P1_units": idx1.tolist(), "P2_units": idx2.tolist(),
        "timings": {"load_s": load_s, "source_fit_s": source_s,
                    "transfer_s": time.perf_counter() - t0},
    })
    with open(path, "x") as f:
        json.dump(result, f, indent=1)
    arms = result["arms"]
    print("frozen source decoder R² (position): " + ", ".join(
        f"{a}={arms[a]['frozen_source_decoder']['position']['r2_mean']:.4f}" for a in arms)
        + f" | direct latent R²: " + ", ".join(
        f"{a}={arms[a]['agreement']['direct_r2']:.4f}" for a in arms) + f" -> {path}", flush=True)

    if args.identifiability:
        t0 = time.perf_counter()
        ident = identifiability(source, X1, Y, device, args.seed, args.recovery_inits,
                                args.recovery_steps)
        ident.update({"tag": tag, "timings": {"identifiability_s": time.perf_counter() - t0}})
        with open(id_path, "x") as f:
            json.dump(ident, f, indent=1)
        print(f"identifiability -> {id_path}", flush=True)


# ---------------------------------------------------------------------------
# Summary and Gate 2 checks
# ---------------------------------------------------------------------------

def frozen_position(arm):
    return arm["frozen_source_decoder"]["position"]["r2_mean"]


def direct_agreement(arm):
    return arm["agreement"]["direct_r2"]


def summarize(results_dir):
    results_dir = Path(results_dir)
    # "arms" excludes pseudo_pop_summary.json, which this function itself writes into the same folder.
    runs = [r for r in (json.loads(p.read_text()) for p in sorted(results_dir.glob("pseudo_*.json")))
            if "arms" in r]
    if not runs:
        print(f"No pseudo-population results in {results_dir}")
        return
    summary = {}
    for key in sorted({(r["pipeline"], r["overlap"]) for r in runs}):
        group = [r for r in runs if (r["pipeline"], r["overlap"]) == key]
        print(f"\n## pipeline={key[0]} overlap={key[1]:g}  ({len(group)} runs)")
        print("| arm | frozen source decoder R² pos | frozen R² vel | refit R² pos | direct latent R² "
              "| procrustes R² | CKA | oracle-procrustes frozen R² pos | test bound (own critic) |")
        print("|---" * 9 + "|")
        for arm in list(group[0]["arms"]) + ["target_specific_cpic"]:
            get = (lambda r, a=arm: r["arms"][a]) if arm != "target_specific_cpic" else \
                (lambda r: r["target_specific_cpic"])
            rows = [get(r) for r in group if arm in r["arms"] or arm == "target_specific_cpic"]

            def ms(fn):
                v = np.array([fn(x) for x in rows])
                return f"{v.mean():.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():.4f}"
            print(f"| {arm} | {ms(frozen_position)} | "
                  f"{ms(lambda x: x['frozen_source_decoder']['velocity']['r2_mean'])} | "
                  f"{ms(lambda x: x['refit_probe']['position']['r2_mean'])} | {ms(direct_agreement)} | "
                  f"{ms(lambda x: x['agreement']['procrustes_r2'])} | {ms(lambda x: x['agreement']['cka'])} | "
                  f"{ms(lambda x: x['oracle_procrustes_frozen_source_decoder']['position']['r2_mean'])} | "
                  f"{ms(lambda x: x['bound_own_critic_test'])} |")
        source_r2 = np.mean([r["source"]["refit_probe"]["position"]["r2_mean"] for r in group])
        print(f"source P1 refit R² pos (reference): {source_r2:.4f}")

        gaps = {}
        for name, metric in (("frozen_r2_pos", frozen_position), ("direct_latent_r2", direct_agreement)):
            for control in ("random", "fresh"):
                g = paired_gaps(group, metric, "pretrained", control, lambda r: r["partition_seed"])
                gaps[f"{name}: pretrained - {control}"] = g
                ci = g["ci95"]
                print(f"Δ {name} pretrained − {control}: {g['mean']:+.4f} "
                      f"(95% t-CI {ci[0]:+.4f}, {ci[1]:+.4f}; n={g['n']}) "
                      f"| partitions with positive mean gap: {g['n_groups_positive']}/{g['n_groups']}"
                      if ci else f"Δ {name} pretrained − {control}: {g['mean']:+.4f} (n={g['n']})")
        if key[1] == 0:
            checks = {k: bool(g["ci95"] and g["ci95"][0] > 0 and g["n_groups_positive"] >= 2)
                      for k, g in gaps.items()}
            print("Gate 2 / I3 (disjoint): " + ("PASS" if all(checks.values()) else "FAIL")
                  + " " + json.dumps(checks))
        summary[f"{key[0]}_ov{key[1]:g}"] = gaps

    ident_runs = [json.loads(p.read_text()) for p in sorted(results_dir.glob("identifiability_*.json"))]
    if ident_runs:
        print(f"\n## Identifiability ({len(ident_runs)} source models)")
        families = {}
        for run in ident_runs:
            for critic, rows in run["profile"].items():
                for row in rows:
                    families.setdefault((critic, row["family"], row["param"]), []).append(row["delta"])
        print("| critic | family | param | mean ΔI (identity − map) |\n|---|---|---|---|")
        for (critic, family, param), deltas in sorted(families.items()):
            print(f"| {critic} | {family} | {param:g} | {np.mean(deltas):+.4f} |")
        print("\n| critic | parametrization | median ‖G−I‖/√d (init → final) | median direct R² "
              "| median frozen R² pos after map | identity frozen R² pos |\n|---|---|---|---|---|---|")
        identity_r2 = np.mean([r["identity_frozen_decoder"]["position"]["r2_mean"] for r in ident_runs])
        for critic in ("pretrained", "random"):
            for param in ("orthogonal", "linear"):
                recs = [x for r in ident_runs for x in r["recovery"]
                        if x["critic"] == critic and x["parametrization"] == param]
                if not recs:
                    continue
                med = lambda fn: float(np.median([fn(x) for x in recs]))
                print(f"| {critic} | {param} | {med(lambda x: x['initial_distance_from_identity']):.3f} → "
                      f"{med(lambda x: x['distance_from_identity']):.3f} | "
                      f"{med(lambda x: x['direct_r2_vs_identity']):.4f} | "
                      f"{med(lambda x: x['frozen_decoder_after_map']['position']['r2_mean']):.4f} | "
                      f"{identity_r2:.4f} |")
    (results_dir / "pseudo_pop_summary.json").write_text(json.dumps(summary, indent=1))


def main():
    parser = argparse.ArgumentParser(description="Gate 2 pseudo-population and identifiability tests")
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--partition-seed", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pipeline", choices=["corrected", "legacy_units"], default="corrected")
    parser.add_argument("--identifiability", action="store_true")
    parser.add_argument("--recovery-inits", type=int, default=10)
    parser.add_argument("--recovery-steps", type=int, default=500)
    parser.add_argument("--no-dca", action="store_true", help="Debug only: skip DCA initialization")
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "pseudo_pop"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", metavar="RESULTS_DIR", default=None)
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize)
    else:
        run(args)


if __name__ == "__main__":
    main()
