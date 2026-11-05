"""
Gate 3: genuine cross-session transfer, Indy session A -> session B, in electrode space.

Electrode columns (all threshold crossings on one Utah-array channel) correspond
PHYSICALLY across sessions of the same array, not cellularly: which neurons a channel
picks up, thresholds and impedance all change between days. That correspondence gives a
read-in initialization near A's coordinates; Gate 2 suggested the frozen critic might
then act as a local refiner. All representation training and adaptation is neural-only.

Arms on session B (one model each; A's frozen decoder is the coordinate-transfer readout):
    zero_shot_source_norm       D1: A's frozen model with A's normalization (zero-shot)
    zero_shot_target_norm       D1': A's frozen model, B's own train-split centering
                                (normalization adaptation, unlabeled B)
    zero_shot_permuted          D1' with B's electrodes randomly permuted (control: is
                                zero-shot driven by the electrode correspondence?)
    refine_pretrained           read-in initialized from A's encoder, refined against A's
                                frozen critic on unlabeled B (unsupervised adaptation)
    refine_random               same init and steps, matched frozen random critic
    refine_fresh                same init and steps, fresh critic trained jointly
    refine_pretrained_shuffled  as refine_pretrained but trained on B with every electrode
                                independently circularly shifted (no shared dynamics);
                                evaluated on real B
    permuted_init_pretrained    read-in initialized from A's encoder with electrodes
                                permuted, refined against A's frozen critic
    random_init_pretrained      random read-in init vs A's frozen critic (Gate 2 setting)
    target_specific_cpic        standard CPIC on B alone in B's own unit-filtered electrode
                                space: an upper reference for refit probes, NOT a transfer

Readouts: A's frozen decoder on B test latents for cursor velocity, cursor position
re-standardized with each session's own train-split statistics (an evaluation-side
adjustment for workspace offsets), and raw position; refit probes on B (representation
content only, and the ceiling for any linear alignment + frozen decoder); bounds under
A's critic; relative change of the read-in from its initialization.

Pre-registered Gate 3 reading: refine_pretrained must beat refine_random and refine_fresh
on frozen-decoder R² (re-standardized position; velocity reported alongside), with the
same sign of the seed-mean gap in every target session. Seeds quantify within-session
variability only; with 2 sessions any cross-session statement is descriptive.

    python experiments/cross_session/transfer_sessions.py --target $SCRATCH/cpic-data/real_data/M1/indy_20160624_03.mat --seed 0
    python experiments/cross_session/transfer_sessions.py --summarize results/transfer_sessions
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from scipy.stats import rankdata

from cpic.evaluation import cursor_velocity, encode, heldout_bounds
from cpic.exp_utils.data_util import load_sabes_session
from cpic.exp_utils.sessions import SessionPreprocessor, chronological_splits
from cpic.identifiability import critic_bound_under_transform
from cpic.multisession import BENCHMARK, fit_readin, fit_standard, transformed_encoder_state
from cpic.utils.helpers import resolve_device
from transfer_common import fit_source_decoders, paired_gaps, refit_probes, score_decoders

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = REPO_ROOT / "data" / "real_data" / "M1" / "indy_20160627_01.mat"
BIN_WIDTH_S = 0.05
SPLIT = {"fractions": (0.7, 0.1, 0.2), "gap": 30}
MIN_RATE_HZ = 0.5
PRIMARY = "position_restandardized"
SEGMENTS = ("train", "test")


def segments(counts, splits, pre):
    return {s: pre.transform(counts[splits.segment(s)], dtype=np.float64) for s in SEGMENTS}


def targets_for(session, splits):
    """Velocity, raw position, and position standardized with this session's train stats."""
    pos = {s: session["cursor_pos"][splits.segment(s)] for s in SEGMENTS}
    mu, sd = pos["train"].mean(axis=0), pos["train"].std(axis=0)
    return {"velocity": {s: cursor_velocity(p, BIN_WIDTH_S) for s, p in pos.items()},
            "position_raw": pos,
            "position_restandardized": {s: (p - mu) / sd for s, p in pos.items()}}


def behavior_conventions(session, splits):
    pos = session["cursor_pos"][splits.segment("train")]
    return {"duration_s": session["duration_s"], "n_bins": int(len(session["counts"])),
            "cursor_train_mean": pos.mean(axis=0).tolist(), "cursor_train_std": pos.std(axis=0).tolist(),
            "cursor_train_min": pos.min(axis=0).tolist(), "cursor_train_max": pos.max(axis=0).tolist(),
            "finger_pos_dims": int(session["finger_pos"].shape[1]) if "finger_pos" in session else None}


def circular_shuffle(X, rng):
    """Independently circularly shift every column (keeps marginals, breaks shared dynamics)."""
    n = len(X)
    shifts = rng.integers(n // 10, n - n // 10, size=X.shape[1])
    return np.stack([np.roll(X[:, j], shifts[j]) for j in range(X.shape[1])], axis=1)


def relative_change(model, init_state):
    W = model.encoder._mean.weight.detach()
    W0 = init_state["_mean.weight"].to(W.device)
    return float(torch.linalg.norm(W - W0) / torch.linalg.norm(W0))


def run_pair(session_a, session_b, device, seed, cfg=BENCHMARK, use_dca=True, source_dropout=0.0):
    """
    All Gate 3 arms for one (source, target) electrode-space session pair and seed.

    ``source_dropout`` > 0 trains the SOURCE model with zero-filled whole-electrode
    dropout, to ask whether masking-style training helps with real session change.
    """
    names_a = [session_a["chan_names"][c] for c in session_a["channel"]]
    names_b = [session_b["chan_names"][c] for c in session_b["channel"]]
    if names_a != names_b:
        raise ValueError("Electrode columns of the two sessions do not refer to the same channels")
    splits_a = chronological_splits(len(session_a["counts"]), **SPLIT)
    splits_b = chronological_splits(len(session_b["counts"]), **SPLIT)
    counts_a, counts_b = session_a["counts"], session_b["counts"]

    pre_a = SessionPreprocessor(bin_width_s=BIN_WIDTH_S, min_rate_hz=MIN_RATE_HZ)
    pre_a.fit(counts_a[splits_a.segment("train")])
    keep = pre_a.keep  # A defines the coordinate system; electrodes silent in B are zero-filled
    pre_b = SessionPreprocessor(bin_width_s=BIN_WIDTH_S, min_rate_hz=None)
    pre_b.fit(counts_b[splits_b.segment("train")], keep=keep)
    pre_b_own = SessionPreprocessor(bin_width_s=BIN_WIDTH_S, min_rate_hz=MIN_RATE_HZ)
    pre_b_own.fit(counts_b[splits_b.segment("train")])

    X_a = segments(counts_a, splits_a, pre_a)
    X_b = segments(counts_b, splits_b, pre_b)             # B's own centering, A's columns
    X_b_strict = segments(counts_b, splits_b, pre_a)      # A's centering applied to B
    X_b_own = segments(counts_b, splits_b, pre_b_own)     # B-only coordinate system
    targets_a, targets_b = targets_for(session_a, splits_a), targets_for(session_b, splits_b)

    rng = np.random.default_rng(seed)
    permutation = rng.permutation(int(keep.sum()))
    X_b_permuted = {s: x[:, permutation] for s, x in X_b.items()}
    X_b_shuffled_train = circular_shuffle(X_b["train"], rng)

    t0 = time.perf_counter()
    source = fit_standard(X_a["train"], device, seed, cfg, use_dca=use_dca,
                          neuron_dropout_p=source_dropout)
    source_fit_s = time.perf_counter() - t0
    Z_a = {s: encode(source, X_a[s]) for s in SEGMENTS}
    decoders = fit_source_decoders(Z_a["train"], {k: v["train"] for k, v in targets_a.items()})
    test_targets_b = {k: v["test"] for k, v in targets_b.items()}

    def readout(model, X, init_state=None):
        Z = {s: encode(model, X[s]) for s in SEGMENTS}
        out = {"frozen_source_decoder": score_decoders(decoders, Z["test"], test_targets_b),
               "refit_probe": refit_probes(Z, targets_b),
               "bound_source_critic_test_mean_latent": critic_bound_under_transform(
                   model, X["test"], critic=source.critic, batch_size=cfg["batch_size"], seed=seed),
               "bound_own_critic_test": heldout_bounds(model, X["test"], cfg["batch_size"],
                                                       seed=seed)["I_predictive_mean"]}
        if init_state is not None:
            out["relative_change_from_init"] = relative_change(model, init_state)
        return out

    init_identity = transformed_encoder_state(source.encoder)
    init_permuted = transformed_encoder_state(source.encoder, permutation=permutation)
    arms = {"zero_shot_source_norm": readout(source, X_b_strict),
            "zero_shot_target_norm": readout(source, X_b),
            "zero_shot_permuted": readout(source, X_b_permuted)}
    adapters = {
        "refine_pretrained": (X_b["train"], "pretrained", init_identity),
        "refine_random": (X_b["train"], "random", init_identity),
        "refine_fresh": (X_b["train"], "fresh", init_identity),
        "refine_pretrained_shuffled": (X_b_shuffled_train, "pretrained", init_identity),
        "permuted_init_pretrained": (X_b["train"], "pretrained", init_permuted),
        "random_init_pretrained": (X_b["train"], "pretrained", None),
    }
    timings = {"source_fit_s": source_fit_s}
    for name, (train, critic_arm, init_state) in adapters.items():
        t0 = time.perf_counter()
        model, history = fit_readin(train, device, seed, critic_arm, source_critic=source.critic,
                                    cfg=cfg, init_encoder_state=init_state)
        arms[name] = {"history": history, **readout(model, X_b, init_state)}
        timings[f"{name}_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    reference = fit_standard(X_b_own["train"], device, seed, cfg, use_dca=use_dca)
    arms["target_specific_cpic"] = readout(reference, X_b_own)
    timings["target_specific_cpic_s"] = time.perf_counter() - t0

    return {
        "seed": int(seed), "device": str(device), "cfg": cfg, "arms": arms, "timings": timings,
        "source_dropout": float(source_dropout),
        "source": {"refit_probe": refit_probes(Z_a, targets_a),
                   "bound_test": heldout_bounds(source, X_a["test"], cfg["batch_size"],
                                                seed=seed)["I_predictive_mean"]},
        "n_electrodes": {"source_kept": int(keep.sum()),
                         "silent_in_target_train": int(
                             (counts_b[splits_b.segment("train")][:, keep].sum(axis=0) == 0).sum()),
                         "target_own_kept": int(pre_b_own.keep.sum())},
        "splits": {"source": splits_a.as_dict(), "target": splits_b.as_dict()},
        "behavior_conventions": {"source": behavior_conventions(session_a, splits_a),
                                 "target": behavior_conventions(session_b, splits_b)},
    }


def run(args):
    device = resolve_device(args.device)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    target_name = Path(args.target).stem
    dropout_tag = f"_srcdrop{int(round(args.source_dropout * 100)):03d}" if args.source_dropout else ""
    path = out_dir / f"transfer_{Path(args.source).stem}_to_{target_name}{dropout_tag}_seed{args.seed}.json"
    if path.exists():
        raise FileExistsError(f"Result already exists: {path}")
    t0 = time.perf_counter()
    session_a = load_sabes_session(args.source, bin_width_s=BIN_WIDTH_S, representation="electrode")
    session_b = load_sabes_session(args.target, bin_width_s=BIN_WIDTH_S, representation="electrode")
    load_s = time.perf_counter() - t0
    result = run_pair(session_a, session_b, device, args.seed, use_dca=not args.no_dca,
                      source_dropout=args.source_dropout)
    result.update({"source_session": Path(args.source).stem, "target_session": target_name})
    result["timings"]["load_s"] = load_s
    with open(path, "x") as f:
        json.dump(result, f, indent=1)
    arms = result["arms"]
    print(f"{target_name} seed {args.seed}: frozen source decoder R² ({PRIMARY}) " + ", ".join(
        f"{a}={r['frozen_source_decoder'][PRIMARY]['r2_mean']:+.3f}" for a, r in arms.items())
        + f" -> {path}", flush=True)


def session_date(name):
    return datetime.strptime(name.split("_")[1], "%Y%m%d").date()


def rate_correlation(rates_dir, source, target):
    """Pearson correlation of training-split electrode rates on the source's kept electrodes."""
    paths = [Path(rates_dir) / f"electrode_rates_{s}.json" for s in (source, target)]
    if rates_dir is None or not all(p.exists() for p in paths):
        return None
    a, b = (json.loads(p.read_text()) for p in paths)
    if a["chan_names"] != b["chan_names"]:
        raise ValueError(f"Electrode order differs between {source} and {target}")
    ra, rb = np.asarray(a["train_rate_hz"]), np.asarray(b["train_rate_hz"])
    keep = ra >= MIN_RATE_HZ
    return float(np.corrcoef(ra[keep], rb[keep])[0, 1])


def cross_session_table(runs, rates_dir=None):
    """One row per target session (plain source only), seed means of every quantity."""
    plain = [r for r in runs if r.get("source_dropout", 0.0) == 0.0]
    rows = []
    for target in sorted({r["target_session"] for r in plain}):
        group = [r for r in plain if r["target_session"] == target]
        source = group[0]["source_session"]

        def mean(arm, fn):
            return float(np.mean([fn(r["arms"][arm]) for r in group]))
        frozen = lambda a: a["frozen_source_decoder"][PRIMARY]["r2_mean"]
        rows.append({
            "target": target, "n_seeds": len(group),
            "days_apart": (session_date(target) - session_date(source)).days,
            "target_duration_min": group[0]["behavior_conventions"]["target"]["duration_s"] / 60,
            "silent_electrodes": group[0]["n_electrodes"]["silent_in_target_train"],
            "critic_bound_zero_shot": mean("zero_shot_target_norm",
                                           lambda a: a["bound_source_critic_test_mean_latent"]),
            "rate_correlation": rate_correlation(rates_dir, source, target),
            "zero_shot_r2": mean("zero_shot_target_norm", frozen),
            "refined_r2": mean("refine_pretrained", frozen),
            "fresh_r2": mean("refine_fresh", frozen),
            "target_only_refit_r2": mean("target_specific_cpic",
                                         lambda a: a["refit_probe"][PRIMARY]["r2_mean"]),
        })
    return rows


def permutation_spearman(x, y, n_perm=20000, seed=0):
    """
    Spearman rho with a two-sided permutation p-value (exact enough for ~8 sessions).
    Returns (nan, nan) when either input is constant: rho is undefined there, and a NaN rho
    would otherwise yield a spuriously tiny p.
    """
    rx, ry = rankdata(x), rankdata(y)
    if np.ptp(rx) == 0 or np.ptp(ry) == 0:
        return float("nan"), float("nan")
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    perms = np.argsort(rng.random((n_perm, len(ry))), axis=1)
    shuffled = ry[perms]
    rxc = rx - rx.mean()
    null = (shuffled - shuffled.mean(axis=1, keepdims=True)) @ rxc
    null /= np.linalg.norm(rxc) * np.linalg.norm(shuffled - shuffled.mean(axis=1, keepdims=True), axis=1)
    p = (1 + np.sum(np.abs(null) >= abs(rho) - 1e-12)) / (n_perm + 1)
    return rho, float(p)


def predictor_tests(rows):
    """
    Pre-registered (Nov 4): does the zero-shot bound under A's frozen critic predict zero-shot
    frozen-decoder R² across sessions better than neural-only controls (electrode-rate
    correlation, -|days apart|)? Supported if rho > 0, p < 0.05, and |rho| beats both controls.
    """
    predictors = {"critic_bound_zero_shot": lambda r: r["critic_bound_zero_shot"],
                  "rate_correlation": lambda r: r["rate_correlation"],
                  "neg_abs_days": lambda r: -abs(r["days_apart"])}
    tests = {}
    for outcome in ("zero_shot_r2", "refined_r2"):
        for name, fn in predictors.items():
            usable = [r for r in rows if fn(r) is not None]
            if len(usable) < 4:
                continue
            rho, p = permutation_spearman([fn(r) for r in usable], [r[outcome] for r in usable])
            tests[f"{outcome} ~ {name}"] = {"rho": rho, "p": p, "n_sessions": len(usable)}
    primary = tests.get("zero_shot_r2 ~ critic_bound_zero_shot")
    controls = [tests.get(f"zero_shot_r2 ~ {c}") for c in ("rate_correlation", "neg_abs_days")]
    verdict = None
    if primary and all(controls) and all(np.isfinite(t["rho"]) for t in [primary] + controls):
        verdict = bool(primary["rho"] > 0 and primary["p"] < 0.05
                       and all(abs(primary["rho"]) > abs(c["rho"]) for c in controls))
    return tests, verdict


def summarize_sessions(results_dir, runs, rates_dir):
    rows = cross_session_table(runs, rates_dir)
    if not rows:
        return
    print(f"\n## Across target sessions (plain source; seed means; {PRIMARY} for R²)")
    print("| target | days | min | silent el. | critic bound (zero-shot) | rate corr | zero-shot R² "
          "| refined R² | fresh R² | B-only refit R² |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda r: r["days_apart"]):
        rc = "n/a" if r["rate_correlation"] is None else f"{r['rate_correlation']:+.3f}"
        print(f"| {r['target']} | {r['days_apart']:+d} | {r['target_duration_min']:.1f} | "
              f"{r['silent_electrodes']} | {r['critic_bound_zero_shot']:+.3f} | {rc} | "
              f"{r['zero_shot_r2']:+.3f} | {r['refined_r2']:+.3f} | {r['fresh_r2']:+.3f} | "
              f"{r['target_only_refit_r2']:+.3f} |")
    tests, verdict = predictor_tests(rows)
    for name, t in tests.items():
        print(f"Spearman {name}: rho {t['rho']:+.3f}, permutation p {t['p']:.4f} (n={t['n_sessions']} sessions)")
    print("Pre-registered predictor test: " + ("not computable yet (need >= 4 sessions with every predictor)"
                                              if verdict is None else ("SUPPORTED" if verdict else "NOT SUPPORTED")))
    (Path(results_dir) / "distance_summary.json").write_text(
        json.dumps({"rows": rows, "tests": tests, "verdict": verdict}, indent=1))


def summarize(results_dir, rates_dir=None):
    runs = [json.loads(p.read_text()) for p in sorted(Path(results_dir).glob("transfer_*.json"))]
    if not runs:
        print(f"No transfer results in {results_dir}")
        return
    keys = sorted({(r["target_session"], r.get("source_dropout", 0.0)) for r in runs})
    sessions = sorted({k[0] for k in keys})
    gate_signs = {}
    for target, source_dropout in keys:
        group = [r for r in runs if (r["target_session"], r.get("source_dropout", 0.0)) == (target, source_dropout)]
        conv = group[0]["behavior_conventions"]
        print(f"\n## {group[0]['source_session']} -> {target} | source electrode dropout "
              f"{source_dropout:g}  ({len(group)} seeds)")
        print(f"electrodes: {group[0]['n_electrodes']} | cursor train mean source "
              f"{np.round(conv['source']['cursor_train_mean'], 1).tolist()} vs target "
              f"{np.round(conv['target']['cursor_train_mean'], 1).tolist()}; std source "
              f"{np.round(conv['source']['cursor_train_std'], 1).tolist()} vs target "
              f"{np.round(conv['target']['cursor_train_std'], 1).tolist()}")
        print(f"source refit R² (restd position) {np.mean([r['source']['refit_probe'][PRIMARY]['r2_mean'] for r in group]):.4f}")
        print("| arm | frozen R² restd pos | frozen R² vel | frozen R² raw pos | refit R² restd pos "
              "| bound under A critic | change from init |\n|---|---|---|---|---|---|---|")
        for arm in group[0]["arms"]:
            def ms(fn):
                v = np.array([fn(r["arms"][arm]) for r in group], dtype=float)
                return f"{v.mean():+.4f} ± {v.std(ddof=1):.4f}" if len(v) > 1 else f"{v.mean():+.4f}"
            change = (ms(lambda a: a["relative_change_from_init"])
                      if "relative_change_from_init" in group[0]["arms"][arm] else "—")
            print(f"| {arm} | {ms(lambda a: a['frozen_source_decoder'][PRIMARY]['r2_mean'])} | "
                  f"{ms(lambda a: a['frozen_source_decoder']['velocity']['r2_mean'])} | "
                  f"{ms(lambda a: a['frozen_source_decoder']['position_raw']['r2_mean'])} | "
                  f"{ms(lambda a: a['refit_probe'][PRIMARY]['r2_mean'])} | "
                  f"{ms(lambda a: a['bound_source_critic_test_mean_latent'])} | {change} |")
        for target_metric in (PRIMARY, "velocity"):
            metric = lambda arm, m=target_metric: arm["frozen_source_decoder"][m]["r2_mean"]
            for control in ("refine_random", "refine_fresh", "zero_shot_target_norm",
                            "refine_pretrained_shuffled", "permuted_init_pretrained"):
                g = paired_gaps(group, metric, "refine_pretrained", control, lambda r: r["seed"])
                ci = f"95% t-CI {g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}" if g["ci95"] else "n=1"
                print(f"Δ frozen R² {target_metric}: refine_pretrained − {control}: {g['mean']:+.4f} ({ci}; n={g['n']})")
                if (target_metric == PRIMARY and control in ("refine_random", "refine_fresh")
                        and source_dropout == 0.0):  # the pre-registered gate uses the plain source
                    gate_signs.setdefault(control, []).append(g["mean"] > 0)
    passed = all(all(signs) for signs in gate_signs.values()) and len(gate_signs) == 2
    print(f"\nPre-registered Gate 3 reading ({len(sessions)} target sessions): "
          f"{'PASS' if passed else 'FAIL'} {json.dumps({k: v for k, v in gate_signs.items()})}")
    summarize_sessions(results_dir, runs, rates_dir)


def main():
    parser = argparse.ArgumentParser(description="Gate 3 electrode-space cross-session transfer")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE))
    parser.add_argument("--target", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-dca", action="store_true", help="Debug only: skip DCA initialization")
    parser.add_argument("--source-dropout", type=float, default=0.0,
                        help="Train the source with this zero-filled whole-electrode dropout rate")
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "transfer_sessions"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--summarize", metavar="RESULTS_DIR", default=None)
    parser.add_argument("--rates-dir", default=str(REPO_ROOT / "inventory"),
                        help="electrode_rates_*.json for the firing-rate-similarity control predictor")
    args = parser.parse_args()
    if args.summarize:
        summarize(args.summarize, args.rates_dir)
    elif args.target is None:
        parser.error("--target is required unless --summarize is given")
    else:
        run(args)


if __name__ == "__main__":
    main()
