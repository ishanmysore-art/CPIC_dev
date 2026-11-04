"""
Shared readouts for transferring a source CPIC model to a new neural population.

Used by the pseudo-population experiment (simultaneously recorded unit subsets, where
the correct latent alignment is known) and the synthetic two-population sanity check,
and later by genuine cross-session transfer. Representation training is neural-only;
behavior enters only through linear probes fit after every model is frozen.

Readouts, never to be conflated:
    frozen_source_decoder   decoders fit on SOURCE latents, applied to target latents
                            (the only coordinate-transfer claim)
    refit_probe             decoders refit on the target's own latents
                            (representation content only)
    agreement               target vs source latents at the same time points
                            (simultaneous recordings only; see latent_agreement)
    oracle_*                alignment fit using simultaneous source latents; a ceiling,
                            not a method
"""

import numpy as np

from cpic.evaluation import (
    apply_procrustes, cursor_velocity, encode, fit_linear_decoder, fit_procrustes,
    heldout_bounds, latent_agreement, probe_suite, score_linear_decoder,
)
from cpic.identifiability import critic_bound_under_transform
from cpic.multisession import CRITIC_ARMS, fit_readin, fit_standard

OFFSETS = (5, 10, 15)
DECODING_WINDOW = 3


def behavior_targets(Y, bin_width_s):
    """Position and finite-difference velocity, per split."""
    return {"position": Y, "velocity": {s: cursor_velocity(y, bin_width_s) for s, y in Y.items()}}


def fit_source_decoders(Z_src_train, targets_src_train):
    return {name: {o: fit_linear_decoder(Z_src_train, y, DECODING_WINDOW, o) for o in OFFSETS}
            for name, y in targets_src_train.items()}


def score_decoders(decoders, Z_test, targets_test):
    out = {}
    for name, by_offset in decoders.items():
        scores = {str(o): score_linear_decoder(d, Z_test, targets_test[name], DECODING_WINDOW, o)
                  for o, d in by_offset.items()}
        out[name] = {"r2_by_offset": scores, "r2_mean": float(np.mean(list(scores.values())))}
    return out


def refit_probes(Z, targets):
    out = {}
    for name in targets:
        suite = probe_suite(Z["train"], targets[name]["train"], Z["test"], targets[name]["test"],
                            OFFSETS, DECODING_WINDOW)
        out[name] = {"r2_by_offset": {str(k): v for k, v in suite["r2_by_offset"].items()},
                     "r2_mean": suite["r2_mean"]}
    return out


def readouts(model, X_tgt, Z_src, decoders, targets_tgt, source_critic, cfg, seed, simultaneous):
    Z = {split: encode(model, X_tgt[split]) for split in ("train", "test")}
    test_targets = {name: t["test"] for name, t in targets_tgt.items()}
    out = {
        "bound_own_critic_test": heldout_bounds(model, X_tgt["test"], cfg["batch_size"],
                                                seed=seed)["I_predictive_mean"],
        "bound_source_critic_test_mean_latent": critic_bound_under_transform(
            model, X_tgt["test"], critic=source_critic, batch_size=cfg["batch_size"], seed=seed),
        "frozen_source_decoder": score_decoders(decoders, Z["test"], test_targets),
        "refit_probe": refit_probes(Z, targets_tgt),
    }
    if simultaneous:
        out["agreement"] = latent_agreement(Z["train"], Z_src["train"], Z["test"], Z_src["test"])
        procrustes = fit_procrustes(Z["train"], Z_src["train"])
        out["oracle_procrustes_frozen_source_decoder"] = score_decoders(
            decoders, apply_procrustes(Z["test"], procrustes), test_targets)
    return out


def run_transfer(source, X_src, X_tgt, Y_src, Y_tgt, device, seed, cfg, bin_width_s,
                 simultaneous, use_dca=True, arms=CRITIC_ARMS, upper_reference=True,
                 init_encoder_state=None):
    """
    Train a target read-in in each critic arm, plus (optionally) a target-specific CPIC with
    the standard recipe, and compute every readout against the frozen source model.
    ``init_encoder_state`` (shared by all arms) replaces the random read-in initialization.
    """
    Z_src = {split: encode(source, X_src[split]) for split in ("train", "test")}
    targets_src = behavior_targets(Y_src, bin_width_s)
    targets_tgt = behavior_targets(Y_tgt, bin_width_s)
    decoders = fit_source_decoders(Z_src["train"], {k: v["train"] for k, v in targets_src.items()})
    result = {
        "source": {
            "bound_test": heldout_bounds(source, X_src["test"], cfg["batch_size"],
                                         seed=seed)["I_predictive_mean"],
            "refit_probe": refit_probes(Z_src, targets_src),
            "n_units": int(X_src["train"].shape[1]),
        },
        "target_n_units": int(X_tgt["train"].shape[1]),
        "arms": {},
    }
    for arm in arms:
        model, history = fit_readin(X_tgt["train"], device, seed, arm,
                                    source_critic=source.critic, cfg=cfg,
                                    init_encoder_state=init_encoder_state)
        result["arms"][arm] = {"history": history, **readouts(
            model, X_tgt, Z_src, decoders, targets_tgt, source.critic, cfg, seed, simultaneous)}
    if upper_reference:
        reference = fit_standard(X_tgt["train"], device, seed, cfg, use_dca=use_dca)
        result["target_specific_cpic"] = readouts(
            reference, X_tgt, Z_src, decoders, targets_tgt, source.critic, cfg, seed, simultaneous)
    return result


def paired_gaps(runs, metric, arm_a, arm_b, group_key):
    """
    arm_a - arm_b for each run (paired: same partition/session, seed and data), with a
    t-interval over runs and the per-group sign of the mean gap.
    """
    from scipy import stats
    gaps = np.array([metric(r["arms"][arm_a]) - metric(r["arms"][arm_b]) for r in runs])
    groups = {}
    for r, g in zip(runs, gaps):
        groups.setdefault(group_key(r), []).append(g)
    n = len(gaps)
    half = (stats.t.ppf(0.975, n - 1) * gaps.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    return {"n": n, "mean": float(gaps.mean()) if n else np.nan,
            "ci95": [float(gaps.mean() - half), float(gaps.mean() + half)] if n > 1 else None,
            "group_means": {str(k): float(np.mean(v)) for k, v in groups.items()},
            "n_groups_positive": int(sum(np.mean(v) > 0 for v in groups.values())),
            "n_groups": len(groups)}
