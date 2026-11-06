"""
Figures for the Nov 9 CPIC meeting, built only from saved result files.

Each figure is written as PNG and PDF to --out, and its caption (dataset, training
setup, evaluation split, metric definition, uncertainty, n) is appended to captions.md.
Figures whose results do not exist yet are skipped with a message.

    python experiments/cross_session/figures.py --results results --inventory inventory --out results/figures
"""

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# Validated reference palette (first four categorical slots, light surface), fixed order.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
NEUTRAL = "#a3a29c"
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
ARM_ORDER = ["pretrained", "random", "fresh", "target_specific_cpic"]
ARM_LABEL = {"pretrained": "pretrained critic (frozen)", "random": "random critic (frozen)",
             "fresh": "fresh critic (trained)", "target_specific_cpic": "target-only CPIC"}
RUNG_LABEL = {"L0_legacy": "L0 legacy", "L3_split": "+ split/gaps", "L4_train_only_prep": "+ train-only prep",
              "L5_no_wraparound": "+ no wraparound", "L6_train_rate_filter": "+ ≥0.5 Hz on train",
              "L7_all_sorted_rows": "+ all sorted rows"}
SEED_NOTE = "error bars: ±1 SD across seeds"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "text.color": INK, "xtick.color": INK_2,
    "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2,
})


def load(pattern_dir, pattern, require="seed"):
    """Result files only: summary files (e.g. pseudo_pop_summary.json) lack ``require``."""
    runs = [json.loads(p.read_text()) for p in sorted(Path(pattern_dir).glob(pattern))]
    return [r for r in runs if isinstance(r, dict) and require in r]


def mean_sd(values):
    v = np.asarray(values, dtype=float)
    return v.mean(), (v.std(ddof=1) if len(v) > 1 else 0.0)


def bar_labels(ax, bars, values, fmt="{:+.2f}", clip=None, errors=None):
    """
    Value labels beyond the end of each bar's error bar, so they never sit on the whisker
    (two series colors are below 3:1 contrast on the surface, so values must be printed).
    """
    errors = [0.0] * len(values) if errors is None else errors
    for bar, value, err in zip(bars, values, errors):
        end = value + err if value >= 0 else value - err
        y = end if clip is None else max(end, clip)
        clipped = clip is not None and value < clip
        ax.annotate(fmt.format(value) + (" ▼" if clipped else ""), (bar.get_x() + bar.get_width() / 2, y),
                    xytext=(0, 3 if value >= 0 else -11), textcoords="offset points",
                    ha="center", fontsize=8, color=INK_2)


def wrap(label, width=12):
    return "\n".join(textwrap.wrap(label, width)) or label


def legend_in_headroom(ax, ncol, top):
    """Raise the y-limit to ``top`` and put a one-row legend in the empty band above the data."""
    ax.set_ylim(top=top)
    ax.legend(loc="upper center", ncol=ncol, fontsize=9)


class Figures:
    def __init__(self, results, inventory, out):
        self.results, self.inventory, self.out = Path(results), Path(inventory), Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.captions = []

    def save(self, fig, name, title, caption):
        for ext in ("png", "pdf"):
            fig.savefig(self.out / f"{name}.{ext}", dpi=200, bbox_inches="tight")
        plt.close(fig)
        self.captions.append(f"## {name}: {title}\n\n{caption}\n")
        print(f"wrote {self.out / name}.png")

    def skip(self, name, why):
        print(f"skip {name}: {why}")

    # -- 1. inventory ------------------------------------------------------------------
    def inventory_figure(self):
        sessions = []
        for path in sorted(self.inventory.glob("recount_*.json")):
            data = json.loads(path.read_text())
            sessions += data if isinstance(data, list) else [data]
        if not sessions:
            return self.skip("fig1_inventory", f"no recount_*.json in {self.inventory}")
        sessions = sorted({Path(s["file"]).stem: s for s in sessions}.values(), key=lambda s: Path(s["file"]).stem)
        names, series = [], {"sorted units with spikes": [], "≥ 0.5 Hz (rate cut, used now)": [],
                             "> 5000 spikes (legacy count cut)": []}
        for s in sessions:
            sorted_rows = [r for r in s["rows"] if r["role"] == "sorted"]
            names.append(f"{Path(s['file']).stem}\n{s['duration_s'] / 60:.1f} min")
            series["sorted units with spikes"].append(sum(r["n_with_spikes"] for r in sorted_rows))
            series["≥ 0.5 Hz (rate cut, used now)"].append(sum(r["n_at_least_0.5_hz"] for r in sorted_rows))
            series["> 5000 spikes (legacy count cut)"].append(sum(r["n_above_5000_spikes"] for r in sorted_rows))
        fig, ax = plt.subplots(figsize=(7.5, 4))
        x, w = np.arange(len(names)), 0.26
        for i, (label, values) in enumerate(series.items()):
            bars = ax.bar(x + (i - 1) * (w + 0.01), values, w, color=SERIES[i], label=label)
            bar_labels(ax, bars, values, fmt="{:d}")
        ax.set_xticks(x, names)
        ax.set_ylabel("sorted units")
        ax.set_title("Sorted units per Indy session: the legacy count cut depends on session length")
        ax.legend(loc="upper right")
        self.save(fig, "fig1_inventory", "neural population inventory",
                  "Sabes/O'Doherty Indy M1 sessions (Zenodo 3854034), 96-channel Utah array, sorted units = "
                  "rows 1..u-1 of the spikes cell (hash excluded), counted directly from the HDF5 files by "
                  "recount_units.py. Rate cut: >= 0.5 Hz over the whole file (the pipeline applies it to the "
                  "training split). Legacy cut: > 5000 total spikes, which equals ~1.5 Hz in the 56-min "
                  "benchmark session but ~10 Hz in the 8.3-min session. Counts, no uncertainty.")

    # -- 2. ladder ---------------------------------------------------------------------
    def ladder_figure(self):
        runs = load(self.results / "ladder", "L*_seed*.json")
        if not runs:
            return self.skip("fig2_ladder", "no ladder results")
        rungs = [r for r in RUNG_LABEL if any(x["rung"] == r for x in runs)]
        get = lambda rung, fn: mean_sd([fn(x) for x in runs if x["rung"] == rung])
        test = [get(r, lambda x: x["bounds"]["test_sampled"]["I_predictive_mean"]) for r in rungs]
        reported = [get(r, lambda x: x["reported_train_bound"]) for r in rungs]
        r2 = [get(r, lambda x: x["probes"]["position"]["r2_mean"]) for r in rungs]
        units = [int(get(r, lambda x: x["n_units"])[0]) for r in rungs]
        n_seeds = len({x["seed"] for x in runs})
        labels = [f"{wrap(RUNG_LABEL[r], 11)}\n{u} units" for r, u in zip(rungs, units)]
        x = np.arange(len(rungs))
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
        bars = ax1.bar(x, [m for m, _ in test], 0.6, yerr=[s for _, s in test], color=SERIES[0],
                       capsize=3, label="held-out test bound (frozen model)")
        bar_labels(ax1, bars, [m for m, _ in test], fmt="{:.2f}", errors=[s for _, s in test])
        ax1.plot(x, [m for m, _ in reported], "o", markersize=8, markerfacecolor=SURFACE,
                 markeredgecolor=SERIES[1], markeredgewidth=2, linestyle="none",
                 label="legacy-reported training bound")
        ax1.set_xticks(x, labels, fontsize=8)
        ax1.set_ylabel("InfoNCE predictive bound (nats)")
        ax1.set_title("Predictive information: reported vs held-out")
        ax1.set_ylim(top=max(m for m, _ in reported) * 1.4)
        ax1.legend(loc="upper left", fontsize=9)
        # Points, not bars: the differences are small, and bars would need a truncated axis.
        ax2.errorbar(x, [m for m, _ in r2], yerr=[s for _, s in r2], color=SERIES[0], marker="o",
                     markersize=8, capsize=3, linestyle="none")
        for xi, (m, _) in zip(x, r2):
            ax2.annotate(f"{m:.3f}", (xi, m), xytext=(8, -3), textcoords="offset points", fontsize=8, color=INK_2)
        ax2.set_xticks(x, labels, fontsize=8)
        ax2.set_ylabel("test R² (cursor position, mean of 3 offsets)")
        ax2.set_title("Behavioral decoding (refit linear probe)")
        self.save(fig, "fig2_ladder", "attribution ladder: one correction per rung",
                  f"indy_20160627_01 (56 min). Linear DCA-initialized stochastic CPIC, ydim 32, T 4, beta 1e-3, "
                  f"batch 512, 10 epochs, identical across rungs; each rung adds one correction to the one "
                  f"before. Splits: L0 = legacy 80/20; later rungs = chronological 70/10/20 with 30-bin gaps. "
                  f"Bound = InfoNCE lower bound on sampled latents, frozen model, seeded full batches of 512 "
                  f"(ceiling log 512 = 6.24). R² = linear decoder, window 3, offsets 5/10/15 bins, fit on "
                  f"train latents and scored on test. n = {n_seeds} seeds; {SEED_NOTE}. The R² panel is a "
                  f"dot plot on a zoomed axis (differences are ~0.01-0.02).")

    # -- 3. pseudo-populations ---------------------------------------------------------
    def pseudo_population_figure(self):
        runs = load(self.results / "pseudo_pop", "pseudo_*.json")
        if not runs:
            return self.skip("fig3_pseudo_population", "no pseudo-population results")
        overlaps = sorted({r["overlap"] for r in runs})
        metrics = [("frozen source decoder R²", lambda a: a["frozen_source_decoder"]["position"]["r2_mean"]),
                   ("direct latent R² (no alignment)", lambda a: a["agreement"]["direct_r2"]),
                   ("Procrustes latent R² (rotation fit)", lambda a: a["agreement"]["procrustes_r2"])]
        fig, axes = plt.subplots(len(overlaps), len(metrics), figsize=(13, 3.3 * len(overlaps)),
                                 squeeze=False, sharey=True)
        floor = -1.5
        for i, overlap in enumerate(overlaps):
            group = [r for r in runs if r["overlap"] == overlap]
            for j, (title, fn) in enumerate(metrics):
                ax = axes[i, j]
                stats = []
                for arm in ARM_ORDER:
                    arm_runs = [r["target_specific_cpic"] if arm == "target_specific_cpic" else r["arms"][arm]
                                for r in group]
                    stats.append(mean_sd([fn(a) for a in arm_runs]))
                means = [m for m, _ in stats]
                errors = [s if m > floor else 0 for m, s in stats]
                bars = ax.bar(np.arange(4), [max(m, floor) for m in means], 0.7,
                              yerr=errors, capsize=3, color=SERIES)
                bar_labels(ax, bars, means, clip=floor, errors=errors)
                ax.axhline(0, color=INK_2, linewidth=0.8)
                ax.set_ylim(floor - 0.4, 1.15)  # room below the clip line for the value label
                ax.set_xticks([])
                if i == 0:
                    ax.set_title(title)
                if j == 0:
                    ax.set_ylabel(f"overlap {overlap:g}\nR²")
        axes[0, -1].legend(handles=[Patch(color=c, label=ARM_LABEL[a]) for c, a in zip(SERIES, ARM_ORDER)],
                           loc="upper left", bbox_to_anchor=(1.02, 1.0))
        n = len(runs)
        self.save(fig, "fig3_pseudo_population", "frozen-critic read-ins on simultaneously recorded subsets",
                  f"indy_20160627_01, corrected pipeline (157 units). Units split into P1/P2 (overlap = shared "
                  f"fraction; 1 = identical units). CPIC trained on P1; a new P2 read-in trained from random "
                  f"init against P1's frozen critic, a matched frozen random critic, or a fresh jointly trained "
                  f"critic (same init, data, steps; neural-only); target-only CPIC = standard recipe on P2. "
                  f"Frozen source decoder = linear decoder fit on P1 latents/behavior, applied unchanged to P2 "
                  f"test latents (the only coordinate-transfer readout). Latent R² compares P2 and P1 latents at "
                  f"the same test time points, variance-weighted. Bars below {floor} are clipped (▼); the printed "
                  f"value is the true mean. At overlap 1 the target-only CPIC is the source model itself (same "
                  f"units, data, seed and initialization), so its 1.00 is a pipeline check, not a result. {n} runs (3 partitions x 3 seeds for overlaps 0/0.5; 3 seeds for 1); "
                  f"{SEED_NOTE} and partitions.")

    # -- 4. identifiability ------------------------------------------------------------
    def identifiability_figure(self):
        runs = load(self.results / "pseudo_pop", "identifiability_*.json", require="profile")
        if not runs:
            return self.skip("fig4_identifiability", "no identifiability results")
        rows = [(critic, row) for r in runs for critic, rs in r["profile"].items() for row in rs]

        def delta(critic, family, param=None):
            return mean_sd([row["delta"] for c, row in rows if c == critic and row["family"] == family
                            and (param is None or row["param"] == param)])
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        angles = sorted({row["param"] for _, row in rows if row["family"] == "rotation_global"})
        lines = [("pretrained", "rotation_global", "pretrained critic: all planes"),
                 ("pretrained", "rotation_dynamics_top1", "pretrained critic: top dynamics plane"),
                 ("pretrained", "rotation_dynamics_top3", "pretrained critic: top 3 dynamics planes"),
                 ("random", "rotation_global", "random critic: all planes")]
        for k, (critic, family, label) in enumerate(lines):
            stats = [delta(critic, family, a) for a in angles]
            ax1.errorbar(angles, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o",
                         markersize=8, color=SERIES[k], label=label, capsize=3)
        ax1.set_xlabel("rotation angle applied to held-out latents (degrees)")
        ax1.set_ylabel("ΔI = bound(identity) − bound(rotated)  (nats)")
        ax1.set_title("How strongly the frozen critic penalizes rotation")
        ax1.legend(loc="upper left")
        families = [("sign_flip", None, "sign flip"), ("permutation", None, "permutation"),
                    ("orthogonal_haar", None, "random rotation"), ("random_linear", None, "random linear"),
                    ("scale_isotropic", 2.0, "scale x2"), ("scale_isotropic", 1.25, "scale x1.25"),
                    ("translation", 1.0, "shift 1 SD"), ("translation", 2.0, "shift 2 SD")]
        x, w = np.arange(len(families)), 0.38
        for k, critic in enumerate(("pretrained", "random")):
            stats = [delta(critic, f, p) for f, p, _ in families]
            bars = ax2.bar(x + (k - 0.5) * (w + 0.01), [m for m, _ in stats], w,
                           yerr=[s for _, s in stats], capsize=3, color=SERIES[k], label=f"{critic} critic")
            if critic == "pretrained":  # random-critic values are all ~0; labeling them only collides
                bar_labels(ax2, bars, [m for m, _ in stats], errors=[s for _, s in stats])
        ax2.set_xticks(x, [label for _, _, label in families], rotation=30, ha="right")
        ax2.set_ylabel("ΔI (nats)")
        ax2.set_title("Other latent maps")
        ax2.legend(loc="upper right")
        self.save(fig, "fig4_identifiability", "what a frozen CPIC critic can distinguish",
                  f"Source CPIC on all 157 units of indy_20160627_01 (corrected pipeline). Each map is applied "
                  f"identically at every time bin to the model's held-out test latents; ΔI is the drop in the "
                  f"frozen critic's InfoNCE bound (mean latents, seeded full batches) relative to the identity. "
                  f"Large ΔI = the critic can tell that map from the correct coordinates. Random critic = same "
                  f"architecture, untrained. n = {len(runs)} source models (seeds); {SEED_NOTE} and random draws.")

    # -- 5. basin radius ---------------------------------------------------------------
    def basin_figure(self):
        runs = load(self.results / "basin_radius", "basin_seed*.json")
        if not runs:
            return self.skip("fig5_basin_radius", "no basin-radius results")
        starts = [s for s in ("0", "5", "10", "20", "30", "45", "60", "90", "haar")
                  if any(x["start"] == s for r in runs for x in r["latent_level"])]
        xpos = [100.0 if s == "haar" else float(s) for s in starts]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.2))
        conditions = [("pretrained", "orthogonal", "pretrained critic, rotation map"),
                      ("pretrained", "linear", "pretrained critic, linear map"),
                      ("random", "orthogonal", "random critic, rotation map")]
        for k, (critic, param, label) in enumerate(conditions):
            med = [np.median([x["direct_r2_vs_identity"] for r in runs for x in r["latent_level"]
                              if x["start"] == s and x["critic"] == critic and x["parametrization"] == param])
                   for s in starts]
            ax1.plot(xpos, med, marker="o", markersize=8, color=SERIES[k], label=label)
        ax1.axhline(0.9, color=INK_2, linestyle="--", linewidth=1)
        ax1.annotate("recovered (pre-registered 0.9)", (2, 0.9), xytext=(0, -12), textcoords="offset points",
                     fontsize=8, color=INK_2)
        ax1.set_xticks(xpos, [("Haar" if s == "haar" else s) for s in starts])
        ax1.set_ylim(max(-1.5, ax1.get_ylim()[0]), 1.05)
        ax1.set_xlabel("start: rotation away from the correct latents (degrees)")
        ax1.set_ylabel("median direct R² after optimization")
        ax1.set_title("Latent-level self-recovery")
        ax1.legend(loc="lower left")
        angles = sorted({x["start_angle_deg"] for r in runs for x in r["input_level"]})
        initial = [np.mean([x["initial_direct_r2"] for r in runs for x in r["input_level"]
                            if x["start_angle_deg"] == a]) for a in angles]
        ax2.plot(angles, initial, color=NEUTRAL, linestyle="--", marker="o", markersize=6,
                 label="at initialization")
        for k, arm in enumerate(("pretrained", "random", "fresh")):
            stats = [mean_sd([x["arms"][arm]["agreement"]["direct_r2"] for r in runs
                              for x in r["input_level"] if x["start_angle_deg"] == a]) for a in angles]
            ax2.errorbar(angles, [m for m, _ in stats], yerr=[s for _, s in stats], marker="o",
                         markersize=8, color=SERIES[k], label=f"after training: {ARM_LABEL[arm]}", capsize=3)
        ax2.set_ylim(top=1.05)  # no lower clip: the random-critic arm must stay visible
        ax2.set_xlabel("start: source encoder rotated by (degrees)")
        ax2.set_ylabel("direct latent R² vs source")
        ax2.set_title("Input-level read-in refinement")
        ax2.legend(loc="best", fontsize=8)
        self.save(fig, "fig5_basin_radius", "is the frozen critic a local refiner?",
                  f"Source CPIC on all units of indy_20160627_01 (corrected pipeline), one per seed. Left: a map "
                  f"on the source's own training latents optimized to maximize the frozen critic's InfoNCE from a "
                  f"start rotated by the given angle in every plane of a random basis (Haar = random rotation); "
                  f"direct R² of mapped vs original test latents, median over starts and seeds. Right: a read-in "
                  f"for the same units initialized as the rotated source encoder and trained 10 epochs against "
                  f"each critic; direct latent R² on test vs the source latents. n = {len(runs)} seeds; right "
                  f"panel {SEED_NOTE}.")

    # -- 6. cross-session transfer -----------------------------------------------------
    def transfer_figure(self):
        """Compact slide figure: the four key arms across every target session (plain source)."""
        from transfer_sessions import PRIMARY, session_date
        runs = [r for r in load(self.results / "transfer_sessions", "transfer_*.json")
                if r.get("source_dropout", 0.0) == 0.0]
        if not runs:
            return self.skip("fig6_transfer", "no transfer results")
        targets = sorted({r["target_session"] for r in runs}, key=session_date)
        source = session_date(runs[0]["source_session"])
        arms = [("zero_shot_target_norm", "zero-shot", "o"),
                ("refine_pretrained", "refined: pretrained critic", "s"),
                ("refine_fresh", "refined: fresh critic", "D"),
                ("refine_pretrained_shuffled", "refined: pretrained critic, time-shuffled B", "^")]
        fig, ax = plt.subplots(figsize=(13, 5.2))
        x = np.arange(len(targets))
        dodge = np.linspace(-0.27, 0.27, len(arms))
        highest = 0.0
        for k, (arm, label, marker) in enumerate(arms):  # marker shape is a second encoding beside color
            stats = [mean_sd([r["arms"][arm]["frozen_source_decoder"][PRIMARY]["r2_mean"]
                              for r in runs if r["target_session"] == t]) for t in targets]
            ax.errorbar(x + dodge[k], [m for m, _ in stats], yerr=[s for _, s in stats], fmt=marker,
                        markersize=8, color=SERIES[k], capsize=2, label=label)
            highest = max(highest, max(m + s for m, s in stats))
        refit = [np.mean([r["arms"]["target_specific_cpic"]["refit_probe"][PRIMARY]["r2_mean"]
                          for r in runs if r["target_session"] == t]) for t in targets]
        ax.plot(x, refit, "_", markersize=28, markeredgewidth=2.5, color=NEUTRAL,
                label="decoder refit on B alone (reference)")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        dates = [session_date(t) for t in targets]
        ax.set_xticks(x, [f"{d:%b} {d.day}, {d.year}\n{(d - source).days:+d} days" for d in dates], fontsize=9)
        ax.set_ylabel("A's frozen decoder R² on B\n(re-standardized cursor position)")
        ax.set_title("Electrode-anchored transfer from indy_20160627_01 to 8 later and earlier sessions")
        legend_in_headroom(ax, ncol=3, top=max(highest, max(refit)) + 0.17)
        self.save(fig, "fig6_transfer", "cross-session transfer in electrode space",
                  f"Source indy_20160627_01 -> {len(targets)} Indy sessions, electrode representation (threshold "
                  f"crossings per channel; physical, not cellular, correspondence), plain (non-dropout) source. "
                  f"Points: A's linear decoder (fit on A latents and A behavior, never refit) applied to each "
                  f"target's held-out test latents. Zero-shot = A's frozen model; refined = read-in initialized "
                  f"from A's encoder and trained on unlabeled B against A's frozen critic (pretrained), a fresh "
                  f"jointly trained critic, or A's critic on B with every electrode independently circularly "
                  f"shifted. Position standardized per session with its own training-split statistics "
                  f"(evaluation-side). Grey bars: decoder refit on a CPIC trained on B alone, a reference rather "
                  f"than a ceiling. Mean ± 1 SD over 3 seeds per session; all arms in fig6b_transfer_detail.")

    # -- 6b. per-session detail (appendix) ----------------------------------------------
    def transfer_detail_figure(self):
        runs = load(self.results / "transfer_sessions", "transfer_*.json")
        if not runs:
            return self.skip("fig6b_transfer_detail", "no transfer results")
        keys = sorted({(r["target_session"], r.get("source_dropout", 0.0)) for r in runs})
        arms = list(runs[0]["arms"])
        fig, axes = plt.subplots(len(keys), 2, figsize=(13, 0.45 * len(arms) * len(keys) + 1.2),
                                 squeeze=False, sharex="col")
        for i, (target, dropout) in enumerate(keys):
            group = [r for r in runs if (r["target_session"], r.get("source_dropout", 0.0)) == (target, dropout)]
            for j, (metric, title) in enumerate((("position_restandardized", "re-standardized cursor position"),
                                                 ("velocity", "cursor velocity"))):
                ax = axes[i, j]
                stats = [mean_sd([r["arms"][a]["frozen_source_decoder"][metric]["r2_mean"] for r in group])
                         for a in arms]
                y = np.arange(len(arms))[::-1]
                colors = [SERIES[0] if a == "refine_pretrained" else NEUTRAL for a in arms]
                ax.barh(y, [max(m, -1.0) for m, _ in stats], 0.65, xerr=[s for _, s in stats],
                        color=colors, capsize=3)
                for yy, (m, s) in zip(y, stats):  # label beyond the whisker
                    end = max(m + s, -1.0) if m >= 0 else max(m - s, -1.0)
                    ax.annotate(f"{m:+.2f}", (end, yy), xytext=(4 if m >= 0 else -4, -3),
                                textcoords="offset points", fontsize=8, color=INK_2,
                                ha="left" if m >= 0 else "right")
                reference = np.mean([r["arms"]["target_specific_cpic"]["refit_probe"][metric]["r2_mean"]
                                     for r in group])
                ax.axvline(reference, color=SERIES[1], linestyle="--", linewidth=1.2)
                ax.annotate("B-only CPIC, refit probe", (reference, y[0] + 0.55), fontsize=8,
                            color=INK_2, ha="left", xytext=(4, 0), textcoords="offset points")
                ax.axvline(0, color=INK_2, linewidth=0.8)
                ax.set_xlim(-1.05, 1.0)
                ax.set_yticks(y, arms, fontsize=8)
                ax.tick_params(axis="y", labelleft=(j == 0))
                ax.set_title(f"{target}, source dropout {dropout:g}: {title}", fontsize=10)
        for ax in axes[-1]:
            ax.set_xlabel("R² of session A's frozen decoder on session B (test)")
        self.save(fig, "fig6b_transfer_detail", "cross-session transfer, every arm (appendix)",
                  f"Source indy_20160627_01 -> targets {', '.join(sorted({k[0] for k in keys}))}; electrode "
                  f"representation (threshold crossings per channel; physical, not cellular, correspondence). "
                  f"Bars: A's linear decoder (fit on A latents and A behavior, never refit) applied to B's held-out "
                  f"test latents for each arm; blue = read-in initialized from A's encoder and refined against A's "
                  f"frozen critic on unlabeled B. Position is standardized per session with its own training-split "
                  f"statistics (evaluation-side). Dashed line: refit probe of a CPIC trained on B alone, a "
                  f"reference rather than a ceiling - with a short B session its decoder sees little training "
                  f"data, so A's decoder (fit on the 56-min session) can exceed it. Bars below -1 clipped; "
                  f"printed values are true means. n = {len({r['seed'] for r in runs})} "
                  f"seeds per pair; {SEED_NOTE}; seeds measure within-session variability only.")

    # -- 10. distance curve and neural-only predictor ----------------------------------
    def distance_figure(self):
        from transfer_sessions import cross_session_table, predictor_tests
        runs = load(self.results / "transfer_sessions", "transfer_*.json")
        rows = cross_session_table(runs, self.inventory) if runs else []
        if len(rows) < 3:
            return self.skip("fig10_distance", f"need >= 3 target sessions, have {len(rows)}")
        tests, verdict = predictor_tests(rows)
        days = np.array([r["days_apart"] for r in rows])
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.8))
        fig.subplots_adjust(wspace=0.3)
        ax = axes[0]
        # Small horizontal offsets (in days) keep the three sessions within a week of A from overlapping.
        for k, (key, label, shift) in enumerate((("zero_shot_r2", "zero-shot", -3),
                                                 ("refined_r2", "refined vs A's frozen critic", 3))):
            ax.plot(days + shift, [r[key] for r in rows], "o", markersize=9, color=SERIES[k], label=label)
        ax.plot(days, [r["target_only_refit_r2"] for r in rows], "o", markersize=9, markerfacecolor=SURFACE,
                markeredgecolor=NEUTRAL, markeredgewidth=2, linestyle="none", label="decoder refit on B alone")
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_xlabel("days from source session (indy_20160627_01)")
        ax.set_ylabel("frozen decoder R² on target (re-std. position)")
        ax.set_title("Transfer vs time between sessions")
        ax.set_ylim(top=max(r["target_only_refit_r2"] for r in rows) + 0.15)
        ax.legend(loc="upper left", fontsize=8)
        zero_shot = [r["zero_shot_r2"] for r in rows]
        for i, (ax, (key, xlabel)) in enumerate(zip(axes[1:], (
                ("critic_bound_zero_shot", "zero-shot bound under A's frozen critic (nats)"),
                ("rate_correlation", "electrode firing-rate correlation, A vs B")))):
            usable = [r for r in rows if r[key] is not None]
            xs = [r[key] for r in usable]
            ax.plot(xs, [r["zero_shot_r2"] for r in usable], "o", markersize=9, color=SERIES[0])
            span = (max(xs) - min(xs)) or 1.0
            placed = []
            for r in sorted(usable, key=lambda r: -r["zero_shot_r2"]):  # stack labels of nearby points
                px, py = r[key], r["zero_shot_r2"]
                near = sum(abs(qx - px) < 0.2 * span and abs(qy - py) < 0.03 for qx, qy in placed)
                ax.annotate(f"{r['days_apart']:+d} d", (px, py), xytext=(14, 3 - 13 * near),
                            textcoords="offset points", fontsize=8, color=INK_2, va="center",
                            # a leader line ties a label pushed down by its neighbours to its own point
                            arrowprops=dict(arrowstyle="-", color=NEUTRAL, lw=0.7) if near else None)
                placed.append((px, py))
            ax.set_xlim(min(xs) - 0.06 * span, max(xs) + 0.22 * span)
            ax.set_ylim(min(zero_shot) - 0.05, max(zero_shot) + 0.06)
            test = tests.get(f"zero_shot_r2 ~ {key}")
            stat = f"Spearman ρ = {test['rho']:+.2f}, p = {test['p']:.3f}, n = {test['n_sessions']}" if test else "n < 4"
            ax.set_title(stat, fontsize=10)
            ax.set_xlabel(xlabel)
            if i == 0:
                ax.set_ylabel("zero-shot frozen decoder R²")
            ax.axhline(0, color=INK_2, linewidth=0.8)
        verdict_text = {None: "not yet computable", True: "supported", False: "not supported"}[verdict]
        self.save(fig, "fig10_distance", "transfer across sessions and a neural-only predictor",
                  f"Source indy_20160627_01 -> {len(rows)} Indy target sessions, electrode representation, plain "
                  f"(non-dropout) source; every point is a session mean over seeds (seeds averaged first; sessions "
                  f"are the unit of analysis). R² = A's frozen decoder on the target's held-out test latents "
                  f"(re-standardized position). Predictors use no behavior: the zero-shot InfoNCE bound of target "
                  f"test latents under A's frozen critic (mean latents), and the Pearson correlation of training-"
                  f"split firing rates on A's kept electrodes. Spearman p-values from 20,000 permutations. "
                  f"Pre-registered test (critic bound beats both controls, rho > 0, p < 0.05): {verdict_text}.")

    # -- 7. synthetic ------------------------------------------------------------------
    def synthetic_figure(self):
        runs = load(self.results / "synthetic_two_population", "synthetic_seed*.json")
        if not runs:
            return self.skip("fig7_synthetic", "no synthetic results")
        fig, ax = plt.subplots(figsize=(8, 4))
        x, w = np.arange(len(ARM_ORDER)), 0.38
        lowest = 0.0
        for k, (key, label) in enumerate((("direct_r2", "direct latent R² (no alignment)"),
                                          ("procrustes_r2", "Procrustes latent R² (rotation fit)"))):
            stats = [mean_sd([(r["target_specific_cpic"] if a == "target_specific_cpic" else r["arms"][a])
                              ["agreement"][key] for r in runs]) for a in ARM_ORDER]
            bars = ax.bar(x + (k - 0.5) * (w + 0.01), [m for m, _ in stats], w,
                          yerr=[s for _, s in stats], capsize=3, color=SERIES[k], label=label)
            bar_labels(ax, bars, [m for m, _ in stats], errors=[s for _, s in stats])
            lowest = min([lowest] + [m - s for m, s in stats])
        ax.axhline(0, color=INK_2, linewidth=0.8)
        ax.set_ylim(bottom=lowest - 0.3)  # unclipped, with room for the label under the lowest whisker
        ax.set_xticks(x, [wrap(ARM_LABEL[a], 16) for a in ARM_ORDER])
        ax.set_ylabel("R² vs source latents (test)")
        ax.set_title("Synthetic two-population check (shared Lorenz latent)")
        legend_in_headroom(ax, ncol=2, top=1.5)
        self.save(fig, "fig7_synthetic", "synthetic sanity check",
                  f"Two 30-unit populations observe one 3-D Lorenz trajectory through different random linear "
                  f"loadings plus independent unit-variance noise; CPIC (ydim 3, T 4) trained on population 1, "
                  f"population-2 read-ins trained per arm as in Fig. 3. Chronological 70/10/20 split; latent R² on "
                  f"test time points. n = {len(runs)} seeds; {SEED_NOTE}.")

    # -- 8. objective ablation ---------------------------------------------------------
    def ablation_figure(self):
        runs = load(self.results / "objective_ablation", "ablation_*.json")
        if not runs:
            return self.skip("fig8_objective_ablation", "no objective-ablation results")
        betas = sorted({r["beta"] for r in runs})
        panels = [("held-out test bound (nats)", lambda r: r["test_bound_sampled"]),
                  ("test R² (cursor position)", lambda r: r["refit_probe"]["position"]["r2_mean"]),
                  ("participation ratio (test latents)", lambda r: r["participation_ratio_test"])]
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
        for ax, (label, fn) in zip(axes, panels):
            stats = [mean_sd([fn(r) for r in runs if r["beta"] == b]) for b in betas]
            bars = ax.bar(np.arange(len(betas)), [m for m, _ in stats], 0.6, yerr=[s for _, s in stats],
                          capsize=3, color=SERIES[0])
            bar_labels(ax, bars, [m for m, _ in stats], fmt="{:.3f}")
            ax.set_xticks(np.arange(len(betas)), [f"β = {b:g}" for b in betas])
            ax.set_title(label)
        self.save(fig, "fig8_objective_ablation", "what the compression term buys",
                  f"indy_20160627_01, corrected pipeline, linear DCA-initialized CPIC with only the compression "
                  f"weight changed (β = 0 is no-compression contrastive latent prediction). Held-out test split; "
                  f"bound = InfoNCE on sampled latents; R² as in Fig. 2; participation ratio = (Σλ)²/Σλ² of the test "
                  f"latent covariance. n = {len({r['seed'] for r in runs})} seeds; {SEED_NOTE}.")

    # -- 9. runtime --------------------------------------------------------------------
    def runtime_figure(self):
        runs = load(self.results / "ladder", "L*_seed*.json")
        if not runs:
            return self.skip("fig9_runtime", "no ladder results")
        rungs = [r for r in RUNG_LABEL if any(x["rung"] == r for x in runs)]
        phases = [("dca_init_s", "DCA initialization"), ("train_s", "CPIC training (10 epochs)"),
                  ("eval_s", "held-out evaluation + probes")]
        fig, ax = plt.subplots(figsize=(9, 4))
        bottom = np.zeros(len(rungs))
        for k, (key, label) in enumerate(phases):
            values = np.array([np.mean([x["timings"][key] for x in runs if x["rung"] == r]) for r in rungs])
            ax.bar(np.arange(len(rungs)), values, 0.6, bottom=bottom, color=SERIES[k], label=label,
                   edgecolor=SURFACE, linewidth=2)
            bottom += values
        for i, total in enumerate(bottom):
            ax.annotate(f"{total:.0f} s", (i, total), xytext=(0, 3), textcoords="offset points",
                        ha="center", fontsize=8, color=INK_2)
        peak = np.mean([x["peak_gpu_memory_mb"] for x in runs])
        ax.set_xticks(np.arange(len(rungs)), [wrap(RUNG_LABEL[r], 11) for r in rungs], fontsize=8)
        ax.set_ylabel("seconds per run (mean over seeds)")
        ax.set_title(f"Where a benchmark run spends its time (peak GPU memory ≈ {peak / 1024:.2f} GB)")
        legend_in_headroom(ax, ncol=3, top=bottom.max() * 1.3)
        self.save(fig, "fig9_runtime", "runtime by phase",
                  "Ladder runs on one NERSC Perlmutter A100 (40 GB). Data loading excluded (shared across rungs). "
                  "DCA initialization (5 restarts) costs about as much as CPIC training itself and was not "
                  "included in the earlier ~29 s benchmark figure. Means over seeds; no error bars.")

    def write_captions(self):
        (self.out / "captions.md").write_text("# Figure captions\n\n" + "\n".join(self.captions))
        print(f"wrote {self.out / 'captions.md'}")


def main():
    parser = argparse.ArgumentParser(description="Build Nov 9 figures from saved results")
    parser.add_argument("--results", default="results")
    parser.add_argument("--inventory", default="inventory")
    parser.add_argument("--out", default="results/figures")
    args = parser.parse_args()
    figs = Figures(args.results, args.inventory, args.out)
    for build in (figs.inventory_figure, figs.ladder_figure, figs.pseudo_population_figure,
                  figs.identifiability_figure, figs.basin_figure, figs.transfer_figure,
                  figs.transfer_detail_figure,
                  figs.distance_figure, figs.synthetic_figure, figs.ablation_figure, figs.runtime_figure):
        try:
            build()
        except Exception as error:  # one malformed result set should not block the others
            print(f"FAILED {build.__name__}: {type(error).__name__}: {error}")
    figs.write_captions()


if __name__ == "__main__":
    main()
