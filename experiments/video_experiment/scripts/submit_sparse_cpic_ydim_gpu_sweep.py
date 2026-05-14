#!/usr/bin/env python3
"""
Launch parallel sparse CPIC train+visualize jobs: ydim 4/6/8/10 on CUDA devices 0–3.

Each job passes ``--device cuda:N`` with ``N`` equal to its GPU index (0–3). The base .ini is
copied with ``ydim``, ``User.saved_root`` set to ``res/video_sparse_cpic/ydim_analysis/ydim_<n>``,
and ``Training.device`` (``cuda:0`` … ``cuda:3``) so runs do not overwrite checkpoints (same
seed/signature would otherwise collide on shared filenames).

Usage (from repo root)::

    uv run python experiments/video_experiment/scripts/submit_sparse_cpic_ydim_gpu_sweep.py
    uv run python experiments/video_experiment/scripts/submit_sparse_cpic_ydim_gpu_sweep.py --dry-run

Parallel jobs and speed
-----------------------
Each training process is heavy on **CPU** (large ``xdim`` matmuls, DataLoader collation, and
``compute_encoded_mean_stats`` in ``SparseCPIC.fit``, which does an extra full pass over the
dataset **every epoch**). If you launch four jobs at once, each process still defaults to using
all CPU cores for BLAS/OpenMP, so cores oversubscribe and every job slows down. This script sets
``OMP_NUM_THREADS`` / ``MKL_NUM_THREADS`` / … per child to roughly ``cpu_count // n_parallel``
unless you override with ``--omp-threads`` or those variables are already set in the parent
environment. Four processes also **read the same video .npz** at startup, which can contend on
disk until the file is cached.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import TextIO


_EXPERIMENT_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _EXPERIMENT_ROOT.parents[1]
_DEFAULT_BASE_CONFIG = _EXPERIMENT_ROOT / "config" / "config_video_sparse_cpic.ini"
_RUNNER = _EXPERIMENT_ROOT / "run_sparse_cpic_train_and_visualize.py"
_LOG_DIR = _SCRIPTS_DIR / "logs"
# Sweep outputs (relative to repo root); ignores User.saved_root subdirs like .../test/...
_SWEEP_SAVED_ROOT_PARENT = "res/video_sparse_cpic/ydim_analysis"

# (physical GPU index, ydim)
_DEFAULT_JOBS: tuple[tuple[int, int], ...] = (
    (0, 4),
    (1, 6),
    (2, 8),
    (3, 10),
)


def _patch_config_ini(base: Path, ydim: int, device: str) -> str:
    """Return patched ini: ydim, saved_root under ydim_analysis, and Training device."""
    lines_out: list[str] = []
    saved_root_val: str | None = None
    for line in base.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            lines_out.append(line)
            continue
        if stripped.lower().startswith("ydim"):
            if "=" in line:
                key, _ = line.split("=", 1)
                lines_out.append(f"{key.strip()} = {ydim}")
            else:
                lines_out.append(line)
            continue
        if stripped.lower().startswith("saved_root"):
            if "=" in line:
                _, rhs = line.split("=", 1)
                saved_root_val = rhs.strip()
                lines_out.append(f"saved_root = {_SWEEP_SAVED_ROOT_PARENT}/ydim_{ydim}")
            else:
                lines_out.append(line)
            continue
        if "=" in line and line.split("=", 1)[0].strip().lower() == "device":
            lines_out.append(f"device = {device}")
            continue
        lines_out.append(line)

    if saved_root_val is None:
        raise ValueError(f"No saved_root = line found in {base}")
    return "\n".join(lines_out) + "\n"


def _subprocess_env(*, num_parallel_jobs: int, omp_threads: int | None) -> dict[str, str]:
    """Child env: cap BLAS/OpenMP threads when several jobs share one machine."""
    env = os.environ.copy()
    if omp_threads is None:
        try:
            cores = len(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            cores = os.cpu_count() or 1
        per = max(1, cores // max(1, num_parallel_jobs))
    else:
        per = max(1, omp_threads)

    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if key not in env:
            env[key] = str(per)
    return env


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base-config",
        type=Path,
        default=_DEFAULT_BASE_CONFIG,
        help="Template config (ydim, saved_root→ydim_analysis/ydim_*, Training.device overridden per job).",
    )
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument(
        "--signature",
        type=int,
        default=None,
        help=(
            "If set, every job uses this exact signature (only safe when each job has a distinct "
            "saved_root, as this script ensures). Default: distinct signature per job from wall time."
        ),
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands and exit without starting processes or writing configs.",
    )
    ap.add_argument(
        "--sequential",
        action="store_true",
        help="Run jobs one after another instead of in parallel.",
    )
    ap.add_argument(
        "--omp-threads",
        type=int,
        default=None,
        metavar="N",
        help=(
            "OMP/BLAS thread cap per job (default when parallel: max(1, cpu_affinity_count // 4); "
            "when --sequential: max(1, cpu_affinity_count)). Ignored for keys already set in the environment."
        ),
    )
    args = ap.parse_args()

    base_config = args.base_config.resolve()
    if not base_config.is_file():
        raise FileNotFoundError(base_config)

    if not _RUNNER.is_file():
        raise FileNotFoundError(_RUNNER)

    t0 = int(time.time())
    procs: list[tuple[int, int, subprocess.Popen[str], TextIO]] = []
    cfg_dir = _SCRIPTS_DIR / "generated_configs"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    py = sys.executable
    n_parallel = 1 if args.sequential else len(_DEFAULT_JOBS)
    child_env = _subprocess_env(num_parallel_jobs=n_parallel, omp_threads=args.omp_threads)
    try:
        cores = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        cores = os.cpu_count() or 1
    tcap = child_env.get("OMP_NUM_THREADS", "?")
    print(
        f"Subprocess thread cap (OMP_NUM_THREADS if unset in parent): {tcap} "
        f"(~{cores} visible CPUs, {n_parallel} concurrent job(s); override with --omp-threads)"
    )

    for idx, (phys_gpu, ydim) in enumerate(_DEFAULT_JOBS):
        device = f"cuda:{phys_gpu}"
        cfg_text = _patch_config_ini(base_config, ydim, device)
        cfg_path = cfg_dir / f"config_video_sparse_cpic_ydim{ydim}.ini"
        if not args.dry_run:
            cfg_path.write_text(cfg_text)

        sig = args.signature if args.signature is not None else t0 * 1000 + idx * 100 + ydim

        cmd = [
            py,
            str(_RUNNER),
            "--config",
            str(cfg_path),
            "--seed",
            str(args.seed),
            "--signature",
            str(sig),
            "--device",
            device,
        ]

        log_path = _LOG_DIR / f"ydim{ydim}_gpu{phys_gpu}.log"
        print(
            f"ydim={ydim} device={device} signature={sig} config={cfg_path}"
        )
        print(" ", " ".join(cmd))

        if args.dry_run:
            continue

        log_f = open(log_path, "w", encoding="utf-8")
        p = subprocess.Popen(
            cmd,
            cwd=str(_REPO_ROOT),
            env=child_env,
            stdout=log_f,
            stderr=subprocess.STDOUT,
            text=True,
        )
        procs.append((ydim, phys_gpu, p, log_f))
        if args.sequential:
            rc = p.wait()
            log_f.close()
            if rc != 0:
                raise SystemExit(f"Job ydim={ydim} gpu={phys_gpu} failed with exit code {rc} (see {log_path})")

    if args.dry_run:
        return

    if args.sequential:
        print("All jobs finished successfully.")
        return

    failures: list[tuple[int, int, int]] = []
    for ydim, phys_gpu, p, log_f in procs:
        rc = p.wait()
        log_f.close()
        if rc != 0:
            failures.append((ydim, phys_gpu, rc))

    if failures:
        parts = ", ".join(f"ydim={y}/gpu={g} rc={r}" for y, g, r in failures)
        raise SystemExit(f"Some jobs failed: {parts}. Logs under {_LOG_DIR}")

    print(f"All {len(_DEFAULT_JOBS)} jobs finished successfully. Logs: {_LOG_DIR}")


if __name__ == "__main__":
    main()
