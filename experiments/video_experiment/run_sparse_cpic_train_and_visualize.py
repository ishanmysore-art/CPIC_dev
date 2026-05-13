"""
Run sparse CPIC training and visualization with the same --seed and --signature.

Training writes checkpoints and pickles keyed by seed/signature; visualization
expects the same pair.

Usage::

    uv run python experiments/video_experiment/run_sparse_cpic_train_and_visualize.py
    uv run python experiments/video_experiment/run_sparse_cpic_train_and_visualize.py \\
        --seed 42 --signature 20260513120000 \\
        --config experiments/video_experiment/config/config_video_sparse_cpic.ini
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from configparser import ConfigParser
from pathlib import Path


_EXPERIMENT_ROOT = Path(__file__).resolve().parent
_CONFIG_DIR = _EXPERIMENT_ROOT / "config"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Conf(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


def _resolve_config_arg(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    for alt in (
        _EXPERIMENT_ROOT / arg,
        _CONFIG_DIR / arg,
        _CONFIG_DIR / Path(arg).name,
        _REPO_ROOT / arg,
    ):
        if alt.is_file():
            return alt
    raise FileNotFoundError(arg)


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    if not p.parts:
        raise ValueError("Empty path in config")
    if p.is_absolute():
        return p
    cand = _EXPERIMENT_ROOT / p
    return cand.resolve() if cand.exists() else (_REPO_ROOT / p).resolve()


def _saved_root_from_config(config_path: Path) -> Path:
    cfg = _Conf()
    cfg.read(config_path)
    return _resolve_path(cfg.get("User", "saved_root"))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Train sparse CPIC on video, then visualize outputs with the same seed and signature."
    )
    ap.add_argument("--config", type=str, default=str(_CONFIG_DIR / "config_video_sparse_cpic.ini"))
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument(
        "--signature",
        type=int,
        default=None,
        help="Run id (default: YYYYMMDDHHMMSS at start, shared by train and vis).",
    )
    ap.add_argument(
        "--saved-root",
        type=str,
        default=None,
        help=(
            "Directory passed to visualization only (default: User.saved_root from config, "
            "which must match where training wrote)."
        ),
    )
    ap.add_argument("--device", type=str, default=None, help="Forwarded to run_sparse_cpic.py only.")
    ap.add_argument(
        "--frame-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional H W for visualize_sparse_cpic_outputs.py.",
    )
    args = ap.parse_args()

    config_path = _resolve_config_arg(args.config)
    signature = args.signature if args.signature is not None else int(datetime.now().strftime("%Y%m%d%H%M%S"))
    saved_root = Path(args.saved_root).resolve() if args.saved_root else _saved_root_from_config(config_path)

    train_script = _EXPERIMENT_ROOT / "run_sparse_cpic.py"
    vis_script = _EXPERIMENT_ROOT / "visualize_sparse_cpic_outputs.py"
    py = sys.executable

    train_cmd = [
        py,
        str(train_script),
        "--config",
        str(config_path),
        "--seed",
        str(args.seed),
        "--signature",
        str(signature),
    ]
    if args.device is not None:
        train_cmd.extend(["--device", args.device])

    print("Running training:", " ".join(train_cmd))
    subprocess.run(train_cmd, cwd=str(_REPO_ROOT), check=True)

    vis_cmd = [
        py,
        str(vis_script),
        "--saved-root",
        str(saved_root),
        "--seed",
        str(args.seed),
        "--signature",
        str(signature),
        "--config",
        str(config_path),
    ]
    if args.frame_shape is not None:
        vis_cmd.extend(["--frame-shape", str(args.frame_shape[0]), str(args.frame_shape[1])])

    print("Running visualization:", " ".join(vis_cmd))
    subprocess.run(vis_cmd, cwd=str(_REPO_ROOT), check=True)

    print(f"Done. seed={args.seed} signature={signature} saved_root={saved_root}")


if __name__ == "__main__":
    main()
