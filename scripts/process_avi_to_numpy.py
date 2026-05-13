#!/usr/bin/env python3
"""
Load an AVI (or any OpenCV-supported video), export frames as NumPy, and plot a preview.

Dataset layout, citation, and stimulus descriptions are in the repository ``README.md`` (same
directory as ``pyproject.toml``, resolved from this script’s location). Source: larval salamander
retinal data / Chicago Motion Database (Dryad).

Requires the project video extra: ``uv sync --extra video`` or
``pip install opencv-python-headless``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent
README_MD = _REPO_ROOT / "README.md"


def _require_cv2():
    try:
        import cv2  # type: ignore[import-untyped]
    except ImportError as e:
        print(
            "OpenCV is required. Install with: uv sync --extra video\n"
            "or: pip install opencv-python-headless",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    return cv2


def read_video_frames(
    path: Path,
    *,
    max_frames: int | None,
    stride: int,
) -> tuple[np.ndarray, float]:
    """
    Returns frames as uint8 array (T, H, W, 3) in RGB order, and FPS (0.0 if unknown).
    """
    cv2 = _require_cv2()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frames: list[np.ndarray] = []
    idx = 0
    try:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx % stride != 0:
                idx += 1
                continue
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            idx += 1
            if max_frames is not None and len(frames) >= max_frames:
                break
    finally:
        cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from {path}")

    return np.stack(frames, axis=0), fps


def save_numpy(
    out: Path,
    frames: np.ndarray,
    fps: float,
    *,
    compressed: bool,
) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(
            out,
            frames=frames,
            fps=np.array(fps, dtype=np.float32),
            shape=np.array(frames.shape, dtype=np.int64),
        )
    else:
        np.save(out, frames)


def plot_preview(frames: np.ndarray, preview_path: Path | None, show: bool, grid: int) -> None:
    import matplotlib.pyplot as plt

    t = frames.shape[0]
    n = min(grid * grid, t)
    if t == 1:
        indices = [0]
        nrows, ncols = 1, 1
    else:
        indices = np.linspace(0, t - 1, num=n, dtype=int)
        nrows = int(np.ceil(np.sqrt(len(indices))))
        ncols = int(np.ceil(len(indices) / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.2 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, ax_idx in zip(axes, range(nrows * ncols)):
        if ax_idx < len(indices):
            ax.imshow(frames[indices[ax_idx]])
            ax.set_title(f"t={indices[ax_idx]}")
        ax.axis("off")
    fig.suptitle(f"{t} frames, {frames.shape[1]}x{frames.shape[2]}")
    fig.tight_layout()

    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(preview_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def main() -> None:
    default_input = _REPO_ROOT / "data" / "dryad_chicago_natural_movies" / "MultipleMoviesStim_1_tree.avi"

    p = argparse.ArgumentParser(
        description=__doc__,
        epilog=f"Dataset documentation: {README_MD}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help=f"Path to video file (default: {default_input})",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .npz (default) or .npy path; default is next to input with same stem",
    )
    p.add_argument(
        "--format",
        choices=("npz", "npy"),
        default="npz",
        help="npz stores frames + fps + shape; npy stores only the frame array",
    )
    p.add_argument(
        "--no-compress",
        action="store_true",
        help="With --format npz, use np.savez instead of np.savez_compressed",
    )
    p.add_argument("--max-frames", type=int, default=None, help="Decode at most this many frames")
    p.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every n-th frame after decoding (1 = all frames)",
    )
    p.add_argument(
        "--preview",
        type=Path,
        default=None,
        help="Save a grid preview PNG (default: <output stem>_preview.png if --no-show)",
    )
    p.add_argument(
        "--no-preview",
        action="store_true",
        help="Do not write a preview image",
    )
    p.add_argument("--show", action="store_true", help="Show interactive matplotlib window")
    p.add_argument(
        "--grid",
        type=int,
        default=4,
        help="Grid is grid x grid frames in the preview (capped by frame count)",
    )
    args = p.parse_args()

    inp = args.input.expanduser().resolve()
    if not inp.is_file():
        raise SystemExit(f"Input not found: {inp}")

    if args.output is None:
        out = inp.with_suffix(".npz" if args.format == "npz" else ".npy")
    else:
        out = args.output.expanduser().resolve()

    frames, fps = read_video_frames(inp, max_frames=args.max_frames, stride=args.stride)

    if args.format == "npz":
        save_numpy(out, frames, fps, compressed=not args.no_compress)
    else:
        if out.suffix.lower() != ".npy":
            out = out.with_suffix(".npy")
        np.save(out, frames)

    if not args.no_preview or args.show:
        preview_path = None
        if not args.no_preview:
            if args.preview is not None:
                preview_path = args.preview.expanduser().resolve()
            elif not args.show:
                preview_path = out.with_name(out.stem + "_preview.png")
        plot_preview(frames, preview_path, show=args.show, grid=max(1, args.grid))

    print(f"Wrote {out} shape={frames.shape} dtype={frames.dtype} fps={fps}")


if __name__ == "__main__":
    main()
