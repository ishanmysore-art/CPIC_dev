"""Patch-mode helpers for video sparse CPIC experiments.

Turns one video ``(T, H, W, C)`` into multiple fixed-location patch time series
for ``PastFutureDataset``. Gated by ``[Data] patch_mode``.
"""

from __future__ import annotations

from configparser import ConfigParser

import numpy as np


def patch_mode_on(cfg: ConfigParser) -> bool:
    """True iff [Data] patch_mode is present and truthy (default: off)."""
    return (
        cfg.has_section("Data")
        and cfg.has_option("Data", "patch_mode")
        and cfg.getboolean("Data", "patch_mode")
    )


def extract_patch_series(frames: np.ndarray, cfg: ConfigParser) -> list[np.ndarray]:
    """Extract fixed-location patch time series from one video.

    Args:
        frames: uint8 (or float) video of shape ``(T, H, W, C)``.
        cfg: experiment config; reads optional ``[Data]`` patch keys below.

    Returns:
        ``num_patches`` arrays, each of shape ``(T, patch_dim)`` where
        ``patch_dim = P*P`` if ``patch_grayscale`` else ``P*P*C``
        (``P = patch_size``). List length is ``num_patches``.

    Instead of one flattened ``(T, H*W*C)`` whole-frame series, return a *list*
    of per-location time series. Each list element is a single fixed spatial
    window whose pixel values across all ``T`` frames form one temporal
    sequence, preserving the past->future continuity the CPIC objective needs.
    The list is handed straight to ``PastFutureDataset``, which keeps every
    past/future window inside a single series (it never straddles two
    locations), so the predictive objective stays well-defined.

    Patch-cropping logic adapted from lpjiang97/sparse-coding ``NatPatchDataset``
    (random ``randint`` row/col crop within a border). No preprocessing is applied
    to the patch pixels beyond the global uint8->[0,1] scaling (the ``normalize``
    flag); in particular there is NO per-patch mean/DC subtraction (per Rui: keep
    the raw data, do not preprocess).

    Config keys (all under ``[Data]``, all optional with the defaults below):
        patch_mode        : enable this path (default off)
        patch_size        : square patch side P            (default 10)
        num_patches       : number of sampled locations    (default 20)
        patch_border      : keep-out margin from edges      (default 0)
        patch_grayscale   : RGB->luminance, obs P*P         (default True)
        patch_sample_mode : 'fixed' for the run             (default 'fixed')
        patch_seed        : RNG seed for fixed locations    (default 0)
    """
    g = lambda k, d: cfg.get("Data", k) if cfg.has_option("Data", k) else d
    truthy = lambda v: str(v).strip().lower() in ("1", "true", "yes", "on")

    P = int(g("patch_size", "10"))
    n_patches = int(g("num_patches", "20"))
    border = int(g("patch_border", "0"))
    grayscale = truthy(g("patch_grayscale", "True"))
    sample_mode = str(g("patch_sample_mode", "fixed")).strip().lower()
    patch_seed = int(g("patch_seed", "0"))
    # NOTE: `normalize` below is the GLOBAL uint8->[0,1] scaling flag (the same one the
    # whole-frame path honors) -- NOT a per-patch normalization. There is deliberately no
    # per-patch mean/DC subtraction here; patches are the raw [0,1]-scaled pixel values.
    normalize = (
        not (cfg.has_section("Data") and cfg.has_option("Data", "normalize"))
        or cfg.getboolean("Data", "normalize")
    )

    if sample_mode != "fixed":
        raise NotImplementedError(
            f"patch_sample_mode={sample_mode!r}: only 'fixed' (locations fixed for the whole "
            "run) is wired up. Per-epoch resampling would need a hook inside the training loop, "
            "which is outside the data-loader scope of this task."
        )
    if P <= 0 or n_patches <= 0:
        raise ValueError(f"patch_size and num_patches must be positive; got {P}, {n_patches}")

    T, H, W, C = frames.shape
    if P + 2 * border > H or P + 2 * border > W:
        raise ValueError(
            f"patch_size {P} (+ 2*border {border}) does not fit frame {H}x{W}; shrink patch_size/border."
        )

    f = frames.astype(np.float32)
    if normalize:
        f = f / 255.0
    if grayscale:
        # ITU-R BT.601 luminance; keep a trailing channel axis so reshape -> P*P.
        wts = np.array([0.299, 0.587, 0.114], dtype=np.float32)
        f = (f * wts).sum(axis=-1, keepdims=True)  # (T, H, W, 1)

    rng = np.random.default_rng(patch_seed)
    r_hi, c_hi = H - P - border, W - P - border  # inclusive upper bounds
    series: list[np.ndarray] = []
    for _ in range(n_patches):
        r = int(rng.integers(border, r_hi + 1))
        c = int(rng.integers(border, c_hi + 1))
        patch = f[:, r : r + P, c : c + P, :]  # (T, P, P, C')
        s = patch.reshape(T, -1).astype(np.float32)  # (T, P*P*C') raw [0,1] pixels, no mean removal
        series.append(s)
    return series
