"""
Visualize saved sparse CPIC encoded representations and decoder weights.

Usage:
    python experiments/video_experiment/visualize_sparse_cpic_outputs.py \
        --saved-root res/video_sparse_cpic/xxx \
        --seed 22 --signature 22 --config experiments/video_experiment/config/xxx.ini
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
from configparser import ConfigParser


_THIS_DIR = Path(__file__).resolve().parent
_CONFIG_DIR = _THIS_DIR / "config"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class _Conf(ConfigParser):
    def optionxform(self, optionstr):
        return optionstr


def _import_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt  # local import to avoid failing on --help in broken envs
        return plt
    except Exception as exc:
        raise RuntimeError(
            "Failed to import matplotlib. Use the project environment via `uv run ...` "
            "or fix your local NumPy/matplotlib installation."
        ) from exc


def _resolve_path(raw: str) -> Path:
    p = Path(raw.strip())
    if p.is_absolute():
        return p
    cand = (_THIS_DIR / p).resolve()
    return cand if cand.exists() else (_REPO_ROOT / p).resolve()


def _resolve_saved_root(raw: str, seed: int, signature: int) -> Path:
    p = Path(raw.strip())
    if p.is_absolute():
        return p

    candidates = [(_THIS_DIR / p).resolve(), (_REPO_ROOT / p).resolve()]
    for cand in candidates:
        enc = cand / f"encoded_representations_seed{seed}.pkl"
        ckpt = cand / f"sparse_cpic_checkpoint_sig{signature}_seed{seed}.pt"
        if enc.is_file() or ckpt.is_file():
            return cand

    # Fall back to previous behavior when no output files are found yet.
    return candidates[0]


def _resolve_config_arg(arg: str) -> Path:
    p = Path(arg)
    if p.is_file():
        return p
    for alt in (_THIS_DIR / arg, _CONFIG_DIR / arg, _CONFIG_DIR / Path(arg).name):
        if alt.is_file():
            return alt.resolve()
    raise FileNotFoundError(f"Could not resolve config path: {arg}")


def _load_encoded_repr(saved_root: Path, seed: int) -> np.ndarray:
    pkl_path = saved_root / f"encoded_representations_seed{seed}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Missing encoded representation file: {pkl_path}\n"
            "Check --saved-root and --seed. "
            "If your outputs are under repo root, pass e.g. --saved-root res/video_sparse_cpic."
        )
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    encoded_list = payload["encoded_representations"]
    encoded = np.asarray(encoded_list[0], dtype=np.float32)
    if encoded.ndim == 3:
        encoded = encoded[:, -1, :]
    if encoded.ndim != 2:
        raise ValueError(f"Expected encoded repr to be 2D after squeeze, got shape {encoded.shape}")
    return encoded


def _load_decoder_weights(saved_root: Path, signature: int, seed: int) -> np.ndarray:
    import torch

    ckpt = saved_root / f"sparse_cpic_checkpoint_sig{signature}_seed{seed}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Missing checkpoint file: {ckpt}\n"
            "Check --saved-root, --signature, and --seed."
        )
    data = torch.load(ckpt, map_location="cpu")
    state = data["model_state_dict"]
    if "decoder.weight" not in state:
        raise KeyError("decoder.weight not found in checkpoint.")
    return state["decoder.weight"].detach().cpu().numpy().astype(np.float32)


def _load_reconstructed(saved_root: Path, seed: int) -> np.ndarray:
    pkl_path = saved_root / f"inferred_trials_seed{seed}.pkl"
    if not pkl_path.is_file():
        raise FileNotFoundError(
            f"Missing reconstructed trial file: {pkl_path}\n"
            "Check --saved-root and --seed."
        )
    with pkl_path.open("rb") as f:
        payload = pickle.load(f)
    inferred = payload["inferred_sparse_CPIC_trials"]
    recon = np.asarray(inferred[0], dtype=np.float32)
    if recon.ndim != 2:
        raise ValueError(f"Expected reconstructed video to be 2D [time, xdim], got shape {recon.shape}")
    return recon


def _infer_hw_from_config(config_path: Path) -> tuple[int, int] | None:
    cfg = _Conf()
    cfg.read(config_path)
    if cfg.has_option("Data", "resize_height") and cfg.has_option("Data", "resize_width"):
        h = cfg.get("Data", "resize_height").strip()
        w = cfg.get("Data", "resize_width").strip()
        if h and w:
            return int(h), int(w)
    return None


def _plot_encoded(encoded: np.ndarray, out_dir: Path) -> None:
    plt = _import_matplotlib_pyplot()
    n_time, n_latent = encoded.shape
    # Use the true raw activation range for the heatmap color scale (no clipping).
    vmax = float(np.max(np.abs(encoded)))
    vmax = max(vmax, 1e-12)

    fig, ax = plt.subplots(figsize=(12, 5))
    im = ax.imshow(
        encoded.T,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(f"Encoded Representation (time x latent), shape={encoded.shape}")
    ax.set_xlabel("time index")
    ax.set_ylabel("latent dimension")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="activation")
    fig.tight_layout()
    fig.savefig(out_dir / "encoded_repr_heatmap.png", dpi=180)
    plt.close(fig)

    per_latent_std = encoded.std(axis=0)
    order = np.argsort(per_latent_std)[::-1]
    top_k = min(10, n_latent)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(np.arange(top_k), per_latent_std[order[:top_k]])
    ax.set_xticks(np.arange(top_k))
    ax.set_xticklabels([str(i) for i in order[:top_k]])
    ax.set_xlabel("latent dimension (sorted by std)")
    ax.set_ylabel("std over time")
    ax.set_title("Top varying latent dimensions")
    fig.tight_layout()
    fig.savefig(out_dir / "encoded_repr_top_std_dims.png", dpi=180)
    plt.close(fig)

    centered = encoded - encoded.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    pc2 = u[:, :2] * s[:2]

    fig, ax = plt.subplots(figsize=(6, 6))
    sc = ax.scatter(pc2[:, 0], pc2[:, 1], c=np.arange(n_time), s=10, cmap="viridis")
    ax.set_title("Encoded trajectory projected to first 2 PCs")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04, label="time index")
    fig.tight_layout()
    fig.savefig(out_dir / "encoded_repr_pc_trajectory.png", dpi=180)
    plt.close(fig)


def _plot_decoder(
    decoder_w: np.ndarray,
    out_dir: Path,
    frame_hw: tuple[int, int] | None,
) -> None:
    plt = _import_matplotlib_pyplot()
    xdim, ydim = decoder_w.shape
    vmax = np.percentile(np.abs(decoder_w), 99.0)
    vmax = max(vmax, 1e-6)

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        decoder_w,
        aspect="auto",
        cmap="coolwarm",
        vmin=-vmax,
        vmax=vmax,
    )
    ax.set_title(f"Decoder Linear Mapping W (xdim x ydim), shape={decoder_w.shape}")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("observation dimension")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="weight")
    fig.tight_layout()
    fig.savefig(out_dir / "decoder_weight_matrix.png", dpi=180)
    plt.close(fig)

    col_norm = np.linalg.norm(decoder_w, axis=0)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(np.arange(ydim), col_norm, marker="o", markersize=3, linewidth=1)
    ax.set_title("Decoder column norm per latent dimension")
    ax.set_xlabel("latent dimension")
    ax.set_ylabel("L2 norm of W[:, j]")
    fig.tight_layout()
    fig.savefig(out_dir / "decoder_column_norms.png", dpi=180)
    plt.close(fig)

    # Deprecated output (kept for backward compatibility cleanup only).
    deprecated_top_latents_png = out_dir / "decoder_basis_top_latents_rgb.png"
    if deprecated_top_latents_png.exists():
        deprecated_top_latents_png.unlink()

    if frame_hw is not None and xdim == frame_hw[0] * frame_hw[1] * 3:
        h, w = frame_hw
        all_idx = np.arange(ydim)
        ncols = min(8, max(1, ydim))
        max_per_fig = 48
        page = 0
        for start in range(0, ydim, max_per_fig):
            page += 1
            idx_chunk = all_idx[start : start + max_per_fig]
            n_show = len(idx_chunk)
            nrows = int(np.ceil(n_show / ncols))
            fig, axes = plt.subplots(nrows, ncols, figsize=(2.2 * ncols, 2.4 * nrows))
            axes = np.atleast_1d(axes).ravel()
            for i, ax in enumerate(axes):
                ax.axis("off")
                if i >= n_show:
                    continue
                j = int(idx_chunk[i])
                basis = decoder_w[:, j].reshape(h, w, 3)
                bmax = np.percentile(np.abs(basis), 99.0)
                bmax = max(bmax, 1e-6)
                ax.imshow(np.clip((basis / (2.0 * bmax)) + 0.5, 0.0, 1.0))
                ax.set_title(f"latent {j}", fontsize=8)
            fig.suptitle(
                (
                    f"Decoder basis vectors reshaped to RGB frame "
                    f"(shape={h}x{w}x3), latents {start}-{start + n_show - 1}"
                ),
                fontsize=11,
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"decoder_basis_all_latents_rgb_page{page:02d}.png", dpi=180)
            plt.close(fig)


def _plot_reconstructed_video(
    recon: np.ndarray,
    out_dir: Path,
    frame_hw: tuple[int, int] | None,
    max_frames_for_gif: int = 200,
) -> None:
    if frame_hw is None:
        print("Skip reconstructed video plotting: frame shape is not available.")
        return
    h, w = frame_hw
    xdim = recon.shape[1]
    if xdim != h * w * 3:
        print(
            "Skip reconstructed video plotting: reconstructed feature dimension "
            f"{xdim} != {h}*{w}*3."
        )
        return

    plt = _import_matplotlib_pyplot()
    recon_img = np.clip(recon.reshape(-1, h, w, 3), 0.0, 1.0)
    n_time = recon_img.shape[0]

    n_show = min(12, n_time)
    idx = np.linspace(0, n_time - 1, n_show, dtype=int)
    ncols = 4
    nrows = int(np.ceil(n_show / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.0 * ncols, 2.7 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n_show:
            continue
        t = int(idx[i])
        ax.imshow(recon_img[t])
        ax.set_title(f"t={t}", fontsize=9)
    fig.suptitle(f"Reconstructed video samples (shape={h}x{w}x3, n_frames={n_time})", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "reconstructed_video_samples.png", dpi=180)
    plt.close(fig)

    # Save a lightweight animated preview (GIF).
    try:
        from matplotlib.animation import FuncAnimation, PillowWriter

        n_gif = min(max_frames_for_gif, n_time)
        gif_idx = np.linspace(0, n_time - 1, n_gif, dtype=int)
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.axis("off")
        im = ax.imshow(recon_img[int(gif_idx[0])])

        def _update(i: int):
            im.set_data(recon_img[int(gif_idx[i])])
            ax.set_title(f"t={int(gif_idx[i])}", fontsize=9)
            return (im,)

        anim = FuncAnimation(fig, _update, frames=n_gif, interval=80, blit=False)
        anim.save(out_dir / "reconstructed_video.gif", writer=PillowWriter(fps=12))
        plt.close(fig)
    except Exception as exc:
        print(f"Warning: failed to save reconstructed_video.gif ({exc})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize encoded repr and SparseCPIC decoder mapping.")
    parser.add_argument("--saved-root", type=str, required=True, help="Directory with .pkl and checkpoint outputs.")
    parser.add_argument("--seed", type=int, required=True, help="Seed used during run_sparse_cpic.py")
    parser.add_argument("--signature", type=int, required=True, help="Signature used during run_sparse_cpic.py")
    parser.add_argument(
        "--config",
        type=str,
        default=str(_CONFIG_DIR / "config_video_sparse_cpic.ini"),
        help="Config path used for training (only used for optional frame shape inference).",
    )
    parser.add_argument(
        "--frame-shape",
        type=int,
        nargs=2,
        metavar=("H", "W"),
        default=None,
        help="Optional frame shape to reshape decoder rows into RGB image basis.",
    )
    args = parser.parse_args()

    saved_root = _resolve_saved_root(args.saved_root, seed=args.seed, signature=args.signature)
    out_dir = saved_root / f"visualizations_sig{args.signature}_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    encoded = _load_encoded_repr(saved_root, args.seed)
    decoder_w = _load_decoder_weights(saved_root, args.signature, args.seed)
    recon = _load_reconstructed(saved_root, args.seed)

    if args.frame_shape is not None:
        frame_hw = (int(args.frame_shape[0]), int(args.frame_shape[1]))
    else:
        frame_hw = _infer_hw_from_config(_resolve_config_arg(args.config))

    _plot_encoded(encoded, out_dir)
    _plot_decoder(decoder_w, out_dir, frame_hw=frame_hw)
    _plot_reconstructed_video(recon, out_dir, frame_hw=frame_hw)

    print(f"Saved visualizations to: {out_dir}")
    print("Generated files:")
    for p in sorted(list(out_dir.glob("*.png")) + list(out_dir.glob("*.gif"))):
        print(f"  - {p.name}")


if __name__ == "__main__":
    main()
