#!/usr/bin/env python3
"""
Convert LLaVA-Video-178K clips under ``data/llava_video_178k`` into ``.npz`` frame
arrays for ``experiments/video_experiment/run_sparse_cpic.py``.

Outputs match ``scripts/process_avi_to_numpy.py``: uint8 RGB ``frames`` with shape
``(T, H, W, 3)``, plus ``fps`` and ``shape`` metadata.

Requires the project video extra: ``uv sync --extra video``.

Example::

    uv run python experiments/video_experiment/scripts/process_llava_video_178k.py
    uv run python experiments/video_experiment/scripts/process_llava_video_178k.py \\
        --annotation data/llava_video_178k/annotations/0_30_s_academic_v0_1/caption_n1.json
    uv run python experiments/video_experiment/scripts/process_llava_video_178k.py --concat
    uv run python experiments/video_experiment/scripts/process_llava_video_178k.py --dry-run

Point ``[User] video_path`` in ``config/config_video_sparse_cpic.ini`` at a processed
``.npz`` (per-video or concatenated). Paths may be relative to the repo root.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.process_avi_to_numpy import read_video_frames, save_numpy


def _default_input_dir() -> Path:
    return _REPO_ROOT / "data" / "llava_video_178k"


def _default_output_dir() -> Path:
    return _default_input_dir() / "processed"


def _load_annotation_rows(path: Path) -> list[dict[str, object]]:
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {path}")
    rows: list[dict[str, object]] = []
    for row in payload:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _video_rel_paths(rows: list[dict[str, object]]) -> list[str]:
    paths: list[str] = []
    seen: set[str] = set()
    for row in rows:
        video = row.get("video")
        if not isinstance(video, str) or not video:
            continue
        if video not in seen:
            seen.add(video)
            paths.append(video)
    return paths


def _discover_annotations(input_dir: Path, annotation: Path | None) -> list[Path]:
    if annotation is not None:
        ann = annotation.expanduser().resolve()
        if not ann.is_file():
            raise FileNotFoundError(ann)
        return [ann]

    ann_root = input_dir / "annotations"
    if not ann_root.is_dir():
        raise FileNotFoundError(f"No annotations directory: {ann_root}")
    paths = sorted(ann_root.rglob("*.json"))
    if not paths:
        raise FileNotFoundError(f"No annotation JSON files under {ann_root}")
    return paths


def _resolve_video_path(input_dir: Path, video_rel: str) -> Path:
    return input_dir / "videos" / video_rel


def _output_path_for_video(output_dir: Path, video_rel: str) -> Path:
    rel = Path(video_rel)
    if rel.suffix.lower() == ".mp4":
        rel = rel.with_suffix(".npz")
    else:
        rel = rel.with_name(rel.name + ".npz")
    return output_dir / "videos" / rel


def _concat_output_path(output_dir: Path, annotation_path: Path) -> Path:
    stem = annotation_path.stem
    parent = annotation_path.parent.name
    if parent and parent != "annotations":
        name = f"{parent}__{stem}.npz"
    else:
        name = f"{stem}.npz"
    return output_dir / "concat" / name


def _process_one_video(
    video_path: Path,
    out_path: Path,
    *,
    max_frames: int | None,
    stride: int,
    compressed: bool,
    force: bool,
    dry_run: bool,
) -> dict[str, object] | None:
    if not video_path.is_file():
        print(f"  skip missing: {video_path}", file=sys.stderr)
        return None
    if out_path.is_file() and not force:
        print(f"  exists: {out_path}")
        with np.load(out_path, allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["shape"])
            fps = float(z["fps"].item()) if "fps" in z.files else 0.0
        return {
            "video": str(video_path),
            "output": str(out_path),
            "shape": shape,
            "fps": fps,
            "skipped": True,
        }

    if dry_run:
        print(f"  would write: {out_path} <- {video_path}")
        return {
            "video": str(video_path),
            "output": str(out_path),
            "dry_run": True,
        }

    frames, fps = read_video_frames(video_path, max_frames=max_frames, stride=stride)
    save_numpy(out_path, frames, fps, compressed=compressed)
    print(f"  wrote {out_path} shape={frames.shape} fps={fps:.3f}")
    return {
        "video": str(video_path),
        "output": str(out_path),
        "shape": tuple(int(x) for x in frames.shape),
        "fps": fps,
    }


def _concat_videos(
    entries: list[dict[str, object]],
    out_path: Path,
    *,
    compressed: bool,
    force: bool,
    dry_run: bool,
) -> dict[str, object] | None:
    valid = [e for e in entries if e is not None and not e.get("dry_run") and "shape" in e]
    if not valid:
        if dry_run:
            print(f"  would concatenate {len(entries)} video(s) -> {out_path}")
            return {"output": str(out_path), "dry_run": True, "num_videos": len(entries)}
        print("  no videos to concatenate", file=sys.stderr)
        return None

    if out_path.is_file() and not force:
        print(f"  exists: {out_path}")
        with np.load(out_path, allow_pickle=False) as z:
            shape = tuple(int(x) for x in z["shape"])
        return {
            "output": str(out_path),
            "shape": shape,
            "num_videos": len(valid),
            "skipped": True,
        }

    shapes = [tuple(e["shape"]) for e in valid]  # type: ignore[index]
    hwc = {(s[1], s[2], s[3]) for s in shapes}
    if len(hwc) > 1:
        detail = ", ".join(f"{s[1]}x{s[2]}x{s[3]}" for s in shapes[:5])
        raise RuntimeError(
            f"Cannot concatenate videos with different frame sizes ({detail}). "
            "Process per-video outputs and use resize_* in the sparse CPIC config, "
            "or preprocess to a common resolution first."
        )

    if dry_run:
        total_frames = sum(int(s[0]) for s in shapes)
        print(f"  would concatenate {len(valid)} video(s), T={total_frames} -> {out_path}")
        return {
            "output": str(out_path),
            "dry_run": True,
            "num_videos": len(valid),
            "shape": (total_frames, *next(iter(hwc))),
        }

    arrays: list[np.ndarray] = []
    fps_vals: list[float] = []
    for entry in valid:
        out = Path(str(entry["output"]))
        with np.load(out, allow_pickle=False) as z:
            arrays.append(np.asarray(z["frames"]))
            fps_vals.append(float(z["fps"].item()) if "fps" in z.files else 0.0)

    frames = np.concatenate(arrays, axis=0)
    fps = float(np.median(fps_vals)) if fps_vals else 0.0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_numpy(out_path, frames, fps, compressed=compressed)
    print(f"  wrote {out_path} shape={frames.shape} fps={fps:.3f} ({len(valid)} clips)")
    return {
        "output": str(out_path),
        "shape": tuple(int(x) for x in frames.shape),
        "fps": fps,
        "num_videos": len(valid),
        "source_outputs": [str(e["output"]) for e in valid],
    }


def _process_annotation(
    annotation_path: Path,
    *,
    input_dir: Path,
    output_dir: Path,
    max_frames: int | None,
    stride: int,
    compressed: bool,
    force: bool,
    dry_run: bool,
    limit: int | None,
    concat: bool,
) -> dict[str, object]:
    rows = _load_annotation_rows(annotation_path)
    video_rels = _video_rel_paths(rows)
    if limit is not None:
        video_rels = video_rels[:limit]

    print(f"\nAnnotation: {annotation_path}")
    print(f"  {len(rows)} row(s), {len(video_rels)} unique video(s) to process")

    per_video: list[dict[str, object] | None] = []
    for video_rel in video_rels:
        video_path = _resolve_video_path(input_dir, video_rel)
        out_path = _output_path_for_video(output_dir, video_rel)
        print(f"Video: {video_rel}")
        entry = _process_one_video(
            video_path,
            out_path,
            max_frames=max_frames,
            stride=stride,
            compressed=compressed,
            force=force,
            dry_run=dry_run,
        )
        per_video.append(entry)

    result: dict[str, object] = {
        "annotation": str(annotation_path),
        "videos": [e for e in per_video if e is not None],
    }

    if concat:
        concat_out = _concat_output_path(output_dir, annotation_path)
        print(f"Concat: {concat_out.name}")
        concat_entry = _concat_videos(
            per_video,
            concat_out,
            compressed=compressed,
            force=force,
            dry_run=dry_run,
        )
        if concat_entry is not None:
            result["concat"] = concat_entry

    return result


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process LLaVA-Video-178K clips into .npz arrays for sparse CPIC.",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=_default_input_dir(),
        help=f"LLaVA dataset root containing videos/ and annotations/ (default: {_default_input_dir()})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help=f"Write processed .npz files here (default: {_default_output_dir()})",
    )
    parser.add_argument(
        "--annotation",
        type=Path,
        default=None,
        help="Annotation JSON file (default: all JSON files under input-dir/annotations/)",
    )
    parser.add_argument(
        "--video",
        action="append",
        default=[],
        metavar="REL_PATH",
        help="Process a specific video path relative to input-dir/videos/ (repeatable). "
        "Ignores --annotation when set.",
    )
    parser.add_argument(
        "--concat",
        action="store_true",
        help="Also write one concatenated .npz per annotation under output-dir/concat/",
    )
    parser.add_argument(
        "-n",
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N unique videos per annotation file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Decode at most this many frames per video (before optional concat).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Keep every n-th decoded frame per video (default: 1).",
    )
    parser.add_argument(
        "--no-compress",
        action="store_true",
        help="Use np.savez instead of np.savez_compressed.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-decode and overwrite existing outputs.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned work without decoding or writing files.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Write a JSON manifest of processed outputs (default: output-dir/manifest.json).",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    compressed = not args.no_compress

    if args.stride < 1:
        raise SystemExit("--stride must be >= 1")

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    manifest: dict[str, object] = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "annotations": [],
    }

    if args.video:
        per_video: list[dict[str, object] | None] = []
        for video_rel in args.video:
            video_path = _resolve_video_path(input_dir, video_rel)
            out_path = _output_path_for_video(output_dir, video_rel)
            print(f"\nVideo: {video_rel}")
            entry = _process_one_video(
                video_path,
                out_path,
                max_frames=args.max_frames,
                stride=args.stride,
                compressed=compressed,
                force=args.force,
                dry_run=args.dry_run,
            )
            per_video.append(entry)
        manifest["videos"] = [e for e in per_video if e is not None]
        if args.concat:
            concat_out = output_dir / "concat" / "selected_videos.npz"
            print(f"\nConcat: {concat_out.name}")
            concat_entry = _concat_videos(
                per_video,
                concat_out,
                compressed=compressed,
                force=args.force,
                dry_run=args.dry_run,
            )
            if concat_entry is not None:
                manifest["concat"] = concat_entry
    else:
        annotations = _discover_annotations(input_dir, args.annotation)
        for ann_path in annotations:
            entry = _process_annotation(
                ann_path,
                input_dir=input_dir,
                output_dir=output_dir,
                max_frames=args.max_frames,
                stride=args.stride,
                compressed=compressed,
                force=args.force,
                dry_run=args.dry_run,
                limit=args.limit,
                concat=args.concat,
            )
            manifest["annotations"].append(entry)

    if not args.dry_run:
        manifest_path = (
            args.manifest.expanduser().resolve()
            if args.manifest is not None
            else output_dir / "manifest.json"
        )
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nWrote manifest: {manifest_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
