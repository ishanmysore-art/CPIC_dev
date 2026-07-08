#!/usr/bin/env python3
"""
Download annotations and videos from the LLaVA-Video-178K Hugging Face dataset.

Source: https://huggingface.co/datasets/lmms-lab/LLaVA-Video-178K

Annotations are JSON files per subset/split. Videos are shipped as ``.tar.gz`` archives
inside each subset directory (not as individual files on the Hub). For a subsample of size
``n``, the script downloads the tar archives that contain the selected videos. A local
video-to-archive index is built automatically while probing archives; run
``--build-video-index`` once per subset to index all archives up front (faster repeat
subsampling).

Files are written under the repository ``data/`` directory by default (``data/`` is gitignored).

Example::

    uv run python experiments/video_experiment/scripts/download_llava_video_178k.py --dry-run
    uv run python experiments/video_experiment/scripts/download_llava_video_178k.py -n 10
    uv run python experiments/video_experiment/scripts/download_llava_video_178k.py \\
        --subset 0_30_s_academic_v0_1 --split caption -n 50 --seed 0
    uv run python experiments/video_experiment/scripts/download_llava_video_178k.py \\
        --subset 0_30_s_academic_v0_1 --build-video-index --keep-archives
    uv run python experiments/video_experiment/scripts/download_llava_video_178k.py --all-subsets

Optional: set ``HF_TOKEN`` for authenticated Hub access (not required for this public dataset).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import tarfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REPO_ID = "lmms-lab/LLaVA-Video-178K"
HF_API_BASE = f"https://huggingface.co/api/datasets/{REPO_ID}"
HF_RESOLVE_BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main"

_SPLIT_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("caption", re.compile(r"_cap_processed\.json$")),
    ("open_ended", re.compile(r"_oe_.*_qa_processed\.json$")),
    ("multi_choice", re.compile(r"_mc_.*_qa_processed\.json$")),
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _default_output_dir() -> Path:
    return _repo_root() / "data" / "llava_video_178k"


def _hf_headers() -> dict[str, str]:
    headers = {"User-Agent": "cpic-download-script/1.0"}
    token = os.environ.get("HF_TOKEN", "").strip() or os.environ.get("HUGGING_FACE_HUB_TOKEN", "").strip()
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _request_json(url: str) -> object:
    req = urllib.request.Request(url, headers=_hf_headers())
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {detail[:500]}") from e
    return json.loads(body)


def _download_file(
    url: str,
    dest: Path,
    *,
    chunk: int = 1024 * 1024,
    expected_size: int | None = None,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    offset = tmp.stat().st_size if tmp.exists() else 0
    headers = _hf_headers()
    if offset:
        headers["Range"] = f"bytes={offset}-"
    req = urllib.request.Request(url, headers=headers)
    mode = "ab" if offset else "wb"
    try:
        with urllib.request.urlopen(req, timeout=600) as resp, tmp.open(mode) as f:
            cl = resp.headers.get("Content-Length")
            total = expected_size
            if cl and cl.isdigit():
                total = offset + int(cl) if resp.status == 206 else int(cl)
            done = offset
            next_report = ((done // (100 * 1024 * 1024)) + 1) * 100 * 1024 * 1024
            while True:
                block = resp.read(chunk)
                if not block:
                    break
                f.write(block)
                done += len(block)
                if total:
                    while done >= next_report:
                        pct = 100.0 * min(next_report, total) / total
                        print(
                            f"    {dest.name}: {min(next_report, done) / (1024**3):.2f} "
                            f"/ {total / (1024**3):.2f} GiB ({pct:.1f}%)",
                            flush=True,
                        )
                        next_report += 100 * 1024 * 1024
    except urllib.error.HTTPError as e:
        if e.code == 416 and offset:
            tmp.replace(dest)
            return
        tmp.unlink(missing_ok=True)
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} downloading {url}: {detail[:500]}") from e
    tmp.replace(dest)


def _resolve_url(relative_path: str) -> str:
    return f"{HF_RESOLVE_BASE}/{urllib.parse.quote(relative_path, safe='/')}"


def _list_subset_names() -> list[str]:
    payload = _request_json(f"{HF_API_BASE}/tree/main")
    if not isinstance(payload, list):
        raise RuntimeError("Unexpected Hub tree response")
    names: list[str] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        if item.get("type") != "directory":
            continue
        path = item.get("path")
        if isinstance(path, str) and not path.startswith("."):
            names.append(path)
    return sorted(names)


def _list_subset_files(subset: str) -> list[dict[str, object]]:
    payload = _request_json(f"{HF_API_BASE}/tree/main/{urllib.parse.quote(subset, safe='')}")
    if not isinstance(payload, list):
        raise RuntimeError(f"Unexpected file listing for subset {subset!r}")
    out: list[dict[str, object]] = []
    for item in payload:
        if isinstance(item, dict):
            out.append(item)
    return out


def _annotation_paths_by_split(subset: str) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in _list_subset_files(subset):
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".json"):
            continue
        for split_name, pattern in _SPLIT_PATTERNS:
            if pattern.search(path):
                mapping[split_name] = path
                break
    return mapping


def _video_archives(subset: str) -> list[tuple[str, int]]:
    archives: list[tuple[str, int]] = []
    for item in _list_subset_files(subset):
        path = item.get("path")
        if not isinstance(path, str) or not path.endswith(".tar.gz"):
            continue
        size = item.get("size")
        archives.append((path, int(size) if isinstance(size, int) else 0))
    archives.sort(key=lambda pair: pair[1])
    return archives


def _load_annotations(relative_path: str, cache_dir: Path) -> list[dict[str, object]]:
    local = cache_dir / "raw_annotations" / relative_path.replace("/", "__")
    if not local.exists():
        print(f"Downloading annotations: {relative_path}")
        _download_file(_resolve_url(relative_path), local)
    with local.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected a JSON list in {relative_path}")
    rows: list[dict[str, object]] = []
    for row in payload:
        if isinstance(row, dict):
            rows.append(row)
    return rows


def _subsample_rows(
    rows: list[dict[str, object]],
    n: int,
    *,
    seed: int | None,
) -> list[dict[str, object]]:
    if n <= 0:
        raise ValueError("--num-samples must be positive")
    if n >= len(rows):
        return list(rows)
    rng = random.Random(seed)
    indices = rng.sample(range(len(rows)), n)
    indices.sort()
    return [rows[i] for i in indices]


def _video_paths(rows: list[dict[str, object]]) -> list[str]:
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


def _index_path(cache_dir: Path, subset: str) -> Path:
    return cache_dir / "video_tar_index" / f"{subset}.json"


def _indexed_archives_path(cache_dir: Path, subset: str) -> Path:
    return cache_dir / "video_tar_index" / f"{subset}_indexed_archives.json"


def _load_indexed_archives(cache_dir: Path, subset: str) -> set[str]:
    path = _indexed_archives_path(cache_dir, subset)
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, list):
        return set()
    return {item for item in payload if isinstance(item, str)}


def _save_indexed_archives(cache_dir: Path, subset: str, archives: set[str]) -> None:
    path = _indexed_archives_path(cache_dir, subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(sorted(archives), f, indent=2)


def _index_archive(
    subset: str,
    archive_rel: str,
    *,
    cache_dir: Path,
    force: bool,
    keep_archives: bool,
    archive_size: int,
) -> dict[str, str]:
    archive_cache = cache_dir / "archives" / subset
    archive_cache.mkdir(parents=True, exist_ok=True)
    archive_name = Path(archive_rel).name
    local_tar = archive_cache / archive_name

    if not local_tar.exists() or force:
        print(f"Indexing archive: {archive_rel}")
        _download_file(_resolve_url(archive_rel), local_tar, expected_size=archive_size or None)
    else:
        print(f"Listing cached archive: {archive_name}")

    members = _list_tar_members(local_tar)
    updates = {member: archive_rel for member in members}
    tar_index = _load_tar_index(cache_dir, subset)
    tar_index.update(updates)
    _save_tar_index(cache_dir, subset, tar_index)

    indexed = _load_indexed_archives(cache_dir, subset)
    indexed.add(archive_rel)
    _save_indexed_archives(cache_dir, subset, indexed)

    if not keep_archives:
        local_tar.unlink(missing_ok=True)

    print(f"  indexed {len(members)} video path(s) in {archive_name}")
    return updates


def _build_video_index(
    subset: str,
    *,
    cache_dir: Path,
    force: bool,
    keep_archives: bool,
) -> None:
    archives = _video_archives(subset)
    if not archives:
        print(f"No video archives found for subset {subset!r}.", file=sys.stderr)
        return
    print(f"Building video index for {subset} ({len(archives)} archive(s))...")
    for archive_rel, size in archives:
        _index_archive(
            subset,
            archive_rel,
            cache_dir=cache_dir,
            force=force,
            keep_archives=keep_archives,
            archive_size=size,
        )
    tar_index = _load_tar_index(cache_dir, subset)
    print(f"Video index complete: {len(tar_index)} path(s) for {subset}")


def _load_tar_index(cache_dir: Path, subset: str) -> dict[str, str]:
    path = _index_path(cache_dir, subset)
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        return {}
    out: dict[str, str] = {}
    for video, archive in payload.items():
        if isinstance(video, str) and isinstance(archive, str):
            out[video] = archive
    return out


def _save_tar_index(cache_dir: Path, subset: str, index: dict[str, str]) -> None:
    path = _index_path(cache_dir, subset)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _list_tar_members(tar_path: Path) -> list[str]:
    with tarfile.open(tar_path, mode="r:gz") as tf:
        return [m.name for m in tf.getmembers() if m.isfile()]


def _extract_members(tar_path: Path, members: set[str], videos_dir: Path) -> int:
    extracted = 0
    with tarfile.open(tar_path, mode="r:gz") as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = member.name.lstrip("./")
            if name not in members:
                continue
            dest = videos_dir / name
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            extracted_obj = tf.extractfile(member)
            if extracted_obj is None:
                continue
            with extracted_obj, dest.open("wb") as out:
                while True:
                    block = extracted_obj.read(1024 * 1024)
                    if not block:
                        break
                    out.write(block)
            extracted += 1
    return extracted


def _download_videos_for_paths(
    subset: str,
    needed_videos: list[str],
    *,
    out_dir: Path,
    cache_dir: Path,
    force: bool,
    keep_archives: bool,
) -> None:
    if not needed_videos:
        print("No video paths to download.")
        return

    videos_dir = out_dir / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    remaining = {v for v in needed_videos if not (videos_dir / v).exists() or force}
    if not remaining:
        print(f"All {len(needed_videos)} video(s) already present.")
        return

    tar_index = _load_tar_index(cache_dir, subset)
    indexed_archives = _load_indexed_archives(cache_dir, subset)
    archives = _video_archives(subset)
    archive_sizes = {rel: size for rel, size in archives}
    if not archives:
        print(f"No video archives found for subset {subset!r}.", file=sys.stderr)
        return

    archive_cache = cache_dir / "archives" / subset
    archive_cache.mkdir(parents=True, exist_ok=True)

    missing_from_index = {v for v in remaining if v not in tar_index}
    if missing_from_index:
        probe_archives = [rel for rel, _ in archives if rel not in indexed_archives]
        if probe_archives:
            print(
                f"{len(missing_from_index)} video(s) not in index; "
                f"probing up to {len(probe_archives)} unindexed archive(s)..."
            )
        for archive_rel in probe_archives:
            if not missing_from_index:
                break
            _index_archive(
                subset,
                archive_rel,
                cache_dir=cache_dir,
                force=force,
                keep_archives=keep_archives,
                archive_size=archive_sizes.get(archive_rel, 0),
            )
            tar_index = _load_tar_index(cache_dir, subset)
            indexed_archives = _load_indexed_archives(cache_dir, subset)
            missing_from_index = {v for v in remaining if v not in tar_index}

    needed_archives = {tar_index[v] for v in remaining if v in tar_index}
    if not needed_archives:
        missing = sorted(remaining)[:10]
        suffix = "" if len(remaining) <= 10 else f" (and {len(remaining) - 10} more)"
        print(
            f"Warning: {len(remaining)} video(s) were not found in archives for subset {subset!r}: "
            f"{missing}{suffix}",
            file=sys.stderr,
        )
        return

    print(f"Need {len(remaining)} video(s) from {len(needed_archives)} archive(s).")

    for archive_rel in sorted(needed_archives):
        members_for_archive = {v for v in remaining if tar_index.get(v) == archive_rel}
        if not members_for_archive:
            continue

        archive_name = Path(archive_rel).name
        local_tar = archive_cache / archive_name
        if not local_tar.exists() or force:
            print(f"Downloading archive: {archive_rel}")
            _download_file(
                _resolve_url(archive_rel),
                local_tar,
                expected_size=archive_sizes.get(archive_rel) or None,
            )
        else:
            print(f"Using cached archive: {archive_name}")

        print(f"Extracting {len(members_for_archive)} video(s) from {archive_name}")
        _extract_members(local_tar, members_for_archive, videos_dir)
        remaining -= members_for_archive

        if not keep_archives:
            local_tar.unlink(missing_ok=True)

    if remaining:
        missing = sorted(remaining)[:10]
        suffix = "" if len(remaining) <= 10 else f" (and {len(remaining) - 10} more)"
        print(
            f"Warning: {len(remaining)} video(s) were not found in archives for subset {subset!r}: "
            f"{missing}{suffix}",
            file=sys.stderr,
        )


def _download_subset(
  subset: str,
  split: str | None,
  *,
  out_dir: Path,
  cache_dir: Path,
  num_samples: int | None,
  seed: int | None,
  skip_videos: bool,
  force: bool,
  keep_archives: bool,
  dry_run: bool,
) -> None:
    annotation_map = _annotation_paths_by_split(subset)
    if not annotation_map:
        print(f"Skip subset with no annotations: {subset}")
        return

    splits = [split] if split else sorted(annotation_map)
    for split_name in splits:
        if split_name not in annotation_map:
            raise RuntimeError(
                f"Split {split_name!r} not available for subset {subset!r}. "
                f"Choices: {', '.join(sorted(annotation_map))}"
            )

        rel_path = annotation_map[split_name]
        archives = _video_archives(subset)
        print(f"\nSubset: {subset}")
        print(f"Split: {split_name}")
        print(f"Annotations: {rel_path}")
        print(f"Video archives: {len(archives)}")

        if dry_run:
            rows = _load_annotations(rel_path, cache_dir)
            selected = _subsample_rows(rows, num_samples, seed=seed) if num_samples else rows
            videos = _video_paths(selected)
            print(f"  rows: {len(rows)} total, {len(selected)} selected, {len(videos)} unique video(s)")
            for archive_rel, size in archives:
                print(f"  - {archive_rel} ({size / (1024**3):.2f} GiB)")
            continue

        rows = _load_annotations(rel_path, cache_dir)
        selected = _subsample_rows(rows, num_samples, seed=seed) if num_samples else rows
        videos = _video_paths(selected)

        ann_dir = out_dir / "annotations" / subset
        ann_dir.mkdir(parents=True, exist_ok=True)
        suffix = f"_n{len(selected)}" if num_samples else ""
        ann_out = ann_dir / f"{split_name}{suffix}.json"
        with ann_out.open("w", encoding="utf-8") as f:
            json.dump(selected, f, indent=2)
        print(f"Wrote {len(selected)} annotation row(s) to {ann_out}")

        if skip_videos:
            print("Skipping video download (--skip-videos).")
            continue

        _download_videos_for_paths(
            subset,
            videos,
            out_dir=out_dir,
            cache_dir=cache_dir,
            force=force,
            keep_archives=keep_archives,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download LLaVA-Video-178K annotations and videos from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help=f"Destination root (default: {_default_output_dir()})",
    )
    parser.add_argument(
        "--subset",
        default="0_30_s_academic_v0_1",
        help="Dataset subset/config name (default: 0_30_s_academic_v0_1). Ignored with --all-subsets.",
    )
    parser.add_argument(
        "--split",
        choices=["caption", "open_ended", "multi_choice"],
        default="caption",
        help="Annotation split to download (default: caption). With --all-subsets, each subset uses its available splits.",
    )
    parser.add_argument(
        "-n",
        "--num-samples",
        type=int,
        default=None,
        metavar="N",
        help="Download only a random subsample of N annotation rows (and their videos).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for subsampling (default: 0).",
    )
    parser.add_argument(
        "--all-subsets",
        action="store_true",
        help="Download all subsets (full dataset; very large).",
    )
    parser.add_argument(
        "--skip-videos",
        action="store_true",
        help="Download annotations only.",
    )
    parser.add_argument(
        "--build-video-index",
        action="store_true",
        help="Download and index all video archives for the selected subset(s), then exit.",
    )
    parser.add_argument(
        "--keep-archives",
        action="store_true",
        help="Keep downloaded .tar.gz archives in the cache after extraction.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even when outputs already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned downloads without writing videos.",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.expanduser().resolve()
    cache_dir = out_dir / ".cache"

    print(f"Dataset: {REPO_ID}")
    print(f"Destination: {out_dir}")
    if args.num_samples is not None:
        print(f"Subsample: n={args.num_samples}, seed={args.seed}")
    print()

    if args.all_subsets:
        subsets = _list_subset_names()
        # Prompt assets are not video subsets.
        subsets = [s for s in subsets if s not in {"gpt4o_caption_prompt", "gpt4o_qa_prompt"}]
    else:
        subsets = [args.subset]
    split = args.split

    for subset in subsets:
        try:
            if args.build_video_index:
                if args.dry_run:
                    archives = _video_archives(subset)
                    print(f"\nSubset: {subset}")
                    print(f"Would index {len(archives)} archive(s).")
                    for archive_rel, size in archives:
                        print(f"  - {archive_rel} ({size / (1024**3):.2f} GiB)")
                    continue
                _build_video_index(
                    subset,
                    cache_dir=cache_dir,
                    force=args.force,
                    keep_archives=args.keep_archives,
                )
                continue

            _download_subset(
                subset,
                split,
                out_dir=out_dir,
                cache_dir=cache_dir,
                num_samples=args.num_samples,
                seed=args.seed,
                skip_videos=args.skip_videos,
                force=args.force,
                keep_archives=args.keep_archives,
                dry_run=args.dry_run,
            )
        except RuntimeError as e:
            print(f"Error for subset {subset}: {e}", file=sys.stderr)
            return 1

    print("\nDone.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
