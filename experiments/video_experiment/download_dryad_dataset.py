#!/usr/bin/env python3
"""
Download the Chicago Motion / natural-movie Dryad package referenced in the repository README.

Source: https://doi.org/10.5061/dryad.4qrfj6qm8 (see README.md, section "Natural movie stimulus data (Dryad)").

Files are written under the repository ``data/`` directory by default (``data/`` is gitignored).

Dryad requires a bearer token for file downloads. Create a Dryad account (ORCID), then add an API
application under *My account* and set environment variables::

    export DRYAD_CLIENT_ID="..."
    export DRYAD_CLIENT_SECRET="..."

Documentation: https://github.com/datadryad/dryad-app/blob/main/documentation/apis/api_accounts.md

Example::

    uv run python experiments/video_experiment/download_dryad_dataset.py
    uv run python experiments/video_experiment/download_dryad_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

DATASET_DOI = "doi:10.5061/dryad.4qrfj6qm8"
DATASET_API_PATH = "/api/v2/datasets/" + urllib.parse.quote(DATASET_DOI, safe="")


class _DryadDownloadRedirectHandler(urllib.request.HTTPRedirectHandler):
    """
    Dryad file downloads redirect to presigned object-store URLs. urllib would forward the
    ``Authorization: Bearer`` header on redirect; S3 rejects that when query-string signing is used.
    """

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp,
        code: int,
        msg: str,
        headers,
        newurl: str,
    ) -> urllib.request.Request | None:
        new_req = super().redirect_request(req, fp, code, msg, headers, newurl)
        if new_req is None:
            return None
        old_host = urllib.parse.urlparse(req.full_url).hostname
        new_host = urllib.parse.urlparse(newurl).hostname
        if old_host != new_host:
            return urllib.request.Request(newurl, unverifiable=True)
        return new_req


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_output_dir() -> Path:
    return _repo_root() / "data" / "dryad_chicago_natural_movies"


def _join_url(base: str, href: str) -> str:
    if href.startswith("http://") or href.startswith("https://"):
        return href
    if not base.endswith("/") and not href.startswith("/"):
        return f"{base}/{href}"
    if base.endswith("/") and href.startswith("/"):
        return f"{base.rstrip('/')}{href}"
    return f"{base}{href}"


def _request_json(url: str, *, method: str = "GET", data: bytes | None = None, headers: dict[str, str] | None = None) -> object:
    h = {"Accept": "application/json"}
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = resp.read().decode()
    except urllib.error.HTTPError as e:
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} for {url}: {detail[:500]}") from e
    return json.loads(body)


def _oauth_token(domain: str, client_id: str, client_secret: str) -> str:
    token_url = f"https://{domain}/oauth/token"
    form = urllib.parse.urlencode(
        {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
    ).encode()
    headers = {"Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"}
    payload = _request_json(token_url, method="POST", data=form, headers=headers)
    if not isinstance(payload, dict) or "access_token" not in payload:
        raise RuntimeError(f"Unexpected token response: {payload!r}")
    token = payload["access_token"]
    if not isinstance(token, str):
        raise RuntimeError("access_token is not a string")
    return token


def _hal_get_link(links: object, rel: str) -> str | None:
    if not isinstance(links, dict):
        return None
    block = links.get(rel)
    if isinstance(block, dict):
        href = block.get("href")
        if isinstance(href, str):
            return href
    return None


def _list_dataset_files(domain: str) -> list[dict[str, object]]:
    base = f"https://{domain}"
    ds_url = _join_url(base, DATASET_API_PATH)
    ds = _request_json(ds_url)
    if not isinstance(ds, dict):
        raise RuntimeError("Dataset response is not a JSON object")

    version_href = _hal_get_link(ds.get("_links"), "stash:version")
    if not version_href:
        raise RuntimeError("Could not find stash:version link on dataset resource")

    files_url = _join_url(base, version_href.rstrip("/") + "/files")
    listing = _request_json(files_url)
    if not isinstance(listing, dict):
        raise RuntimeError("Files listing is not a JSON object")

    embedded = listing.get("_embedded")
    if not isinstance(embedded, dict):
        raise RuntimeError("Files listing has no _embedded object")

    files = embedded.get("stash:files")
    if not isinstance(files, list):
        raise RuntimeError("Could not find stash:files array in listing")

    out: list[dict[str, object]] = []
    for item in files:
        if isinstance(item, dict):
            out.append(item)
    return out


def _download_file(
    url: str,
    dest: Path,
    token: str,
    *,
    expected_size: int | None,
    chunk: int = 1024 * 1024,
) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    headers = {"Authorization": f"Bearer {token}"}
    req = urllib.request.Request(url, headers=headers, method="GET")
    opener = urllib.request.build_opener(_DryadDownloadRedirectHandler())
    try:
        with opener.open(req, timeout=600) as resp, tmp.open("wb") as f:
            cl = resp.headers.get("Content-Length")
            total = int(cl) if cl and cl.isdigit() else expected_size
            done = 0
            next_report = 100 * 1024 * 1024
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
        tmp.unlink(missing_ok=True)
        detail = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code} downloading {url}: {detail[:500]}") from e

    got = tmp.stat().st_size
    if expected_size is not None and got != expected_size:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Size mismatch for {dest.name}: got {got} bytes, expected {expected_size}")

    tmp.replace(dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download Dryad Chicago natural-movie dataset (see README.md).")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_default_output_dir(),
        help=f"Directory to write files (default: {_default_output_dir()})",
    )
    parser.add_argument(
        "--domain",
        default=os.environ.get("DRYAD_OAUTH_DOMAIN", "datadryad.org"),
        help="Dryad host (default: datadryad.org or DRYAD_OAUTH_DOMAIN)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List remote files and exit without downloading.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even when an existing file has the expected size.",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir.expanduser().resolve()
    print(f"Dataset: {DATASET_DOI}")
    print(f"Destination: {out_dir}")
    print(f"API host: https://{args.domain}")
    print()

    client_id = os.environ.get("DRYAD_CLIENT_ID", "").strip()
    client_secret = os.environ.get("DRYAD_CLIENT_SECRET", "").strip()
    if not args.dry_run and (not client_id or not client_secret):
        print(
            "Dryad file downloads require OAuth client credentials.\n"
            "Set DRYAD_CLIENT_ID and DRYAD_CLIENT_SECRET (see script docstring and Dryad API docs), "
            "or run with --dry-run to list files only.",
            file=sys.stderr,
        )
        return 1

    files_meta = _list_dataset_files(args.domain)
    print(f"Found {len(files_meta)} file(s) on Dryad.")

    if args.dry_run:
        for meta in files_meta:
            path = meta.get("path")
            size = meta.get("size")
            print(f"  - {path} ({size} bytes)")
        return 0

    token = _oauth_token(args.domain, client_id, client_secret)
    base = f"https://{args.domain}"

    for meta in files_meta:
        name = meta.get("path")
        if not isinstance(name, str) or not name:
            continue
        size = meta.get("size")
        expected = int(size) if isinstance(size, int) else None

        links = meta.get("_links") if isinstance(meta.get("_links"), dict) else {}
        dl_href = _hal_get_link(links, "stash:download")
        if not dl_href:
            print(f"Skip (no download link): {name}")
            continue

        dest = out_dir / name
        if dest.exists() and expected is not None and not args.force:
            if dest.stat().st_size == expected:
                print(f"OK (already present): {name}")
                continue

        url = _join_url(base, dl_href)
        print(f"Downloading: {name}")
        _download_file(url, dest, token, expected_size=expected)
        print(f"  wrote {dest}")

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
