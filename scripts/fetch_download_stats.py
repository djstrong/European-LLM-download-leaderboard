#!/usr/bin/env python3
"""Fetch download stats and parameter counts for manifest repos; write history + leaderboard."""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi
from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

from leaderboard_common import die, load_json, utc_now_iso, write_json

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = REPO_ROOT / "data" / "manifest" / "latest.json"
DEFAULT_METRICS_DIR = REPO_ROOT / "data" / "metrics"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "output"
EXPAND = ["downloads", "downloadsAllTime", "safetensors", "gguf"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--metrics-dir", type=Path, default=DEFAULT_METRICS_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument(
        "--date",
        default=None,
        help="Snapshot date YYYY-MM-DD (UTC). Default: today UTC.",
    )
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=3, help="Retries after first failure")
    p.add_argument("--retry-base-seconds", type=float, default=1.0)
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write outputs even if some repos fail; exit 0",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or HF_TOKEN env). Recommended to reduce rate limits.",
    )
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Log progress every N completed repos",
    )
    return p.parse_args()


def snapshot_date_utc(value: str | None) -> str:
    if value:
        datetime.strptime(value, "%Y-%m-%d")
        return value
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def extract_parameters(info: Any) -> tuple[int | None, str | None]:
    safetensors = getattr(info, "safetensors", None)
    if safetensors is not None:
        total = getattr(safetensors, "total", None)
        if total is None and isinstance(safetensors, dict):
            total = safetensors.get("total")
        if total is not None:
            return int(total), "safetensors"

    gguf = getattr(info, "gguf", None)
    if gguf is not None:
        total = getattr(gguf, "total", None)
        if total is None and isinstance(gguf, dict):
            total = gguf.get("total")
        if total is not None:
            return int(total), "gguf"

    return None, None


def is_permanent_error(exc: BaseException) -> bool:
    if isinstance(exc, RepositoryNotFoundError):
        return True
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc.response, "status_code", None) if getattr(exc, "response", None) else None
        if status in {401, 403, 404}:
            return True
    return False


def retry_after_seconds(exc: BaseException, attempt: int, base: float) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        header = response.headers.get("Retry-After") or response.headers.get("retry-after")
        if header:
            try:
                return float(header)
            except ValueError:
                pass
    delay = base * (2**attempt)
    jitter = random.uniform(0, delay * 0.25)
    return delay + jitter


def fetch_one(
    api: HfApi,
    repo_id: str,
    *,
    max_retries: int,
    retry_base_seconds: float,
) -> tuple[str, dict[str, Any] | None, dict[str, Any] | None]:
    attempts = 0
    last_error = ""
    while True:
        attempts += 1
        try:
            info = api.model_info(repo_id, expand=EXPAND)
            parameters, parameters_source = extract_parameters(info)
            downloads_all_time = getattr(info, "downloads_all_time", None)
            if downloads_all_time is None:
                downloads_all_time = getattr(info, "downloadsAllTime", None)
            result = {
                "downloads_30d": int(getattr(info, "downloads", 0) or 0),
                "downloads_all_time": int(downloads_all_time or 0),
                "parameters": parameters,
                "parameters_source": parameters_source,
            }
            return repo_id, result, None
        except Exception as exc:  # noqa: BLE001 - collect per-repo failures
            last_error = f"{type(exc).__name__}: {exc}"
            if is_permanent_error(exc) or attempts > max_retries + 1:
                return (
                    repo_id,
                    None,
                    {"repo_id": repo_id, "error": last_error, "attempts": attempts},
                )
            time.sleep(retry_after_seconds(exc, attempts - 1, retry_base_seconds))


def fetch_all(
    api: HfApi,
    repo_ids: list[str],
    *,
    concurrency: int,
    max_retries: int,
    retry_base_seconds: float,
    quiet: bool,
    progress_every: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    repos: dict[str, dict[str, Any]] = {}
    errors: list[dict[str, Any]] = []
    done = 0
    total = len(repo_ids)

    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as pool:
        futures = {
            pool.submit(
                fetch_one,
                api,
                repo_id,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
            ): repo_id
            for repo_id in repo_ids
        }
        for future in as_completed(futures):
            repo_id, result, error = future.result()
            if result is not None:
                repos[repo_id] = result
            if error is not None:
                errors.append(error)
            done += 1
            if not quiet and progress_every > 0 and (done % progress_every == 0 or done == total):
                print(f"Fetched {done}/{total} repos ({len(errors)} errors)", file=sys.stderr)

    errors.sort(key=lambda e: e["repo_id"])
    return repos, errors


def aggregate_rows(
    manifest: dict,
    repos: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    rows_out: dict[str, dict[str, Any]] = {}

    for row_id, row in manifest["rows"].items():
        member_repos = row.get("repos") or []
        total_30d = 0
        total_all = 0
        missing = 0
        best_params: int | None = None
        best_repo: str | None = None

        for repo_id in member_repos:
            stats = repos.get(repo_id)
            if stats is None:
                missing += 1
                continue
            total_30d += stats["downloads_30d"]
            total_all += stats["downloads_all_time"]
            params = stats.get("parameters")
            if params is not None and (best_params is None or params > best_params):
                best_params = params
                best_repo = repo_id

        rows_out[row_id] = {
            "total_downloads_30d": total_30d,
            "total_downloads_all_time": total_all,
            "repo_count": len(member_repos),
            "fetched_repo_count": len(member_repos) - missing,
            "missing_repo_count": missing,
            "complete": missing == 0,
            "parameters": best_params,
            "parameters_source_repo": best_repo,
        }
    return rows_out


def previous_snapshot_path(snapshots_dir: Path, snapshot_date: str) -> Path | None:
    dates = sorted(
        p.stem
        for p in snapshots_dir.glob("????-??-??.json")
        if p.stem != snapshot_date
    )
    if not dates:
        return None
    # Prefer latest date strictly before current when possible
    before = [d for d in dates if d < snapshot_date]
    chosen = before[-1] if before else dates[-1]
    return snapshots_dir / f"{chosen}.json"


def build_leaderboard(
    manifest: dict,
    snapshot: dict,
    previous: dict | None,
) -> dict[str, Any]:
    prev_rows = (previous or {}).get("rows") or {}
    entries = []
    for row_id, agg in snapshot["rows"].items():
        meta = manifest["rows"].get(row_id, {})
        prev = prev_rows.get(row_id) or {}
        prev_30d = prev.get("total_downloads_30d")
        delta = None
        if prev_30d is not None:
            delta = agg["total_downloads_30d"] - prev_30d
        entries.append(
            {
                "row_id": row_id,
                "display_name": meta.get("display_name"),
                "version": meta.get("version"),
                "country": meta.get("country"),
                "developer": meta.get("developer"),
                "official_org": meta.get("official_org"),
                "total_downloads_30d": agg["total_downloads_30d"],
                "total_downloads_all_time": agg["total_downloads_all_time"],
                "parameters": agg.get("parameters"),
                "parameters_source_repo": agg.get("parameters_source_repo"),
                "repo_count": agg["repo_count"],
                "complete": agg["complete"],
                "delta_downloads_30d": delta,
                "repos": meta.get("repos") or [],
            }
        )

    entries.sort(key=lambda e: (-e["total_downloads_30d"], e["row_id"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    return {
        "generated_at": snapshot["fetched_at"],
        "snapshot_date": snapshot["snapshot_date"],
        "manifest_content_hash": snapshot["manifest_content_hash"],
        "ranking_metric": "total_downloads_30d",
        "rows": entries,
    }


def write_leaderboard_csv(path: Path, leaderboard: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "rank",
        "row_id",
        "display_name",
        "version",
        "country",
        "developer",
        "official_org",
        "total_downloads_30d",
        "total_downloads_all_time",
        "parameters",
        "parameters_source_repo",
        "repo_count",
        "complete",
        "delta_downloads_30d",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in leaderboard["rows"]:
            writer.writerow(row)


def append_timeseries(path: Path, snapshot: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    compact_rows = {
        row_id: {
            "downloads_30d": agg["total_downloads_30d"],
            "downloads_all_time": agg["total_downloads_all_time"],
            "repo_count": agg["repo_count"],
            "complete": agg["complete"],
            "parameters": agg.get("parameters"),
        }
        for row_id, agg in snapshot["rows"].items()
    }
    line = {
        "fetched_at": snapshot["fetched_at"],
        "snapshot_date": snapshot["snapshot_date"],
        "manifest_hash": snapshot["manifest_content_hash"],
        "rows": compact_rows,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(line, ensure_ascii=False, separators=(",", ":")) + "\n")


def main() -> None:
    args = parse_args()
    if args.concurrency < 1:
        die("--concurrency must be >= 1")
    if args.max_retries < 0:
        die("--max-retries must be >= 0")
    if not args.manifest.is_file():
        die(f"Manifest not found: {args.manifest}. Run resolve_model_repos.py first.")

    manifest = load_json(args.manifest)
    repo_ids = list(manifest.get("all_repo_ids") or [])
    if not repo_ids:
        die("Manifest has no all_repo_ids")

    snapshot_date = snapshot_date_utc(args.date)
    snapshots_dir = args.metrics_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi(token=args.token)
    if not args.quiet:
        print(
            f"Fetching stats for {len(repo_ids)} repos "
            f"(concurrency={args.concurrency}, max_retries={args.max_retries})",
            file=sys.stderr,
        )

    repos, fetch_errors = fetch_all(
        api,
        repo_ids,
        concurrency=args.concurrency,
        max_retries=args.max_retries,
        retry_base_seconds=args.retry_base_seconds,
        quiet=args.quiet,
        progress_every=args.progress_every,
    )
    rows = aggregate_rows(manifest, repos)

    snapshot = {
        "snapshot_date": snapshot_date,
        "fetched_at": utc_now_iso(),
        "manifest_content_hash": manifest.get("manifest_content_hash"),
        "config_version": manifest.get("config_version"),
        "fetch_concurrency": args.concurrency,
        "repos": dict(sorted(repos.items())),
        "rows": rows,
        "fetch_errors": fetch_errors,
    }

    snapshot_path = snapshots_dir / f"{snapshot_date}.json"
    write_json(snapshot_path, snapshot)
    if not args.quiet:
        print(f"Wrote snapshot: {snapshot_path}", file=sys.stderr)

    timeseries_path = args.metrics_dir / "timeseries.jsonl"
    append_timeseries(timeseries_path, snapshot)
    if not args.quiet:
        print(f"Appended timeseries: {timeseries_path}", file=sys.stderr)

    prev_path = previous_snapshot_path(snapshots_dir, snapshot_date)
    previous = load_json(prev_path) if prev_path and prev_path.is_file() else None
    leaderboard = build_leaderboard(manifest, snapshot, previous)
    out_json = args.output_dir / "leaderboard.json"
    out_csv = args.output_dir / "leaderboard.csv"
    write_json(out_json, leaderboard)
    write_leaderboard_csv(out_csv, leaderboard)
    if not args.quiet:
        print(f"Wrote {out_json} and {out_csv}", file=sys.stderr)
        print(
            f"Done: {len(repos)} ok, {len(fetch_errors)} errors, "
            f"top={leaderboard['rows'][0]['display_name'] if leaderboard['rows'] else 'n/a'}",
            file=sys.stderr,
        )

    if fetch_errors and not args.allow_partial:
        print(
            f"{len(fetch_errors)} repo(s) failed after retries. "
            "Re-run with --allow-partial to exit 0, or fix errors and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
