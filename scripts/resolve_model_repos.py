#!/usr/bin/env python3
"""Resolve Hugging Face repo IDs for leaderboard rows via org catalogs + regexes."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

from huggingface_hub import HfApi

from leaderboard_common import (
    active_models,
    assign_repo_to_row,
    compile_row_patterns,
    compute_manifest_content_hash,
    die,
    load_json,
    organizations_to_scan,
    sha256_hex,
    utc_now_iso,
    validate_config,
    write_json,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "european-llm-hf-download-leaderboard.json"
DEFAULT_OUT = REPO_ROOT / "data" / "manifest" / "latest.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Leaderboard config JSON")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Manifest output path")
    p.add_argument(
        "--history-dir",
        type=Path,
        default=REPO_ROOT / "data" / "manifest" / "history",
        help="Directory for previous manifest snapshots when content hash changes",
    )
    p.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token (or HF_TOKEN env). Optional for public orgs.",
    )
    p.add_argument(
        "--validate-config",
        action="store_true",
        help="Validate config and exit without Hub calls",
    )
    p.add_argument(
        "--audit",
        action="store_true",
        help="Print unmatched repos and empty rows to stderr",
    )
    p.add_argument(
        "--diff",
        action="store_true",
        help="Print a short diff vs existing --out manifest if present",
    )
    return p.parse_args()


def list_org_model_ids(api: HfApi, org: str) -> list[str]:
    ids: list[str] = []
    for model in api.list_models(author=org):
        model_id = getattr(model, "id", None) or getattr(model, "modelId", None)
        if model_id:
            ids.append(model_id)
    return sorted(ids)


def build_manifest(
    config: dict,
    config_path: Path,
    org_repos: dict[str, list[str]],
) -> dict:
    models = active_models(config)
    compiled = compile_row_patterns(models)
    orgs = organizations_to_scan(config, models)

    rows: dict[str, dict] = {}
    for row_id, row in models.items():
        rows[row_id] = {
            "display_name": row["display_name"],
            "version": row.get("version"),
            "country": row.get("country"),
            "developer": row.get("developer"),
            "official_org": row["official_org"],
            "repos": [],
        }

    assigned: set[str] = set()
    unmatched_by_org: dict[str, list[str]] = defaultdict(list)

    for org in orgs:
        for repo_id in org_repos.get(org, []):
            row_id = assign_repo_to_row(repo_id, compiled)
            if row_id is None:
                unmatched_by_org[org].append(repo_id)
                continue
            if repo_id in assigned:
                continue
            assigned.add(repo_id)
            rows[row_id]["repos"].append(repo_id)

    for row in rows.values():
        row["repos"] = sorted(row["repos"])

    all_repo_ids = sorted(assigned)
    unmatched_sorted = {org: sorted(repos) for org, repos in sorted(unmatched_by_org.items())}

    config_bytes = config_path.read_bytes()
    return {
        "generated_at": utc_now_iso(),
        "config_version": config.get("version"),
        "config_path": str(config_path.name),
        "config_content_hash": sha256_hex(config_bytes),
        "organizations_scanned": orgs,
        "manifest_content_hash": compute_manifest_content_hash(rows, all_repo_ids),
        "rows": rows,
        "unmatched_repos_by_org": unmatched_sorted,
        "all_repo_ids": all_repo_ids,
    }


def print_audit(manifest: dict) -> None:
    empty = [rid for rid, row in manifest["rows"].items() if not row["repos"]]
    if empty:
        print(f"Rows with zero matched repos ({len(empty)}):", file=sys.stderr)
        for rid in empty:
            print(f"  - {rid}", file=sys.stderr)
    unmatched = manifest.get("unmatched_repos_by_org") or {}
    total = sum(len(v) for v in unmatched.values())
    print(f"Unmatched org repos ({total}):", file=sys.stderr)
    for org, repos in unmatched.items():
        print(f"  {org}: {len(repos)}", file=sys.stderr)
        for repo_id in repos:
            print(f"    - {repo_id}", file=sys.stderr)


def print_diff(old: dict, new: dict) -> None:
    old_repos = set(old.get("all_repo_ids") or [])
    new_repos = set(new.get("all_repo_ids") or [])
    added = sorted(new_repos - old_repos)
    removed = sorted(old_repos - new_repos)
    print(f"Manifest diff vs previous:", file=sys.stderr)
    print(f"  +{len(added)} repos, -{len(removed)} repos", file=sys.stderr)
    for repo_id in added[:50]:
        print(f"  + {repo_id}", file=sys.stderr)
    if len(added) > 50:
        print(f"  + … ({len(added) - 50} more)", file=sys.stderr)
    for repo_id in removed[:50]:
        print(f"  - {repo_id}", file=sys.stderr)
    if len(removed) > 50:
        print(f"  - … ({len(removed) - 50} more)", file=sys.stderr)

    old_hash = old.get("manifest_content_hash")
    new_hash = new.get("manifest_content_hash")
    if old_hash != new_hash:
        print(f"  hash: {old_hash} -> {new_hash}", file=sys.stderr)


def maybe_write_history(manifest: dict, out_path: Path, history_dir: Path) -> None:
    if not out_path.exists():
        return
    try:
        previous = load_json(out_path)
    except (OSError, json.JSONDecodeError):
        return
    if previous.get("manifest_content_hash") == manifest["manifest_content_hash"]:
        return
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = previous.get("generated_at", utc_now_iso()).replace(":", "").replace("-", "")
    hist_path = history_dir / f"{stamp}.json"
    write_json(hist_path, previous)
    print(f"Wrote previous manifest to {hist_path}", file=sys.stderr)


def main() -> None:
    args = parse_args()
    if not args.config.is_file():
        die(f"Config not found: {args.config}")

    config = load_json(args.config)
    errors = validate_config(config)
    if errors:
        for err in errors:
            print(f"config error: {err}", file=sys.stderr)
        raise SystemExit(1)

    if args.validate_config:
        print(f"Config OK: {args.config} ({len(active_models(config))} active rows)")
        return

    models = active_models(config)
    orgs = organizations_to_scan(config, models)
    api = HfApi(token=args.token)

    org_repos: dict[str, list[str]] = {}
    for org in orgs:
        print(f"Listing models for org: {org}", file=sys.stderr)
        org_repos[org] = list_org_model_ids(api, org)
        print(f"  {len(org_repos[org])} repos", file=sys.stderr)

    manifest = build_manifest(config, args.config, org_repos)

    if args.diff and args.out.exists():
        try:
            print_diff(load_json(args.out), manifest)
        except (OSError, json.JSONDecodeError) as exc:
            print(f"Could not diff previous manifest: {exc}", file=sys.stderr)

    if args.audit:
        print_audit(manifest)

    maybe_write_history(manifest, args.out, args.history_dir)
    write_json(args.out, manifest)
    print(
        f"Wrote manifest: {args.out} "
        f"({len(manifest['all_repo_ids'])} repos, {len(manifest['rows'])} rows, "
        f"hash={manifest['manifest_content_hash']})",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
