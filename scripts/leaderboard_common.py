"""Shared helpers for European LLM Hugging Face leaderboard scripts."""

from __future__ import annotations

import hashlib
import json
import re
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_CONFIG_KEYS = {
    "version",
    "description",
    "match_target",
    "regex_flavor",
    "official_only",
    "regex_scope",
    "downloads_metric",
    "downloads_api_note",
    "deduplication",
    "organizations",
    "models",
}

REQUIRED_ROW_KEYS = {"display_name", "official_org", "regex"}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return "sha256:" + hashlib.sha256(data).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
        f.write("\n")


def validate_config(config: dict[str, Any], *, warn: bool = True) -> list[str]:
    """Validate leaderboard config. Returns list of error messages (empty if ok)."""
    errors: list[str] = []

    if not isinstance(config, dict):
        return ["config root must be a JSON object"]

    if warn:
        unknown = set(config) - KNOWN_CONFIG_KEYS
        for key in sorted(unknown):
            warnings.warn(f"Unknown config key ignored: {key}", stacklevel=2)

    if "version" not in config:
        errors.append("missing required field: version")
    elif not isinstance(config["version"], int):
        errors.append("version must be an integer")

    models = config.get("models")
    if not isinstance(models, dict) or not models:
        errors.append("models must be a non-empty object")
        return errors

    orgs = config.get("organizations")
    if orgs is not None and (
        not isinstance(orgs, list) or not all(isinstance(o, str) and o for o in orgs)
    ):
        errors.append("organizations must be a list of non-empty strings when present")

    for row_id, row in models.items():
        if not isinstance(row, dict):
            errors.append(f"models.{row_id}: must be an object")
            continue
        missing = REQUIRED_ROW_KEYS - set(row)
        if missing:
            errors.append(f"models.{row_id}: missing keys {sorted(missing)}")
            continue
        if not isinstance(row["display_name"], str) or not row["display_name"]:
            errors.append(f"models.{row_id}: display_name must be a non-empty string")
        if not isinstance(row["official_org"], str) or not row["official_org"]:
            errors.append(f"models.{row_id}: official_org must be a non-empty string")
        if row.get("active") is not None and not isinstance(row["active"], bool):
            errors.append(f"models.{row_id}: active must be a boolean when present")
        regexes = row["regex"]
        if not isinstance(regexes, list) or not regexes:
            errors.append(f"models.{row_id}: regex must be a non-empty list")
            continue
        for i, pattern in enumerate(regexes):
            if not isinstance(pattern, str) or not pattern:
                errors.append(f"models.{row_id}: regex[{i}] must be a non-empty string")
                continue
            try:
                re.compile(pattern)
            except re.error as exc:
                errors.append(f"models.{row_id}: regex[{i}] invalid: {exc}")

    return errors


def active_models(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row_id, row in config["models"].items():
        if row.get("active", True) is False:
            continue
        result[row_id] = row
    return result


def organizations_to_scan(config: dict[str, Any], models: dict[str, dict[str, Any]]) -> list[str]:
    orgs: set[str] = set()
    for row in models.values():
        orgs.add(row["official_org"])
    extra = config.get("organizations") or []
    for org in extra:
        orgs.add(org)
    return sorted(orgs)


def compile_row_patterns(
    models: dict[str, dict[str, Any]],
) -> list[tuple[str, dict[str, Any], list[re.Pattern[str]]]]:
    compiled: list[tuple[str, dict[str, Any], list[re.Pattern[str]]]] = []
    for row_id, row in models.items():
        patterns = [re.compile(p) for p in row["regex"]]
        compiled.append((row_id, row, patterns))
    return compiled


def assign_repo_to_row(
    repo_id: str,
    compiled_rows: list[tuple[str, dict[str, Any], list[re.Pattern[str]]]],
) -> str | None:
    for row_id, _row, patterns in compiled_rows:
        for pattern in patterns:
            if pattern.match(repo_id):
                return row_id
    return None


def manifest_content_payload(rows: dict[str, Any], all_repo_ids: list[str]) -> dict[str, Any]:
    """Stable subset used for manifest_content_hash."""
    return {
        "rows": {
            row_id: {
                "display_name": row.get("display_name"),
                "repos": row.get("repos", []),
            }
            for row_id, row in rows.items()
        },
        "all_repo_ids": all_repo_ids,
    }


def compute_manifest_content_hash(rows: dict[str, Any], all_repo_ids: list[str]) -> str:
    payload = manifest_content_payload(rows, all_repo_ids)
    return sha256_hex(json.dumps(payload, sort_keys=True, separators=(",", ":")))


def die(message: str, code: int = 1) -> None:
    print(message, file=sys.stderr)
    raise SystemExit(code)
