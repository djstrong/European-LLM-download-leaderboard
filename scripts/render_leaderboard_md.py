#!/usr/bin/env python3
"""Render output/leaderboard.json as a Markdown table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from leaderboard_common import die, load_json

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "output" / "leaderboard.json"
DEFAULT_ORGS_INPUT = REPO_ROOT / "output" / "leaderboard_orgs.json"
DEFAULT_OUTPUT = REPO_ROOT / "README.md"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Leaderboard JSON path")
    p.add_argument(
        "--orgs-input",
        type=Path,
        default=DEFAULT_ORGS_INPUT,
        help="Organization leaderboard JSON path (optional; section skipped if missing)",
    )
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Markdown output path")
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional max number of rows to include (default: all)",
    )
    return p.parse_args()


def fmt_int(value: int | None) -> str:
    if value is None:
        return "—"
    return f"{value:,}"


def fmt_params(value: int | None) -> str:
    if value is None:
        return "—"
    if value >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2f}B"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    return fmt_int(value)


def fmt_delta(value: int | None) -> str:
    if value is None:
        return "—"
    if value > 0:
        return f"+{value:,}"
    return f"{value:,}"


def fmt_momentum(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def escape_cell(text: str | None) -> str:
    if text is None:
        return "—"
    return str(text).replace("|", "\\|").replace("\n", " ")


def hf_org_link(org: str | None) -> str:
    if not org:
        return "—"
    return f"[{escape_cell(org)}](https://huggingface.co/{org})"


def render_org_table(org_rows: list[dict]) -> list[str]:
    lines = [
        "| Rank | Organization | Developer | Country | Downloads (30d) | All-time | Models | Momentum |",
        "| ---: | --- | --- | :---: | ---: | ---: | ---: | ---: |",
    ]
    for row in org_rows:
        incomplete = "" if row.get("complete", True) else " \\*"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank", "")),
                    hf_org_link(row.get("official_org")) + incomplete,
                    escape_cell(row.get("developer")),
                    escape_cell(row.get("country")),
                    fmt_int(row.get("total_downloads_30d")),
                    fmt_int(row.get("total_downloads_all_time")),
                    str(row.get("model_count", 0)),
                    fmt_momentum(row.get("momentum_score")),
                ]
            )
            + " |"
        )
    return lines


def render_top_model_table(org_rows: list[dict]) -> list[str]:
    ranked = sorted(org_rows, key=lambda r: r.get("rank_by_top_model", 10**9))
    lines = [
        "| Rank | Organization | Best model | Downloads (30d) | All-time | Overall model rank |",
        "| ---: | --- | --- | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank_by_top_model", "")),
                    hf_org_link(row.get("official_org")),
                    escape_cell(row.get("top_model_display_name")),
                    fmt_int(row.get("top_model_downloads_30d")),
                    fmt_int(row.get("top_model_downloads_all_time")),
                    str(row.get("top_model_overall_rank", "")),
                ]
            )
            + " |"
        )
    return lines


def render_momentum_table(org_rows: list[dict]) -> list[str]:
    ranked = sorted(org_rows, key=lambda r: r.get("momentum_rank", 10**9))
    lines = [
        "| Momentum rank | Organization | Momentum | Downloads (30d) | All-time | Downloads rank |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in ranked:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("momentum_rank", "")),
                    hf_org_link(row.get("official_org")),
                    fmt_momentum(row.get("momentum_score")),
                    fmt_int(row.get("total_downloads_30d")),
                    fmt_int(row.get("total_downloads_all_time")),
                    str(row.get("rank", "")),
                ]
            )
            + " |"
        )
    return lines


def render_markdown(
    leaderboard: dict,
    *,
    org_leaderboard: dict | None = None,
    limit: int | None = None,
) -> str:
    rows = list(leaderboard.get("rows") or [])
    if limit is not None:
        rows = rows[:limit]

    momentum_smoothing = leaderboard.get("momentum_smoothing")

    lines = [
        "# European LLM Hugging Face download leaderboard",
        "",
        f"- **Snapshot date:** {leaderboard.get('snapshot_date', '—')}",
        f"- **Generated at:** {leaderboard.get('generated_at', '—')}",
        f"- **Ranking metric:** `{leaderboard.get('ranking_metric', 'total_downloads_30d')}` "
        "(last 30 days, summed across official format variants)",
        f"- **Rows:** {len(rows)}"
        + (f" (of {len(leaderboard.get('rows') or [])})" if limit is not None else ""),
        "",
        "| Rank | Model | Country | Developer | Org | Downloads (30d) | Δ 30d | All-time | Momentum | Params | Repos |",
        "| ---: | --- | :---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for row in rows:
        incomplete = "" if row.get("complete", True) else " \\*"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("rank", "")),
                    escape_cell(row.get("display_name")) + incomplete,
                    escape_cell(row.get("country")),
                    escape_cell(row.get("developer")),
                    hf_org_link(row.get("official_org")),
                    fmt_int(row.get("total_downloads_30d")),
                    fmt_delta(row.get("delta_downloads_30d")),
                    fmt_int(row.get("total_downloads_all_time")),
                    fmt_momentum(row.get("momentum_score")),
                    fmt_params(row.get("parameters")),
                    str(row.get("repo_count", 0)),
                ]
            )
            + " |"
        )

    if any(not row.get("complete", True) for row in rows):
        lines.extend(
            [
                "",
                "\\* Row marked incomplete: one or more member repos failed during the last fetch.",
            ]
        )

    if org_leaderboard and org_leaderboard.get("rows"):
        org_rows = org_leaderboard["rows"]
        lines.extend(
            [
                "",
                "## Organizations",
                "",
                "Same downloads, aggregated across every model version an organization publishes.",
                "",
            ]
        )
        lines.extend(render_org_table(org_rows))

        lines.extend(
            [
                "",
                "### By best single model",
                "",
                "Ranked by each organization's single highest-downloading model "
                "(no summing across versions) — who has the biggest individual hit.",
                "",
            ]
        )
        lines.extend(render_top_model_table(org_rows))

        lines.extend(
            [
                "",
                "### By momentum",
                "",
                "Sorted by momentum (highest first). Momentum highlights orgs whose recent downloads "
                "are large *relative to their lifetime total* — i.e. accelerating adoption, not legacy "
                "long-tail traffic from an old release.",
                "",
            ]
        )
        lines.extend(render_momentum_table(org_rows))

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Downloads are Hugging Face Hub **rolling last-30-day** counts, aggregated over official "
            "repos matched by config regexes (GGUF / MLX / FP8 / etc. when published by the same org).",
            "- **Params** is the max reported parameter count among member repos "
            "(`safetensors.total`, else `gguf.total`).",
            "- **Momentum** = `downloads_30d / (downloads_all_time + "
            + (f"{momentum_smoothing:,.0f})`" if momentum_smoothing is not None else "C)`")
            + " — a recency ratio damped by a smoothing constant so low-volume/brand-new "
            "rows can't look like they have outsized momentum from a handful of downloads. "
            "Higher = more of its lifetime downloads happened in the last 30 days.",
            "- Machine-readable data: [`output/leaderboard.json`](output/leaderboard.json), "
            "[`output/leaderboard.csv`](output/leaderboard.csv), "
            "[`output/leaderboard_orgs.json`](output/leaderboard_orgs.json), "
            "[`output/leaderboard_orgs.csv`](output/leaderboard_orgs.csv).",
            "- How this is built / how to add models: [`DEVELOPMENT.md`](DEVELOPMENT.md).",
            "- This file is regenerated by CI (`fetch` workflow) via `scripts/render_leaderboard_md.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        die(f"Leaderboard JSON not found: {args.input}. Run fetch_download_stats.py first.")
    if args.limit is not None and args.limit < 1:
        die("--limit must be >= 1")

    leaderboard = load_json(args.input)
    org_leaderboard = load_json(args.orgs_input) if args.orgs_input.is_file() else None
    markdown = render_markdown(leaderboard, org_leaderboard=org_leaderboard, limit=args.limit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown, encoding="utf-8")
    print(f"Wrote {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
