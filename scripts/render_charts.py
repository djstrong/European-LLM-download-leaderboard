#!/usr/bin/env python3
"""Generate PNG charts from leaderboard JSON."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from leaderboard_common import die, load_json

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "output" / "leaderboard.json"
DEFAULT_ORGS_INPUT = REPO_ROOT / "output" / "leaderboard_orgs.json"
DEFAULT_OUT_DIR = REPO_ROOT / "docs" / "charts"

BAR_COLOR = "#4a6fa5"
FIG_DPI = 150


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--orgs-input", type=Path, default=DEFAULT_ORGS_INPUT)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    p.add_argument(
        "--top-models",
        type=int,
        default=20,
        help="How many models to show on top-N bar charts",
    )
    return p.parse_args()


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close()
    print(f"Wrote {path}", file=sys.stderr)


def subtitle(snapshot_date: str | None) -> str:
    if snapshot_date:
        return f"Snapshot: {snapshot_date} · HF rolling 30-day downloads"
    return "HF rolling 30-day downloads"


def set_title_with_snapshot(ax: plt.Axes, title: str, snapshot_date: str | None) -> None:
    ax.set_title(f"{title}\n{subtitle(snapshot_date)}", loc="left", fontsize=12)


def chart_top_models_30d(rows: list[dict], out_dir: Path, *, top_n: int, snapshot_date: str | None) -> None:
    subset = sorted(rows, key=lambda r: -r["total_downloads_30d"])[:top_n]
    subset.reverse()
    labels = [r["display_name"] or r["row_id"] for r in subset]
    values = [r["total_downloads_30d"] for r in subset]

    fig_h = max(4.0, 0.28 * len(subset) + 1.5)
    fig, ax = plt.subplots(figsize=(10, fig_h))
    ax.barh(labels, values, color=BAR_COLOR)
    ax.set_xlabel("Downloads (last 30 days)")
    set_title_with_snapshot(ax, f"Top {len(subset)} models by 30-day downloads", snapshot_date)
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    save_fig(out_dir / "top-models-30d.png")


def chart_orgs_best_single_model(org_rows: list[dict], out_dir: Path, *, snapshot_date: str | None) -> None:
    ordered = sorted(org_rows, key=lambda r: r.get("top_model_downloads_30d") or 0)
    labels = [
        r.get("top_model_display_name") or r.get("top_model_row_id") or r["official_org"]
        for r in ordered
    ]
    values = [r.get("top_model_downloads_30d") or 0 for r in ordered]

    fig_h = max(4.0, 0.32 * len(ordered) + 1.5)
    fig, ax = plt.subplots(figsize=(11, fig_h))
    ax.barh(labels, values, color=BAR_COLOR)
    ax.set_xlabel("Downloads (last 30 days)")
    set_title_with_snapshot(
        ax,
        "Best single model per organization (30-day downloads)",
        snapshot_date,
    )
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
    save_fig(out_dir / "orgs-best-single-model-30d.png")


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        die(f"Leaderboard JSON not found: {args.input}")
    if args.top_models < 1:
        die("--top-models must be >= 1")

    leaderboard = load_json(args.input)
    rows = list(leaderboard.get("rows") or [])
    snapshot_date = leaderboard.get("snapshot_date")

    if not args.orgs_input.is_file():
        die(f"Organization leaderboard not found: {args.orgs_input}")

    org_leaderboard = load_json(args.orgs_input)
    org_rows = list(org_leaderboard.get("rows") or [])

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chart_top_models_30d(rows, args.out_dir, top_n=args.top_models, snapshot_date=snapshot_date)
    chart_orgs_best_single_model(org_rows, args.out_dir, snapshot_date=snapshot_date)


if __name__ == "__main__":
    main()
