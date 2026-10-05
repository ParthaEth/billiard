#!/usr/bin/env python3
"""Bring league and Einzel results in data/games.csv up to today.

The season window runs from the newest match already stored through the
season that contains today. A BVBW season starts in September. Older
seasons are not opened. Reports already in the file are skipped by the
league and Einzel scrapers.

    python sync.py
    python sync.py --leagues-only
    python sync.py --einzel-only

League-only is the same walk as ``python scrape_new.py --season …``.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

from elo import parse_match_date
from scrape_bvbw import TARGET_LEAGUES, StepScraper, run_auto
from scrape_einzel import scrape_seasons


def season_of(when: datetime) -> str:
    """BVBW season containing this date. September opens the new season."""
    if when.month >= 9:
        return f"{when.year}/{when.year + 1}"
    return f"{when.year - 1}/{when.year}"


def seasons_from(start: str, end: str) -> tuple[str, ...]:
    """Inclusive season list. ``start`` and ``end`` look like ``2025/2026``."""
    year = int(start.split("/", 1)[0])
    last = int(end.split("/", 1)[0])
    if year > last:
        year = last
    return tuple(f"{y}/{y + 1}" for y in range(year, last + 1))


def newest_date(path: Path, *, einzel: bool) -> datetime | None:
    """Newest match date of league games, or of Einzel games."""
    if not path.exists() or path.stat().st_size == 0:
        return None
    latest: datetime | None = None
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            league = (row.get("league") or "").strip()
            is_einzel = league == "Einzel"
            if is_einzel != einzel:
                continue
            when = parse_match_date(row.get("match_date") or "")
            if when is not None and (latest is None or when > latest):
                latest = when
    return latest


def seasons_to_sync(path: Path, *, einzel: bool, today: datetime) -> tuple[str, ...]:
    """From the newest stored match of this kind through the current season."""
    current = season_of(today)
    latest = newest_date(path, einzel=einzel)
    if latest is None:
        return (current,)
    return seasons_from(season_of(latest), current)


def sync_leagues(path: Path, seasons: tuple[str, ...], min_wait: float, max_wait: float) -> None:
    print(f"Leagues: {', '.join(TARGET_LEAGUES)}")
    print(f"Seasons: {', '.join(seasons)}")
    run_auto(
        StepScraper(
            path,
            min_wait,
            max_wait,
            leagues=TARGET_LEAGUES,
            seasons=seasons,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--leagues-only", action="store_true")
    parser.add_argument("--einzel-only", action="store_true")
    parser.add_argument("--min-wait", type=float, default=0.4)
    parser.add_argument("--max-wait", type=float, default=0.8)
    parser.add_argument("--einzel-min-wait", type=float, default=1.5)
    parser.add_argument("--einzel-max-wait", type=float, default=3.0)
    args = parser.parse_args()
    if args.leagues_only and args.einzel_only:
        parser.error("choose one of --leagues-only and --einzel-only")

    today = datetime.now()
    do_leagues = not args.einzel_only
    do_einzel = not args.leagues_only
    print(f"Today is {today:%d.%m.%Y}, season {season_of(today)}")

    if do_leagues:
        league_seasons = seasons_to_sync(args.csv, einzel=False, today=today)
        sync_leagues(args.csv, league_seasons, args.min_wait, args.max_wait)
    if do_einzel:
        einzel_seasons = seasons_to_sync(args.csv, einzel=True, today=today)
        scrape_seasons(einzel_seasons, args.csv, args.einzel_min_wait, args.einzel_max_wait)


if __name__ == "__main__":
    main()
