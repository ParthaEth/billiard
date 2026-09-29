#!/usr/bin/env python3
"""Append newly played BVBW league games to data/games.csv.

Walks the season currently shown on the Spielplan page (the latest one)
for every region of Kreisliga A, Bezirksliga, Landesliga and Verbandsliga.
Reports already stored are skipped, so this is safe to run after each Spieltag.

    python scrape_new.py
    python scrape_new.py --season 2026/2027
"""

from __future__ import annotations

import argparse
from pathlib import Path

from scrape_bvbw import TARGET_LEAGUES, StepScraper, run_auto


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/games.csv"))
    parser.add_argument(
        "--season",
        help="Season to update, e.g. 2026/2027. Default: the season open on the site",
    )
    parser.add_argument("--min-wait", type=float, default=0.4)
    parser.add_argument("--max-wait", type=float, default=0.8)
    args = parser.parse_args()
    seasons = (args.season,) if args.season else ()
    run_auto(
        StepScraper(
            args.csv,
            args.min_wait,
            args.max_wait,
            leagues=TARGET_LEAGUES,
            seasons=seasons,
        )
    )


if __name__ == "__main__":
    main()
