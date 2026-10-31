#!/usr/bin/env python3
"""Plot a player's Elo at the end of each month they have a dated game.

The fit is joint: every dated game on or before that month-end is included,
with the same half-life on results and on the virtual draw. Matplotlib is
the system python3, not the project venv.

Example:
  python3 plot_elo.py "Fabian Cecotti"
  python3 plot_elo.py --half-life-years 3 "Arash Dayyani" "Marcel Haab"
"""

from __future__ import annotations

import argparse
import calendar
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt

import elo

HALF_LIFE_YEARS = 8 / 12
ROOT = Path(__file__).resolve().parent


def half_life_text(years: float) -> str:
    """Title phrase for a half-life, e.g. ``8-month`` or ``3-year``."""
    months = years * 12
    if abs(months - round(months)) < 1e-6:
        n = int(round(months))
        if n > 0 and n % 12 == 0:
            years_n = n // 12
            return "1-year" if years_n == 1 else f"{years_n}-year"
        return "1-month" if n == 1 else f"{n}-month"
    if abs(years - round(years)) < 1e-6:
        years_n = int(round(years))
        return "1-year" if years_n == 1 else f"{years_n}-year"
    return f"{years:g}-year"


def month_end(when: datetime) -> datetime:
    last = calendar.monthrange(when.year, when.month)[1]
    return datetime(when.year, when.month, last)


def known_names(rows: list[dict[str, str]]) -> list[str]:
    found: set[str] = set()
    for row in rows:
        for side in ("home", "away"):
            name = (row.get(f"{side}_player") or "").strip()
            if name:
                found.add(name)
    return sorted(found)


def resolve_name(query: str, known: list[str]) -> str:
    """Identity display name, the same string written into the games."""
    exact = [name for name in known if name.casefold() == query.casefold()]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise SystemExit(f"{query!r} matches more than one player: {', '.join(exact)}")
    partial = [name for name in known if query.casefold() in name.casefold()]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        raise SystemExit(f"{query!r} matches more than one player: {', '.join(partial)}")
    matches = get_close_matches(query, known, n=5, cutoff=0.6)
    hint = f" Close names: {', '.join(matches)}." if matches else ""
    raise SystemExit(f"No player named {query!r}.{hint}")


def slug_for(name: str) -> str:
    first = name.split()[0].casefold()
    safe = "".join(ch for ch in first if ch.isalnum())
    if not safe:
        raise SystemExit(f"Cannot build a file name from {name!r}.")
    return safe


def player_months(rows: list[dict[str, str]], name: str) -> list[datetime]:
    """Last day of each month in which `name` has a dated, decisive game."""
    months: set[datetime] = set()
    for row in rows:
        home = (row.get("home_player") or "").strip()
        away = (row.get("away_player") or "").strip()
        if name not in (home, away):
            continue
        when = elo.parse_match_date(row.get("match_date", ""))
        if when is None:
            continue
        winner = (row.get("winner") or "").strip()
        loser = (row.get("loser") or "").strip()
        if not winner or not loser or winner == loser:
            continue
        if name not in (winner, loser):
            continue
        months.add(month_end(when))
    return sorted(months)


def games_as_of(
    played: list[tuple[str, str, dict[str, str]]],
    as_of: datetime,
    half_life_years: float,
) -> list[elo.Game]:
    """Dated games on or before `as_of`, weighted back from that day."""
    half_life_days = half_life_years * 365.25
    weighted: list[elo.Game] = []
    for winner, loser, row in played:
        when = elo.parse_match_date(row.get("match_date", ""))
        if when is None or when > as_of:
            continue
        if half_life_years <= 0:
            decay = 1.0
        else:
            decay = elo.time_weight(when, as_of, half_life_days)
        weight = decay * elo._row_weight(row)
        if weight > 0:
            weighted.append((winner, loser, row, weight))
    return weighted


def fit_months(
    played: list[tuple[str, str, dict[str, str]]],
    months: list[datetime],
    half_life_years: float,
) -> dict[datetime, tuple[dict[str, elo.PlayerRecord], dict[str, float]]]:
    """Joint fit at each month-end. Later months start from the previous fit."""
    fitted: dict[datetime, tuple[dict[str, elo.PlayerRecord], dict[str, float]]] = {}
    start: dict[str, float] | None = None
    for index, as_of in enumerate(months, start=1):
        players = elo.collect(games_as_of(played, as_of, half_life_years))
        rating = elo.fit_elo(
            players,
            as_of=as_of,
            half_life_years=half_life_years,
            start=start,
        )
        fitted[as_of] = (players, rating)
        start = rating
        print(f"  fit {index}/{len(months)}  as of {as_of:%b %Y}", flush=True)
    return fitted


def plot_player(
    name: str,
    months: list[datetime],
    fitted: dict[datetime, tuple[dict[str, elo.PlayerRecord], dict[str, float]]],
    half_life_years: float,
    offset: float,
    path: Path,
    *,
    tick_months: int | None = None,
) -> None:
    dates: list[datetime] = []
    values: list[float] = []
    errors: list[float] = []
    for as_of in months:
        players, rating = fitted[as_of]
        record = players.get(name)
        if record is None or name not in rating:
            raise SystemExit(f"{name} has a game in {as_of:%b %Y} but is missing from the fit.")
        se = elo.rating_se(
            name,
            record,
            rating,
            prior_rating=elo.PRIOR_RATING,
            prior_games=1.0,
            scale=elo.SCALE,
            as_of=as_of,
            half_life_years=half_life_years,
        )
        # Plot mid-month so the point sits under its %b label (month-end
        # coordinates hug the next month's tick and look like a missing Sep).
        dates.append(datetime(as_of.year, as_of.month, 15))
        values.append(rating[name] + offset)
        errors.append(se)

    span = (dates[-1].year - dates[0].year) * 12 + (dates[-1].month - dates[0].month)
    if tick_months is not None:
        interval = tick_months
    else:
        interval = 1 if span <= 24 else 2
    label = half_life_text(half_life_years)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.axhline(elo.PRIOR_RATING + offset, color="grey", linestyle="--", label="1500 prior")
    ax.errorbar(
        dates,
        values,
        yerr=errors,
        fmt="-o",
        color="#1f77b4",
        ecolor="#1f77b4",
        markersize=4.5,
        linewidth=1.6,
        capsize=3,
        label=f"{name} ± 1 SE",
    )
    ax.annotate(
        f"{values[-1]:.0f} ± {errors[-1]:.0f}",
        xy=(dates[-1], values[-1]),
        xytext=(-36, 12),
        textcoords="offset points",
    )
    ax.set_title(f"{name} — Elo over time ({label} half-life, joint fit)")
    ax.set_xlabel(f"As of (month of {name}'s games)")
    ax.set_ylabel("Elo")
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=15, interval=interval))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True)
    ax.legend(loc="best")
    fig.autofmt_xdate(rotation=30, ha="right")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def run(args: argparse.Namespace) -> None:
    rows = elo.load_games(args.input)
    known = known_names(rows)
    names = [resolve_name(query, known) for query in args.player]
    slugs = [slug_for(name) for name in names]
    if len(slugs) != len(set(slugs)):
        raise SystemExit("Two players share a first name, so the plot files would overwrite each other.")

    by_player = {name: player_months(rows, name) for name in names}
    for name, months in by_player.items():
        if not months:
            raise SystemExit(f"{name} has no dated games.")

    months = sorted({month for series in by_player.values() for month in series})
    print(f"Joint fit at {len(months)} month-ends, half-life {half_life_text(args.half_life_years)}.")
    fitted = fit_months(elo.outcomes(rows), months, args.half_life_years)
    offset = elo.scale_offset()
    out_dir = args.output_dir
    for name, slug in zip(names, slugs):
        path = out_dir / f"{slug}_elo.png"
        plot_player(
            name,
            by_player[name],
            fitted,
            args.half_life_years,
            offset,
            path,
            tick_months=args.tick_months,
        )
        print(f"Wrote {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "player",
        nargs="+",
        help="Player name. A unique part of the identity display name is enough.",
    )
    parser.add_argument(
        "--half-life-years",
        type=float,
        default=HALF_LIFE_YEARS,
        help="Age at which a game, and the virtual draw, keep half their weight (default: 8/12)",
    )
    parser.add_argument(
        "--tick-months",
        type=int,
        choices=(1, 2),
        default=None,
        help="X-axis month tick spacing. Default: every month if span ≤ 24 months, else every 2",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "games.csv",
        help="Games CSV (default: data/games.csv)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "data",
        help="Directory for data/{firstname}_elo.png (default: data)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
