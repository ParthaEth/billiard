#!/usr/bin/env python3
"""Per-player half-life, chosen on games the fit has not seen.

The standard ratings (elo.py, one half-life for everyone) are left alone.
This script reads games.csv and writes only data/personal_half_life.csv.

Method
- On the first day of each of the last 24 months, fit the whole field with
  the standard 8-month half-life on the games before that day.
- For each player and each candidate half-life, refit only that player on
  his own earlier games, with the field held at that fit, and score the
  rack share of his games in the month that follows.
- The first 12 months pick a half-life per player; the last 12 check the
  pick against the standard 8 months. The half-life written to the file is
  picked on all 24 months.
- The current rating with that half-life is fitted at the newest match,
  field held at the standard fit, and published with the frozen offset.

A player needs MIN_SCORED scored games to leave the standard half-life.
Needs numpy, so run it with python3, not .venv.

    python3 personal_half_life.py "Dominik Heinkele" "Marcel Haab"
"""

from __future__ import annotations

import argparse
import csv
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import elo

STANDARD_MONTHS = 8
CANDIDATE_MONTHS = (3, 4, 6, 8, 12, 18, 24, 36, 60)
MONTHS_BACK = 24
MIN_SCORED = 20
DAYS_PER_MONTH = 365.25 / 12
LN2 = math.log(2)
SLOPE = elo.LN10 / elo.SCALE
ROOT = Path(__file__).resolve().parent


def add_month(day: datetime, step: int = 1) -> datetime:
    month = day.month - 1 + step
    return datetime(day.year + month // 12, month % 12 + 1, 1)


def dated_games(rows: list[dict[str, str]]) -> list[tuple[str, str, dict[str, str], datetime | None]]:
    return [
        (winner, loser, row, elo.parse_match_date(row.get("match_date", "")))
        for winner, loser, row in elo.outcomes(rows)
    ]


def field_fit(games, cutoff: datetime, start: dict[str, float] | None) -> dict[str, float]:
    """Standard joint fit on games before `cutoff`, weighted back from it."""
    half_years = STANDARD_MONTHS / 12
    half_days = half_years * 365.25
    weighted = []
    for winner, loser, row, when in games:
        if when is not None and when >= cutoff:
            continue
        weight = elo.time_weight(when, cutoff, half_days) * elo._row_weight(row)
        if weight > 0:
            weighted.append((winner, loser, row, weight))
    players = elo.collect(weighted)
    return elo.fit_elo(players, as_of=cutoff, half_life_years=half_years, start=start)


class History:
    """One player's games in date order. Undated games count as current."""

    def __init__(self) -> None:
        self.days: list[int] = []
        self.opponents: list[str] = []
        self.shares: list[float] = []
        self.weights: list[float] = []
        self.undated: list[tuple[str, float, float]] = []

    def add(self, when: datetime | None, opponent: str, share: float, weight: float) -> None:
        if when is None:
            self.undated.append((opponent, share, weight))
            return
        self.days.append(when.toordinal())
        self.opponents.append(opponent)
        self.shares.append(share)
        self.weights.append(weight)

    def finish(self) -> None:
        order = sorted(range(len(self.days)), key=self.days.__getitem__)
        self.days = [self.days[i] for i in order]
        self.opponents = [self.opponents[i] for i in order]
        self.shares = [self.shares[i] for i in order]
        self.weights = [self.weights[i] for i in order]

    def before(self, day: datetime, field: dict[str, float], inclusive: bool = False):
        """Arrays for games before `day`, ages in days, and the ghost's age."""
        cut = bisect_right if inclusive else bisect_left
        end = cut(self.days, day.toordinal())
        opponents = self.opponents[:end] + [name for name, _s, _w in self.undated]
        if not opponents:
            return None
        ratings = np.array([field.get(name, elo.PRIOR_RATING) for name in opponents])
        shares = np.array(self.shares[:end] + [s for _n, s, _w in self.undated])
        weights = np.array(self.weights[:end] + [w for _n, _s, w in self.undated])
        ages = np.array([day.toordinal() - d for d in self.days[:end]] + [0] * len(self.undated), dtype=float)
        if self.undated or end == 0:
            ghost_age = 0.0
        else:
            ghost_age = float(day.toordinal() - self.days[end - 1])
        return ratings, shares, weights, ages, ghost_age


def histories(games) -> dict[str, History]:
    found: dict[str, History] = defaultdict(History)
    for winner, loser, row, when in games:
        share = elo.winner_share(row)
        weight = elo._row_weight(row)
        if weight <= 0:
            continue
        found[winner].add(when, loser, share, weight)
        found[loser].add(when, winner, 1.0 - share, weight)
    for history in found.values():
        history.finish()
    return found


def solo_fit(arrays, half_months: float, start: float) -> tuple[float, float]:
    """This player's rating and SE with the field fixed."""
    ratings, shares, weights, ages, ghost_age = arrays
    half_days = half_months * DAYS_PER_MONTH
    w = weights * np.exp(-LN2 * np.maximum(ages, 0.0) / half_days)
    ghost = math.exp(-LN2 * max(ghost_age, 0.0) / half_days)
    rating = start
    curvature = 0.0
    for _ in range(100):
        chance = 1.0 / (1.0 + 10.0 ** ((ratings - rating) / elo.SCALE))
        ghost_chance = elo.expected_score(rating, elo.PRIOR_RATING)
        gradient = float(np.sum(w * (shares - chance))) + ghost * (0.5 - ghost_chance)
        curvature = float(np.sum(w * chance * (1.0 - chance))) + ghost * ghost_chance * (1.0 - ghost_chance)
        if curvature <= 1e-12:
            break
        step = max(-100.0, min(100.0, gradient / (curvature * SLOPE)))
        rating += step
        if abs(step) < 0.01:
            break
    se = 1.0 / math.sqrt(curvature * SLOPE * SLOPE) if curvature > 1e-12 else float("inf")
    return rating, se


def score_months(games, found: dict[str, History], cutoffs: list[datetime]):
    """Log loss per player, cutoff month and candidate half-life."""
    scores: dict[str, dict[datetime, tuple[int, np.ndarray]]] = defaultdict(dict)
    start = None
    for number, cutoff in enumerate(cutoffs, start=1):
        field = field_fit(games, cutoff, start)
        start = field
        end = add_month(cutoff)
        tests: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
        for winner, loser, row, when in games:
            if when is None or not cutoff <= when < end:
                continue
            if winner not in field or loser not in field:
                continue
            share = elo.winner_share(row)
            weight = elo._row_weight(row)
            tests[winner].append((loser, share, weight))
            tests[loser].append((winner, 1.0 - share, weight))
        for name, played in tests.items():
            arrays = found[name].before(cutoff, field)
            if arrays is None:
                continue
            opp = np.array([field[o] for o, _s, _w in played])
            share = np.array([s for _o, s, _w in played])
            weight = np.array([w for _o, _s, w in played])
            losses = np.empty(len(CANDIDATE_MONTHS))
            for k, months in enumerate(CANDIDATE_MONTHS):
                rating, _se = solo_fit(arrays, months, field[name])
                chance = np.clip(1.0 / (1.0 + 10.0 ** ((opp - rating) / elo.SCALE)), 1e-9, 1 - 1e-9)
                losses[k] = -float(np.sum(weight * (share * np.log(chance) + (1 - share) * np.log(1 - chance))))
            scores[name][cutoff] = (len(played), losses)
        print(f"  month {number}/{len(cutoffs)}  {cutoff:%b %Y}  {len(tests)} players scored", flush=True)
    return scores


def pick(scores_for_player: dict[datetime, tuple[int, np.ndarray]], months: list[datetime]) -> tuple[int, int, np.ndarray]:
    """Half-life with the lowest summed loss over `months`, and the evidence."""
    count = 0
    total = np.zeros(len(CANDIDATE_MONTHS))
    for cutoff in months:
        if cutoff in scores_for_player:
            n, losses = scores_for_player[cutoff]
            count += n
            total += losses
    if count < MIN_SCORED:
        return STANDARD_MONTHS, count, total
    return CANDIDATE_MONTHS[int(np.argmin(total))], count, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("player", nargs="*", help="Players to print")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "games.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "personal_half_life.csv")
    args = parser.parse_args()

    rows = elo.load_games(args.input)
    games = dated_games(rows)
    found = histories(games)
    newest = max(when for *_rest, when in games if when is not None)
    current_month = datetime(newest.year, newest.month, 1)
    cutoffs = [add_month(current_month, -step) for step in range(MONTHS_BACK - 1, -1, -1)]
    half = len(cutoffs) // 2
    select, check = cutoffs[:half], cutoffs[half:]
    print(f"Scoring {len(cutoffs)} months, {cutoffs[0]:%b %Y} to {cutoffs[-1]:%b %Y}.")
    scores = score_months(games, found, cutoffs)

    standard = CANDIDATE_MONTHS.index(STANDARD_MONTHS)
    base_loss = 0.0
    picked_loss = 0.0
    scored = 0
    moved = 0
    for name, by_month in scores.items():
        chosen, _count, _total = pick(by_month, select)
        k = CANDIDATE_MONTHS.index(chosen)
        for cutoff in check:
            if cutoff in by_month:
                n, losses = by_month[cutoff]
                scored += n
                base_loss += losses[standard]
                picked_loss += losses[k]
        if chosen != STANDARD_MONTHS:
            moved += 1
    print(
        f"\nCheck on {check[0]:%b %Y}–{check[-1]:%b %Y}, half-life picked on "
        f"{select[0]:%b %Y}–{select[-1]:%b %Y}:"
    )
    print(f"  {scored} scored player-games, {moved} players moved off {STANDARD_MONTHS} months")
    print(f"  standard {STANDARD_MONTHS} months  log loss {base_loss / scored:.5f}")
    print(f"  personal half-life  log loss {picked_loss / scored:.5f}")

    weighted, as_of = elo.assign_weights(elo.outcomes(rows), STANDARD_MONTHS / 12)
    field = elo.fit_elo(elo.collect(weighted), as_of=as_of, half_life_years=STANDARD_MONTHS / 12)
    offset = elo.scale_offset()
    fields = ["player", "scored_games", "half_life_months", "elo_standard", "se_standard", "elo_personal", "se_personal"]
    fields += [f"loss_{months}m" for months in CANDIDATE_MONTHS]
    table: dict[str, dict[str, str]] = {}
    for name in sorted(field):
        arrays = found[name].before(as_of, field, inclusive=True) if name in found else None
        if arrays is None:
            continue
        chosen, count, total = pick(scores.get(name, {}), cutoffs)
        base, base_se = solo_fit(arrays, STANDARD_MONTHS, field[name])
        rating, se = solo_fit(arrays, chosen, field[name])
        row = {
            "player": name,
            "scored_games": str(count),
            "half_life_months": str(chosen),
            "elo_standard": f"{field[name] + offset:.0f}",
            "se_standard": f"{base_se:.0f}",
            "elo_personal": f"{rating + offset:.0f}",
            "se_personal": f"{se:.0f}",
        }
        for months, loss in zip(CANDIDATE_MONTHS, total):
            row[f"loss_{months}m"] = f"{loss / count:.4f}" if count else ""
        table[name] = row
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table.values())
    print(f"\nWrote {args.output} ({len(table)} players). Standard ratings are unchanged.")

    if args.player:
        print(f"\n{'Player':24} {'scored':>6} {'half-life':>9} {'standard':>11} {'personal':>11}")
    for name in args.player:
        row = table.get(name)
        if row is None:
            print(f"{name}: not in the fit")
            continue
        print(
            f"{name:24} {row['scored_games']:>6} {row['half_life_months']:>7} mo "
            f"{row['elo_standard']:>5} ±{row['se_standard']:>3} {row['elo_personal']:>5} ±{row['se_personal']:>3}"
        )


if __name__ == "__main__":
    main()
