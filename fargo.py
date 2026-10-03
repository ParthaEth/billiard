#!/usr/bin/env python3
"""Fargo-style ratings from the same BVBW file as the Elo model.

Each 8-ball, 9-ball and 10-ball rack is one result. A 7:4 is eleven racks
and a 0:1 is one rack. Nothing is capped at a full match, and 14/1 is left
out. 100 points means the higher player wins twice as many racks: the win
probability on one rack is 1 / (1 + 2^(-difference/100)).

500 is the same ghost as Elo 1500, held there by one virtual rack. The
numbers are on the Fargo scale. They are not published FargoRate ratings.

    python fargo.py
    python fargo.py --half-life-years 0
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from bt_elo import PRIOR_GAMES, fit_joint_bt
from elo import (
    PRIOR_RATING,
    SCALE,
    assign_weights,
    collect,
    load_games,
    outcomes,
    parse_match_date,
    rating_se,
    time_weight,
)
from team_elo import club_games

FARGO_ANCHOR = 500.0
FARGO_SCALE = 100.0 / math.log10(2.0)
RACK_DISCIPLINES = frozenset({"8-Ball", "9-Ball", "10-Ball"})


def rack_count(row: dict[str, str]) -> int | None:
    """Racks in an 8-ball, 9-ball or 10-ball game. 14/1 returns None."""
    if row.get("discipline", "").strip() not in RACK_DISCIPLINES:
        return None
    text = (row.get("frame_score") or "").strip()
    if ":" not in text:
        return 1
    left, right = text.split(":", 1)
    try:
        home, away = int(left), int(right)
    except ValueError:
        return 1
    total = home + away
    return total if total > 0 else 1


def rack_games(rows: list[dict[str, str]], half_life_years: float):
    """Weight each rack game by its rack count and, optionally, by age."""
    selected = []
    for winner, loser, row in outcomes(rows):
        racks = rack_count(row)
        if racks is not None:
            selected.append((winner, loser, row, racks, parse_match_date(row.get("match_date", ""))))
    known = [when for *_rest, when in selected if when is not None]
    as_of = max(known) if known else None
    half_life_days = half_life_years * 365.25
    weighted = []
    for winner, loser, row, racks, when in selected:
        if as_of is None or half_life_years <= 0:
            decay = 1.0
        else:
            decay = time_weight(when, as_of, half_life_days)
        weighted.append((winner, loser, row, decay * racks))
    return weighted, as_of


def rack_win_probability(difference: float) -> float:
    """Probability the higher player wins one rack."""
    return 1.0 / (1.0 + 2.0 ** (-difference / 100.0))


def race_win_probability(difference: float, race_to: int) -> float:
    """Probability the higher player wins a race, given the rack probability."""
    chance = rack_win_probability(difference)
    other = 1.0 - chance
    total = 0.0
    for lost in range(race_to):
        total += math.comb(race_to - 1 + lost, lost) * chance**race_to * other**lost
    return total


def probability_table() -> list[tuple[int, float, float, float]]:
    gaps = (10, 20, 30, 40, 50, 100, 200)
    return [
        (gap, rack_win_probability(gap), race_win_probability(gap, 5), race_win_probability(gap, 7))
        for gap in gaps
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--club", default="Tübinger BC")
    parser.add_argument("--half-life-years", type=float, default=8 / 12)
    parser.add_argument("--prior-racks", type=float, default=PRIOR_GAMES)
    parser.add_argument("--output", type=Path, default=Path("data/tbc_fargo.csv"))
    args = parser.parse_args()

    rows = load_games(args.input)
    played, as_of = rack_games(rows, args.half_life_years)
    players = collect(played)
    rating = fit_joint_bt(
        players,
        prior_rating=FARGO_ANCHOR,
        prior_games=args.prior_racks,
        scale=FARGO_SCALE,
        max_iter=400,
        tol=0.02,
    )
    se = {
        name: rating_se(
            name,
            record,
            rating,
            prior_rating=FARGO_ANCHOR,
            prior_games=args.prior_racks,
            scale=FARGO_SCALE,
        )
        for name, record in players.items()
    }

    elo_played, _ = assign_weights(outcomes(rows), args.half_life_years)
    elo_players = collect(elo_played)
    elo_rating = fit_joint_bt(
        elo_players,
        prior_rating=PRIOR_RATING,
        prior_games=PRIOR_GAMES,
        scale=SCALE,
        max_iter=400,
        tol=0.02,
    )
    elo_se = {
        name: rating_se(
            name,
            record,
            elo_rating,
            prior_rating=PRIOR_RATING,
            prior_games=PRIOR_GAMES,
            scale=SCALE,
        )
        for name, record in elo_players.items()
    }

    members = club_games(rows, args.club)
    table = []
    for player, teams in members.items():
        if player not in rating and player not in elo_rating:
            continue
        rack_record = players.get(player)
        elo_record = elo_players.get(player)
        table.append(
            {
                "player": player,
                "teams": ", ".join(f"{team}({count})" for team, count in teams.most_common()),
                "matches": elo_record.games if elo_record else 0,
                "racks": f"{rack_record.weighted_games:.1f}" if rack_record else "",
                "elo": round(elo_rating[player]) if player in elo_rating else "",
                "elo_se": round(elo_se[player]) if player in elo_se else "",
                "fargo": round(rating[player]) if player in rating else "",
                "fargo_se": round(se[player]) if player in se else "",
            }
        )
    table.sort(key=lambda row: (-(row["elo"] or 0), row["player"]))

    fields = ["player", "teams", "matches", "racks", "elo", "elo_se", "fargo", "fargo_se"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(table)

    if as_of is not None and args.half_life_years > 0:
        print(f"As of {as_of:%d.%m.%Y}, half-life {args.half_life_years * 12:.0f} months.")
    else:
        print("All games at full weight.")
    print(f"{len(table)} {args.club} players -> {args.output}")
    print()
    print(f"{'Elo':>5} {'±':>4}  {'Fargo':>5} {'±':>4}  {'racks':>7}  Player")
    for row in table:
        racks = f"{float(row['racks']):>7.1f}" if row["racks"] else f"{'':>7}"
        fargo = f"{row['fargo']:>5}" if row["fargo"] != "" else f"{'':>5}"
        fargo_se = f"{row['fargo_se']:>4}" if row["fargo_se"] != "" else f"{'':>4}"
        print(f"{row['elo']:>5} {row['elo_se']:>4}  {fargo} {fargo_se}  {racks}  {row['player']}")
    print()
    print(f"{'Gap':>5}  {'One rack':>8}  {'Race to 5':>9}  {'Race to 7':>9}")
    for gap, rack, race5, race7 in probability_table():
        print(f"{gap:>5}  {rack:>7.1%}  {race5:>8.1%}  {race7:>8.1%}")


if __name__ == "__main__":
    main()
