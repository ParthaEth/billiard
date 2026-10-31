#!/usr/bin/env python3
"""Match-day lineup search for a BVBW Kombi-Mannschaft match. Standard library only.

A match is 8 games in two rounds. Each round plays 14/1, 8-ball, 9-ball and
10-ball once, so every player plays two different disciplines.

Model
- Each player has one overall Elo from players.csv (no per-discipline Elo,
  those samples are too small). An unrated player gets 1504, the prior.
- Chance of winning one rack (or ball) = 1 / (1 + 10 ** ((opp - me) / 400)).
- Chance of winning a game = chance of winning a race to N with that rack
  chance. N is calibrated on real league results, not the nominal race:
  14/1 behaves like a race to 8, 8-ball 5, 9-ball 3, 10-ball 3.
- The opponent lineup is unknown. Every legal opponent lineup is weighted by
  how often those players played each discipline recently (the recent_*
  columns, time-weighted game counts, plus 0.5 each).
- Match win = 5 or more points out of 8. 4 is a draw.

Usage
  python3 matchday_lineup.py --us Partha Alex Mark Patrick \\
      --them Fabian Dominik Arash Marcel

First names or any unique part of a name are enough.
"""

from __future__ import annotations

import argparse
import csv
import math
from itertools import combinations
from pathlib import Path

DISCIPLINES = ("14/1", "8-ball", "9-ball", "10-ball")
RACE = {"14/1": 8, "8-ball": 5, "9-ball": 3, "10-ball": 3}
PRIOR_ELO = 1504.0
PREF_PRIOR = 0.5
WIN_POINTS = 5


def load_players(path: Path) -> dict[str, dict]:
    players = {}
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            players[row["player"]] = {
                "team": row["team"],
                "elo": float(row["elo"]) if row["elo"] else PRIOR_ELO,
                "se": row["elo_se"],
                "pref": {disc: float(row[f"recent_{disc}"]) for disc in DISCIPLINES},
            }
    return players


def resolve(query: str, players: dict[str, dict]) -> str:
    q = query.casefold()
    exact = [name for name in players if name.casefold() == q]
    if exact:
        return exact[0]
    hits = [name for name in players if q in name.casefold()]
    if query.casefold() == "alex":
        hits = [name for name in hits if name.startswith("Alexander Fischer")] or hits
    if len(hits) == 1:
        return hits[0]
    if not hits:
        raise SystemExit(f"No player matches {query!r}. Add them to players.csv.")
    raise SystemExit(f"{query!r} matches {', '.join(hits)}. Use more of the name.")


def rack_chance(me: float, opp: float) -> float:
    return 1.0 / (1.0 + 10 ** ((opp - me) / 400.0))


def race_win(p: float, target: int) -> float:
    """Chance of reaching `target` racks first."""
    q = 1.0 - p
    return sum(math.comb(target - 1 + lost, lost) * p**target * q**lost for lost in range(target))


def assignments() -> list[tuple[frozenset[str], ...]]:
    """All ways to give 4 players two disciplines each, every discipline twice."""
    pairs = [frozenset(pair) for pair in combinations(DISCIPLINES, 2)]
    found = []

    def rec(picked, used):
        if len(picked) == 4:
            if all(used[d] == 2 for d in DISCIPLINES):
                found.append(tuple(picked))
            return
        for pair in pairs:
            if any(used[d] == 2 for d in pair):
                continue
            for d in pair:
                used[d] += 1
            rec(picked + [pair], used)
            for d in pair:
                used[d] -= 1

    rec([], {d: 0 for d in DISCIPLINES})
    return found


def round_splits(assignment) -> list[tuple[dict[str, int], dict[str, int]]]:
    """Split a lineup into two rounds; each round has every discipline once."""
    edges = [(player, disc) for player, picks in enumerate(assignment) for disc in sorted(picks)]
    splits = []

    def rec(index, rounds):
        if index == len(edges):
            if all(len(r) == 4 for r in rounds):
                splits.append((dict(rounds[0]), dict(rounds[1])))
            return
        player, disc = edges[index]
        for r in rounds:
            if disc in r or player in r.values():
                continue
            r[disc] = player
            rec(index + 1, rounds)
            del r[disc]

    rec(0, ({}, {}))
    unique, seen = [], set()
    for one, two in splits:
        key = frozenset((tuple(sorted(one.items())), tuple(sorted(two.items()))))
        if key not in seen:
            seen.add(key)
            unique.append((one, two))
    return unique


def opponent_seatings(them: list[str], players) -> list[tuple[tuple[dict, dict], float]]:
    lineups = assignments()
    logs = [
        sum(math.log(players[them[i]]["pref"][d] + PREF_PRIOR) for i, picks in enumerate(a) for d in picks)
        for a in lineups
    ]
    peak = max(logs)
    weights = [math.exp(value - peak) for value in logs]
    total = sum(weights)
    seatings = []
    for lineup, weight in zip(lineups, weights):
        splits = round_splits(lineup)
        share = weight / total / (2 * len(splits))
        for one, two in splits:
            seatings.append(((one, two), share))
            seatings.append(((two, one), share))
    return seatings


def points_distribution(our_rounds, their_rounds, chance) -> list[float]:
    dist = [1.0]
    for ours, theirs in zip(our_rounds, their_rounds):
        for disc in DISCIPLINES:
            p = chance[ours[disc], theirs[disc], disc]
            nxt = [0.0] * (len(dist) + 1)
            for points, mass in enumerate(dist):
                nxt[points] += mass * (1 - p)
                nxt[points + 1] += mass * p
            dist = nxt
    return dist


def summary(dist: list[float]) -> tuple[float, float, float, float]:
    win = sum(dist[WIN_POINTS:])
    draw = dist[4]
    return win, draw, 1 - win - draw, sum(points * mass for points, mass in enumerate(dist))


def score(our_rounds, seatings, chance):
    total = [0.0] * 9
    for their_rounds, weight in seatings:
        for points, mass in enumerate(points_distribution(our_rounds, their_rounds, chance)):
            total[points] += weight * mass
    return summary(total)


def short(name: str) -> str:
    first = name.split()[0]
    return "Alex" if first == "Alexander" and "Fischer" in name else first


def print_plan(title, us, them, our_rounds, likely, seatings, chance) -> None:
    win, draw, loss, expected = score(our_rounds, seatings, chance)
    lw, ld, ll, le = summary(points_distribution(our_rounds, likely, chance))
    print(f"{title}")
    print(f"  Averaged over their likely seatings: win {win:.0%}, draw {draw:.0%}, loss {loss:.0%}, expected {expected:.2f}-{8 - expected:.2f}")
    print(f"  Against their most likely seating:   win {lw:.0%}, draw {ld:.0%}, loss {ll:.0%}, expected {le:.2f}-{8 - le:.2f}")
    for number, (ours, theirs) in enumerate(zip(our_rounds, likely), start=1):
        print(f"  Runde {number}")
        for disc in DISCIPLINES:
            p = chance[ours[disc], theirs[disc], disc]
            print(f"    {disc + ':':9} {short(us[ours[disc]])} vs {short(them[theirs[disc]])}  {p:.0%}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--us", nargs=4, required=True)
    parser.add_argument("--them", nargs=4, required=True)
    parser.add_argument("--players", type=Path, default=Path(__file__).with_name("players.csv"))
    args = parser.parse_args()

    players = load_players(args.players)
    us = [resolve(q, players) for q in args.us]
    them = [resolve(q, players) for q in args.them]
    chance = {
        (i, j, disc): race_win(rack_chance(players[a]["elo"], players[b]["elo"]), RACE[disc])
        for i, a in enumerate(us)
        for j, b in enumerate(them)
        for disc in DISCIPLINES
    }
    seatings = opponent_seatings(them, players)
    likely = max(seatings, key=lambda item: item[1])[0]

    ranked = []
    for lineup in assignments():
        best_split = max(round_splits(lineup), key=lambda rounds: score(rounds, seatings, chance)[::3])
        ranked.append((score(best_split, seatings, chance), best_split))
    ranked.sort(key=lambda item: (item[0][0], item[0][3]), reverse=True)

    print("Elo used")
    for name in us + them:
        p = players[name]
        se = f" ± {p['se']}" if p["se"] else " (unrated, prior)"
        print(f"  {name:28} {p['elo']:.0f}{se}")
    print()
    print_plan("Best lineup", us, them, ranked[0][1], likely, seatings, chance)
    print_plan("Worst lineup", us, them, ranked[-1][1], likely, seatings, chance)


if __name__ == "__main__":
    main()
