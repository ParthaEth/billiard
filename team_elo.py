#!/usr/bin/env python3
"""Joint Bradley–Terry Elo for one club's players, overall and per discipline.

Every rating is fitted on the whole BVBW file (all clubs), then the club's
players are picked out. Settings follow backtest_bt.py: rack/ball share,
1 virtual draw against 1500, no age decay.

    python team_elo.py                        # all Tübinger BC teams -> data/tbc_elo.csv
    python team_elo.py --team 2               # only TBC 2 players  -> data/tbc2_elo.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

from bt_elo import PRIOR_GAMES, fit_joint_bt
from elo import PRIOR_RATING, SCALE, assign_weights, collect, load_games, outcomes, rating_se

DISCIPLINES = (("14_1", "14/1e"), ("8", "8-Ball"), ("9", "9-Ball"), ("10", "10-Ball"))


def fit_scope(played, prior_games: float) -> tuple[dict[str, float], dict[str, float]]:
    players = collect(played)
    rating = fit_joint_bt(players, prior_games=prior_games)
    se = {
        name: rating_se(
            name, record, rating, prior_rating=PRIOR_RATING, prior_games=prior_games, scale=SCALE
        )
        for name, record in players.items()
    }
    return rating, se


def club_games(rows, club: str) -> dict[str, Counter]:
    """player -> Counter(team number -> games) for teams named '<club> N'."""
    pattern = re.compile(rf"^{re.escape(club)}\s*(\d+)$")
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        for side in ("home", "away"):
            match = pattern.match(row[f"{side}_team"].strip())
            if match and row[f"{side}_player"].strip():
                counts[row[f"{side}_player"].strip()][match.group(1)] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--club", default="Tübinger BC")
    parser.add_argument("--prefix", default="tbc", help="Column and file name prefix")
    parser.add_argument("--team", help="Only players who played for this team number")
    parser.add_argument("--prior-games", type=float, default=PRIOR_GAMES)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = load_games(args.input)
    played, _ = assign_weights(outcomes(rows), 0.0)
    scopes = {"all": fit_scope(played, args.prior_games)}
    for key, discipline in DISCIPLINES:
        subset = [g for g in played if g[2]["discipline"].strip() == discipline]
        scopes[key] = fit_scope(subset, args.prior_games)

    members = club_games(rows, args.club)
    label = f"{args.prefix}{args.team or ''}"
    output = args.output or Path(f"data/{label}_elo.csv")
    fields = ["player"] + ([] if args.team else ["teams"]) + [f"{label}_games"]
    for key in ("all",) + tuple(k for k, _d in DISCIPLINES):
        fields += [f"elo_{key}", f"se_{key}"]

    out = []
    all_rating = scopes["all"][0]
    for player, teams in members.items():
        games = teams[args.team] if args.team else sum(teams.values())
        if not games or player not in all_rating:
            continue
        row = {"player": player, f"{label}_games": games}
        if not args.team:
            row["teams"] = ", ".join(f"{t}({n})" for t, n in teams.most_common())
        for key, (rating, se) in scopes.items():
            if player in rating:
                row[f"elo_{key}"] = round(rating[player])
                row[f"se_{key}"] = round(se[player])
        out.append(row)
    out.sort(key=lambda r: -r["elo_all"])

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out)

    print(f"{len(out)} {args.club}{' ' + args.team if args.team else ''} players -> {output}")
    print(f"{'Elo':>5} {'±':>4}  {'14/1':>5} {'8':>5} {'9':>5} {'10':>5}  {'games':>5}  Player")
    for r in out:
        cells = " ".join(f"{r.get(f'elo_{k}', ''):>5}" for k, _d in DISCIPLINES)
        print(f"{r['elo_all']:>5} {r['se_all']:>4}  {cells}  {r[f'{label}_games']:>5}  {r['player']}")


if __name__ == "__main__":
    main()
