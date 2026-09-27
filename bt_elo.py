#!/usr/bin/env python3
"""Joint Bradley–Terry ratings over the full BVBW games file.

One model, every decisive game. Strengths are estimated together with the
Hunter (2004) MM algorithm, then written on the usual Elo scale where 400
points is 10:1. A small number of virtual draws against a 1500 ghost keeps
unbeaten one-game records from flying off to infinity.

Rack and ball scores still count as a share, not a binary win: 5:0 is a
full win, 5:4 is 5/9, 60:30 in 14/1 is 60/90. Optional age decay is off
by default so the whole file is used at full weight.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from elo import (
    PRIOR_RATING,
    SCALE,
    assign_weights,
    collect,
    expected_score,
    load_games,
    outcomes,
    rating_se,
    winner_share,
)

PRIOR_GAMES = 1.0


def fit_joint_bt(
    players: dict,
    *,
    prior_rating: float = PRIOR_RATING,
    prior_games: float = PRIOR_GAMES,
    scale: float = SCALE,
    max_iter: int = 800,
    tol: float = 0.01,
) -> dict[str, float]:
    """Simultaneous MM updates of every player's strength."""
    names = list(players)
    index = {name: i for i, name in enumerate(names)}
    wins = [0.0] * len(names)
    edges: list[list[tuple[int, float]]] = [[] for _ in names]
    pair_weight: list[dict[int, float]] = [defaultdict(float) for _ in names]

    for name, record in players.items():
        i = index[name]
        for opponent, outcome, weight in record.opponents:
            j = index[opponent]
            wins[i] += weight * outcome
            pair_weight[i][j] += weight
    for i, pairs in enumerate(pair_weight):
        edges[i] = list(pairs.items())

    strength = [1.0] * len(names)
    dummy = 1.0
    for _ in range(max_iter):
        updated = [0.0] * len(names)
        max_elo_step = 0.0
        for i, current in enumerate(strength):
            denom = 0.0
            for j, n_ij in edges[i]:
                denom += n_ij / (current + strength[j])
            total_wins = wins[i]
            if prior_games > 0:
                denom += prior_games / (current + dummy)
                total_wins += 0.5 * prior_games
            if denom <= 1e-15 or total_wins <= 0:
                nxt = current
            else:
                nxt = total_wins / denom
            nxt = max(1e-12, nxt)
            updated[i] = nxt
            old_elo = scale * math.log10(current)
            new_elo = scale * math.log10(nxt)
            max_elo_step = max(max_elo_step, abs(new_elo - old_elo))
        strength = updated
        if max_elo_step < tol:
            break

    return {
        name: prior_rating + scale * math.log10(strength[index[name]])
        for name in names
    }


def write_ratings(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "player",
        "team",
        "league",
        "staffel",
        "games",
        "wins",
        "losses",
        "weighted_games",
        "elo",
        "elo_low",
        "elo_high",
        "elo_se",
        "expected_wins",
        "residual",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def rating_rows(
    players: dict,
    rating: dict[str, float],
    *,
    prior_rating: float,
    prior_games: float,
    scale: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, record in players.items():
        elo = rating[name]
        expected = sum(
            weight * expected_score(elo, rating[opponent], scale)
            for opponent, _outcome, weight in record.opponents
        )
        observed = sum(weight * outcome for _opponent, outcome, weight in record.opponents)
        se = rating_se(
            name,
            record,
            rating,
            prior_rating=prior_rating,
            prior_games=prior_games,
            scale=scale,
        )
        teams = [team for team in sorted(record.teams) if team]
        rows.append(
            {
                "player": name,
                "team": " | ".join(teams),
                "league": " | ".join(sorted(x for x in record.leagues if x)),
                "staffel": " | ".join(sorted(x for x in record.staffeln if x)),
                "games": str(record.games),
                "wins": str(record.wins),
                "losses": str(record.games - record.wins),
                "weighted_games": f"{record.weighted_games:.2f}",
                "elo": str(round(elo)),
                "elo_low": str(round(elo - 1.96 * se)),
                "elo_high": str(round(elo + 1.96 * se)),
                "elo_se": str(round(se)),
                "expected_wins": f"{expected:.2f}",
                "residual": f"{observed - expected:.2f}",
            }
        )
    rows.sort(key=lambda row: (-int(row["elo"]), row["player"]))
    return rows


def run(args: argparse.Namespace) -> None:
    source = load_games(args.input)
    played, as_of = assign_weights(outcomes(source), args.half_life_years)
    if not played:
        raise SystemExit(f"No decisive games in {args.input}")

    players = collect(played)
    if args.outcome == "binary":
        for record in players.values():
            record.opponents = [
                (opponent, 1.0 if share > 0.5 else 0.0 if share < 0.5 else 0.5, weight)
                for opponent, share, weight in record.opponents
            ]
    rating = fit_joint_bt(
        players,
        prior_rating=args.prior_rating,
        prior_games=args.prior_games,
        scale=args.scale,
    )
    rows = rating_rows(
        players,
        rating,
        prior_rating=args.prior_rating,
        prior_games=args.prior_games,
        scale=args.scale,
    )
    write_ratings(args.output, rows)

    mean_abs = sum(abs(float(row["residual"])) for row in rows) / len(rows)
    print(
        f"Joint Bradley–Terry on {len(played)} games, {len(rows)} players -> {args.output}"
    )
    if as_of is not None and args.half_life_years > 0:
        print(
            f"As of {as_of:%d.%m.%Y}; half-life {args.half_life_years:g} years."
        )
    else:
        print("All games at full weight (no age decay).")
    print(
        f"Regularization: {args.prior_games:g} virtual draws vs {args.prior_rating:.0f}."
    )
    print(f"Outcome: {args.outcome} ({'win = 1, loss = 0' if args.outcome == 'binary' else 'share of racks/balls'}).")
    print(f"Mean |observed − expected| per player (in games): {mean_abs:.3f}")
    print()
    print(f"{'Elo':>5}  {'95% range':>13}  {'±':>4}  {'W-L':>8}  Player")
    for row in rows[: args.top]:
        bounds = f"{row['elo_low']}–{row['elo_high']}"
        record = f"{row['wins']}-{row['losses']}"
        team = f" ({row['team']})" if row["team"] else ""
        print(
            f"{row['elo']:>5}  {bounds:>13}  {row['elo_se']:>4}  "
            f"{record:>8}  {row['player']}{team}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/player_bt_elo.csv"))
    parser.add_argument(
        "--prior-games",
        type=float,
        default=PRIOR_GAMES,
        help="Virtual draws against 1500 (default: 1, best in backtest_bt.py)",
    )
    parser.add_argument("--prior-rating", type=float, default=PRIOR_RATING)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument(
        "--half-life-years",
        type=float,
        default=0.0,
        help="Age decay half-life. 0 = use every game equally (default)",
    )
    parser.add_argument(
        "--outcome",
        choices=("share", "binary"),
        default="share",
        help="share: 5:4 counts 5/9 (default). binary: any win counts 1",
    )
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
