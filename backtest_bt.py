#!/usr/bin/env python3
"""Compare Bradley–Terry variants by how well they predict unseen games.

For every test season, ratings are fitted on all earlier seasons only, then
used to predict who wins each game of the test season. Each variant gets one
calibration slope (p = logistic(slope * elo_diff)), so the scoring choice
(share vs binary) does not decide the result by its scale alone.

Stability is a split-half check: the full file is split at random into two
halves, both are fitted, and the ranking of regular players is compared.
"""

from __future__ import annotations

import argparse
import math
import random
from datetime import datetime
from pathlib import Path

from bt_elo import fit_joint_bt
from elo import collect, load_games, outcomes, parse_match_date, time_weight, _row_weight

TEST_SEASONS = ("2023/2024", "2024/2025", "2025/2026")


def game_date(row: dict[str, str]) -> datetime:
    when = parse_match_date(row.get("match_date", ""))
    if when is not None:
        return when
    return datetime(int(row["season"][:4]), 11, 1)


def weighted(played, half_life_years: float):
    as_of = max(game_date(row) for _w, _l, row in played)
    days = half_life_years * 365.25
    return [
        (w, l, row, (1.0 if half_life_years <= 0 else time_weight(game_date(row), as_of, days)) * _row_weight(row))
        for w, l, row in played
    ]


def fit(played, *, outcome: str, prior: float, half_life: float) -> dict[str, float]:
    players = collect(weighted(played, half_life))
    if outcome == "binary":
        for record in players.values():
            record.opponents = [
                (o, 1.0 if s > 0.5 else 0.0 if s < 0.5 else 0.5, wt)
                for o, s, wt in record.opponents
            ]
    return fit_joint_bt(players, prior_games=prior)


def calibrated_log_loss(diffs: list[float]) -> tuple[float, float]:
    """diffs are winner-minus-loser Elo. Fit one slope, return (log loss, slope)."""
    k = math.log(10) / 400
    slope = 1.0
    for _ in range(50):
        grad = hess = 0.0
        for d in diffs:
            p = 1 / (1 + math.exp(-slope * k * d))
            grad += (1 - p) * k * d
            hess += p * (1 - p) * (k * d) ** 2
        if hess <= 0:
            break
        step = grad / hess
        slope += step
        if abs(step) < 1e-6:
            break
    loss = sum(math.log1p(math.exp(-slope * k * d)) for d in diffs) / len(diffs)
    return loss, slope


def spearman(a: list[float], b: list[float]) -> float:
    def ranks(x):
        order = sorted(range(len(x)), key=x.__getitem__)
        r = [0.0] * len(x)
        for pos, i in enumerate(order):
            r[i] = pos
        return r

    ra, rb = ranks(a), ranks(b)
    n = len(a)
    ma = sum(ra) / n
    cov = sum((x - ma) * (y - ma) for x, y in zip(ra, rb))
    var = sum((x - ma) ** 2 for x in ra)
    return cov / var


def evaluate(played, variant: dict, min_games: int) -> dict[str, float]:
    diffs: list[float] = []
    for season in TEST_SEASONS:
        train = [g for g in played if g[2]["season"] < season]
        test = [g for g in played if g[2]["season"] == season]
        rating = fit(train, **variant)
        seen: dict[str, int] = {}
        for w, l, _row in train:
            seen[w] = seen.get(w, 0) + 1
            seen[l] = seen.get(l, 0) + 1
        for w, l, _row in test:
            if seen.get(w, 0) >= min_games and seen.get(l, 0) >= min_games:
                diffs.append(rating[w] - rating[l])
    loss, slope = calibrated_log_loss(diffs)
    accuracy = sum(1.0 if d > 0 else 0.5 if d == 0 else 0.0 for d in diffs) / len(diffs)
    return {"n": len(diffs), "log_loss": loss, "accuracy": accuracy, "slope": slope}


def split_half(played, variant: dict, min_games: int, seed: int = 7) -> float:
    rng = random.Random(seed)
    halves: list[list] = [[], []]
    for game in played:
        halves[rng.random() < 0.5].append(game)
    fits = [fit(half, **variant) for half in halves]
    counts = [{}, {}]
    for c, half in zip(counts, halves):
        for w, l, _row in half:
            c[w] = c.get(w, 0) + 1
            c[l] = c.get(l, 0) + 1
    common = [p for p in fits[0] if counts[0].get(p, 0) >= min_games and counts[1].get(p, 0) >= min_games]
    return spearman([fits[0][p] for p in common], [fits[1][p] for p in common])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--min-games", type=int, default=5, help="Both players need this many training games")
    args = parser.parse_args()

    played = outcomes(load_games(args.input))
    variants = [
        {"outcome": o, "prior": p, "half_life": h}
        for o in ("share", "binary")
        for p in (0.5, 1.0, 2.0, 4.0, 8.0)
        for h in (0.0, 1.0, 2.0, 4.0)
    ]
    print(f"Test seasons: {', '.join(TEST_SEASONS)}; both players need ≥{args.min_games} earlier games.")
    print("Log loss: lower is better (0.693 = coin flip). Split-half: ranking agreement, higher is more stable.\n")
    print(f"{'outcome':>7} {'prior':>5} {'half-life':>9} {'games':>6} {'log loss':>9} {'accuracy':>8} {'split-half':>10}")
    results = []
    for v in variants:
        r = evaluate(played, v, args.min_games)
        r["stability"] = split_half(played, v, 20)
        results.append((v, r))
        hl = "none" if v["half_life"] <= 0 else f"{v['half_life']:g}y"
        print(
            f"{v['outcome']:>7} {v['prior']:>5g} {hl:>9} {r['n']:>6} "
            f"{r['log_loss']:>9.4f} {r['accuracy']:>8.3f} {r['stability']:>10.3f}",
            flush=True,
        )
    best_v, best_r = min(results, key=lambda x: x[1]["log_loss"])
    print(f"\nBest by log loss: {best_v} -> {best_r['log_loss']:.4f}, accuracy {best_r['accuracy']:.3f}")


if __name__ == "__main__":
    main()
