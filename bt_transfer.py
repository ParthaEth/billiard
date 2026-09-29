#!/usr/bin/env python3
"""Bradley–Terry with a shared transfer matrix across disciplines.

Each player has four skills. The diagonal of the transfer matrix is fixed
at 1, so a discipline always counts in full. An off-diagonal entry is the
extra influence of another skill. It is kept strictly below 1 and pulled
toward 0. The saved pull is TRANSFER_TAU, chosen on a random 5-fold holdout.

    python bt_transfer.py
"""

from __future__ import annotations

import csv
import math
import random
from pathlib import Path

from elo import PRIOR_RATING, assign_weights, load_games, outcomes

from bt_multi import (
    DISCIPLINES,
    SHORT,
    SLOPE,
    _expected,
    _inv,
    _solve,
    fit_pooled,
    log_loss,
    prepare,
)

PRIOR_PRECISION = 1.0 / (180.0 ** 2)
SLOPE2 = SLOPE * SLOPE
FOLDS = 5
# Random 5-fold holdout, diagonal fixed at 1, off-diagonals strictly below 1.
# tau 0.10 log loss 0.6521, tau 0.25 0.6529, tau 0.50 0.6532.
TRANSFER_TAU = 0.10
SHOW = (
    "Wadea Al Mahamid",
    "Konrad Bau",
    "Alexander Fischer",
    "Mark Hast",
    "Partha Ghosh",
    "Patrick Coklica",
    "Nikolaos Seristatidis",
)


def _identity() -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(4)] for i in range(4)]


def _lock_diagonal(weights: list[list[float]]) -> list[list[float]]:
    locked = [row[:] for row in weights]
    for disc in range(4):
        locked[disc][disc] = 1.0
    return locked


def _dot(row: list[float], vec: list[float]) -> float:
    return row[0] * vec[0] + row[1] * vec[1] + row[2] * vec[2] + row[3] * vec[3]


def _solve3(matrix: list[list[float]], vector: list[float]) -> list[float]:
    big = [[0.0] * 4 for _ in range(4)]
    rhs = [0.0] * 4
    for i in range(3):
        for j in range(3):
            big[i][j] = matrix[i][j]
        rhs[i] = vector[i]
    big[3][3] = 1.0
    return _solve(big, rhs)[:3]


def _player_step(u_i: list[float], bucket, u, weights) -> tuple[list[float], list[list[float]]]:
    current = u_i[:]
    hess = [[0.0] * 4 for _ in range(4)]
    for _step in range(4):
        grad = [-PRIOR_PRECISION * current[k] for k in range(4)]
        hess = [[-PRIOR_PRECISION if a == b else 0.0 for b in range(4)] for a in range(4)]
        for j, disc, share, weight in bucket:
            row = weights[disc]
            left = _dot(row, current)
            right = _dot(row, u[j])
            chance = min(1.0 - 1e-6, max(1e-6, _expected(left, right)))
            pull = weight * (share - chance) * SLOPE
            curve = weight * chance * (1.0 - chance) * SLOPE2
            for a in range(4):
                grad[a] += pull * row[a]
                for b in range(4):
                    hess[a][b] -= curve * row[a] * row[b]
        try:
            step = _solve(hess, [-value for value in grad])
        except ZeroDivisionError:
            break
        step = [min(40.0, max(-40.0, value)) for value in step]
        current = [current[k] + step[k] for k in range(4)]
        if max(abs(value) for value in step) < 0.05:
            break
    return current, hess


def _row_update(disc: int, alpha: list[float], games, u, prior_precision: float) -> list[float]:
    """One Gauss–Newton step on the off-diagonal influences. Diagonal stays 1."""
    others = [k for k in range(4) if k != disc]
    grad_w = [0.0] * 4
    outer = [[0.0] * 4 for _ in range(4)]
    for i, j, _game_disc, share, weight in games:
        left = _dot(alpha, u[i])
        right = _dot(alpha, u[j])
        chance = min(1.0 - 1e-6, max(1e-6, _expected(left, right)))
        pull = weight * (share - chance) * SLOPE
        curve = weight * chance * (1.0 - chance) * SLOPE2
        for a in others:
            delta = u[i][a] - u[j][a]
            grad_w[a] += pull * delta
            for b in others:
                outer[a][b] -= curve * delta * (u[i][b] - u[j][b])
    grad_b = [grad_w[k] - prior_precision * alpha[k] for k in others]
    hess_b = []
    for a in others:
        hess_b.append([outer[a][b] - (prior_precision if a == b else 0.0) for b in others])
    try:
        step = _solve3(hess_b, [-value for value in grad_b])
    except ZeroDivisionError:
        return alpha
    updated = alpha[:]
    updated[disc] = 1.0
    for index, k in enumerate(others):
        # Off-diagonal influence stays strictly below the discipline's own weight of 1.
        updated[k] = min(math.nextafter(1.0, 0.0), max(-0.5, alpha[k] + min(0.3, max(-0.3, step[index]))))
    return updated


def fit_transfer(played, *, rounds: int = 10, learn: bool = True, weights=None, tau: float = TRANSFER_TAU):
    names, edges, games = prepare(played)
    u = [[0.0] * 4 for _ in names]
    weights = _identity() if weights is None else _lock_diagonal(weights)
    prior_precision = 1.0 / (tau * tau)
    last_hess = [[[-PRIOR_PRECISION if a == b else 0.0 for b in range(4)] for a in range(4)] for _ in names]
    by_disc: list[list] = [[], [], [], []]
    for game in games:
        by_disc[game[2]].append(game)
    for _ in range(rounds):
        for i, bucket in enumerate(edges):
            if not bucket:
                continue
            u[i], last_hess[i] = _player_step(u[i], bucket, u, weights)
        if not learn:
            continue
        for disc in range(4):
            for _step in range(2):
                weights[disc] = _row_update(disc, weights[disc], by_disc[disc], u, prior_precision)
            weights[disc][disc] = 1.0
    predictive = []
    posterior = []
    for i in range(len(names)):
        try:
            cov = _inv([[-last_hess[i][a][b] for b in range(4)] for a in range(4)])
        except ZeroDivisionError:
            cov = [[(180.0 ** 2) if a == b else 0.0 for b in range(4)] for a in range(4)]
        rating = []
        var = [[0.0] * 4 for _ in range(4)]
        for disc in range(4):
            row = weights[disc]
            rating.append(PRIOR_RATING + _dot(row, u[i]))
            variance = 0.0
            for a in range(4):
                for b in range(4):
                    variance += row[a] * cov[a][b] * row[b]
            var[disc][disc] = max(variance, 1e-8)
        predictive.append(rating)
        posterior.append(var)
    return names, predictive, posterior, weights


def _score(played, fit) -> tuple[float, float, int]:
    loss, acc, count = log_loss(played, fit)
    return loss, acc, count


def _folds(played, folds: int, seed: int) -> list[list[int]]:
    order = list(range(len(played)))
    random.Random(seed).shuffle(order)
    return [order[index::folds] for index in range(folds)]


def validate(played) -> float:
    folds = _folds(played, FOLDS, seed=0)
    taus = (0.10, 0.25, 0.50)
    print(f"Random {FOLDS}-fold holdout, {len(played)} games. Diagonal fixed at 1.\n")
    print(f"{'model':22} {'games':>7} {'log loss':>9} {'accuracy':>9}")
    totals = {f"transfer τ={tau:.2f}": [0.0, 0.0, 0] for tau in taus}
    totals["separate"] = [0.0, 0.0, 0]
    totals["one rating"] = [0.0, 0.0, 0]
    for fold, test_index in enumerate(folds, start=1):
        print(f"  fold {fold}...", flush=True)
        test_set = set(test_index)
        train = [played[i] for i in range(len(played)) if i not in test_set]
        test = [played[i] for i in test_index]
        for tau in taus:
            learned = fit_transfer(train, learn=True, tau=tau)
            loss, acc, count = _score(test, learned)
            bucket = totals[f"transfer τ={tau:.2f}"]
            bucket[0] += loss * count
            bucket[1] += acc * count
            bucket[2] += count
            off = max(abs(learned[3][d][k]) for d in range(4) for k in range(4) if d != k)
            print(f"    τ={tau:.2f} largest off-diagonal {off:.3f}", flush=True)
        for name, fit in (("separate", fit_transfer(train, learn=False)), ("one rating", fit_pooled(train))):
            loss, acc, count = _score(test, fit)
            bucket = totals[name]
            bucket[0] += loss * count
            bucket[1] += acc * count
            bucket[2] += count
    best_tau = taus[0]
    best_loss = float("inf")
    for name, (loss_sum, acc_sum, count) in totals.items():
        loss = loss_sum / count
        print(f"{name:22} {count:7} {loss:9.4f} {acc_sum / count:9.3f}")
        if name.startswith("transfer") and loss < best_loss:
            best_loss = loss
            best_tau = float(name.split("=")[1])
    return best_tau


def _check_gradient() -> None:
    """Finite-difference check on a toy league before the real fit."""
    rows = []
    pairs = (("A", "B", "5:3"), ("A", "C", "5:4"), ("B", "C", "5:2"), ("D", "A", "5:1"))
    for disc in DISCIPLINES:
        for home, away, score in pairs:
            rows.append(
                {
                    "winner": home,
                    "loser": away,
                    "home_player": home,
                    "away_player": away,
                    "frame_score": score,
                    "discipline": disc,
                    "match_date": "01.06.2025",
                    "season": "2024/2025",
                }
            )
    played, _ = assign_weights(outcomes(rows), 0.0)
    names, _edges, games = prepare(played)
    u = [[(hash(name + str(k)) % 50) - 25 for k in range(4)] for name in names]
    alpha = [1.0, 0.2, -0.1, 0.15]
    own = [game for game in games if game[2] == 0]

    def loss_for(row: list[float]) -> float:
        total = 0.0
        for i, j, _disc, share, weight in own:
            chance = min(1.0 - 1e-6, max(1e-6, _expected(_dot(row, u[i]), _dot(row, u[j]))))
            total += -(share * math.log(chance) + (1.0 - share) * math.log(1.0 - chance))
        return total

    grad_w = [0.0] * 4
    for i, j, _disc, share, weight in own:
        chance = min(1.0 - 1e-6, max(1e-6, _expected(_dot(alpha, u[i]), _dot(alpha, u[j]))))
        pull = weight * (share - chance) * SLOPE
        for a in (1, 2, 3):
            grad_w[a] += pull * (u[i][a] - u[j][a])
    numeric = []
    eps = 1e-4
    for m in (1, 2, 3):
        bumped = alpha[:]
        bumped[m] += eps
        numeric.append((loss_for(alpha) - loss_for(bumped)) / eps)
    analytic = [grad_w[m] for m in (1, 2, 3)]
    gap = max(abs(analytic[i] - numeric[i]) for i in range(3))
    updated = _row_update(0, alpha, own, u, prior_precision=1.0 / (0.25 ** 2))
    if gap > 1e-3 or updated[0] != 1.0:
        raise SystemExit(f"transfer gradient check failed, max gap {gap:.4g}")
    print(f"Gradient check ok, max gap {gap:.3g}")


def write_ratings(path: Path, names, theta, posterior) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["player", "elo", "se"]
    for short in SHORT:
        fields += [f"elo_{short}", f"se_{short}"]
    rows = []
    for i, name in enumerate(names):
        precision = [1.0 / posterior[i][d][d] for d in range(4)]
        total = sum(precision)
        elo = sum(precision[d] * theta[i][d] for d in range(4)) / total
        se = math.sqrt(1.0 / total)
        row = {"player": name, "elo": str(round(elo)), "se": str(round(se))}
        for d, short in enumerate(SHORT):
            row[f"elo_{short}"] = str(round(theta[i][d]))
            row[f"se_{short}"] = str(round(math.sqrt(posterior[i][d][d])))
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["elo"]), row["player"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    by_name = {row["player"]: row for row in rows}
    print("\nPlayer ratings (discipline columns are the transferred ratings):")
    print(f"{'player':24} {'elo':>5} {'±':>4}  " + " ".join(f"{short:>12}" for short in ("8", "9", "10", "14/1")))
    shown = sorted((by_name[name] for name in SHOW if name in by_name), key=lambda row: -int(row["elo"]))
    for row in shown:
        name = row["player"]
        cells = " ".join(
            f"{row[f'elo_{short}']:>5}±{row[f'se_{short}']:<5}" for short in SHORT
        )
        print(f"{name:24} {row['elo']:>5} {row['se']:>4}  {cells}")


def main() -> None:
    _check_gradient()
    rows = load_games(Path("data/games.csv"))
    played, _ = assign_weights(outcomes(rows), 0.0)
    tau = TRANSFER_TAU
    print(f"\nFull-data transfer matrix at τ={tau:.2f}. Diagonal is fixed at 1.")
    print("Row = discipline being predicted, column = extra skill used.")
    names, theta, posterior, weights = fit_transfer(played, learn=True, tau=tau)
    header = " ".join(f"{name:>8}" for name in ("8", "9", "10", "14/1"))
    print(f"{'':8} {header}")
    for disc, name in enumerate(("8", "9", "10", "14/1")):
        print(f"{name:>8} " + " ".join(f"{weights[disc][k]:8.3f}" for k in range(4)))
    output = Path("data/player_bt_transfer.csv")
    write_ratings(output, names, theta, posterior)
    print(f"\nWrote {output}")


if __name__ == "__main__":
    main()
