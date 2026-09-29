#!/usr/bin/env python3
"""Bradley–Terry with a shared covariance across the four disciplines.

Each player has one strength per discipline. Those four numbers are tied
together by a single 4x4 covariance, estimated jointly with the strengths.
A 10-Ball game updates 10-Ball directly and the other three through that
covariance. The error bar is the curvature of the same fit, so borrowed
games shrink it.

The shrinkage toward independent disciplines is chosen by predicting later
seasons, then the winning value is refit on the whole file.

    python bt_multi.py --backtest
    python bt_multi.py --shrink 0.25
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from elo import (
    PRIOR_RATING,
    SCALE,
    LN10,
    assign_weights,
    load_games,
    outcomes,
    winner_share,
)

DISCIPLINES = ("8-Ball", "9-Ball", "10-Ball", "14/1e")
DISC_INDEX = {name: i for i, name in enumerate(DISCIPLINES)}
SLOPE = LN10 / SCALE
TEST_SEASONS = ("2023/2024", "2024/2025", "2025/2026")
SHORT = ("8", "9", "10", "14_1")


def prepare(played):
    """Index players and attach each game to both ends."""
    names = sorted({name for winner, loser, _row, _w in played for name in (winner, loser)})
    index = {name: i for i, name in enumerate(names)}
    edges: list[list[tuple[int, int, float, float]]] = [[] for _ in names]
    games = []
    for winner, loser, row, weight in played:
        disc = DISC_INDEX.get(row.get("discipline", "").strip())
        if disc is None or weight <= 0:
            continue
        share = float(winner_share(row))
        i, j = index[winner], index[loser]
        edges[i].append((j, disc, share, weight))
        edges[j].append((i, disc, 1.0 - share, weight))
        games.append((i, j, disc, share, weight))
    return names, edges, games


def _expected(left: float, right: float) -> float:
    return 1.0 / (1.0 + 10 ** ((right - left) / SCALE))


def _eye(value: float = 1.0) -> list[list[float]]:
    return [[value if i == j else 0.0 for j in range(4)] for i in range(4)]


def _matvec(matrix: list[list[float]], vector: list[float]) -> list[float]:
    return [sum(matrix[i][j] * vector[j] for j in range(4)) for i in range(4)]


def _solve(matrix: list[list[float]], vector: list[float]) -> list[float]:
    rows = [matrix[i][:] + [vector[i]] for i in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda row: abs(rows[row][col]))
        rows[col], rows[pivot] = rows[pivot], rows[col]
        scale = rows[col][col]
        if abs(scale) < 1e-12:
            raise ZeroDivisionError
        for j in range(col, 5):
            rows[col][j] /= scale
        for row in range(4):
            if row == col:
                continue
            factor = rows[row][col]
            for j in range(col, 5):
                rows[row][j] -= factor * rows[col][j]
    return [rows[i][4] for i in range(4)]


def _inv(matrix: list[list[float]]) -> list[list[float]]:
    columns = [_solve(matrix, [1.0 if i == k else 0.0 for i in range(4)]) for k in range(4)]
    return [[columns[j][i] for j in range(4)] for i in range(4)]


def _add(left: list[list[float]], right: list[list[float]]) -> list[list[float]]:
    return [[left[i][j] + right[i][j] for j in range(4)] for i in range(4)]


def _initial_sigma() -> list[list[float]]:
    spread = 80.0 ** 2
    corr = 0.9
    return [
        [spread * (corr if i != j else 1.0) for j in range(4)]
        for i in range(4)
    ]


def _ridge_sigma(theta, posterior, shrink: float, active: list[bool]) -> list[list[float]]:
    """Empirical covariance of the strengths, with posterior variance added back."""
    used = [i for i, flag in enumerate(active) if flag]
    if len(used) < 20:
        return _initial_sigma()
    scatter = _eye(0.0)
    for i in used:
        resid = [theta[i][d] - PRIOR_RATING for d in range(4)]
        for a in range(4):
            for b in range(4):
                scatter[a][b] += resid[a] * resid[b]
    scatter = [[scatter[a][b] / len(used) for b in range(4)] for a in range(4)]
    extra = _eye(0.0)
    for i in used:
        extra = _add(extra, posterior[i])
    extra = [[extra[a][b] / len(used) for b in range(4)] for a in range(4)]
    scatter = _add(scatter, extra)
    sigma = _eye(0.0)
    for a in range(4):
        for b in range(4):
            diagonal = scatter[a][b] if a == b else 0.0
            coupled = scatter[a][b]
            sigma[a][b] = (1.0 - shrink) * coupled + shrink * diagonal
    # Keep the prior from collapsing: shift eigenvalues up by retrying inversion.
    jitter = 0.0
    while jitter <= 400.0:
        trial = _add(sigma, _eye(jitter))
        try:
            _inv(trial)
            return trial
        except ZeroDivisionError:
            jitter = 400.0 if jitter == 0.0 else jitter * 2
    return _add(sigma, _eye(400.0))


def fit_multi(played, shrink: float, *, rounds: int = 12):
    """MAP fit. shrink=1 keeps disciplines independent. shrink=0 uses the full covariance."""
    names, edges, _games = prepare(played)
    theta = [[PRIOR_RATING] * 4 for _ in names]
    sigma = _initial_sigma()
    active = [sum(weight for _j, _d, _s, weight in bucket) >= 10 for bucket in edges]
    posterior = [_eye(80.0 ** 2) for _ in names]

    for _ in range(rounds):
        precision = _inv(sigma)
        for i, bucket in enumerate(edges):
            current = theta[i][:]
            hess = [[-precision[a][b] for b in range(4)] for a in range(4)]
            for _step in range(6):
                gap = [current[d] - PRIOR_RATING for d in range(4)]
                pulled = _matvec(precision, gap)
                grad = [-pulled[d] for d in range(4)]
                hess = [[-precision[a][b] for b in range(4)] for a in range(4)]
                for j, disc, share, weight in bucket:
                    chance = _expected(current[disc], theta[j][disc])
                    chance = min(1.0 - 1e-6, max(1e-6, chance))
                    grad[disc] += weight * (share - chance) * SLOPE
                    hess[disc][disc] -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
                try:
                    step = _solve(hess, [-value for value in grad])
                except ZeroDivisionError:
                    break
                step = [min(40.0, max(-40.0, value)) for value in step]
                current = [current[d] + step[d] for d in range(4)]
                if max(abs(value) for value in step) < 0.05:
                    break
            theta[i] = current
            try:
                posterior[i] = _inv([[-hess[a][b] for b in range(4)] for a in range(4)])
            except ZeroDivisionError:
                posterior[i] = _eye(80.0 ** 2)
        if shrink >= 0:
            sigma = _ridge_sigma(theta, posterior, shrink, active)
    return names, theta, posterior, sigma


def fit_pooled(played, *, rounds: int = 25):
    """One strength per player, used for every discipline."""
    names, edges, _games = prepare(played)
    theta = [PRIOR_RATING for _ in names]
    prior_precision = 1.0 / (180.0 ** 2)
    for _ in range(rounds):
        max_step = 0.0
        for i, bucket in enumerate(edges):
            current = theta[i]
            for _step in range(6):
                grad = -prior_precision * (current - PRIOR_RATING)
                hess = -prior_precision
                for j, _disc, share, weight in bucket:
                    chance = min(1.0 - 1e-6, max(1e-6, _expected(current, theta[j])))
                    grad += weight * (share - chance) * SLOPE
                    hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
                if abs(hess) < 1e-12:
                    break
                step = min(40.0, max(-40.0, -grad / hess))
                current += step
                max_step = max(max_step, abs(step))
                if abs(step) < 0.05:
                    break
            theta[i] = current
        if max_step < 0.1:
            break
    wide = [[value] * 4 for value in theta]
    posterior = []
    for i, bucket in enumerate(edges):
        hess = -prior_precision
        for j, _disc, _share, weight in bucket:
            chance = min(1.0 - 1e-6, max(1e-6, _expected(theta[i], theta[j])))
            hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
        variance = 1.0 / max(-hess, 1e-12)
        posterior.append(_eye(variance))
    return names, wide, posterior, _eye(180.0 ** 2)


def fit_sparse(played, min_gap: float, *, rounds: int = 12):
    """One shared rating, plus a discipline offset kept only for a repeated gap.

    min_gap is the Elo gap a player with 30 even games must clear before the
    offset survives. Players with fewer games need a larger gap.
    """
    names, edges, _games = prepare(played)
    theta = [PRIOR_RATING for _ in names]
    delta = [[0.0] * 4 for _ in names]
    prior_precision = 1.0 / (180.0 ** 2)
    curvature = 30.0 * 0.25 * SLOPE * SLOPE
    penalty = min_gap * curvature

    def level(player: int, disc: int) -> float:
        return theta[player] + delta[player][disc]

    for _ in range(rounds):
        max_step = 0.0
        for i, bucket in enumerate(edges):
            current = theta[i]
            for _step in range(4):
                grad = -prior_precision * (current - PRIOR_RATING)
                hess = -prior_precision
                for j, disc, share, weight in bucket:
                    chance = min(1.0 - 1e-6, max(1e-6, _expected(current + delta[i][disc], level(j, disc))))
                    grad += weight * (share - chance) * SLOPE
                    hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
                if abs(hess) < 1e-12:
                    break
                step = min(40.0, max(-40.0, -grad / hess))
                current += step
                max_step = max(max_step, abs(step))
                if abs(step) < 0.05:
                    break
            theta[i] = current
        for i, bucket in enumerate(edges):
            for disc in range(4):
                own = [(j, share, weight) for j, d, share, weight in bucket if d == disc]
                if not own:
                    delta[i][disc] = 0.0
                    continue
                current = delta[i][disc]
                grad = hess = 0.0
                for _step in range(4):
                    grad = 0.0
                    hess = 0.0
                    for j, share, weight in own:
                        chance = min(1.0 - 1e-6, max(1e-6, _expected(theta[i] + current, level(j, disc))))
                        grad += weight * (share - chance) * SLOPE
                        hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
                    if abs(hess) < 1e-12:
                        current = 0.0
                        break
                    unpenalized = current - grad / hess
                    thresh = penalty / abs(hess)
                    if abs(unpenalized) <= thresh:
                        current = 0.0
                    else:
                        current = math.copysign(abs(unpenalized) - thresh, unpenalized)
                    current = min(80.0, max(-80.0, current))
                    if abs(current - delta[i][disc]) < 0.05:
                        break
                delta[i][disc] = current
                max_step = max(max_step, abs(current))
        if max_step < 0.2:
            break

    wide = [[theta[i] + delta[i][d] for d in range(4)] for i in range(len(names))]
    posterior = []
    for i, bucket in enumerate(edges):
        hess = -prior_precision
        for j, disc, _share, weight in bucket:
            chance = min(1.0 - 1e-6, max(1e-6, _expected(level(i, disc), level(j, disc))))
            hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
        base = 1.0 / max(-hess, 1e-12)
        cov = _eye(base)
        for disc in range(4):
            if delta[i][disc] == 0.0:
                continue
            disc_hess = 0.0
            for j, d, _share, weight in bucket:
                if d != disc:
                    continue
                chance = min(1.0 - 1e-6, max(1e-6, _expected(level(i, disc), level(j, disc))))
                disc_hess -= weight * chance * (1.0 - chance) * SLOPE * SLOPE
            extra = 1.0 / max(-disc_hess, 1e-12)
            cov[disc][disc] = base + extra
        posterior.append(cov)
    kept = sum(value != 0.0 for row in delta for value in row)
    return names, wide, posterior, kept


def log_loss(played, fit) -> tuple[float, float, int]:
    names, theta, _posterior, _sigma = fit
    index = {name: i for i, name in enumerate(names)}
    total = 0.0
    correct = 0.0
    count = 0
    for winner, loser, row, weight in played:
        disc = DISC_INDEX.get(row.get("discipline", "").strip())
        if disc is None:
            continue
        left = theta[index[winner]][disc] if winner in index else PRIOR_RATING
        right = theta[index[loser]][disc] if loser in index else PRIOR_RATING
        chance = min(1.0 - 1e-6, max(1e-6, _expected(left, right)))
        share = float(winner_share(row))
        total += -(share * math.log(chance) + (1.0 - share) * math.log(1.0 - chance))
        correct += 1.0 if chance > 0.5 else 0.5 if chance == 0.5 else 0.0
        count += 1
    if count == 0:
        return float("nan"), float("nan"), 0
    return total / count, correct / count, count


def backtest(rows) -> float:
    played, _ = assign_weights(outcomes(rows), 0.0)
    options: list[tuple[str, float]] = [("pooled", -1.0)] + [(f"shrink {s:g}", s) for s in (0.0, 0.25, 0.5, 0.75, 1.0)]
    print(f"Held-out seasons: {', '.join(TEST_SEASONS)}. Lower log loss is better.\n")
    print(f"{'model':12} {'games':>7} {'log loss':>9} {'accuracy':>9}")
    scores = []
    for label, shrink in options:
        losses = []
        correct = []
        counts = []
        for season in TEST_SEASONS:
            train = [g for g in played if g[2]["season"] < season]
            test = [g for g in played if g[2]["season"] == season]
            fit = fit_pooled(train) if shrink < 0 else fit_multi(train, shrink)
            loss, acc, count = log_loss(test, fit)
            losses.append(loss * count)
            correct.append(acc * count)
            counts.append(count)
        total = sum(counts)
        loss = sum(losses) / total
        acc = sum(correct) / total
        scores.append((loss, label, shrink))
        print(f"{label:12} {total:7} {loss:9.4f} {acc:9.3f}", flush=True)
    best_loss, best_label, best_shrink = min(scores)
    print(f"\nBest: {best_label}  log loss {best_loss:.4f}")
    return best_shrink


def overall(theta_row: list[float], cov: list[list[float]]) -> tuple[float, float]:
    precision = _inv(cov)
    weight = [sum(precision[d]) for d in range(4)]
    total = sum(weight)
    if total <= 0:
        return sum(theta_row) / 4.0, float("inf")
    elo = sum(weight[d] * theta_row[d] for d in range(4)) / total
    return elo, math.sqrt(1.0 / total)


def write_ratings(path: Path, names, theta, posterior) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["player", "elo", "se"]
    for short in SHORT:
        fields += [f"elo_{short}", f"se_{short}"]
    rows = []
    for i, name in enumerate(names):
        tied = len({round(value, 2) for value in theta[i]}) == 1
        elo, se = (theta[i][0], math.sqrt(posterior[i][0][0])) if tied else overall(theta[i], posterior[i])
        row = {"player": name, "elo": str(round(elo)), "se": str(round(se))}
        for d, short in enumerate(SHORT):
            row[f"elo_{short}"] = str(round(float(theta[i][d])))
            row[f"se_{short}"] = str(round(math.sqrt(max(posterior[i][d][d], 0.0))))
        rows.append(row)
    rows.sort(key=lambda row: (-int(row["elo"]), row["player"]))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_sparse(rows, output: Path) -> None:
    played, _ = assign_weights(outcomes(rows), 0.0)
    gaps = (15.0, 25.0, 40.0, 60.0)
    print(f"Held-out seasons: {', '.join(TEST_SEASONS)}. Lower log loss is better.\n")
    print(f"{'model':16} {'games':>7} {'log loss':>9} {'accuracy':>9} {'offsets':>8}")
    scores = []
    for gap in gaps:
        losses, correct, counts, kept = [], [], [], []
        for season in TEST_SEASONS:
            train = [g for g in played if g[2]["season"] < season]
            test = [g for g in played if g[2]["season"] == season]
            fit = fit_sparse(train, gap)
            loss, acc, count = log_loss(test, fit)
            losses.append(loss * count)
            correct.append(acc * count)
            counts.append(count)
            kept.append(fit[3])
        total = sum(counts)
        loss = sum(losses) / total
        acc = sum(correct) / total
        scores.append((loss, gap))
        print(f"gap {gap:4.0f} Elo     {total:7} {loss:9.4f} {acc:9.3f} {sum(kept)/len(kept):8.0f}", flush=True)
    pooled_losses, pooled_correct, pooled_counts = [], [], []
    for season in TEST_SEASONS:
        train = [g for g in played if g[2]["season"] < season]
        test = [g for g in played if g[2]["season"] == season]
        loss, acc, count = log_loss(test, fit_pooled(train))
        pooled_losses.append(loss * count)
        pooled_correct.append(acc * count)
        pooled_counts.append(count)
    total = sum(pooled_counts)
    pooled_loss = sum(pooled_losses) / total
    print(f"{'one rating':16} {total:7} {pooled_loss:9.4f} {sum(pooled_correct)/total:9.3f} {'0':>8}")
    best_loss, best_gap = min(scores)
    use_sparse = best_loss < pooled_loss - 1e-4
    print(f"\nBest specialty gap: {best_gap:.0f} Elo, log loss {best_loss:.4f}")
    if use_sparse:
        fit = fit_sparse(played, best_gap)
        print(f"Specialty offsets kept on the full file: {fit[3]}")
    else:
        print("One shared rating still predicts better. Writing that fit.")
        fit = fit_pooled(played)
    write_ratings(output, fit[0], fit[1], fit[2])
    print(f"Wrote {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/player_bt_multi.csv"))
    parser.add_argument("--backtest", action="store_true")
    parser.add_argument("--shrink", type=float, default=None, help="0 = full covariance, 1 = independent disciplines")
    parser.add_argument("--sparse", action="store_true", help="Shared rating plus a specialty offset, chosen on held-out seasons")
    args = parser.parse_args()
    rows = load_games(args.input)
    if args.sparse:
        run_sparse(rows, args.output)
        return
    if args.backtest or args.shrink is None:
        shrink = backtest(rows)
    else:
        shrink = args.shrink
    if args.backtest and args.shrink is None:
        # The backtest already reported the winner. Refit only when asked to write ratings
        # by running without --backtest, or when --shrink was not the sole goal.
        pass
    played, _ = assign_weights(outcomes(rows), 0.0)
    if shrink < 0:
        fit = fit_pooled(played)
    else:
        fit = fit_multi(played, shrink)
    names, theta, posterior, sigma = fit
    write_ratings(args.output, names, theta, posterior)
    scale = [math.sqrt(max(sigma[i][i], 1e-9)) for i in range(4)]
    print(f"\nFinal fit shrink={shrink:g} -> {args.output}")
    print("Correlation of the four discipline strengths:")
    print("       " + " ".join(f"{name:>8}" for name in SHORT))
    for i, name in enumerate(SHORT):
        print(f"{name:>7} " + " ".join(f"{sigma[i][j] / (scale[i] * scale[j]):8.2f}" for j in range(4)))


if __name__ == "__main__":
    main()
