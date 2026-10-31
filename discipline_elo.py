#!/usr/bin/env python3
"""Overall Elo plus a shrunk adjustment per discipline.

A player's rating in discipline d is R + delta_d. R is fitted as in elo.py:
rack or ball share, one virtual draw against 1500 that ages from the
player's last match, and the same half-life on every game. Each delta has a
normal prior at 0 with spread --tau, so a player with few games in a
discipline keeps almost exactly the overall Elo there. Games without one of
the four disciplines (TournamentApp Multiball) count toward R only.

--holdout fits on games before a date and scores the rack share of the
games after it, for several tau values. tau 0 is the plain overall Elo.

    python discipline_elo.py "Partha Ghosh" "Marcel Haab"
    python discipline_elo.py --holdout 2026-01-01
"""

from __future__ import annotations

import argparse
import csv
import math
from datetime import datetime
from pathlib import Path

import elo

DISCIPLINES = ("14/1e", "8-Ball", "9-Ball", "10-Ball")
DISC_INDEX = {name: i for i, name in enumerate(DISCIPLINES)}
SLOPE = elo.LN10 / elo.SCALE
HALF_LIFE_YEARS = 8 / 12
# Holdout rack-share log loss, tau 0 / 40 / 80:
# cutoff 2025-07-01 0.65319 / 0.65324 / 0.65359
# cutoff 2026-01-01 0.64957 / 0.64959 / 0.64987
# cutoff 2026-06-01 0.64400 / 0.64393 / 0.64401
TAU = 40.0
ROOT = Path(__file__).resolve().parent

Edge = tuple[int, int, float, float]


def build(played: list[elo.Game]) -> tuple[list[str], list[list[Edge]], list[list[int]], list[elo.PlayerRecord]]:
    """Index players and attach each game to both ends."""
    records = elo.collect(played)
    names = sorted(records)
    index = {name: i for i, name in enumerate(names)}
    edges: list[list[Edge]] = [[] for _ in names]
    counts = [[0] * len(DISCIPLINES) for _ in names]
    for winner, loser, row, weight in played:
        disc = DISC_INDEX.get(row.get("discipline", "").strip(), -1)
        share = elo.winner_share(row)
        w, l = index[winner], index[loser]
        edges[w].append((l, disc, share, weight))
        edges[l].append((w, disc, 1.0 - share, weight))
        if disc >= 0:
            counts[w][disc] += 1
            counts[l][disc] += 1
    return names, edges, counts, [records[name] for name in names]


def fit(
    edges: list[list[Edge]],
    ghost: list[float],
    tau: float,
    start: list[float] | None = None,
    max_iter: int = 400,
    tol: float = 0.01,
) -> tuple[list[float], list[list[float]]]:
    """Joint Newton step per player on R and the four deltas."""
    n = len(edges)
    base = list(start) if start else [elo.PRIOR_RATING] * n
    delta = [[0.0] * len(DISCIPLINES) for _ in range(n)]
    prec = 1.0 / (tau * tau) if tau > 0 else 0.0
    for _ in range(max_iter):
        max_step = 0.0
        for i in range(n):
            own = base[i]
            own_delta = delta[i]
            g_all = 0.0
            c_all = 0.0
            g_disc = [0.0] * len(DISCIPLINES)
            c_disc = [0.0] * len(DISCIPLINES)
            for j, disc, share, weight in edges[i]:
                if disc >= 0 and tau > 0:
                    mine = own + own_delta[disc]
                    theirs = base[j] + delta[j][disc]
                else:
                    mine = own
                    theirs = base[j]
                chance = elo.expected_score(mine, theirs)
                g = weight * (share - chance)
                c = weight * chance * (1.0 - chance)
                g_all += g
                c_all += c
                if disc >= 0:
                    g_disc[disc] += g
                    c_disc[disc] += c
            if ghost[i] > 0:
                chance = elo.expected_score(own, elo.PRIOR_RATING)
                g_all += ghost[i] * (0.5 - chance)
                c_all += ghost[i] * chance * (1.0 - chance)
            grad_r = g_all * SLOPE
            hess_r = c_all * SLOPE * SLOPE
            if tau <= 0:
                if hess_r <= 1e-12:
                    continue
                step = max(-100.0, min(100.0, grad_r / hess_r))
                base[i] = own + step
                max_step = max(max_step, abs(step))
                continue
            grad_d = [g_disc[d] * SLOPE - prec * own_delta[d] for d in range(len(DISCIPLINES))]
            hess_d = [c_disc[d] * SLOPE * SLOPE for d in range(len(DISCIPLINES))]
            schur = hess_r - sum(h * h / (h + prec) for h in hess_d)
            if schur <= 1e-12:
                continue
            rhs = grad_r - sum(h * g / (h + prec) for h, g in zip(hess_d, grad_d))
            step_r = max(-100.0, min(100.0, rhs / schur))
            base[i] = own + step_r
            max_step = max(max_step, abs(step_r))
            for d in range(len(DISCIPLINES)):
                step_d = (grad_d[d] - hess_d[d] * step_r) / (hess_d[d] + prec)
                step_d = max(-100.0, min(100.0, step_d))
                own_delta[d] += step_d
                max_step = max(max_step, abs(step_d))
        if max_step < tol:
            break
    return base, delta


def errors(
    edges: list[list[Edge]],
    ghost: list[float],
    base: list[float],
    delta: list[list[float]],
    tau: float,
) -> tuple[list[float], list[list[float]]]:
    """One standard deviation for R and each delta, opponents held fixed."""
    prec = 1.0 / (tau * tau)
    se_r: list[float] = []
    se_d: list[list[float]] = []
    for i, player_edges in enumerate(edges):
        c_all = 0.0
        c_disc = [0.0] * len(DISCIPLINES)
        for j, disc, _share, weight in player_edges:
            if disc >= 0:
                chance = elo.expected_score(base[i] + delta[i][disc], base[j] + delta[j][disc])
            else:
                chance = elo.expected_score(base[i], base[j])
            c = weight * chance * (1.0 - chance)
            c_all += c
            if disc >= 0:
                c_disc[disc] += c
        if ghost[i] > 0:
            chance = elo.expected_score(base[i], elo.PRIOR_RATING)
            c_all += ghost[i] * chance * (1.0 - chance)
        hess_r = c_all * SLOPE * SLOPE
        hess_d = [c * SLOPE * SLOPE for c in c_disc]
        schur = hess_r - sum(h * h / (h + prec) for h in hess_d)
        var_r = 1.0 / schur if schur > 1e-12 else float("inf")
        se_r.append(math.sqrt(var_r))
        se_d.append(
            [math.sqrt(1.0 / (h + prec) + (h / (h + prec)) ** 2 * var_r) for h in hess_d]
        )
    return se_r, se_d


def ghosts(records: list[elo.PlayerRecord], as_of: datetime | None, half_life_years: float) -> list[float]:
    return [elo.ghost_weight(record, 1.0, as_of, half_life_years) for record in records]


def weighted_before(
    played: list[tuple[str, str, dict[str, str]]],
    cutoff: datetime,
    half_life_years: float,
) -> tuple[list[elo.Game], list[tuple[str, str, dict[str, str]]]]:
    """Games before the cutoff, weighted back from it, and the dated games after it."""
    half_life_days = half_life_years * 365.25
    train: list[elo.Game] = []
    test: list[tuple[str, str, dict[str, str]]] = []
    for winner, loser, row in played:
        when = elo.parse_match_date(row.get("match_date", ""))
        if when is not None and when >= cutoff:
            test.append((winner, loser, row))
            continue
        weight = elo.time_weight(when, cutoff, half_life_days) * elo._row_weight(row)
        if weight > 0:
            train.append((winner, loser, row, weight))
    return train, test


def holdout(played: list[tuple[str, str, dict[str, str]]], cutoff: datetime, taus: list[float], half_life_years: float) -> None:
    train, test = weighted_before(played, cutoff, half_life_years)
    names, edges, _counts, records = build(train)
    index = {name: i for i, name in enumerate(names)}
    ghost = ghosts(records, cutoff, half_life_years)
    scored = [
        (index[winner], index[loser], DISC_INDEX[row["discipline"].strip()], elo.winner_share(row), elo._row_weight(row))
        for winner, loser, row in test
        if winner in index and loser in index and row.get("discipline", "").strip() in DISC_INDEX
    ]
    print(f"Train {len(train)} games before {cutoff:%d.%m.%Y}, score {len(scored)} later games.")
    start = None
    for tau in taus:
        base, delta = fit(edges, ghost, tau, start=start)
        if tau == 0:
            start = base
        loss = 0.0
        total = 0.0
        for w, l, disc, share, weight in scored:
            chance = elo.expected_score(base[w] + delta[w][disc], base[l] + delta[l][disc])
            chance = min(1 - 1e-9, max(1e-9, chance))
            loss -= weight * (share * math.log(chance) + (1 - share) * math.log(1 - chance))
            total += weight
        print(f"  tau {tau:5.0f}  rack-share log loss {loss / total:.5f}")


def write_csv(path: Path, names, counts, records, base, delta, se_r, se_d, offset: float) -> None:
    fields = ["player", "pass_nr", "elo", "elo_se", "games"]
    for disc in DISCIPLINES:
        fields += [f"elo_{disc}", f"delta_{disc}", f"delta_se_{disc}", f"games_{disc}"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, name in enumerate(names):
            row = {
                "player": name,
                "pass_nr": records[i].pass_nr,
                "elo": f"{base[i] + offset:.0f}",
                "elo_se": f"{se_r[i]:.0f}",
                "games": records[i].games,
            }
            for d, disc in enumerate(DISCIPLINES):
                row[f"elo_{disc}"] = f"{base[i] + delta[i][d] + offset:.0f}"
                row[f"delta_{disc}"] = f"{delta[i][d]:+.0f}"
                row[f"delta_se_{disc}"] = f"{se_d[i][d]:.0f}"
                row[f"games_{disc}"] = counts[i][d]
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("player", nargs="*", help="Players to print")
    parser.add_argument("--tau", type=float, default=TAU, help=f"Spread of the delta prior in Elo (default {TAU:g})")
    parser.add_argument("--half-life-years", type=float, default=HALF_LIFE_YEARS)
    parser.add_argument("--holdout", help="Cutoff date YYYY-MM-DD; scores several tau values")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "games.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "discipline_elo.csv")
    args = parser.parse_args()

    rows = elo.load_games(args.input)
    played = elo.outcomes(rows)
    if args.holdout:
        cutoff = datetime.strptime(args.holdout, "%Y-%m-%d")
        holdout(played, cutoff, [0, 20, 40, 60, 80, 120], args.half_life_years)
        return

    weighted, as_of = elo.assign_weights(played, args.half_life_years)
    names, edges, counts, records = build(weighted)
    ghost = ghosts(records, as_of, args.half_life_years)
    start, _ = fit(edges, ghost, 0.0)
    base, delta = fit(edges, ghost, args.tau, start=start)
    se_r, se_d = errors(edges, ghost, base, delta, args.tau)
    offset = elo.scale_offset()
    write_csv(args.output, names, counts, records, base, delta, se_r, se_d, offset)
    print(f"Wrote {args.output} ({len(names)} players, tau {args.tau:g}).")

    index = {name: i for i, name in enumerate(names)}
    if args.player:
        print(f"\n{'Player':24} {'Elo':>9}  " + "  ".join(f"{disc:>14}" for disc in DISCIPLINES))
    for query in args.player:
        if query not in index:
            print(f"{query}: not in the fit")
            continue
        i = index[query]
        cells = [
            f"{delta[i][d]:+4.0f} ±{se_d[i][d]:3.0f} ({counts[i][d]:3})"
            for d in range(len(DISCIPLINES))
        ]
        print(f"{query:24} {base[i] + offset:5.0f} ±{se_r[i]:3.0f}  " + "  ".join(f"{cell:>14}" for cell in cells))


if __name__ == "__main__":
    main()
