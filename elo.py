#!/usr/bin/env python3
"""Calibrated Elo ratings from scraped BVBW game results.

Each rating is the Bradley–Terry / Elo maximum-likelihood value for that
player. Fitting stops when every player's expected wins match their actual
wins, aside from a small pull toward 1500. A single pass with a K-factor
does not have that property, so those ratings are not calibrated.

The scale is classical Elo: 400 points is 10:1 odds on one game. Ratings
describe only the games in the input file. They are not FargoRate scores
and they are not a BVBW or DBU ranking.

elo_low and elo_high are a 95% interval around elo. elo_se is the typical
error, one standard deviation. Both come from the curvature of the fit.

Older games count less. Weight halves every three years, measured back
from the newest match in the file, which is the same decay FargoRate uses.
Adding an older season therefore moves current ratings less than adding
a recent one.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

PRIOR_RATING = 1500.0
SCALE = 400.0
HALF_LIFE_YEARS = 3.0
LN10 = math.log(10)
Game = tuple[str, str, dict[str, str], float]


@dataclass
class PlayerRecord:
    games: int = 0
    wins: int = 0
    weighted_games: float = 0.0
    teams: set[str] = field(default_factory=set)
    leagues: set[str] = field(default_factory=set)
    staffeln: set[str] = field(default_factory=set)
    opponents: list[tuple[str, float, float]] = field(default_factory=list)


def expected_score(rating: float, opponent: float, scale: float = SCALE) -> float:
    """Probability that `rating` beats `opponent` on the Elo scale."""
    diff = (opponent - rating) / scale
    if diff > 20:
        return 0.0
    if diff < -20:
        return 1.0
    return 1.0 / (1.0 + 10**diff)


def load_games(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def outcomes(rows: list[dict[str, str]]) -> list[tuple[str, str, dict[str, str]]]:
    """Return (winner, loser, row) for decisive individual games."""
    played: list[tuple[str, str, dict[str, str]]] = []
    for row in rows:
        winner = row.get("winner", "").strip()
        loser = row.get("loser", "").strip()
        home = row.get("home_player", "").strip()
        away = row.get("away_player", "").strip()
        if not winner or not loser or winner == loser:
            continue
        if winner not in {home, away} or loser not in {home, away}:
            continue
        played.append((winner, loser, row))
    return played


def parse_match_date(text: str) -> datetime | None:
    text = text.strip()
    for fmt in ("%d.%m.%Y", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def time_weight(when: datetime | None, as_of: datetime, half_life_days: float) -> float:
    """Fargo-style decay: half the weight after one half-life, a quarter after two."""
    if when is None or half_life_days <= 0:
        return 1.0
    age_days = (as_of - when).days
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def assign_weights(
    played: list[tuple[str, str, dict[str, str]]],
    half_life_years: float,
) -> tuple[list[Game], datetime | None]:
    """Weight each game by age relative to the newest match in the file."""
    dated = [
        (winner, loser, row, parse_match_date(row.get("match_date", "")))
        for winner, loser, row in played
    ]
    known = [when for *_rest, when in dated if when is not None]
    as_of = max(known) if known else None
    half_life_days = half_life_years * 365.25
    weighted: list[Game] = []
    for winner, loser, row, when in dated:
        if as_of is None or half_life_years <= 0:
            weight = 1.0
        else:
            weight = time_weight(when, as_of, half_life_days)
        weighted.append((winner, loser, row, weight))
    return weighted, as_of


def collect(played: list[Game]) -> dict[str, PlayerRecord]:
    players: dict[str, PlayerRecord] = defaultdict(PlayerRecord)
    for winner, loser, row, weight in played:
        for name, won in ((winner, True), (loser, False)):
            record = players[name]
            record.games += 1
            record.wins += int(won)
            record.weighted_games += weight
            if name == row["home_player"].strip():
                record.teams.add(row["home_team"].strip())
            else:
                record.teams.add(row["away_team"].strip())
            if row.get("league"):
                record.leagues.add(row["league"].strip())
            if row.get("staffel"):
                record.staffeln.add(row["staffel"].strip())
        players[winner].opponents.append((loser, 1.0, weight))
        players[loser].opponents.append((winner, 0.0, weight))
    return players


def fit_elo(
    players: dict[str, PlayerRecord],
    *,
    prior_rating: float = PRIOR_RATING,
    prior_games: float = 1.0,
    scale: float = SCALE,
    max_iter: int = 300,
    tol: float = 0.01,
) -> dict[str, float]:
    """Newton updates until expected wins match actual wins.

    `prior_games` is a fractional draw against `prior_rating`. It keeps an
    undefeated player with one or two games from running off to infinity,
    which is what unregularized Elo does on this league graph.
    """
    rating = {name: prior_rating for name in players}
    slope = LN10 / scale
    for _ in range(max_iter):
        max_step = 0.0
        for name, record in players.items():
            current = rating[name]
            gradient = 0.0
            curvature = 0.0
            for opponent, outcome, weight in record.opponents:
                chance = expected_score(current, rating[opponent], scale)
                gradient += weight * (outcome - chance) * slope
                curvature += weight * chance * (1.0 - chance) * slope * slope
            if prior_games > 0:
                chance = expected_score(current, prior_rating, scale)
                gradient += prior_games * (0.5 - chance) * slope
                curvature += prior_games * chance * (1.0 - chance) * slope * slope
            if curvature <= 1e-12:
                continue
            step = gradient / curvature
            step = max(-100.0, min(100.0, step))
            rating[name] = current + step
            max_step = max(max_step, abs(step))
        if max_step < tol:
            break
    return rating


def player_expected(
    name: str,
    record: PlayerRecord,
    rating: dict[str, float],
    scale: float,
) -> float:
    return sum(
        weight * expected_score(rating[name], rating[opponent], scale)
        for opponent, _outcome, weight in record.opponents
    )


def rating_se(
    name: str,
    record: PlayerRecord,
    rating: dict[str, float],
    *,
    prior_rating: float,
    prior_games: float,
    scale: float,
) -> float:
    """One standard deviation for this player's rating at the fitted point."""
    slope = LN10 / scale
    current = rating[name]
    curvature = 0.0
    for opponent, _outcome, weight in record.opponents:
        chance = expected_score(current, rating[opponent], scale)
        curvature += weight * chance * (1.0 - chance) * slope * slope
    if prior_games > 0:
        chance = expected_score(current, prior_rating, scale)
        curvature += prior_games * chance * (1.0 - chance) * slope * slope
    if curvature <= 1e-12:
        return float("inf")
    return 1.0 / math.sqrt(curvature)


def calibration_rows(
    scope: str,
    players: dict[str, PlayerRecord],
    rating: dict[str, float],
    scale: float,
    *,
    prior_rating: float,
    prior_games: float,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, record in players.items():
        expected = player_expected(name, record, rating, scale)
        weighted_wins = sum(weight * outcome for _opponent, outcome, weight in record.opponents)
        residual = weighted_wins - expected
        elo = rating[name]
        se = rating_se(
            name,
            record,
            rating,
            prior_rating=prior_rating,
            prior_games=prior_games,
            scale=scale,
        )
        rows.append(
            {
                "scope": scope,
                "player": name,
                "team": " | ".join(sorted(record.teams)),
                "league": " | ".join(sorted(record.leagues)),
                "staffel": " | ".join(sorted(record.staffeln)),
                "games": str(record.games),
                "wins": str(record.wins),
                "losses": str(record.games - record.wins),
                "weighted_games": f"{record.weighted_games:.2f}",
                "elo": str(round(elo)),
                "elo_low": str(round(elo - 1.96 * se)),
                "elo_high": str(round(elo + 1.96 * se)),
                "elo_se": str(round(se)),
                "expected_wins": f"{expected:.2f}",
                "residual": f"{residual:.2f}",
            }
        )
    rows.sort(key=lambda row: (-int(row["elo"]), row["player"]))
    return rows


def grouped_games(played: list[Game]) -> dict[str, list[Game]]:
    groups: dict[str, list[Game]] = {"all": played}
    for item in played:
        row = item[2]
        discipline = row.get("discipline", "").strip() or "unknown"
        league = row.get("league", "").strip()
        groups.setdefault(discipline, []).append(item)
        if league:
            groups.setdefault(f"league:{league}", []).append(item)
    return groups


def reliability(
    played: list[Game],
    rating: dict[str, float],
    scale: float,
) -> list[tuple[str, float, float]]:
    """Home win rate against predicted probability, with older games down-weighted."""
    edges = [index / 10 for index in range(0, 11)]
    counts = [0.0] * (len(edges) - 1)
    wins = [0.0] * (len(edges) - 1)
    for winner, _loser, row, weight in played:
        home = row["home_player"].strip()
        away = row["away_player"].strip()
        chance = expected_score(rating[home], rating[away], scale)
        won = 1.0 if home == winner else 0.0
        index = min(int(chance * 10), 9)
        counts[index] += weight
        wins[index] += weight * won
    table: list[tuple[str, float, float]] = []
    for index, (count, won) in enumerate(zip(counts, wins)):
        low = edges[index]
        high = edges[index + 1]
        observed = won / count if count else float("nan")
        table.append((f"{low:.1f}–{high:.1f}", count, observed))
    return table


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope",
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


def mean_abs_residual(rows: list[dict[str, str]], scope: str) -> float:
    scoped = [row for row in rows if row["scope"] == scope]
    if not scoped:
        return float("nan")
    return sum(abs(float(row["residual"])) for row in scoped) / len(scoped)


def run(args: argparse.Namespace) -> None:
    source = load_games(args.input)
    played, as_of = assign_weights(outcomes(source), args.half_life_years)
    if not played:
        raise SystemExit(f"No decisive games in {args.input}")

    all_rows: list[dict[str, str]] = []
    fitted: dict[str, dict[str, float]] = {}
    groups = grouped_games(played)
    for scope, games in groups.items():
        players = collect(games)
        rating = fit_elo(
            players,
            prior_rating=args.prior_rating,
            prior_games=args.prior_games,
            scale=args.scale,
        )
        fitted[scope] = rating
        all_rows.extend(
            calibration_rows(
                scope,
                players,
                rating,
                args.scale,
                prior_rating=args.prior_rating,
                prior_games=args.prior_games,
            )
        )

    all_rows.sort(key=lambda row: (-int(row["elo"]), row["player"], row["scope"]))
    write_csv(args.output, all_rows)
    overall = [row for row in all_rows if row["scope"] == "all"]
    print(
        f"Wrote {len(overall)} players to {args.output} "
        f"from {len(played)} games"
    )
    if as_of is not None and args.half_life_years > 0:
        oldest = min(
            parse_match_date(row.get("match_date", "")) or as_of
            for _winner, _loser, row, _weight in played
        )
        oldest_weight = time_weight(oldest, as_of, args.half_life_years * 365.25)
        print(
            f"Ratings as of {as_of:%d.%m.%Y}. "
            f"A game loses half its weight every {args.half_life_years:g} years "
            f"(oldest game in this file counts as {oldest_weight:.0%})."
        )
    print(
        "Mean |weighted wins − expected wins| on all games: "
        f"{mean_abs_residual(all_rows, 'all'):.3f}"
    )
    print(
        "That gap is the prior pulling sparse records toward "
        f"{args.prior_rating:.0f}. It shrinks as more games are scraped."
    )
    print()
    print(f"{'Elo':>5}  {'95% range':>13}  {'±':>4}  {'W-L':>7}  Player")
    for row in overall[: args.top]:
        record = f"{row['wins']}-{row['losses']}"
        bounds = f"{row['elo_low']}–{row['elo_high']}"
        print(
            f"{row['elo']:>5}  {bounds:>13}  {row['elo_se']:>4}  "
            f"{record:>7}  {row['player']} ({row['team']})"
        )
    print()
    print("Calibration of predicted win probabilities (all games):")
    print(f"{'P(win)':>8}  {'games':>6}  {'observed':>8}")
    for label, count, observed in reliability(played, fitted["all"], args.scale):
        if count == 0:
            continue
        print(f"{label:>8}  {count:6.0f}  {observed:8.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/games.csv"),
        help="Scraped games CSV (default: data/games.csv)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/player_elo.csv"),
        help="Ratings CSV (default: data/player_elo.csv)",
    )
    parser.add_argument(
        "--prior-games",
        type=float,
        default=1.0,
        help="Virtual draws against 1500. Use 0 for unshrunk Elo (default: 1)",
    )
    parser.add_argument("--prior-rating", type=float, default=PRIOR_RATING)
    parser.add_argument("--scale", type=float, default=SCALE)
    parser.add_argument(
        "--half-life-years",
        type=float,
        default=HALF_LIFE_YEARS,
        help="Age at which a game keeps half its weight. 0 disables decay (default: 3)",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=15,
        help="How many overall ratings to print",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
