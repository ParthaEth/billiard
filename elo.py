#!/usr/bin/env python3
"""Calibrated Elo ratings from scraped BVBW game results.

Each rating is the Bradley–Terry / Elo maximum-likelihood value for that
player. Fitting stops when every player's expected share of racks and balls
matches the share they actually took, aside from a small pull toward 1500.
A single pass with a K-factor does not have that property, so those ratings
are not calibrated.

The scale is classical Elo: 400 points is 10:1 odds on one rack, or on one
ball in 14/1. A 5:0 is every rack; a 5:4 is five racks out of nine. A
60:30 in straight pool is a larger share than 60:52. Ratings describe only
the games in the input file. They are not FargoRate scores and they are not
a BVBW or DBU ranking.

elo_low and elo_high are a 95% interval around elo. elo_se is the typical
error, one standard deviation. Both come from the curvature of the fit.

Older games count less. Weight halves every three years, measured back
from the newest match in the file, which is the same decay FargoRate uses.
The one virtual draw against 1500 is dated on each player's last match and
ages on that same clock. After a player stops, the rating stays where the
results left it and the error bar grows.
"""

from __future__ import annotations

import argparse
import configparser
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
    pass_nr: str = ""
    teams: set[str] = field(default_factory=set)
    leagues: set[str] = field(default_factory=set)
    staffeln: set[str] = field(default_factory=set)
    opponents: list[tuple[str, float, float]] = field(default_factory=list)
    last_played: datetime | None = None
    open_result: bool = False


def scale_offset(path: Path | None = None) -> float:
    """Frozen shift from fitted Elo to the published scale.

    The fit still uses a ghost at 1500. This constant was measured once,
    when the higher BW leagues were added, and is not recomputed.
    The +4.30 offset in scale.cfg stays fixed, so leave that file alone.
    """
    path = path or Path(__file__).with_name("scale.cfg")
    parser = configparser.ConfigParser()
    if not parser.read(path, encoding="utf-8"):
        return 0.0
    try:
        return float(parser["scale"]["offset"])
    except (KeyError, ValueError):
        return 0.0


def expected_score(rating: float, opponent: float, scale: float = SCALE) -> float:
    """Probability that `rating` beats `opponent` on the Elo scale."""
    diff = (opponent - rating) / scale
    if diff > 20:
        return 0.0
    if diff < -20:
        return 1.0
    return 1.0 / (1.0 + 10**diff)


def load_games(path: Path) -> list[dict[str, str]]:
    """Load a results file, plus data/inhouse.csv when loading games.csv.

    The inhouse file is the hand-edited club list. Its columns are
    date, player, opponent, score, discipline. Single frames (1:0 or 0:1)
    between the same two players, on the same day and in the same discipline,
    are added into one score. A short club set is weighted by club_sets.cfg:
    racks out of 5, 7 or 5, and 14/1 balls out of 70. League rows are left
    as scraped.
    """
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    extra = path.parent / "inhouse.csv"
    if path.name == "games.csv" and extra.exists():
        rows.extend(_inhouse_games(extra))
    if path.name == "games.csv":
        from identity import apply_identities

        apply_identities(rows, identity_path=path.parent / "player_identity.csv")
    return rows


def _inhouse_games(path: Path) -> list[dict[str, str]]:
    pending: list[dict] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for index, row in enumerate(csv.DictReader(handle), start=2):
            player = (row.get("player") or "").strip()
            opponent = (row.get("opponent") or "").strip()
            score = (row.get("score") or "").strip().replace(" ", "")
            discipline = (row.get("discipline") or "").strip()
            date = (row.get("date") or "").strip()
            if not player and not opponent and not score:
                continue
            if not player or not opponent or ":" not in score:
                raise SystemExit(f"{path}:{index}: need player, opponent, and a score like 5:3")
            left, right = score.split(":", 1)
            try:
                taken, conceded = int(left), int(right)
            except ValueError:
                raise SystemExit(f"{path}:{index}: score {score!r} is not two numbers") from None
            if taken == conceded:
                continue
            when = parse_match_date(date)
            if when is None:
                season = ""
            elif when.month >= 9:
                season = f"{when.year}/{when.year + 1}"
            else:
                season = f"{when.year - 1}/{when.year}"
            if discipline in {"8", "8-Ball", "8-ball"}:
                discipline = "8-Ball"
            elif discipline in {"9", "9-Ball", "9-ball"}:
                discipline = "9-Ball"
            elif discipline in {"10", "10-Ball", "10-ball"}:
                discipline = "10-Ball"
            elif discipline.lower().startswith("14"):
                discipline = "14/1e"
            pending.append(
                {
                    "index": index,
                    "player": player,
                    "opponent": opponent,
                    "taken": taken,
                    "conceded": conceded,
                    "discipline": discipline,
                    "date": date,
                    "season": season,
                }
            )
    return [_inhouse_row(item) for item in _aggregate_single_frames(pending)]


def _aggregate_single_frames(rows: list[dict]) -> list[dict]:
    """Add same-day 1:0 and 0:1 rows between one pair into a single score."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    order: list[tuple] = []
    for row in rows:
        if row["taken"] + row["conceded"] == 1:
            key = ("frame", row["date"], row["discipline"], frozenset((row["player"], row["opponent"])))
        else:
            key = ("row", row["index"])
        if key not in groups:
            order.append(key)
        groups[key].append(row)
    merged: list[dict] = []
    for key in order:
        group = groups[key]
        if key[0] != "frame" or len(group) == 1:
            merged.extend(group)
            continue
        player = group[0]["player"]
        taken = conceded = 0
        for item in group:
            if item["player"] == player:
                taken += item["taken"]
                conceded += item["conceded"]
            else:
                taken += item["conceded"]
                conceded += item["taken"]
        if taken == conceded:
            continue
        combined = dict(group[0])
        combined["taken"] = taken
        combined["conceded"] = conceded
        merged.append(combined)
    return merged


def _inhouse_row(item: dict) -> dict[str, str]:
    player = item["player"]
    opponent = item["opponent"]
    taken = item["taken"]
    conceded = item["conceded"]
    score = f"{taken}:{conceded}"
    discipline = item["discipline"]
    date = item["date"]
    index = item["index"]
    winner, loser = (player, opponent) if taken > conceded else (opponent, player)
    straight = score if discipline == "14/1e" else ""
    return {
        "season": item["season"],
        "league": "Intern",
        "staffel": "Tübinger BC",
        "spieltag": str(index - 1),
        "match_date": date,
        "match_time": "",
        "home_team": "Tübinger BC",
        "away_team": "Tübinger BC",
        "match_score": score,
        "round": "",
        "game_no": "1",
        "discipline": discipline,
        "home_player": player,
        "away_player": opponent,
        "frame_score": score,
        "home_points": "1" if winner == player else "0",
        "away_points": "1" if winner == opponent else "0",
        "winner": winner,
        "loser": loser,
        "winner_side": "home" if winner == player else "away",
        "straight_pool_punkte": straight,
        "straight_pool_aufnahmen": "",
        "straight_pool_hs": "",
        "straight_pool_gd": "",
        "report_url": f"internal://tbc/{date}/{index - 1}",
        "game_weight": f"{_set_weight(discipline, taken, conceded):.4f}",
    }


def load_set_lengths(path: Path | None = None) -> dict[str, float]:
    """Read full-set lengths from club_sets.cfg. Keys are discipline names."""
    path = path or Path(__file__).with_name("club_sets.cfg")
    parser = configparser.ConfigParser()
    if not parser.read(path, encoding="utf-8"):
        raise SystemExit(f"missing club set lengths: {path}")
    if "full_set" not in parser:
        raise SystemExit(f"{path}: need a [full_set] section")
    lengths: dict[str, float] = {}
    for name, text in parser["full_set"].items():
        try:
            length = float(text)
        except ValueError:
            raise SystemExit(f"{path}: {name} = {text!r} is not a number") from None
        if length <= 0:
            raise SystemExit(f"{path}: {name} must be greater than 0")
        lengths[name] = length
    return lengths


def _set_weight(discipline: str, taken: int, conceded: int) -> float:
    """Short club sets count as the fraction of club_sets.cfg they filled."""
    length = SET_LENGTHS.get(discipline.lower())
    if not length:
        return 1.0
    return min(1.0, (taken + conceded) / length)


SET_LENGTHS = load_set_lengths()


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


def alex_time_weight(age_days: float) -> float:
    """Alex's curve: 1 now, 0.75 after a year, 0.20 after two years, then 0.

    The three points (0, 2), (1, 1.5), (2, 0.4) divided by 2. The quadratic
    through them reaches 0 at about 2.3 years.
    """
    if age_days <= 0:
        return 1.0
    years = age_days / 365.25
    return max(0.0, 1.0 - 0.1 * years - 0.15 * years * years)


def assign_alex_weights(
    played: list[tuple[str, str, dict[str, str]]],
) -> tuple[list[Game], datetime | None]:
    """Weight each game with Alex's curve, then by the club set length."""
    dated = [
        (winner, loser, row, parse_match_date(row.get("match_date", "")))
        for winner, loser, row in played
    ]
    known = [when for *_rest, when in dated if when is not None]
    as_of = max(known) if known else None
    weighted: list[Game] = []
    for winner, loser, row, when in dated:
        if as_of is None or when is None:
            decay = 1.0
        else:
            decay = alex_time_weight((as_of - when).days)
        weight = decay * _row_weight(row)
        if weight > 0:
            weighted.append((winner, loser, row, weight))
    return weighted, as_of


def time_weight(when: datetime | None, as_of: datetime, half_life_days: float) -> float:
    """Fargo-style decay: half the weight after one half-life, a quarter after two."""
    if when is None or half_life_days <= 0:
        return 1.0
    age_days = (as_of - when).days
    if age_days <= 0:
        return 1.0
    return 0.5 ** (age_days / half_life_days)


def ghost_weight(
    record: PlayerRecord,
    prior_games: float,
    as_of: datetime | None,
    half_life_years: float,
) -> float:
    """Virtual draw against 1500, played on the player's last match day.

    It then ages with the same half-life as a real result. An undated result
    counts as current, so the draw stays at full weight too.
    """
    if prior_games <= 0:
        return 0.0
    if record.open_result or as_of is None or half_life_years <= 0:
        return prior_games
    return prior_games * time_weight(record.last_played, as_of, half_life_years * 365.25)


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
        weighted.append((winner, loser, row, weight * _row_weight(row)))
    return weighted, as_of


def _row_weight(row: dict[str, str]) -> float:
    text = (row.get("game_weight") or "").strip()
    if not text:
        return 1.0
    try:
        return float(text)
    except ValueError:
        return 1.0


def _score_pair(text: str) -> tuple[int, int] | None:
    if ":" not in text:
        return None
    left, right = text.split(":", 1)
    try:
        home_score, away_score = int(left), int(right)
    except ValueError:
        return None
    if home_score < 0 or away_score < 0 or home_score + away_score == 0:
        return None
    return home_score, away_score


def winner_share(row: dict[str, str]) -> float:
    """Fraction of racks or balls taken by the winner.

    5:0 is 1.00, 5:4 is 5/9, 60:30 is 60/90, 60:52 is 60/112. A game with
    no usable score stays a plain win.
    """
    discipline = row.get("discipline", "").strip()
    text = row.get("straight_pool_punkte", "") if discipline == "14/1e" else ""
    parsed = _score_pair(text.strip()) or _score_pair(row.get("frame_score", "").strip())
    if parsed is None:
        return 1.0
    home_score, away_score = parsed
    winner = row.get("winner", "").strip()
    if winner == row.get("home_player", "").strip():
        taken, conceded = home_score, away_score
    elif winner == row.get("away_player", "").strip():
        taken, conceded = away_score, home_score
    else:
        return 1.0
    if taken < conceded or taken + conceded <= 0:
        return 1.0
    return taken / (taken + conceded)


def collect(played: list[Game]) -> dict[str, PlayerRecord]:
    players: dict[str, PlayerRecord] = defaultdict(PlayerRecord)
    for winner, loser, row, weight in played:
        share = winner_share(row)
        when = parse_match_date(row.get("match_date", ""))
        for name, won in ((winner, True), (loser, False)):
            record = players[name]
            record.games += 1
            record.wins += int(won)
            record.weighted_games += weight
            if when is None:
                record.open_result = True
            elif record.last_played is None or when > record.last_played:
                record.last_played = when
            if name == row["home_player"].strip():
                record.teams.add(row["home_team"].strip())
                if row.get("home_pass"):
                    record.pass_nr = row["home_pass"]
            else:
                record.teams.add(row["away_team"].strip())
                if row.get("away_pass"):
                    record.pass_nr = row["away_pass"]
            if row.get("league"):
                record.leagues.add(row["league"].strip())
            if row.get("staffel"):
                record.staffeln.add(row["staffel"].strip())
        players[winner].opponents.append((loser, share, weight))
        players[loser].opponents.append((winner, 1.0 - share, weight))
    return players


def race_win_probability(unit_probability: float, target: int) -> float:
    """Chance of reaching `target` racks or balls before the opponent.

    `unit_probability` is the Elo chance of winning one rack, or one ball
    in 14/1. The match ends when either player reaches the target.
    """
    if target <= 1:
        return min(1.0, max(0.0, unit_probability))
    if unit_probability <= 0.0:
        return 0.0
    if unit_probability >= 1.0:
        return 1.0
    log_term = target * math.log(unit_probability)
    total = math.exp(log_term)
    log_opponent = math.log(1.0 - unit_probability)
    for lost in range(1, target):
        log_term += log_opponent + math.log(target - 1 + lost) - math.log(lost)
        if log_term < -700:
            continue
        total += math.exp(log_term)
    return min(1.0, max(0.0, total))


def fit_elo(
    players: dict[str, PlayerRecord],
    *,
    prior_rating: float = PRIOR_RATING,
    prior_games: float = 1.0,
    scale: float = SCALE,
    max_iter: int = 300,
    tol: float = 0.01,
    as_of: datetime | None = None,
    half_life_years: float = 0.0,
) -> dict[str, float]:
    """Newton updates until expected score share matches the score taken.

    `prior_games` is a fractional draw against `prior_rating`, dated on each
    player's last match. It keeps an undefeated player with one or two games
    from running off to infinity, which is what unregularized Elo does on
    this league graph. Once those matches age, the draw ages with them.
    """
    rating = {name: prior_rating for name in players}
    slope = LN10 / scale
    for _ in range(max_iter):
        max_step = 0.0
        for name, record in players.items():
            current = rating[name]
            gradient = 0.0
            curvature = 0.0
            draw = ghost_weight(record, prior_games, as_of, half_life_years)
            for opponent, outcome, weight in record.opponents:
                chance = expected_score(current, rating[opponent], scale)
                gradient += weight * (outcome - chance) * slope
                curvature += weight * chance * (1.0 - chance) * slope * slope
            if draw > 0:
                chance = expected_score(current, prior_rating, scale)
                gradient += draw * (0.5 - chance) * slope
                curvature += draw * chance * (1.0 - chance) * slope * slope
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
    as_of: datetime | None = None,
    half_life_years: float = 0.0,
) -> float:
    """One standard deviation for this player's rating at the fitted point."""
    slope = LN10 / scale
    current = rating[name]
    curvature = 0.0
    draw = ghost_weight(record, prior_games, as_of, half_life_years)
    for opponent, _outcome, weight in record.opponents:
        chance = expected_score(current, rating[opponent], scale)
        curvature += weight * chance * (1.0 - chance) * slope * slope
    if draw > 0:
        chance = expected_score(current, prior_rating, scale)
        curvature += draw * chance * (1.0 - chance) * slope * slope
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
    as_of: datetime | None = None,
    half_life_years: float = 0.0,
    offset: float = 0.0,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for name, record in players.items():
        expected = player_expected(name, record, rating, scale)
        weighted_wins = sum(weight * outcome for _opponent, outcome, weight in record.opponents)
        residual = weighted_wins - expected
        elo = rating[name] + offset
        se = rating_se(
            name,
            record,
            rating,
            prior_rating=prior_rating,
            prior_games=prior_games,
            scale=scale,
            as_of=as_of,
            half_life_years=half_life_years,
        )
        rows.append(
            {
                "scope": scope,
                "player": name,
                "pass_nr": record.pass_nr,
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
    """Home score share against predicted rack or ball probability."""
    edges = [index / 10 for index in range(0, 11)]
    counts = [0.0] * (len(edges) - 1)
    wins = [0.0] * (len(edges) - 1)
    for winner, _loser, row, weight in played:
        home = row["home_player"].strip()
        away = row["away_player"].strip()
        chance = expected_score(rating[home], rating[away], scale)
        share = winner_share(row)
        home_share = share if home == winner else 1.0 - share
        index = min(int(chance * 10), 9)
        counts[index] += weight
        wins[index] += weight * home_share
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
        "pass_nr",
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


def is_tbc_team(team_field: str) -> bool:
    """True when any of the player's teams is Tübinger BC."""
    for part in team_field.split("|"):
        name = part.strip()
        if name == "Tübinger BC" or name.startswith("Tübinger BC "):
            return True
    return False


def season_of(when: datetime) -> str:
    """BVBW season. September opens the new year."""
    if when.month >= 9:
        return f"{when.year}/{when.year + 1}"
    return f"{when.year - 1}/{when.year}"


def recent_seasons(as_of: datetime, count: int = 2) -> list[str]:
    """The season of `as_of` and the ones just before it, oldest first."""
    current = season_of(as_of)
    year = int(current.split("/", 1)[0])
    return [f"{year - i}/{year - i + 1}" for i in range(count - 1, -1, -1)]


def active_tbc_players(rows: list[dict[str, str]], seasons: set[str]) -> set[str]:
    """Tübinger BC players with a game in one of `seasons`.

    Club membership is a seat on a Tübinger BC team. A championship game has
    no team, so a recent Einzel result still counts once the player has
    represented the club.
    """
    members: set[str] = set()
    played: set[str] = set()
    for row in rows:
        season = (row.get("season") or "").strip()
        for side in ("home", "away"):
            name = (row.get(f"{side}_player") or "").strip()
            if not name:
                continue
            team = (row.get(f"{side}_team") or "").strip()
            if is_tbc_team(team):
                members.add(name)
            if season in seasons:
                played.add(name)
    return members & played


def tbc_markdown(rows: list[dict[str, str]], active: set[str], seasons: list[str]) -> str:
    """Markdown table of active Tübinger BC players, highest Elo first."""
    club = [row for row in rows if row["scope"] == "all" and row["player"] in active]
    club.sort(key=lambda row: (-int(row["elo"]), row["player"]))
    window = " and ".join(seasons)
    lines = [
        "## Tübinger BC",
        "",
        f"Players with a game in {window}.",
        "",
        "| Elo | ± | 95% | W-L | Games | Player |",
        "|---:|---:|---|---:|---:|---|",
    ]
    for row in club:
        player = row["player"].replace("|", "\\|")
        lines.append(
            f"| {row['elo']} | {row['elo_se']} | {row['elo_low']}–{row['elo_high']} | "
            f"{row['wins']}-{row['losses']} | {row['games']} | {player} |"
        )
    if len(lines) == 6:
        lines.append("| | | | | | |")
    return "\n".join(lines)


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
            as_of=as_of,
            half_life_years=args.half_life_years,
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
                as_of=as_of,
                half_life_years=args.half_life_years,
                offset=scale_offset(),
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
            f"(oldest game in this file counts as {oldest_weight:.0%}). "
            f"The virtual draw ages from each player's last match. "
            f"Published ratings add the frozen offset of {scale_offset():+.2f}."
        )
    print(
        "Mean |score share − expected share| on all games: "
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
    print("Calibration of predicted rack/ball share (all games):")
    print(f"{'P(unit)':>8}  {'games':>6}  {'taken':>8}")
    for label, count, observed in reliability(played, fitted["all"], args.scale):
        if count == 0:
            continue
        print(f"{label:>8}  {count:6.0f}  {observed:8.2f}")
    tbc_path = Path(__file__).resolve().parent / "tbc players.md"
    seasons = recent_seasons(as_of) if as_of is not None else recent_seasons(datetime.now())
    active = active_tbc_players(source, set(seasons))
    tbc_path.write_text(tbc_markdown(all_rows, active, seasons) + "\n", encoding="utf-8")
    print(f"\nWrote {tbc_path.name}")


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
        help="Age at which a game, and the virtual draw, keep half their weight. 0 disables decay (default: 3)",
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
