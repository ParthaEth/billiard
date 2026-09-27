#!/usr/bin/env python3
"""Pick a BVBW lineup that maximises the chance of winning the team match.

A match is eight games: two each of 14/1e, 8-ball, 9-ball and 10-ball.
Each player plays two different disciplines. The opponent's assignment is
unknown, so every valid opponent lineup is weighted by how often those
players have actually played each discipline (newer games count more).
The recommended lineup is the one with the highest chance of scoring 5
or more points under that mixture.

Example:
  .venv/bin/python lineup.py \\
    --us "Partha Ghosh" "Mark Hast" "Alexander Fischer" "Patrick Coklica" \\
    --them "Peter Baur" "Name Two" "Name Three" "Name Four"
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from difflib import get_close_matches
from itertools import combinations
from pathlib import Path

from elo import (
    HALF_LIFE_YEARS,
    PRIOR_RATING,
    SCALE,
    assign_weights,
    collect,
    expected_score,
    fit_elo,
    load_games,
    outcomes,
    race_win_probability,
)

DISCIPLINES = ("14/1e", "8-Ball", "9-Ball", "10-Ball")
PRIOR_DISCIPLINE = 0.5
WIN_POINTS = 5


@dataclass
class SideModel:
    names: list[str]
    assignments: list[tuple[frozenset[str], ...]]
    probabilities: list[float]
    coverage: float


@dataclass
class LineupScore:
    assignment: tuple[frozenset[str], ...]
    distribution: list[float]
    win: float
    draw: float
    loss: float
    expected: float


def split_names(parts: list[str]) -> list[str]:
    names: list[str] = []
    for part in parts:
        names.extend(piece.strip() for piece in part.split(",") if piece.strip())
    if len(names) != len(set(names)):
        raise SystemExit("A player is listed twice.")
    return names


def resolve_name(name: str, known: list[str]) -> str:
    exact = {player.casefold(): player for player in known}
    if name.casefold() in exact:
        return exact[name.casefold()]
    matches = get_close_matches(name, known, n=5, cutoff=0.72)
    hint = f" Close names: {', '.join(matches)}." if matches else ""
    raise SystemExit(f"No player named {name!r} in the scraped games.{hint}")


def enumerate_assignments(n_players: int) -> list[tuple[frozenset[str], ...]]:
    """Lineups where each discipline is played twice and nobody repeats one."""
    if n_players == 4:
        sizes = (2,)
    elif n_players == 5:
        sizes = (1, 2)
    else:
        raise SystemExit(
            "Pass 4 players per side, or 5 when someone plays only one game. "
            f"Got {n_players}."
        )
    choices = [
        frozenset(combo)
        for size in sizes
        for combo in combinations(DISCIPLINES, size)
    ]
    found: list[tuple[frozenset[str], ...]] = []

    def rec(index: int, used: dict[str, int], picked: tuple[frozenset[str], ...]) -> None:
        if index == n_players:
            if all(used[disc] == 2 for disc in DISCIPLINES):
                found.append(picked)
            return
        remaining = n_players - index
        slots_left = sum(2 - used[disc] for disc in DISCIPLINES)
        if slots_left < remaining or slots_left > 2 * remaining:
            return
        for choice in choices:
            if any(used[disc] == 2 for disc in choice):
                continue
            nxt = dict(used)
            for disc in choice:
                nxt[disc] += 1
            rec(index + 1, nxt, picked + (choice,))

    rec(0, {disc: 0 for disc in DISCIPLINES}, ())
    return found


def preference_table(
    played: list[tuple[str, str, dict[str, str], float]],
) -> dict[str, dict[str, float]]:
    """Time-weighted games per player and discipline, plus a half-game prior."""
    counts: dict[str, dict[str, float]] = defaultdict(
        lambda: {disc: 0.0 for disc in DISCIPLINES}
    )
    for winner, loser, row, weight in played:
        disc = row.get("discipline", "").strip()
        if disc not in DISCIPLINES:
            continue
        for name in (winner, loser):
            counts[name][disc] += weight
    return counts


def history_counts(counts: dict[str, dict[str, float]], name: str) -> dict[str, float]:
    raw = counts.get(name)
    if raw is None:
        return {disc: 0.0 for disc in DISCIPLINES}
    return raw


def log_likelihood(
    assignment: tuple[frozenset[str], ...],
    preferences: list[dict[str, float]],
) -> float:
    total = 0.0
    for picks, prefs in zip(assignment, preferences):
        for disc in picks:
            total += math.log(prefs[disc] + PRIOR_DISCIPLINE)
    return total


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    return peak + math.log(sum(math.exp(value - peak) for value in values))


def side_model(
    names: list[str],
    counts: dict[str, dict[str, float]],
    keep: int,
) -> SideModel:
    assignments = enumerate_assignments(len(names))
    preferences = [
        history_counts(counts, name) for name in names
    ]
    logs = [log_likelihood(assignment, preferences) for assignment in assignments]
    normalizer = logsumexp(logs)
    order = sorted(range(len(assignments)), key=lambda index: logs[index], reverse=True)
    chosen: list[int] = []
    covered = 0.0
    for index in order:
        chosen.append(index)
        covered += math.exp(logs[index] - normalizer)
        if covered >= 0.99 or len(chosen) >= keep:
            break
    kept_logs = [logs[index] for index in chosen]
    kept_norm = logsumexp(kept_logs)
    probabilities = [math.exp(log_p - kept_norm) for log_p in kept_logs]
    return SideModel(
        names=names,
        assignments=[assignments[index] for index in chosen],
        probabilities=probabilities,
        coverage=covered,
    )


def fit_ratings(
    played: list[tuple[str, str, dict[str, str], float]],
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, int]]]:
    scopes: dict[str, list] = {"all": played}
    for game in played:
        disc = game[2].get("discipline", "").strip()
        if disc in DISCIPLINES:
            scopes.setdefault(disc, []).append(game)
    elo: dict[str, dict[str, float]] = {}
    games: dict[str, dict[str, int]] = {}
    for scope, games_in_scope in scopes.items():
        players = collect(games_in_scope)
        elo[scope] = fit_elo(players)
        games[scope] = {name: record.games for name, record in players.items()}
    return elo, games


def player_rating(
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
    name: str,
    discipline: str,
) -> tuple[float, int, str]:
    if name in elo.get(discipline, {}):
        return elo[discipline][name], games[discipline].get(name, 0), discipline
    if name in elo.get("all", {}):
        return elo["all"][name], games["all"].get(name, 0), "all"
    return PRIOR_RATING, 0, "none"


def infer_race_targets(
    played: list[tuple[str, str, dict[str, str], float]],
    names: list[str],
) -> tuple[str, dict[str, int]]:
    """Race length for the league our players mostly appear in."""
    league_weight: dict[str, float] = defaultdict(float)
    winner_scores: dict[tuple[str, str], dict[int, int]] = defaultdict(lambda: defaultdict(int))
    wanted = set(names)
    defaults = {"8-Ball": 5, "9-Ball": 7, "10-Ball": 5, "14/1e": 70}
    for winner, loser, row, weight in played:
        league = row.get("league", "").strip()
        discipline = row.get("discipline", "").strip()
        if winner in wanted or loser in wanted:
            league_weight[league] += weight
        text = row.get("straight_pool_punkte", "") if discipline == "14/1e" else row.get("frame_score", "")
        if ":" not in text:
            continue
        try:
            home_score, away_score = (int(part) for part in text.split(":", 1))
        except ValueError:
            continue
        target = max(home_score, away_score)
        floor = 40 if discipline == "14/1e" else 4
        if target >= floor:
            winner_scores[(league, discipline)][target] += 1
    league = max(league_weight, key=league_weight.get) if league_weight else ""
    targets = {}
    for discipline in DISCIPLINES:
        bucket = winner_scores.get((league, discipline), {})
        targets[discipline] = max(bucket, key=bucket.get) if bucket else defaults[discipline]
    return league, targets


def win_probabilities(
    us: list[str],
    them: list[str],
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
    targets: dict[str, int],
) -> dict[tuple[int, int, str], float]:
    table: dict[tuple[int, int, str], float] = {}
    our_ratings = {
        (index, disc): player_rating(elo, games, name, disc)[0]
        for index, name in enumerate(us)
        for disc in DISCIPLINES
    }
    their_ratings = {
        (index, disc): player_rating(elo, games, name, disc)[0]
        for index, name in enumerate(them)
        for disc in DISCIPLINES
    }
    for our_index in range(len(us)):
        for their_index in range(len(them)):
            for disc in DISCIPLINES:
                rack_chance = expected_score(
                    our_ratings[our_index, disc],
                    their_ratings[their_index, disc],
                    SCALE,
                )
                table[our_index, their_index, disc] = race_win_probability(
                    rack_chance, targets[disc]
                )
    return table


def convolve(left: list[float], right: list[float]) -> list[float]:
    out = [0.0] * (len(left) + len(right) - 1)
    for i, p_left in enumerate(left):
        for j, p_right in enumerate(right):
            out[i + j] += p_left * p_right
    return out


def summarize(distribution: list[float], assignment: tuple[frozenset[str], ...]) -> LineupScore:
    win = sum(distribution[WIN_POINTS:])
    draw = distribution[4]
    loss = sum(distribution[:4])
    expected = sum(points * mass for points, mass in enumerate(distribution))
    return LineupScore(assignment, distribution, win, draw, loss, expected)


def opponent_schedules(
    opponents: SideModel,
) -> list[tuple[tuple[dict[str, int], dict[str, int]], float]]:
    """Each opponent lineup, spread across its round orders and both alignments."""
    weighted: list[tuple[tuple[dict[str, int], dict[str, int]], float]] = []
    for their, probability in zip(opponents.assignments, opponents.probabilities):
        options = decompositions(their)
        share = probability / (2 * len(options))
        for option in options:
            weighted.append((option, share))
            weighted.append(((option[1], option[0]), share))
    return weighted


def score_schedule(
    assignment: tuple[frozenset[str], ...],
    rounds: tuple[dict[str, int], dict[str, int]],
    their_schedules: list[tuple[tuple[dict[str, int], dict[str, int]], float]],
    chances: dict[tuple[int, int, str], float],
) -> LineupScore:
    total = [0.0] * 9
    for their_rounds, probability in their_schedules:
        distribution = schedule_distribution(rounds, their_rounds, chances)
        for points, mass in enumerate(distribution):
            total[points] += probability * mass
    return summarize(total, assignment)


def decompositions(
    assignment: tuple[frozenset[str], ...],
) -> list[tuple[dict[str, int], dict[str, int]]]:
    """Split a lineup into two rounds. Each round plays every discipline once."""
    edges = [(index, disc) for index, picks in enumerate(assignment) for disc in picks]
    color: dict[tuple[int, str], int] = {}
    found: list[tuple[dict[str, int], dict[str, int]]] = []

    def fits(edge: tuple[int, str], chosen: int) -> bool:
        player, disc = edge
        for (other_player, other_disc), existing in color.items():
            if existing == chosen and (other_player == player or other_disc == disc):
                return False
        return True

    def rec(index: int) -> None:
        if index == len(edges):
            round_one = {disc: player for (player, disc), chosen in color.items() if chosen == 0}
            round_two = {disc: player for (player, disc), chosen in color.items() if chosen == 1}
            if len(round_one) == 4 and len(round_two) == 4:
                found.append((round_one, round_two))
            return
        for chosen in (0, 1):
            if fits(edges[index], chosen):
                color[edges[index]] = chosen
                rec(index + 1)
                del color[edges[index]]

    rec(0)
    unique: list[tuple[dict[str, int], dict[str, int]]] = []
    seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
    for round_one, round_two in found:
        forward = (
            tuple(round_one[disc] for disc in DISCIPLINES),
            tuple(round_two[disc] for disc in DISCIPLINES),
        )
        backward = (forward[1], forward[0])
        signature = forward if forward <= backward else backward
        if signature in seen:
            continue
        seen.add(signature)
        if forward <= backward:
            unique.append((round_one, round_two))
        else:
            unique.append((round_two, round_one))
    return unique


def schedule_distribution(
    our_rounds: tuple[dict[str, int], dict[str, int]],
    their_rounds: tuple[dict[str, int], dict[str, int]],
    chances: dict[tuple[int, int, str], float],
) -> list[float]:
    distribution = [1.0]
    for our_round, their_round in zip(our_rounds, their_rounds):
        for disc in DISCIPLINES:
            p_win = chances[our_round[disc], their_round[disc], disc]
            distribution = convolve(distribution, [1.0 - p_win, p_win])
    return distribution


def rank_lineups(
    us: SideModel,
    their_schedules: list[tuple[tuple[dict[str, int], dict[str, int]], float]],
    chances: dict[tuple[int, int, str], float],
) -> list[tuple[LineupScore, tuple[dict[str, int], dict[str, int]]]]:
    ranked: list[tuple[LineupScore, tuple[dict[str, int], dict[str, int]]]] = []
    for assignment in us.assignments:
        options = decompositions(assignment)
        if not options:
            raise RuntimeError("Could not split a lineup into two rounds")
        scored = [
            (score_schedule(assignment, rounds, their_schedules, chances), rounds)
            for rounds in options
        ]
        ranked.append(max(scored, key=lambda item: (item[0].win, item[0].expected)))
    ranked.sort(key=lambda item: (item[0].win, item[0].expected), reverse=True)
    return ranked


def scheduled_win_chance(
    player: int,
    discipline: str,
    round_index: int,
    their_schedules: list[tuple[tuple[dict[str, int], dict[str, int]], float]],
    chances: dict[tuple[int, int, str], float],
) -> float:
    total = 0.0
    for their_rounds, probability in their_schedules:
        opponent = their_rounds[round_index][discipline]
        total += probability * chances[player, opponent, discipline]
    return total


def format_percent(value: float) -> str:
    return f"{100.0 * value:.0f}%"


SPECIALIST_GAP = 100
SPECIALIST_GAMES = 4


def discipline_cells(
    name: str,
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
) -> list[tuple[str, float | None, int]]:
    cells = []
    for disc in DISCIPLINES:
        rating, played, source = player_rating(elo, games, name, disc)
        if source == disc:
            cells.append((disc, rating, played))
        else:
            cells.append((disc, None, played if source == "all" else 0))
    return cells


def specialist_label(cells: list[tuple[str, float | None, int]]) -> str:
    """Flag a real gap between disciplines, not a thin sample."""
    rated = [(disc, rating, played) for disc, rating, played in cells if rating is not None and played >= SPECIALIST_GAMES]
    if len(rated) < 2:
        return ""
    best = max(rated, key=lambda item: item[1])
    worst = min(rated, key=lambda item: item[1])
    if best[1] - worst[1] < SPECIALIST_GAP:
        return ""
    return f"{best[0]} specialist, avoid {worst[0]}"


def print_roster(
    title: str,
    names: list[str],
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
) -> None:
    print(title)
    header = f"  {'Player':<28}"
    for disc in DISCIPLINES:
        header += f" {disc:>10}"
    header += "  Flag"
    print(header)
    for name in names:
        cells = discipline_cells(name, elo, games)
        overall = elo.get("all", {}).get(name)
        line = f"  {name:<28}"
        for _disc, rating, played in cells:
            if rating is None:
                line += f" {'—':>10}"
            else:
                line += f" {f'{rating:.0f} ({played})':>10}"
        note = specialist_label(cells)
        if overall is not None and not note:
            line += f"  overall {overall:.0f}"
        else:
            line += f"  {note}"
        print(line)
    print("  Elo (games in that discipline). A dash means too little there to rate it on its own.")


GIVE_UP_BELOW = 0.45
LOCK_ABOVE = 0.58


def strongest_opponent(
    discipline: str,
    names: list[str],
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
) -> str:
    best_name = ""
    best_rating = -1.0
    for name in names:
        rating, played, source = player_rating(elo, games, name, discipline)
        if source == discipline and played >= SPECIALIST_GAMES and rating > best_rating:
            best_rating = rating
            best_name = name
    if not best_name:
        return ""
    return f"{best_name} ({best_rating:.0f})"


def print_match_plan(
    names: list[str],
    rounds: tuple[dict[str, int], dict[str, int]],
    their_schedules: list[tuple[tuple[dict[str, int], dict[str, int]], float]],
    chances: dict[tuple[int, int, str], float],
    elo: dict[str, dict[str, float]],
    games: dict[str, dict[str, int]],
    opponents: list[str],
) -> None:
    """Show the two rounds, and say when one table is conceded to lock the other three."""
    any_sacrifice = False
    for number, pairing in enumerate(rounds, start=1):
        slots = []
        for disc in DISCIPLINES:
            index = pairing[disc]
            chance = scheduled_win_chance(index, disc, number - 1, their_schedules, chances)
            slots.append((disc, index, chance))
        weakest = min(chance for _disc, _index, chance in slots)
        locked = sum(chance >= LOCK_ABOVE for _disc, _index, chance in slots)
        sacrifice = weakest <= GIVE_UP_BELOW and locked >= 3
        any_sacrifice = any_sacrifice or sacrifice
        print(f"Runde {number}")
        for disc, index, chance in slots:
            if sacrifice and chance <= GIVE_UP_BELOW:
                tag = "GIVE UP"
            elif sacrifice and chance >= LOCK_ABOVE:
                tag = "LOCK"
            elif chance <= GIVE_UP_BELOW:
                tag = "WEAK"
            else:
                tag = ""
            rating, played, source = player_rating(elo, games, names[index], disc)
            if source == disc:
                detail = f"{rating:.0f} from {played} games"
            elif source == "all":
                detail = f"overall {rating:.0f}"
            else:
                detail = "no rating"
            print(
                f"  {tag:<7} {disc:<8} {names[index]:<28} "
                f"win {format_percent(chance):>4}  {detail}"
            )
        if sacrifice:
            given = [slot for slot in slots if slot[2] <= GIVE_UP_BELOW]
            held = [slot for slot in slots if slot[2] >= LOCK_ABOVE]
            for disc, index, chance in given:
                threat = strongest_opponent(disc, opponents, elo, games)
                threat_note = f" Their strongest there is {threat}." if threat else ""
                print(
                    f"  Concede {disc} ({names[index]}, {format_percent(chance)})."
                    f"{threat_note}"
                )
            held_text = ", ".join(f"{disc} {format_percent(chance)}" for disc, _index, chance in held)
            print(f"  The other three tables are the ones to win: {held_text}.")
        elif weakest <= GIVE_UP_BELOW:
            soft = ", ".join(
                f"{disc} {format_percent(chance)}"
                for disc, _index, chance in slots
                if chance <= GIVE_UP_BELOW
            )
            print(f"  Likely loss: {soft}. The other tables in this round are not safe enough to give it away on purpose.")
        print()
    if not any_sacrifice:
        print("No table is given away. The eight games are close enough that conceding one does not help.")
        print()


def print_assignment(names: list[str], assignment: tuple[frozenset[str], ...]) -> None:
    for name, picks in zip(names, assignment):
        ordered = [disc for disc in DISCIPLINES if disc in picks]
        print(f"  {name:<28} {', '.join(ordered)}")


def run(args: argparse.Namespace) -> None:
    us_names = split_names(args.us)
    them_names = split_names(args.them)
    rows = load_games(args.games)
    played, as_of = assign_weights(outcomes(rows), args.half_life_years)
    if not played:
        raise SystemExit(f"No decisive games in {args.games}")
    known = sorted({name for winner, loser, _row, _weight in played for name in (winner, loser)})
    us_names = [resolve_name(name, known) for name in us_names]
    them_names = [resolve_name(name, known) for name in them_names]

    elo, games = fit_ratings(played)
    counts = preference_table(played)
    league, targets = infer_race_targets(played, us_names)
    chances = win_probabilities(us_names, them_names, elo, games, targets)
    # Our search keeps every legal lineup. The opponent mixture keeps the
    # likely ones; four-player sides are small enough that this is all of them.
    our_assignments = enumerate_assignments(len(us_names))
    our_side = SideModel(us_names, our_assignments, [1.0] * len(our_assignments), 1.0)
    their_side = side_model(them_names, counts, keep=400)
    their_schedules = opponent_schedules(their_side)
    ranked = rank_lineups(our_side, their_schedules, chances)
    best, schedule = ranked[0]
    modal_index = max(
        range(len(their_side.probabilities)),
        key=their_side.probabilities.__getitem__,
    )

    as_of_text = as_of.strftime("%d.%m.%Y") if as_of else "the newest match"
    print(
        f"Best chance of winning the match, ratings as of {as_of_text}. "
        f"A win is {WIN_POINTS} or more points out of 8."
    )
    lengths = ", ".join(f"{disc} to {targets[disc]}" for disc in DISCIPLINES)
    print(f"Race lengths from {league}: {lengths}.")
    print(
        f"Win {format_percent(best.win)}   "
        f"draw {format_percent(best.draw)}   "
        f"loss {format_percent(best.loss)}   "
        f"expected score {best.expected:.2f}–{8.0 - best.expected:.2f}"
    )
    if their_side.coverage < 0.999:
        print(
            f"Opponent lineups used: {len(their_side.assignments)} "
            f"({format_percent(their_side.coverage)} of the probability)."
        )
    print()
    print_roster("Us", us_names, elo, games)
    print()
    print_roster("Them", them_names, elo, games)
    print()
    print("Plan")
    print_match_plan(
        us_names,
        schedule,
        their_schedules,
        chances,
        elo,
        games,
        them_names,
    )
    print(
        "Most likely opponent lineup "
        f"({format_percent(their_side.probabilities[modal_index])} of their weighted lineups)"
    )
    print_assignment(them_names, their_side.assignments[modal_index])
    print()
    thin = []
    for disc in DISCIPLINES:
        for index, picks in enumerate(best.assignment):
            if disc not in picks:
                continue
            _rating, played, source = player_rating(elo, games, us_names[index], disc)
            if source != disc:
                thin.append(f"{us_names[index]} has no {disc} games, so that slot uses overall Elo")
            elif played < 4:
                game_word = "game" if played == 1 else "games"
                thin.append(f"{us_names[index]} has {played} {game_word} of {disc}")
    if thin:
        print()
        print(
            "Thin ratings, so these slots can move a lot: "
            + "; ".join(thin)
            + "."
        )
    if len(ranked) > 1 and args.top > 1:
        print()
        print("Other lineups")
        for alternative, _rounds in ranked[1 : args.top]:
            picks = []
            for name, assigned in zip(us_names, alternative.assignment):
                discs = [disc for disc in DISCIPLINES if disc in assigned]
                picks.append(f"{name}: {'/'.join(discs)}")
            print(
                f"  win {format_percent(alternative.win)}  "
                f"expected {alternative.expected:.2f}  "
                + " | ".join(picks)
            )


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--us", nargs="+", required=True, help="Our players, in any order")
    parser.add_argument("--them", nargs="+", required=True, help="Opponent players, in any order")
    parser.add_argument("--games", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--half-life-years", type=float, default=HALF_LIFE_YEARS)
    parser.add_argument("--top", type=int, default=3, help="How many alternative lineups to print")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_args(sys.argv[1:]))
