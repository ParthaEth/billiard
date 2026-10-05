#!/usr/bin/env python3
"""Scrape BVBW Pool individual championships into the same games.csv."""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
import urllib.error
import urllib.request
from html import unescape
from pathlib import Path
from bs4 import BeautifulSoup

from scrape_bvbw import (
    SCORE_RE,
    _split_score,
    _winner_loser_names,
    _winner_side,
    append_csv,
    load_seen_report_urls,
)

LIST_URL = "https://billard-bvbw.de/sb_meisterschaft.php?p=999-6-{season}----1--100000--"
RESULTS_URL = (
    "https://billard-bvbw.de/sb_einzelergebnisse.php"
    "?p=999-6-{season}-{tid}----1-1-100000--"
)
CANCELLED_RE = re.compile(r"entf[aä]llt", re.I)
BYE_RE = re.compile(r"freilos", re.I)
STATS_RE = re.compile(
    r"HS:\s*(?P<hs>[^;\s]+).*?Aufn\.:\s*(?P<aufn>[^;\s]+).*?Ø:\s*(?P<gd>[^;\s]+)",
    re.I,
)
NAME_SUFFIX_RE = re.compile(r"\s*\((?:Gewinner|Verlierer)\s+Partie\s*\d+\)", re.I)
DATE_RE = re.compile(r"(\d{2}\.\d{2}\.\d{4})")
CLOCK_RE = re.compile(r"(\d{2}:\d{2})")
ZERO_DATE_RE = re.compile(r"^00\.00\.00")
DEFAULT_SEASONS = (
    "2026/2027",
    "2025/2026",
    "2024/2025",
    "2023/2024",
    "2022/2023",
    "2021/2022",
)


def fetch(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=40) as response:
        return unescape(response.read().decode("utf-8", "replace"))


def wait(min_wait: float, max_wait: float) -> None:
    time.sleep(random.uniform(min_wait, max_wait))


def discipline_from_name(name: str) -> str:
    text = name.lower().replace(" ", "")
    if "14" in text:
        return "14/1e"
    if "10-ball" in text or "10er" in text or "10ball" in text:
        return "10-Ball"
    if "9-ball" in text or "9er" in text or "9ball" in text:
        return "9-Ball"
    if "8-ball" in text or "8er" in text or "8ball" in text:
        return "8-Ball"
    return ""


def clean_player(text: str) -> str:
    text = NAME_SUFFIX_RE.sub("", text)
    text = re.split(r"\s+HS:", text, maxsplit=1)[0]
    return " ".join(text.split()).strip()


def list_tournaments(html: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    found: list[tuple[str, str]] = []
    seen: set[str] = set()
    for anchor in soup.select("a.cc_bluelink"):
        href = anchor.get("href") or ""
        match = re.search(r"sb_meisterschaft\.php\?p=999-6-[^'\"-]+-(\d+)", href)
        if match is None:
            continue
        tid = match.group(1)
        name = anchor.get_text(" ", strip=True)
        if tid in seen or not name or CANCELLED_RE.search(name):
            continue
        seen.add(tid)
        found.append((tid, name))
    return found


def blank_date(text: str) -> bool:
    """True when the results page left the planned time empty or zero."""
    text = (text or "").strip()
    return not text or ZERO_DATE_RE.match(text) is not None


def tournament_when(soup: BeautifulSoup) -> tuple[str, str]:
    """Start date and clock from the tournament header, not the match row.

    The per-game Planzeit is often ``00.00.0000``. The header still says
    ``Datum 09.10.2022 - 09.10.2022`` and when play started.
    """
    heading = soup.select_one("h6")
    if heading is None:
        return "", ""
    texts: list[str] = []
    for element in heading.next_elements:
        if getattr(element, "name", None) == "table":
            break
        if isinstance(element, str):
            text = element.strip()
            if text:
                texts.append(text)
    date = ""
    clock = ""
    seen_datum = False
    for text in texts:
        if text == "Datum":
            seen_datum = True
            continue
        if not seen_datum:
            continue
        if not date:
            found = DATE_RE.search(text)
            if found:
                date = found.group(1)
        if "Spielbeginn" in text:
            found_clock = CLOCK_RE.search(text)
            if found_clock:
                clock = found_clock.group(1)
        if date and text == "Location":
            break
    return date, clock


def parse_results(
    html: str,
    season: str,
    tournament: str,
    results_url: str,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    discipline = discipline_from_name(tournament)
    title = soup.select_one("h6")
    if title is not None and title.get_text(strip=True):
        tournament = title.get_text(strip=True)
        discipline = discipline or discipline_from_name(tournament)
    event_date, event_time = tournament_when(soup)

    rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        headers = [th.get_text(" ", strip=True) for th in table.find_all("th")]
        if "Partie" not in headers:
            continue
        current_round = ""
        for tr in table.find_all("tr"):
            ths = [th.get_text(" ", strip=True) for th in tr.find_all("th")]
            if len(ths) == 1 and ths[0] and ths[0] != "Partie":
                current_round = ths[0]
                continue
            tds = tr.find_all("td", recursive=False)
            if len(tds) < 5 or not tds[0].get_text(strip=True).isdigit():
                continue
            partie = tds[0].get_text(strip=True)
            home_raw = tds[1].get_text(" ", strip=True)
            score = tds[2].get_text(" ", strip=True).replace("\xa0", "").strip()
            away_raw = tds[3].get_text(" ", strip=True)
            if BYE_RE.search(home_raw) or BYE_RE.search(away_raw):
                continue
            if not SCORE_RE.match(score):
                continue
            home_player = clean_player(home_raw)
            away_player = clean_player(away_raw)
            if not home_player or not away_player:
                continue
            home_points, away_points = _split_score(score)
            winner_side = _winner_side(home_points, away_points)
            winner, loser = _winner_loser_names(home_player, away_player, winner_side)
            termin = tds[4].get_text("\n", strip=True).splitlines()
            match_date = termin[0].strip() if termin else ""
            match_time = termin[1].strip() if len(termin) > 1 else ""
            if blank_date(match_date):
                match_date = event_date
                if blank_date(match_time) or match_time.startswith("00:00"):
                    match_time = f"{event_time} Uhr" if event_time else ""
            report_url = f"{results_url}#{current_round}-{partie}"
            stats = _pair_stats(home_raw, away_raw, score)
            rows.append(
                {
                    "season": season,
                    "league": "Einzel",
                    "staffel": tournament,
                    "spieltag": partie,
                    "match_date": match_date,
                    "match_time": match_time,
                    "home_team": "",
                    "away_team": "",
                    "match_score": score,
                    "round": current_round,
                    "game_no": "1",
                    "discipline": discipline,
                    "home_player": home_player,
                    "away_player": away_player,
                    "frame_score": score,
                    "home_points": home_points,
                    "away_points": away_points,
                    "winner": winner,
                    "loser": loser,
                    "winner_side": winner_side,
                    "straight_pool_punkte": stats["punkte"],
                    "straight_pool_aufnahmen": stats["aufnahmen"],
                    "straight_pool_hs": stats["hs"],
                    "straight_pool_gd": stats["gd"],
                    "report_url": report_url,
                }
            )
    return rows


def _pair_stats(home_raw: str, away_raw: str, score: str) -> dict[str, str]:
    home = STATS_RE.search(home_raw)
    away = STATS_RE.search(away_raw)
    if home is None or away is None:
        return {"punkte": "", "aufnahmen": "", "hs": "", "gd": ""}
    return {
        "punkte": score,
        "aufnahmen": f"{home.group('aufn')}/{away.group('aufn')}",
        "hs": f"{home.group('hs')}/{away.group('hs')}",
        "gd": f"{home.group('gd')}/{away.group('gd')}",
    }


def progress_line(
    season: str,
    season_index: int,
    season_total: int,
    tournament: str,
    tour_index: int,
    tour_total: int,
    saved_games: int,
) -> str:
    width = 22
    filled = int(width * tour_index / tour_total) if tour_total else 0
    bar = "█" * filled + "░" * (width - filled)
    pct = int(100 * tour_index / tour_total) if tour_total else 0
    name = tournament[:42] + ("…" if len(tournament) > 42 else "")
    return (
        f"{season} ({season_index}/{season_total})  "
        f"{bar} {tour_index}/{tour_total} {pct:3d}%  "
        f"{name}  einzel games {saved_games}"
    )


def scrape_seasons(
    seasons: tuple[str, ...],
    csv_path: Path,
    min_wait: float,
    max_wait: float,
) -> None:
    seen = load_seen_report_urls(csv_path)
    saved = 0
    print(f"Einzel scrape -> {csv_path}")
    print(f"Seasons: {', '.join(seasons)}")
    for season_index, season in enumerate(seasons, start=1):
        list_url = LIST_URL.format(season=season)
        print(f"\n{season} ({season_index}/{len(seasons)}) listing tournaments")
        try:
            listing = fetch(list_url)
        except urllib.error.URLError as exc:
            print(f"  failed to list {season}: {exc}")
            continue
        wait(min_wait, max_wait)
        tournaments = list_tournaments(listing)
        print(f"  {len(tournaments)} Pool championships")
        for tour_index, (tid, name) in enumerate(tournaments, start=1):
            results_url = RESULTS_URL.format(season=season, tid=tid)
            line = progress_line(
                season, season_index, len(seasons), name, tour_index, len(tournaments), saved
            )
            print(f"\r{line:<160}", end="", flush=True)
            try:
                html = fetch(results_url)
            except urllib.error.URLError as exc:
                print(f"\n  skip {name}: {exc}")
                continue
            rows = [
                row
                for row in parse_results(html, season, name, results_url)
                if row["report_url"] not in seen
            ]
            if rows:
                append_csv(csv_path, rows)
                for row in rows:
                    seen.add(row["report_url"])
                saved += len(rows)
            wait(min_wait, max_wait)
        print()
    print(f"Finished. Added {saved} individual games -> {csv_path}")


def repair_dates(csv_path: Path, min_wait: float, max_wait: float) -> None:
    """Fill Einzel rows whose Planzeit is empty or 00.00.0000 from the tournament header."""
    with csv_path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    if not fieldnames:
        raise SystemExit(f"No header in {csv_path}")

    pending: dict[str, list[int]] = {}
    for index, row in enumerate(rows):
        if row.get("league") != "Einzel" or not blank_date(row.get("match_date", "")):
            continue
        url = (row.get("report_url") or "").split("#", 1)[0]
        if url:
            pending.setdefault(url, []).append(index)

    games = sum(len(indexes) for indexes in pending.values())
    print(f"{games} games across {len(pending)} tournaments")
    filled = 0
    missed = 0
    for number, (url, indexes) in enumerate(pending.items(), start=1):
        try:
            html = fetch(url)
        except urllib.error.URLError as exc:
            print(f"  skip {url}: {exc}")
            missed += len(indexes)
            wait(min_wait, max_wait)
            continue
        soup = BeautifulSoup(html, "lxml")
        event_date, event_time = tournament_when(soup)
        if blank_date(event_date):
            print(f"  no header date {url}")
            missed += len(indexes)
        else:
            clock = f"{event_time} Uhr" if event_time else ""
            for index in indexes:
                rows[index]["match_date"] = event_date
                current = rows[index].get("match_time") or ""
                if blank_date(current) or current.startswith("00:00"):
                    rows[index]["match_time"] = clock
            filled += len(indexes)
            title = soup.select_one("h6")
            name = title.get_text(strip=True) if title else url
            print(f"  {number}/{len(pending)} {event_date}  {len(indexes):4d}  {name}", flush=True)
        wait(min_wait, max_wait)

    temporary = csv_path.with_suffix(".csv.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(csv_path)
    print(f"Filled {filled}, still missing {missed} -> {csv_path}")


def test_parse(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    rows = parse_results(html, "2025/2026", "test", "local")
    for row in rows[:8]:
        print(
            f"{row['round']:<24} {row['home_player']} {row['frame_score']} "
            f"{row['away_player']}  winner={row['winner']}"
        )
    print(f"parsed {len(rows)} games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/games.csv"))
    parser.add_argument("--seasons", nargs="+", default=list(DEFAULT_SEASONS))
    parser.add_argument("--min-wait", type=float, default=1.5)
    parser.add_argument("--max-wait", type=float, default=3.0)
    parser.add_argument("--test-parse", type=Path)
    parser.add_argument(
        "--repair-dates",
        action="store_true",
        help="Fill Einzel dates that are empty or 00.00.0000 from each tournament header",
    )
    args = parser.parse_args()
    if args.test_parse:
        test_parse(args.test_parse)
        return
    if args.repair_dates:
        repair_dates(args.csv, args.min_wait, args.max_wait)
        return
    scrape_seasons(tuple(args.seasons), args.csv, args.min_wait, args.max_wait)


if __name__ == "__main__":
    main()
