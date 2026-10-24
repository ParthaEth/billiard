#!/usr/bin/env python3
"""Scrape open tournaments in Baden-Württemberg from tournamentapp.de.

Every TournamentApp event held in one of REGION_PLACES is taken: the Sanwald
Cup Mondays at Die Drei Sindelfingen, the BC Stuttgart 1891 house
tournaments, the Magic-Cup and Tito's Cup in Pfullingen, the 10-Ball BULL
CUP in Leonberg, Konstanz, Aalen and the rest. Doubles, team and darts
events are skipped (SKIP_TITLE). The season list is the unfiltered result
search, one page per year.

Results stay in data/tournamentapp_*.csv. They are not written into
data/games.csv. Pages of events from the last REFRESH_DAYS, and the current
season's list, are fetched again on every run; older pages come from the
cache.

Identity is three joins:

1. BVBW league rows join on Pass-Nr. (see identity.py).
2. The first link from a TournamentApp sid to a Pass-Nr. is a record in
   data/tournamentapp_identity.csv. All of these must hold:
   - the full name (any token order, case, accents) matches exactly one
     league player. No fuzzy or first-name-only matching, so Dayyani Aryan
     never joins Arash Dayyani and Haabinger never joins Haab;
   - no other sid with the same name played or entered the same event as
     this sid. Two Heydarpour Saeid sids in one bracket are two people, and
     neither is linked;
   - the TournamentApp-only Elo agrees with the league Elo within
     ELO_Z_LIMIT standard errors, after removing the median shift between
     the two fits. Elo is only this late check, never the reason for a link;
   - the opponent network does not contradict it (network_check). The
     people the sid met at events are compared with the league player's
     league opponents and teammates. More shared people than chance
     (p < NETWORK_P) confirms. Chance-level overlap where a few were
     expected rejects. A common surname needs the confirmation; an uncommon
     name with too little data on either side stays linked on the name.
   A hand-written row (any other evidence note) is kept as it is.
3. Later games join on that sid. They are not matched by name again. A sid
   is one TournamentApp account, so a link made from Sanwald games also
   covers that player's BULL CUP or Stuttgart games.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from html import unescape
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "tournamentapp_cache"
BASE = "https://www.tournamentapp.de/"
# TournamentApp seasons are calendar years. Results start in 2021.
FIRST_SEASON = 2021
SEARCH = BASE + "turnierergebnisse.php?s=&suche=&saison={season}"
# Venues in Baden-Württemberg, where league players can be linked.
REGION_PLACES = {
    "aalen", "bad saulgau", "bietigheim-bissingen", "böblingen", "bruchsal",
    "calw", "denzlingen", "esslingen", "fellbach", "filderstadt", "freiburg",
    "freudenstadt", "friedrichshafen", "göppingen", "heidelberg", "heilbronn",
    "herrenberg", "karlsruhe", "konstanz", "leonberg", "ludwigsburg",
    "mannheim", "markdorf", "nagold", "offenburg", "pforzheim", "pfullingen",
    "rastatt", "ravensburg", "reutlingen", "schwäbisch gmünd",
    "schwäbisch hall", "sindelfingen", "stuttgart", "tübingen", "ulm",
    "villingen-schwenningen", "waldstetten", "weingarten",
}
# Pairs, teams, other games and organiser test events: nothing to rate.
SKIP_TITLE = re.compile(
    r"doppel|double|scotch|team|mannschaft|dart|snooker|karambol|vs\.? "
    r"|bank-?pool|one pocket|demo|test",
    re.I,
)
# Events this recent may still change, so their pages are fetched again.
REFRESH_DAYS = 14

MONTHS = {
    "januar": 1,
    "februar": 2,
    "märz": 3,
    "maerz": 3,
    "april": 4,
    "mai": 5,
    "juni": 6,
    "juli": 7,
    "august": 8,
    "september": 9,
    "oktober": 10,
    "november": 11,
    "dezember": 12,
}
DATE_RE = re.compile(
    r"(\d{1,2})\.\s*([A-Za-zÄÖÜäöü]+)\s+(\d{4})(?:,\s*(\d{2}:\d{2}))?"
)
SID_RE = re.compile(r"sid=(\d+)")
MATCH_ID_RE = re.compile(r"[?&]id=(\d+)")
SCORE_RE = re.compile(r"^(\d+)\s*:\s*(\d+)$")

EVENT_FIELDS = [
    "t_id",
    "event_date",
    "start_time",
    "title",
    "place",
    "discipline",
    "status",
    "tab_id",
    "url",
]
MATCH_FIELDS = [
    "t_id",
    "event_date",
    "tab_id",
    "round",
    "match_code",
    "match_id",
    "end_stamp",
    "discipline",
    "player1",
    "player1_sid",
    "score1",
    "player2",
    "player2_sid",
    "score2",
    "title",
]
ENTRY_FIELDS = [
    "t_id",
    "event_date",
    "tab_id",
    "seed",
    "player",
    "player_sid",
]
PLAYER_FIELDS = [
    "sid",
    "portrait_name",
    "subtitle",
    "result_name",
    "matches",
    "events",
    "ta_elo",
    "ta_se",
    "pass_nr",
]
IDENTITY_FIELDS = ["sid", "pass_nr", "evidence"]
CANDIDATE_FIELDS = [
    "sid",
    "portrait_name",
    "result_name",
    "matches",
    "ta_elo",
    "ta_se",
    "pass_nr",
    "league_name",
    "clubs",
    "league_elo",
    "elo_delta",
    "shared_people",
    "expected_shared",
    "network_p",
    "note",
]
IDENTITY_PATH = DATA / "tournamentapp_identity.csv"
# Same half-life as `elo.py --half-life-years 0.67`.
HALF_LIFE_YEARS = 0.67
# The two fits sit on different graphs; this many SE off the median shift
# flags a name match for manual review instead of linking it.
ELO_Z_LIMIT = 2.5
# Candidates with this many TournamentApp matches set the median shift.
SHIFT_MIN_MATCHES = 20
# Opponent network: shared people (league opponents and teammates of the
# league player, met by the sid at TournamentApp events) against the count a
# random league player would share. Below NETWORK_P the network confirms.
NETWORK_P = 0.05
# Chance-level overlap is a contradiction once this many shared people were
# expected by chance alone; below that the network is too thin to judge.
NETWORK_POWER = 2.0
# A namesake is plausible for these; such a link needs the network to confirm.
COMMON_SURNAMES = {
    "becker", "bauer", "braun", "celik", "demir", "fischer", "hartmann",
    "hoffmann", "hofmann", "huber", "kaya", "klein", "koch", "krause", "krüger",
    "lange", "lehmann", "maier", "mayer", "meier", "meyer", "müller", "neumann",
    "nguyen", "richter", "sahin", "schmidt", "schmitt", "schmitz", "schneider",
    "schröder", "schulz", "schwarz", "tran", "wagner", "weber", "werner", "wolf",
    "yilmaz", "zimmermann",
}
NETWORK_EVIDENCE = "full name, no same-event namesake, Elo consistent, opponent network confirms"
NAME_EVIDENCE = "uncommon full name, no same-event namesake, Elo consistent, network too thin to judge"
# Written by earlier versions; replaced on the next link like the two above.
AUTO_EVIDENCE = {
    NETWORK_EVIDENCE,
    NAME_EVIDENCE,
    "full name, one league player, no same-event namesake, Elo consistent",
    "full name and Elo within 30",
}
LINK_NOTE = "name matches, no namesake at the same event, Elo consistent"
LINK_EVIDENCE = {
    "linked: opponent network confirms": NETWORK_EVIDENCE,
    "linked: uncommon name, network too thin to judge": NAME_EVIDENCE,
}


def fetch(url: str, fresh: bool = False) -> str:
    CACHE.mkdir(parents=True, exist_ok=True)
    key = re.sub(r"[^A-Za-z0-9._-]+", "_", url.replace(BASE, ""))[:180]
    path = CACHE / f"{key}.html"
    if not fresh and path.exists() and path.stat().st_size > 0:
        return path.read_text(encoding="utf-8", errors="replace")
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=45) as response:
                text = unescape(response.read().decode("utf-8", "replace"))
            path.write_text(text, encoding="utf-8")
            return text
        except (urllib.error.URLError, TimeoutError) as exc:
            last_error = exc
            time.sleep(1.2 * (attempt + 1))
    raise RuntimeError(f"failed {url}: {last_error}")


def clean(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").replace("\u00a0", " ")).strip()


def parse_date(text: str) -> tuple[str, str]:
    match = DATE_RE.search(text or "")
    if not match:
        return "", ""
    day, month_name, year, clock = match.groups()
    month = MONTHS.get(month_name.casefold())
    if not month:
        return "", clock or ""
    return f"{int(day):02d}.{month:02d}.{year}", clock or ""


def classes(tag) -> list[str]:
    return tag.get("class") or []


def seasons(today: datetime) -> list[int]:
    return list(range(FIRST_SEASON, today.year + 1))


def is_recent(event_date: str, today: datetime) -> bool:
    """Undated, future, or within REFRESH_DAYS: the page may still change."""
    try:
        when = datetime.strptime(event_date, "%d.%m.%Y")
    except ValueError:
        return True
    return when >= today - timedelta(days=REFRESH_DAYS)


def list_events(season: int, fresh: bool = False) -> list[dict[str, str]]:
    url = SEARCH.format(season=f"{season % 100:02d}")
    soup = BeautifulSoup(fetch(url, fresh=fresh), "html.parser")
    events = []
    for article in soup.find_all("article"):
        onclick = article.get("onclick") or ""
        found = re.search(r"tID=(\d+)", onclick)
        if not found or "kalender-bg" not in classes(article):
            continue
        title_node = article.find("p", class_="post-classic-title")
        title = ""
        place = ""
        if title_node:
            place_node = title_node.find("span")
            place = clean(place_node.get_text(" ", strip=True)) if place_node else ""
            if place_node:
                place_node.extract()
            title = clean(title_node.get_text(" ", strip=True))
        if place.casefold() not in REGION_PLACES or SKIP_TITLE.search(title):
            continue
        date_node = article.find("span", style=re.compile("color"))
        event_date, start_time = parse_date(date_node.get_text(" ", strip=True) if date_node else "")
        badges = [clean(node.get_text(" ", strip=True)) for node in article.find_all("span", class_="badge")]
        discipline = next((badge for badge in badges if "ball" in badge.casefold() or badge == "Diverse"), "")
        status = ""
        for node in article.find_all("span"):
            kind = node.get("class") or []
            if any(item.startswith("badge-widget-resultate") for item in kind):
                status = clean(node.get_text(" ", strip=True)).split()[0]
                break
        events.append(
            {
                "t_id": found.group(1),
                "event_date": event_date,
                "start_time": start_time,
                "title": title,
                "place": place,
                "discipline": discipline,
                "status": status,
                "tab_id": "",
                "url": f"{BASE}kalender-details.php?s=&tID={found.group(1)}",
            }
        )
    return events


def detail_tabs(event: dict[str, str], fresh: bool) -> list[dict[str, str]]:
    soup = BeautifulSoup(fetch(event["url"], fresh=fresh), "html.parser")
    heading = soup.find("h4")
    if heading:
        event["title"] = clean(heading.get_text(" ", strip=True)) or event["title"]
        date_node = heading.find_next("span")
        if date_node:
            event_date, _clock = parse_date(date_node.get_text(" ", strip=True))
            if event_date:
                event["event_date"] = event_date
    tabs = []
    seen = set()
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "rub=rs" not in href or "tabID=" not in href:
            continue
        tab = re.search(r"tabID=(\d+)", href)
        if not tab or tab.group(1) in seen:
            continue
        seen.add(tab.group(1))
        label = clean(link.get_text(" ", strip=True))
        discipline = event["discipline"]
        if "9-Ball" in label or "9 ball" in label.casefold():
            discipline = "9-Ball"
        elif "8-Ball" in label or "8 ball" in label.casefold():
            discipline = "8-Ball"
        elif "10-Ball" in label or "10 ball" in label.casefold():
            discipline = "10-Ball"
        tabs.append({"tab_id": tab.group(1), "discipline": discipline, "label": label})
    if not tabs:
        tabs.append({"tab_id": "", "discipline": event["discipline"], "label": ""})
    return tabs


def parse_matches(html: str, event: dict[str, str], tab_id: str, discipline: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    round_name = ""
    rows = []
    for tr in soup.find_all("tr"):
        header = tr.find("th")
        if header and header.has_attr("colspan"):
            text = clean(header.get_text(" ", strip=True))
            if text and "PLAYER" not in text.upper():
                round_name = text
            continue
        card = tr.find("a", href=re.compile(r"begegnungskarte"))
        players = tr.find_all("a", href=SID_RE)
        if card is None or len(players) < 2:
            continue
        score = ""
        for node in tr.find_all(["div", "td", "span"]):
            text = clean(node.get_text(" ", strip=True))
            if SCORE_RE.match(text):
                score = text
                break
        found = SCORE_RE.match(score) if score else None
        if not found:
            continue
        end_node = tr.find("span", attrs={"title": "Endzeit"})
        ball = tr.find("img", title=re.compile(r"gespielt", re.I))
        row_discipline = discipline
        if ball and ball.get("title"):
            if "9-Ball" in ball["title"]:
                row_discipline = "9-Ball"
            elif "8-Ball" in ball["title"]:
                row_discipline = "8-Ball"
            elif "10-Ball" in ball["title"]:
                row_discipline = "10-Ball"
            elif "durchmischt" in ball["title"]:
                row_discipline = "Multiball"
        match_id = MATCH_ID_RE.search(card["href"])
        rows.append(
            {
                "t_id": event["t_id"],
                "event_date": event["event_date"],
                "tab_id": tab_id,
                "round": round_name,
                "match_code": clean(card.get_text(" ", strip=True)),
                "match_id": match_id.group(1) if match_id else "",
                "end_stamp": clean(end_node.get_text(" ", strip=True)) if end_node else "",
                "discipline": row_discipline,
                "player1": clean(players[0].get_text(" ", strip=True)),
                "player1_sid": SID_RE.search(players[0]["href"]).group(1),
                "score1": found.group(1),
                "player2": clean(players[1].get_text(" ", strip=True)),
                "player2_sid": SID_RE.search(players[1]["href"]).group(1),
                "score2": found.group(2),
                "title": event["title"],
            }
        )
    return rows


def parse_entries(html: str, event: dict[str, str], tab_id: str) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for tr in soup.find_all("tr"):
        name_node = tr.find(class_="team-name")
        link = tr.find("a", href=SID_RE)
        if name_node is None or link is None:
            continue
        seed_node = tr.find("span")
        seed = clean(seed_node.get_text(" ", strip=True)) if seed_node else ""
        if not seed.isdigit():
            seed = ""
        rows.append(
            {
                "t_id": event["t_id"],
                "event_date": event["event_date"],
                "tab_id": tab_id,
                "seed": seed,
                "player": clean(name_node.get_text(" ", strip=True)),
                "player_sid": SID_RE.search(link["href"]).group(1),
            }
        )
    return rows


def scrape_event(
    event: dict[str, str], fresh: bool = False
) -> tuple[dict[str, str], list[dict[str, str]], list[dict[str, str]]]:
    tabs = detail_tabs(event, fresh)
    matches: list[dict[str, str]] = []
    entries: list[dict[str, str]] = []
    tab_ids = []
    for tab in tabs:
        tab_id = tab["tab_id"]
        if tab_id:
            tab_ids.append(tab_id)
            results_url = (
                f"{BASE}kalender-details.php?s=&tID={event['t_id']}&rub=rs&tabID={tab_id}"
            )
            matches.extend(
                parse_matches(fetch(results_url, fresh=fresh), event, tab_id, tab["discipline"])
            )
            start_url = (
                f"{BASE}kalender-details.php?s=&tID={event['t_id']}&rub=sl&tabID={tab_id}"
            )
            entries.extend(parse_entries(fetch(start_url, fresh=fresh), event, tab_id))
    event["tab_id"] = "|".join(tab_ids)
    if matches and not event["discipline"]:
        event["discipline"] = matches[0]["discipline"]
    return event, matches, entries


def write_csv(path: Path, fields: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def token_key(name: str) -> str:
    name = name.replace("_", " ")
    name = name.casefold().replace("\u00a0", " ")
    parts = re.findall(r"[a-z0-9äöüß]+", name)
    return " ".join(sorted(parts))


def load_league_index() -> dict[str, list[dict[str, str]]]:
    index: dict[str, list[dict[str, str]]] = defaultdict(list)
    path = DATA / "player_identity.csv"
    if not path.exists():
        return index
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            name = clean(row.get("display_name") or "")
            pass_nr = (row.get("pass_nr") or "").strip()
            if not name or not pass_nr.isdigit() or name.casefold().startswith("freilos"):
                continue
            key = token_key(name)
            if not key:
                continue
            index[key].append(
                {
                    "pass_nr": pass_nr,
                    "league_name": name,
                    "clubs": clean(row.get("clubs") or ""),
                }
            )
    return index


def portrait(sid: str) -> tuple[str, str]:
    url = f"{BASE}spieler-portrait.php?s=&sid={sid}"
    try:
        soup = BeautifulSoup(fetch(url), "html.parser")
    except RuntimeError:
        return "", ""
    title = soup.find("h4", class_="player-info-title")
    subtitle = soup.find("p", class_="player-info-subtitle")
    return (
        clean(title.get_text(" ", strip=True)) if title else "",
        clean(subtitle.get_text(" ", strip=True)) if subtitle else "",
    )


def load_league_elo() -> dict[str, tuple[int, float]]:
    """Pass-Nr. → (league Elo, its standard error)."""
    path = DATA / "player_elo.csv"
    ratings: dict[str, tuple[int, float]] = {}
    if not path.exists():
        return ratings
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("scope") != "all":
                continue
            pass_nr = (row.get("pass_nr") or "").strip()
            if not pass_nr.isdigit():
                continue
            try:
                ratings[pass_nr] = (int(row["elo"]), float(row.get("elo_se") or 0))
            except (KeyError, ValueError):
                continue
    return ratings


def load_entry_events() -> dict[str, set[str]]:
    """sid → events it was seeded into, from the start lists."""
    events: dict[str, set[str]] = defaultdict(set)
    path = DATA / "tournamentapp_entries.csv"
    if not path.exists():
        return events
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = (row.get("player_sid") or "").strip()
            if sid:
                events[sid].add(row.get("t_id") or "")
    return events


def tournament_ratings(matches: list[dict[str, str]]) -> tuple[dict[str, float], dict[str, float]]:
    """Published TournamentApp-only Elo for each TournamentApp id.

    Finished races only (elo.finished_races). Forfeits and mistyped rows stay
    out of the fit. The frozen offset is added, same as the league table.
    """
    from elo import (
        PRIOR_RATING,
        SCALE,
        assign_weights,
        collect,
        finished_races,
        fit_elo,
        outcomes,
        rating_se,
        scale_offset,
    )

    rows = []
    for match in finished_races(matches):
        score1 = int(match["score1"])
        score2 = int(match["score2"])
        if score1 > score2:
            winner, loser = match["player1_sid"], match["player2_sid"]
            home_score, away_score = score1, score2
        else:
            winner, loser = match["player2_sid"], match["player1_sid"]
            home_score, away_score = score2, score1
        rows.append(
            {
                "match_date": match["event_date"],
                "winner": winner,
                "loser": loser,
                "home_player": winner,
                "away_player": loser,
                "frame_score": f"{home_score}:{away_score}",
                "discipline": match.get("discipline") or "9-Ball",
                "home_team": "",
                "away_team": "",
                "league": "TournamentApp",
                "staffel": match.get("title") or "",
            }
        )
    played, as_of = assign_weights(outcomes(rows), HALF_LIFE_YEARS)
    fitted_players = collect(played)
    rating = fit_elo(fitted_players, as_of=as_of, half_life_years=HALF_LIFE_YEARS)
    offset = scale_offset()
    published: dict[str, float] = {}
    errors: dict[str, float] = {}
    for sid, fitted in rating.items():
        published[sid] = fitted + offset
        errors[sid] = rating_se(
            sid,
            fitted_players[sid],
            rating,
            prior_rating=PRIOR_RATING,
            prior_games=1.0,
            scale=SCALE,
            as_of=as_of,
            half_life_years=HALF_LIFE_YEARS,
        )
    return published, errors


def load_identity() -> list[dict[str, str]]:
    """sid → Pass-Nr. rows. A row needs both ids and an evidence note.

    A club-only player without a Pass-Nr. is linked by the local id from
    data/player_ids.csv, e.g. local:hadies|Tübinger BC.
    """
    if not IDENTITY_PATH.exists():
        return []
    rows = []
    with IDENTITY_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sid = (row.get("sid") or "").strip()
            pass_nr = (row.get("pass_nr") or "").strip()
            evidence = (row.get("evidence") or "").strip()
            if sid.isdigit() and (pass_nr.isdigit() or pass_nr.startswith("local:")) and evidence:
                rows.append({"sid": sid, "pass_nr": pass_nr, "evidence": evidence})
    return rows


def save_name_elo_identity(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    """Record the initial matches. A hand-written evidence note for a sid is kept."""
    kept = [row for row in load_identity() if row["evidence"] not in AUTO_EVIDENCE]
    kept_sids = {row["sid"] for row in kept}
    recorded = list(kept)
    for row in candidates:
        evidence = LINK_EVIDENCE.get(row["note"])
        if evidence is None:
            continue
        if row["sid"] in kept_sids or not row["pass_nr"].isdigit():
            continue
        recorded.append({"sid": row["sid"], "pass_nr": row["pass_nr"], "evidence": evidence})
    recorded.sort(key=lambda row: int(row["sid"]))
    write_csv(IDENTITY_PATH, IDENTITY_FIELDS, recorded)
    return recorded


def league_neighbourhoods() -> dict[str, set[str]]:
    """Pass-Nr. → league opponents and teammates (same team and season)."""
    hood: dict[str, set[str]] = defaultdict(set)
    squads: dict[tuple[str, str], set[str]] = defaultdict(set)
    path = DATA / "games.csv"
    if not path.exists():
        return hood
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            home = (row.get("home_player_id") or "").strip()
            away = (row.get("away_player_id") or "").strip()
            if home.isdigit() and away.isdigit():
                hood[home].add(away)
                hood[away].add(home)
            for player, team in ((home, row.get("home_team")), (away, row.get("away_team"))):
                if player.isdigit() and team:
                    squads[(row.get("season") or "", team)].add(player)
    for squad in squads.values():
        for player in squad:
            hood[player] |= squad - {player}
    return hood


def poisson_tail(observed: int, expected: float) -> float:
    """P(X >= observed) for X ~ Poisson(expected)."""
    if observed <= 0:
        return 1.0
    below, term = 0.0, math.exp(-expected)
    for count in range(observed):
        below += term
        term *= expected / (count + 1)
    return max(0.0, 1.0 - below)


def network_check(matches: list[dict[str, str]], candidates: list[dict[str, str]]) -> None:
    """Second step for name matches: do they move among the same people?

    The sid's opponents are mapped to Pass-Nr. through the hand-written links
    and the other name matches. Shared people are those that are also the
    league player's league opponents or teammates. Updates each note.
    """
    provisional = {
        row["sid"]: row["pass_nr"] for row in load_identity() if row["evidence"] not in AUTO_EVIDENCE
    }
    for row in candidates:
        if row["note"] == LINK_NOTE:
            provisional.setdefault(row["sid"], row["pass_nr"])
    met: dict[str, set[str]] = defaultdict(set)
    for row in matches:
        one, two = row["player1_sid"], row["player2_sid"]
        if two in provisional:
            met[one].add(provisional[two])
        if one in provisional:
            met[two].add(provisional[one])
    hood = league_neighbourhoods()
    population = len(hood) or 1
    for row in candidates:
        if row["note"] != LINK_NOTE:
            continue
        pass_nr = row["pass_nr"]
        theirs = met[row["sid"]] - {pass_nr}
        mine = hood.get(pass_nr, set())
        shared = len(theirs & mine)
        expected = len(theirs) * len(mine) / population
        p_value = poisson_tail(shared, expected)
        row["shared_people"] = str(shared)
        row["expected_shared"] = f"{expected:.1f}"
        row["network_p"] = f"{p_value:.2g}"
        common = any(token.casefold() in COMMON_SURNAMES for token in row["league_name"].split())
        if p_value < NETWORK_P:
            row["note"] = "linked: opponent network confirms"
        elif expected >= NETWORK_POWER and shared <= expected:
            row["note"] = "rejected: opponent network no better than chance"
        elif common:
            row["note"] = "rejected: common surname, opponent network does not confirm"
        else:
            row["note"] = "linked: uncommon name, network too thin to judge"


def link_players(
    matches: list[dict[str, str]],
    portraits: dict[str, tuple[str, str]] | None = None,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    names: dict[str, set[str]] = defaultdict(set)
    match_counts: dict[str, int] = defaultdict(int)
    event_sets: dict[str, set[str]] = defaultdict(set)
    for row in matches:
        if row["score1"] == row["score2"]:
            continue
        for side in ("1", "2"):
            sid = row[f"player{side}_sid"]
            names[sid].add(row[f"player{side}"])
            match_counts[sid] += 1
            event_sets[sid].add(row["t_id"])
    index = load_league_index()
    league_elo = load_league_elo()
    print("fitting TournamentApp-only Elo", flush=True)
    ratings, errors = tournament_ratings(matches)
    sids = sorted(names, key=lambda sid: (-match_counts[sid], int(sid)))
    if portraits is None:
        portraits = {}
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(portrait, sid): sid for sid in sids}
            done = 0
            for future in as_completed(futures):
                sid = futures[future]
                portraits[sid] = future.result()
                done += 1
                if done % 100 == 0:
                    print(f"  portraits {done}/{len(sids)}", flush=True)
    entry_events = load_entry_events()
    keys: dict[str, set[str]] = {}
    sids_by_key: dict[str, set[str]] = defaultdict(set)
    for sid in sids:
        result_name = sorted(names[sid], key=len)[-1]
        portrait_name = portraits.get(sid, ("", ""))[0]
        keys[sid] = {key for key in (token_key(portrait_name), token_key(result_name)) if key}
        for key in keys[sid]:
            sids_by_key[key].add(sid)

    def namesake_clash(sid: str) -> bool:
        mine = event_sets[sid] | entry_events.get(sid, set())
        for key in keys[sid]:
            for other in sids_by_key[key] - {sid}:
                if mine & (event_sets[other] | entry_events.get(other, set())):
                    return True
        return False

    shift_sample = []
    for sid in sids:
        hits = {item["pass_nr"] for key in keys[sid] for item in index.get(key, [])}
        if len(hits) == 1 and match_counts[sid] >= SHIFT_MIN_MATCHES and sid in ratings:
            league = league_elo.get(next(iter(hits)))
            if league is not None:
                shift_sample.append(ratings[sid] - league[0])
    shift = statistics.median(shift_sample) if shift_sample else 0.0
    print(f"TournamentApp minus league Elo, median over {len(shift_sample)} name matches: {shift:+.0f}")

    players = []
    candidates = []
    for sid in sids:
        result_name = sorted(names[sid], key=len)[-1]
        portrait_name, subtitle = portraits.get(sid, ("", ""))
        name_hits = []
        seen = set()
        for label in (portrait_name, result_name):
            key = token_key(label)
            for item in index.get(key, []):
                if item["pass_nr"] in seen:
                    continue
                seen.add(item["pass_nr"])
                name_hits.append(item)
        chosen = {"pass_nr": "", "league_name": "", "clubs": ""}
        note = ""
        if len(name_hits) == 1:
            chosen = name_hits[0]
        elif len(name_hits) > 1:
            note = "several league players share this name"
            chosen = {
                "pass_nr": "|".join(item["pass_nr"] for item in name_hits),
                "league_name": "|".join(item["league_name"] for item in name_hits),
                "clubs": "|".join(item["clubs"] for item in name_hits),
            }
        ta_elo = ratings.get(sid)
        ta_se = errors.get(sid)
        elo_text = "" if ta_elo is None else str(round(ta_elo))
        se_text = "" if ta_se is None else str(round(ta_se))
        players.append(
            {
                "sid": sid,
                "portrait_name": portrait_name,
                "subtitle": subtitle,
                "result_name": result_name,
                "matches": str(match_counts[sid]),
                "events": str(len(event_sets[sid])),
                "ta_elo": elo_text,
                "ta_se": se_text,
            }
        )
        league = league_elo.get(chosen["pass_nr"]) if len(name_hits) == 1 else None
        if len(name_hits) == 1 and ta_elo is not None and league is not None:
            league_rating, league_se = league
            delta_value = round(ta_elo) - league_rating
            spread = math.hypot(ta_se or 0.0, league_se)
            z = (delta_value - shift) / spread if spread else 0.0
            if namesake_clash(sid):
                note = "another sid with this name played the same event"
            elif abs(z) > ELO_Z_LIMIT:
                note = f"name matches; Elo off by {z:+.1f} SE"
            else:
                note = LINK_NOTE
            candidates.append(
                {
                    "sid": sid,
                    "portrait_name": portrait_name,
                    "result_name": result_name,
                    "matches": str(match_counts[sid]),
                    "ta_elo": elo_text,
                    "ta_se": se_text,
                    "pass_nr": chosen["pass_nr"],
                    "league_name": chosen["league_name"],
                    "clubs": chosen["clubs"],
                    "league_elo": str(league_rating),
                    "elo_delta": str(delta_value),
                    "note": note,
                }
            )
        elif note:
            candidates.append(
                {
                    "sid": sid,
                    "portrait_name": portrait_name,
                    "result_name": result_name,
                    "matches": str(match_counts[sid]),
                    "ta_elo": elo_text,
                    "ta_se": se_text,
                    "pass_nr": chosen["pass_nr"],
                    "league_name": chosen["league_name"],
                    "clubs": chosen["clubs"],
                    "league_elo": "",
                    "elo_delta": "",
                    "note": note,
                }
            )
    network_check(matches, candidates)
    return players, candidates


def scrape(workers: int) -> None:
    today = datetime.now()
    listed: dict[str, dict[str, str]] = {}
    for season in seasons(today):
        found = list_events(season, fresh=season == today.year)
        titles = defaultdict(int)
        for event in found:
            titles[f"{event['title']} ({event['place']})"] += 1
        print(f"season {season}: {len(found)} events", flush=True)
        for title, count in sorted(titles.items(), key=lambda item: -item[1]):
            print(f"  {count:3d}  {title}", flush=True)
        for event in found:
            listed.setdefault(event["t_id"], event)
    events = sorted(listed.values(), key=lambda row: (row["event_date"][6:], row["event_date"][3:5], row["event_date"][:2], int(row["t_id"])))
    print(f"scraping {len(events)} events", flush=True)
    scraped_events = []
    matches = []
    entries = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(scrape_event, event, is_recent(event["event_date"], today)): event["t_id"]
            for event in events
        }
        done = 0
        for future in as_completed(futures):
            event, event_matches, event_entries = future.result()
            scraped_events.append(event)
            matches.extend(event_matches)
            entries.extend(event_entries)
            done += 1
            if done % 25 == 0 or done == len(events):
                print(
                    f"  {done}/{len(events)} events, {len(matches)} matches",
                    flush=True,
                )
    scraped_events.sort(key=lambda row: (row["event_date"][6:], row["event_date"][3:5], row["event_date"][:2], int(row["t_id"])))
    matches.sort(key=lambda row: (row["event_date"][6:], row["event_date"][3:5], row["event_date"][:2], int(row["t_id"] or 0), row["match_code"]))
    entries.sort(key=lambda row: (row["event_date"][6:], row["event_date"][3:5], row["event_date"][:2], int(row["t_id"] or 0), int(row["seed"] or 0)))
    write_csv(DATA / "tournamentapp_events.csv", EVENT_FIELDS, scraped_events)
    write_csv(DATA / "tournamentapp_matches.csv", MATCH_FIELDS, matches)
    write_csv(DATA / "tournamentapp_entries.csv", ENTRY_FIELDS, entries)
    publish(matches, events=len(scraped_events), entries=len(entries))


def publish(
    matches: list[dict[str, str]],
    portraits: dict[str, tuple[str, str]] | None = None,
    *,
    events: int | None = None,
    entries: int | None = None,
) -> None:
    print("rating TournamentApp players by sid", flush=True)
    players, candidates = link_players(matches, portraits=portraits)
    identity = save_name_elo_identity(candidates)
    pass_by_sid = {row["sid"]: row["pass_nr"] for row in identity if row["pass_nr"].isdigit()}
    for player in players:
        player["pass_nr"] = pass_by_sid.get(player["sid"], "")
    write_csv(DATA / "tournamentapp_players.csv", PLAYER_FIELDS, players)
    write_csv(DATA / "tournamentapp_candidates.csv", CANDIDATE_FIELDS, candidates)
    decisive = [row for row in matches if row["score1"] != row["score2"]]
    if events is not None:
        print(f"events {events}")
    print(f"matches {len(matches)} decisive {len(decisive)}")
    if entries is not None:
        print(f"start-list seats {entries}")
    print(f"players {len(players)} keyed by sid")
    print(f"initial sid → Pass-Nr. records {len(identity)}")
    print(f"name hits left unmatched {sum(1 for row in candidates if row['sid'] not in pass_by_sid)}")


def link_only() -> None:
    matches = list(csv.DictReader((DATA / "tournamentapp_matches.csv").open(encoding="utf-8")))
    portraits: dict[str, tuple[str, str]] = {}
    player_path = DATA / "tournamentapp_players.csv"
    if player_path.exists():
        for row in csv.DictReader(player_path.open(encoding="utf-8")):
            portraits[row["sid"]] = (row.get("portrait_name") or "", row.get("subtitle") or "")
    publish(matches, portraits=portraits)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--link-only",
        action="store_true",
        help="refit TournamentApp Elo and rewrite data/tournamentapp_players.csv from the saved matches",
    )
    args = parser.parse_args()
    if args.link_only:
        link_only()
    else:
        scrape(args.workers)


if __name__ == "__main__":
    main()
