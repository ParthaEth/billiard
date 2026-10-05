#!/usr/bin/env python3
"""Join BVBW games to people by Pass-Nr., not by name alone.

A Pass-Nr. is the player's DBU number. It is unique: one number is one
person, for every club and season. A name is not unique. Two squad lists
can both say "Alexander Fischer" and be different people.

League games are linked only when that season's team squad lists the name
with exactly one Pass-Nr. Individual games are linked from the tournament
entry list (the option value on the results page is the Pass-Nr.). In-house
games are linked when the name belongs to exactly one Tübinger BC Pass-Nr.
A game with no such link keeps its own identity, name plus club, and is
never folded into a Pass-Nr. just because the spelling matches.
Club nicknames in data/aliases.csv are the exception. Bluber and Blubber
are Hannes Reißner, Pass-Nr. 368517. Matze is Matthias Ludersdorfer,
Pass-Nr. 102864.

    python identity.py fetch     # download squads and write data/roster.csv
    python identity.py attach    # add player ids to games.csv and inhouse.csv
    python identity.py report    # write data/player_identity.csv
"""

from __future__ import annotations

import csv
import hashlib
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
CACHE = DATA / "identity_cache"
ROSTER_PATH = DATA / "roster.csv"
ALIASES_PATH = DATA / "aliases.csv"
IDENTITY_PATH = DATA / "player_identity.csv"
GAMES_PATH = DATA / "games.csv"

BASE = "https://billard-bvbw.de/"
UA = "billiard-identity/1.0 (club rating research)"
CTX = ssl.create_default_context()
FIRST_SEASON_YEAR = 2021


def live_seasons(today: datetime | None = None) -> tuple[str, ...]:
    """2021/2022 through the season containing today. September opens a season."""
    today = today or datetime.now()
    last = today.year if today.month >= 9 else today.year - 1
    return tuple(f"{year}/{year + 1}" for year in range(FIRST_SEASON_YEAR, last + 1))


SEASONS = live_seasons()

ROSTER_FIELDS = (
    "season",
    "team",
    "club_id",
    "pass_nr",
    "name",
    "source",
    "tournament_id",
)

TEAM_KEY_RE = re.compile(r"999--(\d{4}/\d{4})-(\d+)-(\d+)-(\d+)")
PASS_HREF_RE = re.compile(r"p=999--(\d{4}/\d{4})-(\d+)--(\d+)")
LEAGUE_HREF_RE = re.compile(r"sb_spielplan\.php\?p=999\|\|(\d{4}/\d{4})\|(\d+)$")
STAFFEL_HREF_RE = re.compile(r"sb_spielplan\.php\?p=999\|\|(\d{4}/\d{4})\|(\d+)\|(\d+)$")
EINZEL_RE = re.compile(r"einzelergebnisse\.php\?p=[^\"']*?(\d{4}/\d{4})-(\d+)")
TRAILING_TEAM_NO = re.compile(r"\s+\d+$")

_RATE_LOCK = threading.Lock()
_NEXT_REQUEST = 0.0
# Squad pages of a running season gain players, so these seasons bypass the cache once per run.
_REFRESH_SEASONS: frozenset[str] = frozenset()
_REFRESHED: set[str] = set()


def norm_name(name: str) -> str:
    name = (name or "").replace("\u00a0", " ").casefold()
    return re.sub(r"\s+", " ", name).strip()


def club_from_team(team: str) -> str:
    """'Tübinger BC 2' and 'Tübinger BC 1' are one club."""
    team = re.sub(r"\s+", " ", (team or "").replace("\u00a0", " ")).strip()
    return TRAILING_TEAM_NO.sub("", team)


def flip_last_first(label: str) -> str:
    """Tournament lists use 'Fischer, Alexander'. Games use 'Alexander Fischer'."""
    label = re.sub(r"\s+", " ", (label or "").replace("\u00a0", " ")).strip()
    if "," not in label:
        return label
    last, first = label.split(",", 1)
    return f"{first.strip()} {last.strip()}".strip()


def _cache_path(url: str) -> Path:
    digest = hashlib.sha1(url.encode()).hexdigest()
    return CACHE / f"{digest}.html"


def _rate_limit() -> None:
    global _NEXT_REQUEST
    with _RATE_LOCK:
        now = time.monotonic()
        wait = _NEXT_REQUEST - now
        _NEXT_REQUEST = max(now, _NEXT_REQUEST) + 0.2
    if wait > 0:
        time.sleep(wait)


def fetch(url: str, retries: int = 4) -> str:
    dest = _cache_path(url)
    stale = url not in _REFRESHED and any(season in url for season in _REFRESH_SEASONS)
    if not stale and dest.exists() and dest.stat().st_size > 1000:
        return dest.read_text(encoding="utf-8", errors="replace")
    _REFRESHED.add(url)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(retries):
        _rate_limit()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=60, context=CTX) as resp:
                data = resp.read()
            text = data.decode("utf-8", "replace")
            if len(text) > 1000 and "billard" in text.casefold():
                dest.write_text(text, encoding="utf-8")
                return text
            last_error = RuntimeError(f"short page {len(text)}")
        except (urllib.error.URLError, TimeoutError, OSError, RuntimeError) as exc:
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    print(f"  fetch failed {url} ({last_error})")
    if dest.exists() and dest.stat().st_size > 1000:
        return dest.read_text(encoding="utf-8", errors="replace")
    return ""


def _soup(html: str):
    from bs4 import BeautifulSoup

    return BeautifulSoup(html, "html.parser")


def pool_league_urls(html: str) -> list[str]:
    soup = _soup(html)
    found: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a"):
        title = anchor.get("title") or ""
        href = anchor.get("href") or ""
        if not title.startswith("Pool -"):
            continue
        if not LEAGUE_HREF_RE.search(href):
            continue
        url = urljoin(BASE, href)
        if url not in seen:
            seen.add(url)
            found.append(url)
    return found


def staffel_urls(html: str, season: str, league_id: str) -> list[str]:
    soup = _soup(html)
    found: list[str] = []
    seen: set[str] = set()
    for anchor in soup.find_all("a"):
        href = anchor.get("href") or ""
        match = STAFFEL_HREF_RE.search(href)
        if not match:
            continue
        if match.group(1) != season or match.group(2) != league_id:
            continue
        url = urljoin(BASE, href)
        if url not in seen:
            seen.add(url)
            found.append(url)
    return found


def team_keys(html: str) -> set[str]:
    return {match.group(0) for match in TEAM_KEY_RE.finditer(html)}


def parse_team_page(html: str, key: str) -> list[dict[str, str]]:
    match = TEAM_KEY_RE.fullmatch(key)
    if not match or not html:
        return []
    season, _league, _staffel, team_id = match.groups()
    soup = _soup(html)
    team = ""
    for anchor in soup.find_all("a"):
        href = anchor.get("href") or ""
        title = anchor.get("title") or ""
        if not (title.startswith("Mannschaftsspielplan von ") and title.endswith(" anzeigen")):
            continue
        own_page = re.search(rf"(?:^|[?&=]){re.escape(key)}(?:$|[^\d])", href)
        own_id = re.search(rf"(?:^|[?&])t={team_id}(?:&|$)", href)
        if not own_page and not own_id:
            continue
        team = title.removeprefix("Mannschaftsspielplan von ").removesuffix(" anzeigen").strip()
        break
    if not team:
        return []
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for anchor in soup.find_all("a"):
        href = anchor.get("href") or ""
        found = PASS_HREF_RE.search(href)
        name = re.sub(r"\s+", " ", anchor.get_text(" ", strip=True)).strip()
        if not found or not name:
            continue
        link_season, club_id, pass_nr = found.groups()
        if link_season != season:
            continue
        marker = (pass_nr, norm_name(name))
        if marker in seen:
            continue
        seen.add(marker)
        rows.append(
            {
                "season": season,
                "team": team,
                "club_id": club_id,
                "pass_nr": pass_nr,
                "name": name,
                "source": "team",
                "tournament_id": "",
            }
        )
    return rows


def parse_einzel_page(html: str, season: str, tournament_id: str) -> list[dict[str, str]]:
    if not html:
        return []
    soup = _soup(html)
    rows: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for option in soup.find_all("option"):
        pass_nr = (option.get("value") or "").strip()
        raw = re.sub(r"\s+", " ", option.get_text(" ", strip=True)).strip()
        # Player rows are "Last, First". Other options are filters ("10-Ball", "Alle Ergebnisse").
        if "," not in raw or not pass_nr.isdigit() or len(pass_nr) < 5:
            continue
        name = flip_last_first(raw)
        if not name:
            continue
        marker = (pass_nr, norm_name(name))
        if marker in seen:
            continue
        seen.add(marker)
        rows.append(
            {
                "season": season,
                "team": "",
                "club_id": "",
                "pass_nr": pass_nr,
                "name": name,
                "source": "einzel",
                "tournament_id": tournament_id,
            }
        )
    return rows


def einzel_targets(games_path: Path = GAMES_PATH) -> list[tuple[str, str]]:
    found: set[tuple[str, str]] = set()
    if not games_path.exists():
        return []
    with games_path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            url = row.get("report_url") or ""
            match = EINZEL_RE.search(url)
            if match:
                found.add((match.group(1), match.group(2)))
    return sorted(found)


def write_roster(rows: list[dict[str, str]], path: Path = ROSTER_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=ROSTER_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_roster(path: Path = ROSTER_PATH) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_aliases(path: Path = ALIASES_PATH) -> dict[str, tuple[str, str]]:
    """Map a club nickname to (Pass-Nr., official name).

    The key is the normalized nickname. Both spellings of a nickname can
    point at the same Pass-Nr.
    """
    if not path.exists():
        return {}
    aliases: dict[str, tuple[str, str]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            alias = norm_name(row.get("alias") or "")
            pass_nr = (row.get("pass_nr") or "").strip()
            name = (row.get("name") or "").strip()
            if alias and pass_nr and name:
                aliases[alias] = (pass_nr, name)
    return aliases


class Roster:
    """Pass lookups. A Pass-Nr. never maps to two people."""

    def __init__(self, rows: list[dict[str, str]]) -> None:
        self.team: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.einzel: dict[tuple[str, str, str], set[str]] = defaultdict(set)
        self.by_pass_names: dict[str, Counter[str]] = defaultdict(Counter)
        self.pass_clubs: dict[str, set[str]] = defaultdict(set)
        self.name_passes: dict[str, set[str]] = defaultdict(set)
        self.club_name_passes: dict[tuple[str, str], set[str]] = defaultdict(set)
        tbc_ids: set[str] = set()
        for row in rows:
            pass_nr = (row.get("pass_nr") or "").strip()
            name = (row.get("name") or "").strip()
            if not pass_nr or not name:
                continue
            key = norm_name(name)
            self.by_pass_names[pass_nr][name] += 1
            self.name_passes[key].add(pass_nr)
            season = (row.get("season") or "").strip()
            team = (row.get("team") or "").strip()
            club_id = (row.get("club_id") or "").strip()
            source = (row.get("source") or "").strip()
            if team:
                self.team[(season, norm_name(team), key)].add(pass_nr)
                club = club_from_team(team)
                self.pass_clubs[pass_nr].add(club)
                if club_id:
                    self.club_name_passes[(club_id, key)].add(pass_nr)
                if club.casefold() == "tübinger bc":
                    tbc_ids.add(club_id)
            if source == "einzel":
                tid = (row.get("tournament_id") or "").strip()
                if tid:
                    self.einzel[(season, tid, key)].add(pass_nr)
        self.tbc_ids = {item for item in tbc_ids if item}
        self.aliases = load_aliases()
        self.tbc_names: dict[str, set[str]] = defaultdict(set)
        for (club_id, key), passes in self.club_name_passes.items():
            if club_id in self.tbc_ids:
                self.tbc_names[key].update(passes)

    def canonical_name(self, pass_nr: str) -> str:
        names = self.by_pass_names.get(pass_nr)
        if not names:
            return ""
        return names.most_common(1)[0][0]


def tournament_of(url: str) -> tuple[str, str] | None:
    match = EINZEL_RE.search(url or "")
    if not match:
        return None
    return match.group(1), match.group(2)


def _one(passes: set[str]) -> str:
    if len(passes) == 1:
        return next(iter(passes))
    return ""


def lookup_pass(roster: Roster, row: dict[str, str], side: str) -> str:
    """Return the Pass-Nr. for this seat, or '' when the link is not unique."""
    name = (row.get(f"{side}_player") or "").strip()
    if not name:
        return ""
    key = norm_name(name)
    alias = roster.aliases.get(key)
    if alias:
        return alias[0]
    team = (row.get(f"{side}_team") or "").strip()
    season = (row.get("season") or "").strip()
    league = (row.get("league") or "").strip()

    if league == "Intern":
        return _one(roster.tbc_names.get(key, set()))

    if team:
        return _one(roster.team.get((season, norm_name(team), key), set()))

    found = tournament_of(row.get("report_url") or "")
    if found:
        event_season, tid = found
        linked = _one(roster.einzel.get((event_season, tid, key), set()))
        if linked:
            return linked
    if season >= "2021/2022":
        return _one(roster.name_passes.get(key, set()))
    return ""


def official_name(roster: Roster, name: str) -> str:
    """Return the roster name when this spelling is a known nickname."""
    alias = roster.aliases.get(norm_name(name))
    if alias:
        return alias[1]
    return name


def public_player_id(roster: Roster, row: dict[str, str], side: str) -> str:
    """Pass-Nr. when the squad list identifies the player, otherwise a local id.

    A local id is name plus club. It is not a DBU number, and it is not
    joined to a Pass-Nr. that happens to use the same spelling.
    """
    name = (row.get(f"{side}_player") or "").strip()
    if not name:
        return ""
    pass_nr = lookup_pass(roster, row, side)
    if pass_nr:
        return pass_nr
    club = club_from_team(row.get(f"{side}_team") or "")
    if (row.get("league") or "").strip() == "Intern" and not club:
        club = "Tübinger BC"
    return f"local:{norm_name(name)}|{club or 'unassigned'}"


def _person_key(pass_nr: str, name: str, club: str) -> str:
    if pass_nr:
        return f"p:{pass_nr}"
    return f"n:{norm_name(name)}|{club or 'unassigned'}"


def _display_names(people: dict[str, dict]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for key, info in people.items():
        grouped[info["norm"]].append(key)
    labels: dict[str, str] = {}
    for keys in grouped.values():
        pass_keys = [key for key in keys if people[key]["pass"]]
        for key in keys:
            info = people[key]
            if len(keys) == 1:
                labels[key] = info["name"]
            elif info["pass"] and len(pass_keys) == 1:
                labels[key] = info["name"]
            elif info["pass"]:
                labels[key] = f"{info['name']} [{info['pass']}]"
            else:
                labels[key] = f"{info['name']} ({info['club'] or 'unassigned'})"
    return labels


def apply_identities(
    rows: list[dict[str, str]],
    roster_path: Path = ROSTER_PATH,
    identity_path: Path | None = None,
) -> list[dict[str, str]]:
    """Rewrite player names so each person has one label. Pass wins over name."""
    if not rows or rows[0].get("_identity") == "1":
        return rows
    roster_rows = load_roster(roster_path)
    if not roster_rows:
        return rows
    roster = Roster(roster_rows)
    pending: list[tuple[dict[str, str], str, str, str, str]] = []
    people: dict[str, dict] = {}
    for row in rows:
        sides: dict[str, tuple[str, str]] = {}
        for side in ("home", "away"):
            raw = (row.get(f"{side}_player") or "").strip()
            if not raw:
                continue
            pass_nr = lookup_pass(roster, row, side)
            club = club_from_team(row.get(f"{side}_team") or "")
            if (row.get("league") or "").strip() == "Intern" and not club:
                club = "Tübinger BC"
            person = _person_key(pass_nr, raw, club)
            sides[side] = (pass_nr, person)
            info = people.setdefault(
                person,
                {
                    "pass": pass_nr,
                    "norm": norm_name(roster.canonical_name(pass_nr) or raw),
                    "name": roster.canonical_name(pass_nr) or raw,
                    "club": club,
                    "names": Counter(),
                    "clubs": set(),
                    "teams": set(),
                    "games": 0,
                },
            )
            info["names"][raw] += 1
            info["games"] += 1
            if club:
                info["clubs"].add(club)
            team = (row.get(f"{side}_team") or "").strip()
            if team:
                info["teams"].add(team)
            if pass_nr and not info["name"]:
                info["name"] = raw
        pending.append(
            (
                row,
                sides.get("home", ("", ""))[0],
                sides.get("home", ("", ""))[1],
                sides.get("away", ("", ""))[0],
                sides.get("away", ("", ""))[1],
            )
        )
    for info in people.values():
        if info["names"]:
            raw_top = info["names"].most_common(1)[0][0]
            if not info["pass"]:
                info["name"] = raw_top
                info["norm"] = norm_name(raw_top)
            elif info["pass"]:
                info["norm"] = norm_name(info["name"])
    labels = _display_names(people)
    for row, home_pass, home_key, away_pass, away_key in pending:
        mapping: dict[str, str] = {}
        home_raw = (row.get("home_player") or "").strip()
        away_raw = (row.get("away_player") or "").strip()
        if home_raw and home_key:
            mapping[home_raw] = labels[home_key]
        if away_raw and away_key:
            mapping[away_raw] = labels[away_key]
        for field in ("home_player", "away_player", "winner", "loser"):
            current = (row.get(field) or "").strip()
            if current in mapping:
                row[field] = mapping[current]
        row["home_pass"] = home_pass
        row["away_pass"] = away_pass
        row["home_person"] = home_key
        row["away_person"] = away_key
        row["_identity"] = "1"
    if identity_path is not None:
        write_identity(people, labels, identity_path)
    return rows


def write_identity(people: dict[str, dict], labels: dict[str, str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = (
        "person_id",
        "pass_nr",
        "display_name",
        "name",
        "clubs",
        "teams",
        "games",
    )
    records = []
    for key, info in people.items():
        records.append(
            {
                "person_id": key,
                "pass_nr": info["pass"],
                "display_name": labels[key],
                "name": info["name"],
                "clubs": " | ".join(sorted(info["clubs"])),
                "teams": " | ".join(sorted(info["teams"])),
                "games": str(info["games"]),
            }
        )
    records.sort(key=lambda item: (-int(item["games"]), item["display_name"]))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)


def cmd_fetch(refresh: tuple[str, ...] = ()) -> None:
    """Download squads for every live season. Pages of ``refresh`` seasons are fetched anew."""
    global _REFRESH_SEASONS
    _REFRESH_SEASONS = frozenset(refresh)
    if refresh:
        print(f"Re-downloading squad pages for {', '.join(refresh)}")
    roster: list[dict[str, str]] = []
    team_keys_found: set[str] = set()
    for season in SEASONS:
        index_url = f"{BASE}sb_spielplan.php?p=999||{season}"
        print(f"season {season}")
        index = fetch(index_url)
        leagues = pool_league_urls(index)
        print(f"  pool leagues {len(leagues)}")
        staffeln: list[str] = []
        for league_url in leagues:
            match = LEAGUE_HREF_RE.search(league_url)
            if not match:
                continue
            html = fetch(league_url)
            team_keys_found.update(team_keys(html))
            staffeln.extend(staffel_urls(html, match.group(1), match.group(2)))
        print(f"  staffeln {len(staffeln)}")
        for url in staffeln:
            team_keys_found.update(team_keys(fetch(url)))
    print(f"team pages {len(team_keys_found)}")

    def load_team(key: str) -> list[dict[str, str]]:
        # Rangliste is everyone who played that season. The squad page is only
        # who is licensed now, so a player who left is missing from it.
        rows = parse_team_page(fetch(f"{BASE}sb_mannschaft-rangliste.php?p={key}"), key)
        rows.extend(parse_team_page(fetch(f"{BASE}sb_mannschaft.php?p={key}"), key))
        return rows

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(load_team, key) for key in sorted(team_keys_found)]
        for future in as_completed(futures):
            roster.extend(future.result())
            done += 1
            if done % 100 == 0:
                print(f"  teams parsed {done}/{len(team_keys_found)}")

    targets = einzel_targets()
    print(f"einzel tournaments {len(targets)}")

    def load_einzel(item: tuple[str, str]) -> list[dict[str, str]]:
        season, tid = item
        url = f"{BASE}sb_einzelergebnisse.php?p=999-6-{season}-{tid}----1-1-100000--"
        return parse_einzel_page(fetch(url), season, tid)

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(load_einzel, item) for item in targets]
        for future in as_completed(futures):
            roster.extend(future.result())
            done += 1
            if done % 50 == 0:
                print(f"  einzel parsed {done}/{len(targets)}")

    # One pass keeps one canonical row per (source, season, team/tournament, pass, name).
    deduped: list[dict[str, str]] = []
    seen: set[tuple[str, ...]] = set()
    for row in roster:
        marker = (
            row["source"],
            row["season"],
            row["team"],
            row["tournament_id"],
            row["pass_nr"],
            norm_name(row["name"]),
        )
        if marker in seen:
            continue
        seen.add(marker)
        deduped.append(row)
    write_roster(deduped)
    passes = {row["pass_nr"] for row in deduped}
    print(f"Wrote {len(deduped)} roster rows, {len(passes)} pass numbers -> {ROSTER_PATH}")
    names_for_pass: dict[str, set[str]] = defaultdict(set)
    for row in deduped:
        names_for_pass[row["pass_nr"]].add(norm_name(row["name"]))
    multi = {pass_nr: names for pass_nr, names in names_for_pass.items() if len(names) > 1}
    print(f"Pass numbers with more than one spelling: {len(multi)}")
    name_passes: dict[str, set[str]] = defaultdict(set)
    for row in deduped:
        if row["source"] == "team":
            name_passes[norm_name(row["name"])].add(row["pass_nr"])
    collisions = {name: passes for name, passes in name_passes.items() if len(passes) > 1}
    print(f"Names shared by more than one Pass-Nr. on a squad list: {len(collisions)}")
    for name, owned in sorted(collisions.items(), key=lambda item: -len(item[1]))[:15]:
        print(f"  {name}: {', '.join(sorted(owned))}")
    fischer = [
        row
        for row in deduped
        if norm_name(row["name"]) == "alexander fischer" and row["source"] == "team"
    ]
    print(f"Alexander Fischer squad rows: {len(fischer)}")
    for row in fischer[:8]:
        print(f"  {row['season']} {row['team']} pass {row['pass_nr']} club {row['club_id']}")


GAME_ID_FIELDS = (
    "season",
    "league",
    "staffel",
    "spieltag",
    "match_date",
    "match_time",
    "home_team",
    "away_team",
    "match_score",
    "round",
    "game_no",
    "discipline",
    "home_player",
    "home_player_id",
    "away_player",
    "away_player_id",
    "frame_score",
    "home_points",
    "away_points",
    "winner",
    "loser",
    "winner_side",
    "straight_pool_punkte",
    "straight_pool_aufnahmen",
    "straight_pool_hs",
    "straight_pool_gd",
    "report_url",
)
INHOUSE_FIELDS = ("date", "player", "player_id", "opponent", "opponent_id", "score", "discipline")
PLAYER_ID_FIELDS = ("player_id", "pass_nr", "name")
PLAYER_IDS_PATH = DATA / "player_ids.csv"


def _remember_id(catalog: dict[str, dict], player_id: str, name: str) -> None:
    if not player_id or not name:
        return
    info = catalog.setdefault(player_id, {"pass_nr": "", "names": Counter()})
    info["names"][name] += 1
    if player_id.isdigit():
        info["pass_nr"] = player_id


def attach_ids() -> None:
    """Write unique player ids into the league file and the in-house file."""
    if not ROSTER_PATH.exists():
        raise SystemExit(f"Missing {ROSTER_PATH}. Run: python identity.py fetch")
    roster = Roster(load_roster())
    catalog: dict[str, dict] = {}

    with GAMES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        games = list(csv.DictReader(handle))
    for row in games:
        for side in ("home", "away"):
            player_id = public_player_id(roster, row, side)
            row[f"{side}_player_id"] = player_id
            _remember_id(catalog, player_id, (row.get(f"{side}_player") or "").strip())
    games_tmp = GAMES_PATH.with_suffix(".csv.tmp")
    with games_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=GAME_ID_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(games)
    games_tmp.replace(GAMES_PATH)

    inhouse_path = DATA / "inhouse.csv"
    with inhouse_path.open(encoding="utf-8-sig", newline="") as handle:
        inhouse = list(csv.DictReader(handle))
    for row in inhouse:
        row["player"] = official_name(roster, (row.get("player") or "").strip())
        row["opponent"] = official_name(roster, (row.get("opponent") or "").strip())
        carrier = {
            "league": "Intern",
            "season": "",
            "home_player": row["player"],
            "away_player": row["opponent"],
            "home_team": "Tübinger BC",
            "away_team": "Tübinger BC",
            "report_url": "",
        }
        row["player_id"] = public_player_id(roster, carrier, "home")
        row["opponent_id"] = public_player_id(roster, carrier, "away")
        _remember_id(catalog, row["player_id"], carrier["home_player"])
        _remember_id(catalog, row["opponent_id"], carrier["away_player"])
    inhouse_tmp = inhouse_path.with_suffix(".csv.tmp")
    with inhouse_tmp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=INHOUSE_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(inhouse)
    inhouse_tmp.replace(inhouse_path)

    id_rows = []
    for player_id, info in catalog.items():
        if info["pass_nr"]:
            name = roster.canonical_name(info["pass_nr"]) or info["names"].most_common(1)[0][0]
        else:
            name = info["names"].most_common(1)[0][0]
        id_rows.append({"player_id": player_id, "pass_nr": info["pass_nr"], "name": name})
    id_rows.sort(key=lambda item: (item["player_id"][:1] == "l", item["name"].casefold(), item["player_id"]))
    with PLAYER_IDS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PLAYER_ID_FIELDS)
        writer.writeheader()
        writer.writerows(id_rows)

    passes = sum(1 for row in games for side in ("home", "away") if (row.get(f"{side}_player_id") or "").isdigit())
    seats = sum(1 for row in games for side in ("home", "away") if (row.get(f"{side}_player") or "").strip())
    inhouse_passes = sum(1 for row in inhouse if row["player_id"].isdigit()) + sum(
        1 for row in inhouse if row["opponent_id"].isdigit()
    )
    print(f"Wrote {GAMES_PATH} ({len(games)} games)")
    print(f"  player seats with a Pass-Nr.: {passes} / {seats}")
    print(f"Wrote {inhouse_path} ({len(inhouse)} club games)")
    print(f"  player seats with a Pass-Nr.: {inhouse_passes} / {len(inhouse) * 2}")
    print(f"Wrote {PLAYER_IDS_PATH} ({len(id_rows)} people)")


def cmd_report() -> None:
    if not ROSTER_PATH.exists():
        raise SystemExit(f"Missing {ROSTER_PATH}. Run: python identity.py fetch")
    with GAMES_PATH.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    extra = DATA / "inhouse.csv"
    if extra.exists():
        from elo import _inhouse_games

        rows.extend(_inhouse_games(extra))
    apply_identities(rows, identity_path=IDENTITY_PATH)
    print(f"Wrote {IDENTITY_PATH}")
    linked = Counter()
    for row in rows:
        for side in ("home", "away"):
            raw_pass = (row.get(f"{side}_pass") or "").strip()
            name = (row.get(f"{side}_player") or "").strip()
            if not name:
                continue
            linked["with pass" if raw_pass else "name only"] += 1
    print(dict(linked))
    print("Alexander Fischer identities:")
    with IDENTITY_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            if "fischer" in row["display_name"].casefold() and "alex" in row["display_name"].casefold():
                print(
                    f"  {row['display_name']}  pass={row['pass_nr'] or '—'}  "
                    f"games={row['games']}  clubs={row['clubs']}"
                )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cmd", choices=("fetch", "attach", "report"))
    args = parser.parse_args()
    if args.cmd == "fetch":
        cmd_fetch()
    elif args.cmd == "attach":
        attach_ids()
    else:
        cmd_report()


if __name__ == "__main__":
    main()
