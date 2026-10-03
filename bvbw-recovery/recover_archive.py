#!/usr/bin/env python3
"""Recover BillardArea Spielberichte from Wayback and emit games.csv rows.

Filters to players present in data/games.csv (+ inhouse) unless --all-players.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[1]
RECOVERY = Path(__file__).resolve().parent
WAYBACK = RECOVERY / "wayback"
RECOVERED = RECOVERY / "recovered"
EXTRACTED = RECOVERY / "extracted"
MATCHDAYS = RECOVERED / "matchdays"
PLANS = RECOVERED / "plans"

UA = "bvbw-recovery/1.0 (historical sports research)"
CTX = ssl.create_default_context()

GAMES_FIELDS = [
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
]


def load_db_players() -> set[str]:
    players: set[str] = set()
    for path in (ROOT / "data" / "games.csv", ROOT / "data" / "inhouse.csv"):
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            for row in csv.DictReader(f):
                for key in ("home_player", "away_player", "winner", "loser", "player", "opponent"):
                    name = (row.get(key) or "").strip()
                    if name:
                        players.add(name)
    return players


def norm_name(name: str) -> str:
    name = re.sub(r"\s+", " ", name or "").strip().casefold()
    # BillardArea sometimes doubles spaces or uses nbsp variants
    return name.replace("\u00a0", " ")


def build_player_index(players: set[str]) -> dict[str, str]:
    """Map normalized name -> canonical DB spelling."""
    return {norm_name(p): p for p in players}


def match_player(raw: str, index: dict[str, str]) -> str | None:
    """Spelling only. Person identity is Pass-Nr. plus name, in identity.py."""
    key = norm_name(raw)
    if key in index:
        return index[key]
    # tolerate missing middle particles / extra spaces already handled
    return None


def load_cdx_rows() -> list[list[str]]:
    rows: list[list[str]] = []
    for name in (
        "bvbw-billardarea.json",
        "bvbw-billardarea-2020-2022.json",
    ):
        path = WAYBACK / name
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if isinstance(data, list) and len(data) > 1:
            rows.extend(data[1:])
    return rows


def latest_by_pattern(pattern: str) -> dict[str, tuple[str, str]]:
    """Return key -> (timestamp, original_url) for latest HTML capture."""
    rx = re.compile(pattern)
    best: dict[str, tuple[str, str]] = {}
    for row in load_cdx_rows():
        ts, original, mimetype = row[0], row[1], row[2]
        if mimetype != "text/html":
            continue
        m = rx.search(original)
        if not m:
            continue
        key = m.group(0) if m.lastindex is None else m.group(1) if m.lastindex == 1 else "/".join(m.groups())
        if key not in best or ts > best[key][0]:
            best[key] = (ts, original)
    return best


def wayback_url(ts: str, original: str) -> str:
    return f"https://web.archive.org/web/{ts}id_/{original}"


def fetch(url: str, dest: Path, retries: int = 4) -> bool:
    if dest.exists() and dest.stat().st_size > 500:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as resp:
                data = resp.read()
            if len(data) > 500:
                dest.write_bytes(data)
                return True
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2 * (attempt + 1))
    return False


def download_plans(limit: int | None = None) -> tuple[int, int]:
    plans = latest_by_pattern(r"/cms_leagues/plan/(\d+)/(\d+)")
    ok = fail = 0
    items = sorted(plans.items())
    if limit is not None:
        items = items[:limit]
    for key, (ts, original) in items:
        # key is "seasonId/leagueId"
        safe = key.replace("/", "_")
        dest = PLANS / f"plan_{safe}.html"
        if fetch(wayback_url(ts, original), dest):
            ok += 1
        else:
            fail += 1
        time.sleep(0.5)
    return ok, fail


def download_matchdays(limit: int | None = None) -> tuple[int, int]:
    mds = latest_by_pattern(r"/cms_leagues/matchday/(\d+)")
    ok = fail = 0
    items = sorted(mds.items(), key=lambda kv: int(kv[0]))
    if limit is not None:
        items = items[:limit]
    for mid, (ts, original) in items:
        dest = MATCHDAYS / f"{mid}.html"
        if fetch(wayback_url(ts, original), dest):
            ok += 1
        else:
            fail += 1
        if (ok + fail) % 50 == 0:
            print(f"  matchdays progress ok={ok} fail={fail}", flush=True)
        time.sleep(0.45)
    return ok, fail


def map_discipline(raw: str) -> str:
    t = (raw or "").strip()
    low = t.casefold()
    if "14.1" in low or "14/1" in low or "endlos" in low:
        return "14/1e"
    if "8-ball" in low or "8 ball" in low:
        return "8-Ball"
    if "9-ball" in low or "9 ball" in low:
        return "9-Ball"
    if "10-ball" in low or "10 ball" in low:
        return "10-Ball"
    return t


def parse_league_season(text: str) -> tuple[str, str, str]:
    """Return season, league, staffel from BillardArea 'Liga (Saison)' line."""
    m = re.search(r"Liga \(Saison\)\s+(.+?)\s+Partie-Nr", text)
    if not m:
        return "", "", ""
    blob = m.group(1).strip()
    # e.g. "Bezirksliga Ost - Staffel 1 (2019/2020)"
    sm = re.search(r"\((\d{4}/\d{4})\)$", blob)
    season = sm.group(1) if sm else ""
    league_part = blob[: sm.start()].strip() if sm else blob
    # Split staffel-ish suffix
    staffel = ""
    for sep in (" - Staffel ", " Staffel ", " - "):
        if sep in league_part:
            league, staffel = league_part.split(sep, 1)
            staffel = staffel.strip()
            if sep.strip() == "-" and not staffel.lower().startswith(
                ("staffel", "ost", "west", "nord", "mitte", "1", "2")
            ):
                # maybe "Bezirksliga 2 - Ost"
                league_part = league_part  # keep as league
                league, staffel = league_part, ""
            else:
                league_part = league.strip()
            break
    # Prefer keeping full name in league if staffel empty
    league = league_part
    # Normalize seasons with 4-digit second year already
    return season, league, staffel


def parse_matchday(path: Path) -> list[dict[str, str]]:
    raw = path.read_text(errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True))

    season, league, staffel = parse_league_season(text)
    partie = re.search(r"Partie-Nr\.\s*(\d+)", text)
    datum = re.search(r"Datum\s+(\d{2}\.\d{2}\.\d{4})\s+(\d{1,2}:\d{2})", text)
    endstand = re.search(r"Endstand:\s*(\d+)\s*:\s*(\d+)", text)
    hm = re.search(r"Heim\s*:\s*(.+?)\s+Gast\s*:", text)
    gm = re.search(r"Gast\s*:\s*(.+?)\s+Spiel\s+1", text)
    home_team = hm.group(1).strip() if hm else ""
    away_team = gm.group(1).strip() if gm else ""
    match_score = ""
    if endstand:
        match_score = f"{endstand.group(1)}:{endstand.group(2)}"

    original = f"http://bvbw.billardarea.de/cms_leagues/matchday/{path.stem}"
    report_url = f"https://web.archive.org/web/{original}"  # placeholder; better with ts if known
    # Prefer archived id_ URL if we can find timestamp from filename sibling list — keep original path
    report_url = original

    rows: list[dict[str, str]] = []
    for table in soup.find_all("table"):
        trs = table.find_all("tr")
        i = 0
        while i < len(trs):
            cells = [c.get_text(" ", strip=True) for c in trs[i].find_all(["td", "th"])]
            if not cells:
                i += 1
                continue
            m = re.match(r"Spiel\s+(\d+)\s+(.+)", cells[0])
            if not m:
                i += 1
                continue
            game_no = m.group(1)
            discipline = map_discipline(m.group(2))
            home_player = cells[1] if len(cells) > 1 else ""
            away_player = cells[2] if len(cells) > 2 else ""

            home_score = away_score = ""
            sp_punkte = sp_aufn = sp_hs = sp_gd = ""
            j = i + 1
            while j < len(trs):
                scells = [c.get_text(" ", strip=True) for c in trs[j].find_all(["td", "th"])]
                if not any(scells):
                    j += 1
                    continue
                if scells and re.match(r"Spiel\s+\d+", scells[0] or ""):
                    break
                if len(scells) >= 2 and scells[0].isdigit() and scells[1].isdigit():
                    home_score, away_score = scells[0], scells[1]
                    break
                joined = " ".join(scells)
                if "Bälle" in joined:
                    balls = re.findall(r"Bälle:?\s*(\d+)", joined)
                    aufn = re.findall(r"Aufn\.:?\s*(\d+)", joined)
                    hs = re.findall(r"HS:\s*(\d+)", joined)
                    if len(balls) >= 2:
                        home_score, away_score = balls[0], balls[1]
                        sp_punkte = f"{balls[0]}:{balls[1]}"
                    if len(aufn) >= 2:
                        sp_aufn = f"{aufn[0]}/{aufn[1]}"
                    if len(hs) >= 2:
                        sp_hs = f"{hs[0]}/{hs[1]}"
                    break
                j += 1

            if not home_player or not away_player:
                i = max(j, i + 1)
                continue

            # Frame score / points
            if discipline == "14/1e" and home_score and away_score:
                frame_score = "1:0" if int(home_score) > int(away_score) else "0:1"
                if int(home_score) == int(away_score):
                    frame_score = "0:0"
            elif home_score and away_score:
                frame_score = f"{home_score}:{away_score}"
            else:
                frame_score = ""

            winner_side = ""
            home_points = away_points = ""
            if home_score.isdigit() and away_score.isdigit():
                hs_i, as_i = int(home_score), int(away_score)
                if hs_i > as_i:
                    winner_side, home_points, away_points = "home", "1", "0"
                elif as_i > hs_i:
                    winner_side, home_points, away_points = "away", "0", "1"
                else:
                    winner_side, home_points, away_points = "", "0", "0"

            winner = loser = ""
            if winner_side == "home":
                winner, loser = home_player, away_player
            elif winner_side == "away":
                winner, loser = away_player, home_player

            # Infer round from game number (1-4 Runde 1, 5-8 Runde 2) — BillardArea Pool Kombi
            gn = int(game_no)
            round_name = "Runde 1" if gn <= 4 else "Runde 2" if gn <= 8 else f"Spiel {gn}"

            rows.append(
                {
                    "season": season,
                    "league": league,
                    "staffel": staffel,
                    "spieltag": "",
                    "match_date": datum.group(1) if datum else "",
                    "match_time": f"{datum.group(2)} Uhr" if datum else "",
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_score": match_score,
                    "round": round_name,
                    "game_no": game_no,
                    "discipline": discipline,
                    "home_player": home_player,
                    "away_player": away_player,
                    "frame_score": frame_score,
                    "home_points": home_points,
                    "away_points": away_points,
                    "winner": winner,
                    "loser": loser,
                    "winner_side": winner_side,
                    "straight_pool_punkte": sp_punkte,
                    "straight_pool_aufnahmen": sp_aufn,
                    "straight_pool_hs": sp_hs,
                    "straight_pool_gd": sp_gd,
                    "report_url": report_url,
                    "_match_id": partie.group(1) if partie else path.stem,
                }
            )
            i = max(j, i + 1)
    return rows


POOL_DISCIPLINES = {"8-Ball", "9-Ball", "10-Ball", "14/1e"}


def extract_games(
    player_index: dict[str, str] | None,
    min_season: str = "2015/2016",
    max_season: str | None = "2020/2021",
    pool_only: bool = True,
) -> list[dict]:
    out: list[dict] = []
    files = sorted(MATCHDAYS.glob("*.html"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    for path in files:
        try:
            rows = parse_matchday(path)
        except Exception as exc:  # noqa: BLE001
            print(f"parse fail {path.name}: {exc}")
            continue
        for row in rows:
            season = row.get("season") or ""
            if season and season < min_season:
                continue
            if max_season and season and season > max_season:
                continue
            if pool_only and row.get("discipline") not in POOL_DISCIPLINES:
                continue
            hp = row["home_player"]
            ap = row["away_player"]
            if player_index is not None:
                ch = match_player(hp, player_index)
                ca = match_player(ap, player_index)
                if not ch and not ca:
                    continue
                # canonicalize names when matched
                if ch:
                    row["home_player"] = ch
                    if row["winner_side"] == "home":
                        row["winner"] = ch
                    elif row["winner_side"] == "away":
                        row["loser"] = ch
                if ca:
                    row["away_player"] = ca
                    if row["winner_side"] == "away":
                        row["winner"] = ca
                    elif row["winner_side"] == "home":
                        row["loser"] = ca
            clean = {k: row.get(k, "") for k in GAMES_FIELDS}
            out.append(clean)
    return out


def merge_into_games(new_rows: list[dict], games_path: Path) -> tuple[int, int]:
    """Prepend recovered historical rows; skip duplicates by report_url+game_no+players."""
    existing: list[dict] = []
    with games_path.open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or GAMES_FIELDS
        for row in reader:
            existing.append({k: row.get(k, "") for k in fieldnames})

    def key(r: dict) -> tuple:
        return (
            r.get("report_url", ""),
            r.get("game_no", ""),
            norm_name(r.get("home_player", "")),
            norm_name(r.get("away_player", "")),
            r.get("match_date", ""),
            r.get("discipline", ""),
        )

    seen = {key(r) for r in existing}
    added = 0
    merged_new = []
    for r in new_rows:
        k = key(r)
        if k in seen:
            continue
        seen.add(k)
        merged_new.append(r)
        added += 1

    # Historical first (ascending season/date), then existing modern rows
    def sort_key(r: dict):
        return (r.get("season", ""), r.get("match_date", ""), r.get("report_url", ""), r.get("game_no", ""))

    all_rows = sorted(merged_new, key=sort_key) + existing
    with games_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GAMES_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(all_rows)
    return added, len(all_rows)


def cmd_download(args: argparse.Namespace) -> None:
    MATCHDAYS.mkdir(parents=True, exist_ok=True)
    PLANS.mkdir(parents=True, exist_ok=True)
    if args.plans:
        print("Downloading plans...")
        ok, fail = download_plans(args.limit)
        print(f"plans ok={ok} fail={fail}")
    if args.matchdays:
        print("Downloading matchdays...")
        ok, fail = download_matchdays(args.limit)
        print(f"matchdays ok={ok} fail={fail}")


def cmd_extract(args: argparse.Namespace) -> None:
    EXTRACTED.mkdir(parents=True, exist_ok=True)
    players = load_db_players()
    index = None if args.all_players else build_player_index(players)
    print(f"DB players: {len(players)}; filter={'off' if args.all_players else 'on'}")
    rows = extract_games(
        index,
        min_season=args.min_season,
        max_season=None if args.max_season in ("", "none", "None") else args.max_season,
        pool_only=not args.include_non_pool,
    )
    out = EXTRACTED / "recovered_games.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=GAMES_FIELDS)
        w.writeheader()
        w.writerows(rows)
    seasons = Counter(r["season"] for r in rows)
    print(f"Wrote {len(rows)} rows -> {out}")
    print("By season:", dict(sorted(seasons.items())))
    matched_players = set()
    for r in rows:
        matched_players.add(r["home_player"])
        matched_players.add(r["away_player"])
    if index is not None:
        hit = {p for p in matched_players if norm_name(p) in index}
        print(f"DB players with ≥1 recovered game: {len(hit)}")


def cmd_merge(args: argparse.Namespace) -> None:
    src = EXTRACTED / "recovered_games.csv"
    rows = list(csv.DictReader(src.open(encoding="utf-8")))
    games_path = ROOT / "data" / "games.csv"
    backup = ROOT / "data" / "games.pre_archive_merge.csv"
    if not backup.exists():
        backup.write_bytes(games_path.read_bytes())
        print(f"Backup -> {backup}")
    added, total = merge_into_games(rows, games_path)
    print(f"Merged +{added} rows; games.csv now {total} rows")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("download")
    d.add_argument("--plans", action="store_true")
    d.add_argument("--matchdays", action="store_true")
    d.add_argument("--limit", type=int, default=None)
    d.set_defaults(func=cmd_download)

    e = sub.add_parser("extract")
    e.add_argument("--all-players", action="store_true")
    e.add_argument("--min-season", default="2015/2016")
    e.add_argument("--max-season", default="2020/2021",
                   help="Skip seasons already covered by live billard-bvbw.de scrape")
    e.add_argument("--include-non-pool", action="store_true")
    e.set_defaults(func=cmd_extract)

    m = sub.add_parser("merge")
    m.set_defaults(func=cmd_merge)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
