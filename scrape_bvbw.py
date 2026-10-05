#!/usr/bin/env python3
"""Step-through Selenium scraper for BVBW pool match reports.

Each click of Next (or Enter in --cli mode) performs one action:
open a page, click a league/staffel/score, or save games to CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
import re
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

BASE_URL = "https://billard-bvbw.de/"
START_URL = urljoin(BASE_URL, "sb_spielplan.php")
TARGET_LEAGUES = (
    # Names are the exact Pool links on the Spielplan. A season that does
    # not have one of them is skipped.
    "1. Bundesliga",
    "2. Bundesliga Süd",
    "Regionalliga Mitte",
    "Regionalliga Süd",
    "Oberliga",
    "Pokal-Mannschaft",
    "Relegation Oberliga",
    "Relagation Verbandsliga M-O",
    "Relagation Verbandsliga N_W",
    "Kreisliga A",
    "Bezirksliga",
    "Landesliga",
    "Verbandsliga Nord-West",
    "Verbandsliga Mitte-Ost",
    # 2026/2027 renamed the two Verbandsliga divisions.
    "Verbandsliga Nord",
    "Verbandsliga Süd",
)
SCORE_RE = re.compile(r"^\d+:\d+$")
SKIP_SCORES = {"0:0", ":"}
CSV_FIELDS = [
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


@dataclass
class MatchRef:
    spieltag: str
    match_date: str
    match_time: str
    home_team: str
    away_team: str
    score: str
    url: str


@dataclass
class ScraperState:
    league: str = ""
    staffel: str = ""
    season: str = ""
    seen_reports: set[str] = field(default_factory=set)
    saved_games: int = 0
    saved_matches: int = 0


@dataclass
class Progress:
    season_index: int = 0
    season_total: int = 1
    league_index: int = 0
    league_total: int = 1
    staffel_index: int = 0
    staffel_total: int = 0
    match_index: int = 0
    match_total: int = 0


class StepScraper:
    def __init__(
        self,
        csv_path: Path,
        min_wait: float,
        max_wait: float,
        leagues: tuple[str, ...] = TARGET_LEAGUES,
        seasons: tuple[str, ...] = (),
    ) -> None:
        self.csv_path = csv_path
        self.min_wait = min_wait
        self.max_wait = max_wait
        self.leagues = leagues
        self.seasons = seasons
        self.driver: webdriver.Chrome | None = None
        self.state = ScraperState()
        self.state.seen_reports = load_seen_report_urls(csv_path)
        self.progress = Progress(
            season_total=len(seasons) or 1,
            league_total=len(leagues) or 1,
        )
        self.quiet = False
        self.queue: list[tuple[str, Callable[[], None]]] = []
        self.last_status = "Ready. Click Next to start Chrome."
        self.done = False

    def enqueue(self, label: str, action: Callable[[], None]) -> None:
        self.queue.append((label, action))

    def enqueue_front(self, items: list[tuple[str, Callable[[], None]]]) -> None:
        self.queue = items + self.queue

    def remaining(self) -> int:
        return len(self.queue)

    def current_label(self) -> str:
        if self.done:
            return "Finished"
        if not self.queue:
            return "No more steps"
        return self.queue[0][0]

    def step(self) -> str:
        if not self.queue:
            self.done = True
            self.last_status = "All planned steps are done."
            return self.last_status
        label, action = self.queue.pop(0)
        if not self.quiet:
            print(f"\n>>> {label}", flush=True)
        action()
        self.last_status = f"Done: {label}"
        if not self.quiet:
            print(self.last_status, flush=True)
        if not self.queue:
            self.done = True
            self.last_status += "  Nothing left in the queue."
        return self.last_status

    def human_wait(self, extra: float = 0.0) -> None:
        delay = random.uniform(self.min_wait, self.max_wait) + extra
        if not self.quiet:
            print(f"    waiting {delay:.1f}s")
        time.sleep(delay)

    def _progress_line(self) -> str:
        season = self.state.season or "season?"
        league = self.state.league or "liga?"
        staffel = self.state.staffel or "-"
        p = self.progress
        done = p.match_index
        total = p.match_total
        width = 22
        filled = int(width * done / total) if total else 0
        bar = "█" * filled + "░" * (width - filled)
        pct = int(100 * done / total) if total else 0
        return (
            f"{season} ({p.season_index}/{p.season_total})  "
            f"{league} ({p.league_index}/{p.league_total})  "
            f"{staffel} ({p.staffel_index}/{p.staffel_total})  "
            f"{bar} {done}/{total} {pct:3d}%  "
            f"saved {self.state.saved_matches} matches / {self.state.saved_games} games"
        )

    def show_progress(self, newline: bool = False) -> None:
        line = self._progress_line()
        if newline:
            print(f"\n{line}", flush=True)
        else:
            print(f"\r{line:<140}", end="", flush=True)

    def wait_ready(self) -> None:
        assert self.driver is not None
        WebDriverWait(self.driver, 20).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        self.human_wait()

    def start_browser(self) -> None:
        options = Options()
        options.add_argument("--window-size=1400,1000")
        options.add_argument("--disable-blink-features=AutomationControlled")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")
        options.add_argument("--disable-dev-shm-usage")
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option("useAutomationExtension", False)
        profile = Path(__file__).resolve().parent / ".chrome-profile"
        profile.mkdir(exist_ok=True)
        options.add_argument(f"--user-data-dir={profile}")
        self.driver = webdriver.Chrome(options=options)
        self.driver.execute_cdp_cmd(
            "Page.addScriptToEvaluateOnNewDocument",
            {
                "source": "Object.defineProperty(navigator, 'webdriver', {get: () => undefined});"
            },
        )
        self.enqueue("Open the Spielplan start page", self.open_start_page)

    def open_start_page(self) -> None:
        assert self.driver is not None
        self.driver.get(START_URL)
        self.wait_ready()
        if self.seasons:
            items = [
                (f"Select season {season}", lambda s=season: self.select_season(s))
                for season in self.seasons
            ]
            self.enqueue_front(items)
            return
        self.state.season = self._read_season()
        self.progress.season_index = 1
        self._enqueue_leagues()

    def select_season(self, season: str) -> None:
        assert self.driver is not None
        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "select[name='s']"))
        )
        self.driver.execute_script(
            """
            const season = arguments[0];
            const hidden = document.querySelector("input[name='s']");
            const select = document.querySelector("select[name='s']");
            if (hidden) hidden.value = season;
            if (select) select.value = season;
            document.forms[0].action = 'sb_spielplan.php';
            document.forms[0].submit();
            """,
            season,
        )
        WebDriverWait(self.driver, 20).until(
            lambda driver: self._page_season(driver) == season
        )
        self.wait_ready()
        self.state.season = season
        self.progress.season_index += 1
        self.progress.league_index = 0
        self.progress.staffel_index = 0
        self.progress.staffel_total = 0
        self.progress.match_index = 0
        self.progress.match_total = 0
        self.state.league = ""
        self.state.staffel = ""
        self.show_progress(newline=True)
        self._enqueue_leagues()

    def _page_season(self, driver) -> str:
        try:
            heading = driver.find_element(By.CSS_SELECTOR, "h2#jumpy")
            match = re.search(r"(\d{4}/\d{4})", heading.text)
            return match.group(1) if match else ""
        except Exception:
            return ""

    def _enqueue_leagues(self) -> None:
        season = self.state.season
        items = [
            (f"Click league: {name} ({season})", lambda n=name: self.click_league(n))
            for name in self.leagues
        ]
        self.enqueue_front(items)

    def _read_season(self) -> str:
        assert self.driver is not None
        return self._page_season(self.driver) or "2026/2027"

    def click_league(self, name: str) -> None:
        assert self.driver is not None
        matches = self.driver.find_elements(
            By.XPATH,
            f"//a[contains(@class,'cc_bluelink') and normalize-space()='{name}']",
        )
        if not matches:
            print(f"    no '{name}' in {self.state.season}, skipping")
            return
        link = matches[0]
        self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", link)
        time.sleep(0.4)
        link.click()
        self.wait_ready()
        self.state.league = name
        self.progress.league_index += 1
        self.progress.staffel_index = 0
        self.progress.staffel_total = 0
        self.progress.match_index = 0
        self.progress.match_total = 0
        self.state.staffel = ""
        self.show_progress(newline=True)
        self.enqueue_front(
            [(f"Collect all staffeln in {name}", lambda n=name: self.collect_staffeln(n))]
        )

    def collect_staffeln(self, league: str) -> None:
        assert self.driver is not None
        # The Staffel strip is Ost/West/Mitte/Nord/Süd and numbered groups.
        # Sport tabs live in a different strip, so every link here is a region.
        tabs = self.driver.find_elements(
            By.XPATH,
            "//td[normalize-space()='Staffel']/following-sibling::td[1]"
            "//ul[contains(@class,'tabstrip')]//a",
        )
        staffeln: list[tuple[str, str]] = []
        seen: set[str] = set()
        for tab in tabs:
            title = " ".join((tab.get_attribute("title") or tab.text or "").split())
            href = tab.get_attribute("href") or ""
            if not title or title in seen:
                continue
            seen.add(title)
            staffeln.append((title, href))
        if not staffeln:
            # Verbandsliga has one group and no Staffel strip; the Spielplan is on this page.
            self.state.staffel = league
            self.progress.staffel_total = 1
            self.progress.staffel_index = 1
            if not self.quiet:
                print(f"    no staffel tabs; reading the Spielplan for {league}")
            self.enqueue_front(
                [(f"Open Spielplan tab for {league}", self.open_spielplan_tab)]
            )
            return
        self.progress.staffel_total = len(staffeln)
        self.progress.staffel_index = 0
        if not self.quiet:
            print(f"    found staffeln: {[name for name, _ in staffeln]}")
        items = [
            (f"Open {league} / {name}", lambda n=name, h=href: self.open_staffel(n, h))
            for name, href in staffeln
        ]
        self.enqueue_front(items)

    def open_staffel(self, name: str, href: str = "") -> None:
        assert self.driver is not None
        if href:
            self.driver.get(href)
        else:
            xpath = (
                "//td[normalize-space()='Staffel']/following-sibling::td[1]"
                f"//a[normalize-space(@title)='{name}' or normalize-space()='{name}']"
            )
            link = WebDriverWait(self.driver, 15).until(
                EC.element_to_be_clickable((By.XPATH, xpath))
            )
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", link)
            time.sleep(0.4)
            link.click()
        self.wait_ready()
        self.state.staffel = name
        self.progress.staffel_index += 1
        self.progress.match_index = 0
        self.progress.match_total = 0
        self.show_progress(newline=True)
        self.enqueue_front(
            [
                (
                    f"Open Spielplan tab for {self.state.league} / {name}",
                    self.open_spielplan_tab,
                )
            ]
        )

    def open_spielplan_tab(self) -> None:
        assert self.driver is not None
        tab = WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//ul[contains(@class,'tabstrip')]//a[@title='Spielplan']")
            )
        )
        parent_class = tab.find_element(By.XPATH, "..").get_attribute("class") or ""
        if "aktiverReiter" not in parent_class:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", tab)
            time.sleep(0.3)
            tab.click()
            self.wait_ready()
        else:
            print("    Spielplan tab is already active")
        self.enqueue_front(
            [
                (
                    f"Collect finished matches in {self.state.league} / {self.state.staffel}",
                    self.collect_matches,
                )
            ]
        )

    def _spielplan_table(self):
        assert self.driver is not None
        return self.driver.find_element(
            By.XPATH,
            "//ul[contains(@class,'tabstrip')][.//a[@title='Spielplan']]"
            "/following-sibling::table[1]",
        )

    def collect_matches(self) -> None:
        table = self._spielplan_table()
        matches: list[MatchRef] = []
        for row in table.find_elements(By.CSS_SELECTOR, "tr"):
            match = self._match_from_row(row)
            if match is None or match.url in self.state.seen_reports:
                continue
            matches.append(match)
        self.progress.match_total = len(matches)
        self.progress.match_index = 0
        self.show_progress()
        if not self.quiet:
            print(f"    finished matches: {len(matches)}")
            for match in matches:
                print(f"      {match.match_date}  {match.home_team} {match.score} {match.away_team}")
        items: list[tuple[str, Callable[[], None]]] = []
        for match in matches:
            items.append(
                (
                    f"Open report {match.home_team} {match.score} {match.away_team}",
                    lambda m=match: self.open_report(m),
                )
            )
            items.append(
                (
                    f"Parse and save {match.home_team} vs {match.away_team}",
                    lambda m=match: self.parse_and_save(m),
                )
            )
        self.enqueue_front(items)

    def _match_from_row(self, row) -> MatchRef | None:
        cells = row.find_elements(By.XPATH, "./td")
        score_link = None
        score_idx = -1
        for index, cell in enumerate(cells):
            for anchor in cell.find_elements(By.CSS_SELECTOR, "a.cc_bluelink"):
                text = (anchor.text or "").strip()
                if SCORE_RE.match(text) and text not in SKIP_SCORES:
                    score_link = anchor
                    score_idx = index
                    break
            if score_link is not None:
                break
        if score_link is None:
            return None
        href = score_link.get_attribute("href") or ""
        if not href:
            return None
        texts = [cell.text.strip() for cell in cells]
        spieltag = next((text for text in texts if re.fullmatch(r"\d+", text)), "")
        match_date = ""
        match_time = ""
        for text in texts:
            date_match = re.search(r"(\d{2}\.\d{2}\.\d{4})", text)
            if date_match:
                match_date = date_match.group(1)
                leftover = text.replace(match_date, "").strip()
                match_time = leftover.splitlines()[0].strip() if leftover else ""
                break
        before = [
            text
            for text in texts[:score_idx]
            if text and text != spieltag and match_date not in text
        ]
        after = [text for text in texts[score_idx + 1 :] if text]
        return MatchRef(
            spieltag=spieltag,
            match_date=match_date,
            match_time=match_time,
            home_team=before[-1] if before else "",
            away_team=after[0] if after else "",
            score=score_link.text.strip(),
            url=href,
        )

    def open_report(self, match: MatchRef) -> None:
        assert self.driver is not None
        self.driver.get(match.url)
        self.wait_ready()
        WebDriverWait(self.driver, 15).until(
            EC.presence_of_element_located(
                (By.XPATH, "//th[contains(normalize-space(), 'SPIELBERICHT')]")
            )
        )

    def parse_and_save(self, match: MatchRef) -> None:
        assert self.driver is not None
        rows = parse_spielbericht(
            self.driver.page_source,
            season=self.state.season,
            league=self.state.league,
            staffel=self.state.staffel,
            match=match,
            report_url=self.driver.current_url,
        )
        append_csv(self.csv_path, rows)
        self.state.seen_reports.add(match.url)
        self.state.saved_matches += 1
        self.state.saved_games += len(rows)
        self.progress.match_index += 1
        self.show_progress()
        if not self.quiet:
            print(f"    saved {len(rows)} games -> {self.csv_path}")
            print(
                f"    totals: {self.state.saved_matches} matches, "
                f"{self.state.saved_games} games"
            )

    def go_back_to_spielplan(self) -> None:
        assert self.driver is not None
        self.driver.back()
        self.wait_ready()

    def close(self) -> None:
        if self.driver is not None:
            self.driver.quit()
            self.driver = None


def parse_spielbericht(
    html: str,
    season: str,
    league: str,
    staffel: str,
    match: MatchRef,
    report_url: str,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "lxml")
    header = soup.find("th", string=re.compile(r"SPIELBERICHT", re.I))
    if header is None:
        raise RuntimeError("Could not find SPIELBERICHT table")
    table = header.find_parent("table")
    if table is None:
        raise RuntimeError("SPIELBERICHT table is missing")

    home_team, away_team, match_date, match_score = _header_meta(table, match)
    current_round = ""
    games: list[dict[str, str]] = []

    for tr in table.find_all("tr"):
        ths = [th.get_text(" ", strip=True) for th in tr.find_all("th")]
        if len(ths) >= 2 and ths[1].startswith("Runde"):
            current_round = ths[1]
            continue

        home_cell = tr.select_one("td.p1")
        away_cell = tr.select_one("td.p2")
        if home_cell and away_cell:
            tds = tr.find_all("td", recursive=False)
            frame_score = ""
            for td in tds:
                strong = td.find("strong")
                if strong and SCORE_RE.match(strong.get_text(strip=True)):
                    frame_score = strong.get_text(strip=True)
                    break
            points = tds[-1].get_text(strip=True) if tds else ""
            home_points, away_points = _split_score(points)
            home_player = home_cell.get_text(" ", strip=True)
            away_player = away_cell.get_text(" ", strip=True)
            winner_side = _winner_side(home_points, away_points)
            winner, loser = _winner_loser_names(home_player, away_player, winner_side)
            games.append(
                {
                    "season": season,
                    "league": league,
                    "staffel": staffel,
                    "spieltag": match.spieltag,
                    "match_date": match_date or match.match_date,
                    "match_time": match.match_time,
                    "home_team": home_team,
                    "away_team": away_team,
                    "match_score": match_score or match.score,
                    "round": current_round,
                    "game_no": tds[0].get_text(strip=True) if tds else "",
                    "discipline": tds[1].get_text(strip=True) if len(tds) > 1 else "",
                    "home_player": home_player,
                    "away_player": away_player,
                    "frame_score": frame_score,
                    "home_points": home_points,
                    "away_points": away_points,
                    "winner": winner,
                    "loser": loser,
                    "winner_side": winner_side,
                    "straight_pool_punkte": "",
                    "straight_pool_aufnahmen": "",
                    "straight_pool_hs": "",
                    "straight_pool_gd": "",
                    "report_url": report_url,
                }
            )
            continue

        text = tr.get_text(" ", strip=True)
        if games and text.startswith("Punkte:"):
            games[-1].update(_straight_pool_stats(text))

    return games


def _header_meta(table, match: MatchRef) -> tuple[str, str, str, str]:
    home = match.home_team
    away = match.away_team
    date = match.match_date
    score = match.score
    header_row = None
    for tr in table.find_all("tr"):
        labels = [th.get_text(" ", strip=True) for th in tr.find_all("th")]
        if labels and any("Heim-Mannschaft" in label for label in labels):
            header_row = tr.find_next_sibling("tr")
            break
    if header_row is not None:
        cells = header_row.find_all("td", recursive=False)
        # Partie-Nr | Heim | : | Gast | Datum
        if len(cells) >= 5:
            home_text = cells[1].get_text(" ", strip=True)
            away_text = cells[3].get_text(" ", strip=True)
            date_text = cells[4].get_text(" ", strip=True)
            if home_text:
                home = home_text
            if away_text:
                away = away_text
            if date_text:
                date = date_text
    endstand = table.find(string=re.compile(r"Endstand"))
    if endstand is not None:
        row = endstand.find_parent("tr")
        if row is not None:
            strongs = [s.get_text(strip=True) for s in row.find_all("strong")]
            scores = [s for s in strongs if SCORE_RE.match(s)]
            if scores:
                score = scores[-1]
    return home, away, date, score


def _split_score(score: str) -> tuple[str, str]:
    parts = score.split(":")
    if len(parts) != 2:
        return "", ""
    return parts[0].strip(), parts[1].strip()


def _winner_side(home_points: str, away_points: str) -> str:
    try:
        home = float(home_points)
        away = float(away_points)
    except ValueError:
        return ""
    if home > away:
        return "home"
    if away > home:
        return "away"
    return "draw"


def _winner_loser_names(
    home_player: str, away_player: str, winner_side: str
) -> tuple[str, str]:
    if winner_side == "home":
        return home_player, away_player
    if winner_side == "away":
        return away_player, home_player
    return "", ""


def add_winner_loser_columns(path: Path) -> int:
    """Rewrite an existing scrape CSV so winner/loser are player names."""
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        home_player = row.get("home_player", "").strip()
        away_player = row.get("away_player", "").strip()
        side = row.get("winner_side", "").strip() or row.get("winner", "").strip()
        if side not in {"home", "away", "draw"}:
            side = _winner_side(row.get("home_points", ""), row.get("away_points", ""))
        winner, loser = _winner_loser_names(home_player, away_player, side)
        row["winner_side"] = side
        row["winner"] = winner
        row["loser"] = loser
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _straight_pool_stats(text: str) -> dict[str, str]:
    def grab(label: str) -> str:
        match = re.search(rf"{label}:\s*(\S+)", text)
        return match.group(1) if match else ""

    return {
        "straight_pool_punkte": grab("Punkte"),
        "straight_pool_aufnahmen": grab(r"Aufn\."),
        "straight_pool_hs": grab("HS"),
        "straight_pool_gd": grab("GD"),
    }


def load_seen_report_urls(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return {
            row["report_url"]
            for row in csv.DictReader(handle)
            if row.get("report_url")
        }


def append_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def run_gui(scraper: StepScraper) -> None:
    import tkinter as tk

    root = tk.Tk()
    root.title("BVBW scraper")
    root.attributes("-topmost", True)
    root.geometry("640x260+40+40")

    status = tk.StringVar(value=scraper.last_status)
    nxt = tk.StringVar(value=f"Next: {scraper.current_label()}")
    counts = tk.StringVar(value="Saved: 0 matches / 0 games")

    def refresh() -> None:
        nxt.set(f"Next: {scraper.current_label()}  ({scraper.remaining()} left)")
        counts.set(
            f"Saved: {scraper.state.saved_matches} matches / "
            f"{scraper.state.saved_games} games"
        )

    def on_next() -> None:
        try:
            status.set(scraper.step())
        except Exception as exc:  # keep the window open so you can inspect
            status.set(f"Error: {exc}")
            print(f"ERROR: {exc}")
            traceback.print_exc()
        finally:
            refresh()
            if scraper.done:
                next_btn.config(state="disabled")

    def on_close() -> None:
        scraper.close()
        root.destroy()

    tk.Label(root, textvariable=nxt, wraplength=600, justify="left", font=("Sans", 12)).pack(
        anchor="w", padx=12, pady=(12, 6)
    )
    tk.Label(root, textvariable=status, wraplength=600, justify="left").pack(
        anchor="w", padx=12, pady=6
    )
    tk.Label(root, textvariable=counts, justify="left").pack(anchor="w", padx=12, pady=6)
    next_btn = tk.Button(root, text="Next step", width=16, command=on_next)
    next_btn.pack(side="left", padx=12, pady=16)
    tk.Button(root, text="Quit", width=10, command=on_close).pack(side="left", padx=8, pady=16)
    root.protocol("WM_DELETE_WINDOW", on_close)
    refresh()
    root.mainloop()


def run_cli(scraper: StepScraper) -> None:
    scraper.enqueue("Start Chrome", scraper.start_browser)
    print("Press Enter for each step. Type q to quit.")
    while True:
        print(f"\nNext: {scraper.current_label()}  ({scraper.remaining()} left)")
        answer = input("Enter = next step, q = quit > ").strip().lower()
        if answer in {"q", "quit"}:
            break
        scraper.step()
        if scraper.done:
            print(scraper.last_status)
            break
    scraper.close()


def run_auto(scraper: StepScraper) -> None:
    scraper.quiet = True
    scraper.enqueue("Start Chrome", scraper.start_browser)
    seasons = ", ".join(scraper.seasons) or "current page season"
    leagues = ", ".join(scraper.leagues)
    print(f"Auto-run started. Writing {scraper.csv_path}")
    print(f"Seasons: {seasons}")
    print(f"Leagues: {leagues}")
    print("Staffeln: every region on the Staffel strip (Ost, West, Mitte, Nord, Süd, …)")
    print(f"Skipping {len(scraper.state.seen_reports)} report URLs already in the CSV")
    try:
        while scraper.queue:
            label = scraper.current_label()
            try:
                scraper.step()
            except Exception as exc:
                print(f"\nERROR during '{label}': {exc}")
                traceback.print_exc()
                _recover_after_error(scraper)
        print(
            f"\nFinished. {scraper.state.saved_matches} matches, "
            f"{scraper.state.saved_games} games -> {scraper.csv_path}"
        )
    finally:
        scraper.close()


def _recover_after_error(scraper: StepScraper) -> None:
    driver = scraper.driver
    if driver is None:
        return
    try:
        url = driver.current_url or ""
        if "spielbericht" in url:
            print("    recovering: going back to Spielplan")
            scraper.go_back_to_spielplan()
    except Exception as exc:
        print(f"    recovery failed: {exc}")


def test_parse(html_path: Path) -> None:
    html = html_path.read_text(encoding="utf-8", errors="replace")
    match = MatchRef(
        spieltag="1",
        match_date="20.09.2026",
        match_time="10:30 Uhr",
        home_team="BSF Kurpfalz 2",
        away_team="PBV Schwetzingen 6",
        score="2:6",
        url="local",
    )
    rows = parse_spielbericht(
        html,
        season="2026/2027",
        league="Bezirksliga",
        staffel="West 1",
        match=match,
        report_url="local",
    )
    for row in rows:
        print(
            f"{row['game_no']:>2} {row['round']:<8} {row['discipline']:<8} "
            f"{row['home_player']} {row['frame_score']} {row['away_player']} "
            f"pts {row['home_points']}:{row['away_points']}"
        )
    print(f"parsed {len(rows)} games")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        default="data/games.csv",
        type=Path,
        help="CSV output path",
    )
    parser.add_argument("--min-wait", type=float, default=0.4)
    parser.add_argument("--max-wait", type=float, default=0.8)
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Use Enter in the terminal instead of the Next button",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Run every step without clicking and write the CSV",
    )
    parser.add_argument(
        "--leagues",
        nargs="+",
        default=list(TARGET_LEAGUES),
        help="Leagues to scrape. Default includes Verbandsliga Nord-West and Mitte-Ost",
    )
    parser.add_argument(
        "--seasons",
        nargs="+",
        default=[],
        help="Seasons to scrape, e.g. 2025/2026 2024/2025. Default: season already on the page",
    )
    parser.add_argument(
        "--test-parse",
        type=Path,
        help="Parse a saved Spielbericht HTML file and print games",
    )
    parser.add_argument(
        "--postprocess",
        action="store_true",
        help="Add winner/loser player columns to an existing CSV",
    )
    args = parser.parse_args()

    if args.test_parse:
        test_parse(args.test_parse)
        return
    if args.postprocess:
        count = add_winner_loser_columns(args.csv)
        print(f"Updated {count} rows in {args.csv}")
        return

    scraper = StepScraper(
        args.csv,
        args.min_wait,
        args.max_wait,
        leagues=tuple(args.leagues),
        seasons=tuple(args.seasons),
    )
    if args.auto:
        run_auto(scraper)
        return
    if args.cli:
        run_cli(scraper)
        return
    scraper.enqueue("Start Chrome", scraper.start_browser)
    run_gui(scraper)


if __name__ == "__main__":
    main()
