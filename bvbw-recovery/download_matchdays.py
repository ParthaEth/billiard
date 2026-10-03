#!/usr/bin/env python3
"""Resumable Wayback download of all BillardArea matchday pages."""

from __future__ import annotations

import json
import os
import re
import ssl
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

WAYBACK = Path(__file__).parent / "wayback"
OUT = Path(__file__).parent / "recovered" / "matchdays"
OUT.mkdir(parents=True, exist_ok=True)
CTX = ssl.create_default_context()
lock = threading.Lock()
stats = {"ok": 0, "fail": 0, "skip": 0}


def load_best() -> dict[str, list]:
    rows: list = []
    for name in ("bvbw-billardarea.json", "bvbw-billardarea-2020-2022.json"):
        path = WAYBACK / name
        if path.exists():
            data = json.loads(path.read_text())
            if isinstance(data, list) and len(data) > 1:
                rows.extend(data[1:])
    best: dict[str, list] = {}
    for row in rows:
        if row[2] != "text/html":
            continue
        match = re.search(r"/cms_leagues/matchday/(\d+)", row[1])
        if not match:
            continue
        mid = match.group(1)
        if mid not in best or row[0] > best[mid][0]:
            best[mid] = row
    return best


def fetch_one(mid: str, row: list) -> None:
    dest = OUT / f"{mid}.html"
    if dest.exists() and dest.stat().st_size > 500:
        with lock:
            stats["skip"] += 1
        return
    url = f"https://web.archive.org/web/{row[0]}id_/{row[1]}"
    for attempt in range(5):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "bvbw-recovery/1.0"})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as resp:
                data = resp.read()
            if len(data) > 500 and b"Spielergebnisse" in data:
                dest.write_bytes(data)
                with lock:
                    stats["ok"] += 1
                time.sleep(0.8)
                return
        except Exception:
            time.sleep(3 * (attempt + 1))
    with lock:
        stats["fail"] += 1


def main() -> None:
    best = load_best()
    todo = [
        (mid, row)
        for mid, row in best.items()
        if not ((OUT / f"{mid}.html").exists() and (OUT / f"{mid}.html").stat().st_size > 500)
    ]
    todo.sort(key=lambda item: -int(item[0]))
    workers = int(os.environ.get("WORKERS", "3"))
    print(f"to download {len(todo)} / {len(best)}  workers={workers}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch_one, mid, row) for mid, row in todo]
        done = 0
        for future in as_completed(futures):
            future.result()
            done += 1
            if done % 50 == 0:
                with lock:
                    print(f"progress {done}/{len(todo)} {stats}", flush=True)
    print("DONE", stats, flush=True)
    (Path(__file__).parent / "download_parallel.log").write_text(json.dumps(stats))


if __name__ == "__main__":
    main()
