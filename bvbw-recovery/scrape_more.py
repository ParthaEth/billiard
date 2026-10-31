#!/usr/bin/env python3
"""Download BillardArea matchdays that Common Crawl captured and we never saved.

Already-saved matchday HTML is skipped. This script does not write games.csv.
"""

from __future__ import annotations

import gzip
import json
import random
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CC_DIR = ROOT / "commoncrawl"
OUT = ROOT / "recovered" / "matchdays"
UA = "bvbw-recovery/1.0 (historical sports research)"
CTX = ssl.create_default_context()
ARCHIVE_MIN_WAIT = 4.0
ARCHIVE_MAX_WAIT = 9.0
MATCHDAY_RE = re.compile(r"matchday/(\d+)")


def archive_pause(lo: float = ARCHIVE_MIN_WAIT, hi: float = ARCHIVE_MAX_WAIT) -> None:
    delay = random.uniform(lo, hi)
    print(f"    waiting {delay:.1f}s", flush=True)
    time.sleep(delay)


def load_captures() -> dict[str, dict]:
    """Best Common Crawl record for each matchday id not yet on disk."""
    best: dict[str, dict] = {}
    for path in sorted(CC_DIR.glob("*.jsonl")):
        for line in path.read_text(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            match = MATCHDAY_RE.search(row.get("url") or "")
            if not match or str(row.get("status")) not in {"200", "200"}:
                continue
            if row.get("mime") not in {None, "", "text/html"}:
                continue
            mid = match.group(1)
            dest = OUT / f"{mid}.html"
            if dest.exists() and dest.stat().st_size > 500:
                continue
            if not row.get("filename") or row.get("offset") is None or row.get("length") is None:
                continue
            current = best.get(mid)
            if current is None or int(row["length"]) > int(current["length"]):
                best[mid] = row
    return best


def html_from_warc(raw: bytes) -> bytes:
    sep = b"\r\n\r\n"
    first = raw.find(sep)
    if first < 0:
        return b""
    rest = raw[first + len(sep) :]
    second = rest.find(sep)
    if second < 0:
        return rest
    return rest[second + len(sep) :]


def fetch_capture(row: dict, retries: int = 4) -> bytes | None:
    start = int(row["offset"])
    length = int(row["length"])
    url = "https://data.commoncrawl.org/" + row["filename"]
    for attempt in range(retries):
        archive_pause()
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": UA,
                    "Range": f"bytes={start}-{start + length - 1}",
                },
            )
            with urllib.request.urlopen(req, timeout=120, context=CTX) as resp:
                blob = resp.read()
            html = html_from_warc(gzip.decompress(blob))
            if b"Spielergebnisse" in html and len(html) > 500:
                return html
        except (urllib.error.URLError, TimeoutError, OSError, gzip.BadGzipFile, EOFError):
            archive_pause(12.0 * (attempt + 1), 24.0 * (attempt + 1))
    return None


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    captures = load_captures()
    print(
        f"Common Crawl matchdays to fetch: {len(captures)}  "
        f"pause {ARCHIVE_MIN_WAIT:.0f}–{ARCHIVE_MAX_WAIT:.0f}s",
        flush=True,
    )
    ok = fail = 0
    for done, (mid, row) in enumerate(sorted(captures.items(), key=lambda item: int(item[0])), start=1):
        dest = OUT / f"{mid}.html"
        if dest.exists() and dest.stat().st_size > 500:
            continue
        html = fetch_capture(row)
        if html is None:
            fail += 1
            print(f"  fail {mid}", flush=True)
        else:
            dest.write_bytes(html)
            ok += 1
            print(f"  saved {mid} ({len(html)} bytes)", flush=True)
        if done % 10 == 0:
            print(f"progress {done}/{len(captures)} ok={ok} fail={fail}", flush=True)
    print(f"DONE ok={ok} fail={fail}", flush=True)


if __name__ == "__main__":
    main()
