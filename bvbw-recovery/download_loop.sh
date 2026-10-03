#!/usr/bin/env bash
# Keep retrying Wayback matchday downloads until none are missing (or Ctrl-C).
# Safe to run while extract/merge are run separately — only writes new HTML files.
set -euo pipefail
cd "$(dirname "$0")/.."
LOG=bvbw-recovery/download_loop.log
WORKERS="${WORKERS:-3}"
SLEEP_BETWEEN_PASSES="${SLEEP_BETWEEN_PASSES:-30}"

echo "$(date -Is) starting retry loop workers=$WORKERS" | tee -a "$LOG"

pass=0
while true; do
  pass=$((pass + 1))
  echo "$(date -Is) === pass $pass ===" | tee -a "$LOG"
  # One pass over whatever is still missing (failed attempts leave no file)
  python3 bvbw-recovery/download_matchdays.py >>"$LOG" 2>&1 || true

  missing=$(python3 - <<'PY'
import json, re
from pathlib import Path
rows = []
for name in ("bvbw-billardarea.json", "bvbw-billardarea-2020-2022.json"):
    p = Path("bvbw-recovery/wayback") / name
    if p.exists():
        d = json.loads(p.read_text())
        if isinstance(d, list) and len(d) > 1:
            rows.extend(d[1:])
best = set()
for r in rows:
    if r[2] != "text/html":
        continue
    m = re.search(r"/cms_leagues/matchday/(\d+)", r[1])
    if m:
        best.add(m.group(1))
have = {p.stem for p in Path("bvbw-recovery/recovered/matchdays").glob("*.html") if p.stat().st_size > 500}
print(len(best - have))
PY
)
  have=$((4104 - missing))
  echo "$(date -Is) have ~$((4104 - missing))/4104  missing=$missing" | tee -a "$LOG"
  if [[ "$missing" -eq 0 ]]; then
    echo "$(date -Is) DONE — nothing left to download" | tee -a "$LOG"
    break
  fi
  echo "$(date -Is) sleeping ${SLEEP_BETWEEN_PASSES}s before retry..." | tee -a "$LOG"
  sleep "$SLEEP_BETWEEN_PASSES"
done
