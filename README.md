# Billiard

Virtualenv is `.venv` in this directory. Run commands with `.venv/bin/python`.

Update league and Einzel results through today (skips seasons already in `data/games.csv`):

```bash
.venv/bin/python sync.py
```

Sync then downloads the squad lists of those seasons again and attaches Pass-Nr. to every game. No separate `identity.py` step is needed.

Leagues only: `.venv/bin/python sync.py --leagues-only`  
Einzel only: `.venv/bin/python sync.py --einzel-only`  
Without Pass-Nr. attach: `.venv/bin/python sync.py --skip-ids`

If Einzel rows are dated `00.00.0000`, fill them from the tournament header:

```bash
.venv/bin/python scrape_einzel.py --repair-dates
```

Refit ratings after new games:

```bash
.venv/bin/python elo.py --half-life-years 0.67  #this isthe bradly terry fit with 8 months half life
```

