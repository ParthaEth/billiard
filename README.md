# Billiard

Virtualenv is `.venv` in this directory. Run commands with `.venv/bin/python`.

Update league and Einzel results through today (skips seasons already in `data/games.csv`):

```bash
.venv/bin/python sync.py
```

Leagues only: `.venv/bin/python sync.py --leagues-only`  
Einzel only: `.venv/bin/python sync.py --einzel-only`

Refit ratings after new games:

```bash
.venv/bin/python elo.py --half-life-years 0.67  #this isthe bradly terry fit with 8 months half life
```

