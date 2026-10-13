# Billiard

Virtualenv is `.venv` in this directory. Run commands with `.venv/bin/python`.

Update league, Einzel, and Sanwald Cup Monday nights through today (league and Einzel skip seasons already in `data/games.csv`):

```bash
.venv/bin/python sync.py
```

Sync then attaches Pass-Nr. to every game. No separate `identity.py` step is needed. Squad pages are read from `data/identity_cache`. In the synced seasons, a squad page is downloaded again only for a new team, or when a game names a player that the cached page does not list.

Leagues only: `.venv/bin/python sync.py --leagues-only`  
Einzel only: `.venv/bin/python sync.py --einzel-only`  
Without Pass-Nr. attach: `.venv/bin/python sync.py --skip-ids`

If Einzel rows are dated `00.00.0000`, fill them from the tournament header:

```bash
.venv/bin/python scrape_einzel.py --repair-dates
```

The +4.30 offset in `scale.cfg` stays fixed, so leave that file alone. Published Elo is the fitted Elo plus that offset. Never recompute it and never overwrite `data/scale_reference.csv`.

`sync.py` refreshes the Sanwald Cup (Monday 20:00, 9-ball, Die Drei Sindelfingen) into its own files. Those matches stay out of `data/games.csv` and are included in the Elo fit. League rows join on Pass-Nr. The first link from a TournamentApp `sid` to a Pass-Nr. is a full name plus Elo within 30, stored in `data/tournamentapp_identity.csv`. Later Sanwald games join on that `sid`. Anyone without a row there keeps a separate rating. Mondays alone:

```bash
.venv/bin/python scrape_sanwald.py
```

Refit ratings after new games:

```bash
.venv/bin/python elo.py --half-life-years 0.67  #this isthe bradly terry fit with 8 months half life
```

