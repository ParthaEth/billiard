# Billiard

Virtualenv is `.venv` in this directory. Run commands with `.venv/bin/python`.

Update league, Einzel, and TournamentApp open tournaments (Sanwald Cup, BULL CUP, Stuttgart, Pfullingen, …) through today (league and Einzel skip seasons already in `data/games.csv`):

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

`sync.py` refreshes every TournamentApp event held at a Baden-Württemberg venue into `data/tournamentapp_*.csv`: the Sanwald Cup Mondays at Die Drei Sindelfingen, the BC Stuttgart 1891 house tournaments, Magic-Cup and Tito's Cup in Pfullingen, the 10-Ball BULL CUP in Leonberg, Konstanz, Aalen and the opens. Doubles, team, darts, bank-pool, one-pocket and demo events are skipped. Those matches stay out of `data/games.csv` and are included in the Elo fit. League rows join on Pass-Nr. A TournamentApp `sid` is linked to a Pass-Nr. in `data/tournamentapp_identity.csv` when the full name matches exactly one league player, no other `sid` with that name played the same event, the two Elo fits agree within 2.5 standard errors, and the opponent network does not contradict it: the people the `sid` met at events are compared with the league player's league opponents and teammates. More shared people than chance confirms the link. Chance-level overlap where a few were expected rejects it. A common surname (Müller, Schmidt, …) needs that confirmation. An uncommon name with too little data on either side stays linked on the name. Later games join on the `sid`. Anyone without a row there keeps a separate rating. Rejected name matches are listed in `data/tournamentapp_candidates.csv` for review. TournamentApp alone:

```bash
.venv/bin/python scrape_tournamentapp.py
.venv/bin/python scrape_tournamentapp.py --link-only   # relink from the saved matches, no download
```

Refit ratings after new games:

```bash
.venv/bin/python elo.py --half-life-years 0.67  #this isthe bradly terry fit with 8 months half life
```

