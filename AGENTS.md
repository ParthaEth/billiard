# Agents

- The +4.30 offset in `scale.cfg` stays fixed, so leave that file alone. Never edit, recompute or reset it, and never overwrite `data/scale_reference.csv`. Only the user changes it.
- Routine update: `.venv/bin/python sync.py`, then `.venv/bin/python elo.py --half-life-years .67`. Sync refreshes leagues, Einzel, and Sanwald Cup Mondays. The Monday matches stay in `data/sanwald_matches.csv`; the Elo fit reads them from there.
- Open name mappings are in `AGENT_TODO.md`.
