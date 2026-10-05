# Agents

- The +4.30 offset in `scale.cfg` stays fixed, so leave that file alone. Never edit, recompute or reset it, and never overwrite `data/scale_reference.csv`. Only the user changes it.
- Routine update: `.venv/bin/python sync.py`, then `.venv/bin/python elo.py --half-life-years .67`.
- Open name mappings are in `AGENT_TODO.md`.
