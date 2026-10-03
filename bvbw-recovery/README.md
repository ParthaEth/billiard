# BVBW archive recovery

## Pipeline

```bash
# 1) Download archived Spielberichte (resumable; slow due to Wayback rate limits)
python3 bvbw-recovery/download_matchdays.py

# 2) Extract pool games for players already in data/games.csv (+ inhouse)
python3 bvbw-recovery/recover_archive.py extract

# 3) Merge into data/games.csv (backup: data/games.pre_archive_merge.csv)
python3 bvbw-recovery/recover_archive.py merge
```

## Scope

- Seasons **2015/2016–2020/2021** (pool only)
- Live `billard-bvbw.de` already covers **2021/2022+** (no 2020/2021 there — COVID gap)
- A game is kept if **either** player matches the main DB name list

## Status

First merge added archive rows into `data/games.csv`. Re-run extract+merge after more matchdays finish downloading.
