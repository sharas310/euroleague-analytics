"""
Download EuroLeague player data with the euroleague-api package.

Install:
    python -m pip install -r requirements.txt

Run (from the repository root):
    python src/euroleague_download.py                       # 2021-2026, season stats + box scores
    python src/euroleague_download.py --start 2015 --shots --pbp
    python src/euroleague_download.py --delay 3             # slower, if you still get HTTP 429

A season is identified by its starting year: 2025 = 2025-26, 2026 = 2026-27.

Every game is cached as its own file under data/raw/..., so:
  * if the script stops or the server rate-limits you (429), just run it again;
    games already downloaded are skipped and only missing ones are fetched;
  * on a 429 the script waits by itself (30 s, 60 s, 120 s ...) and retries.
Final files: data/*.parquet (or .csv with --csv).
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from requests.exceptions import HTTPError, ConnectionError, Timeout
from euroleague_api.player_stats import PlayerStats
from euroleague_api.boxscore_data import BoxScoreData
from euroleague_api.shot_data import ShotData
from euroleague_api.play_by_play_data import PlayByPlay

COMP = "E"  # "E" = EuroLeague, "U" = EuroCup
ENDPOINTS = ["traditional", "advanced", "misc", "scoring"]
MODES = ["PerGame", "Accumulated", "Per100Possesions"]  # spelled as in the API

OUT = Path("data")
RAW = OUT / "raw"
NO_DATA = "NO_DATA"  # marker file: this game has no data (so it is not retried every run)


class GameMissing(Exception):
    pass


def fetch(fn, *args, retries=6, base_wait=30, **kwargs):
    """Call the API with retries. 429/5xx/connection errors -> wait and try again.
    404 -> GameMissing (no such game)."""
    for attempt in range(retries + 1):
        try:
            return fn(*args, **kwargs)
        except HTTPError as e:
            code = e.response.status_code if e.response is not None else None
            if code == 404:
                raise GameMissing() from e
            if code not in (429, 500, 502, 503, 504) or attempt == retries:
                raise
        except (ConnectionError, Timeout):
            if attempt == retries:
                raise
        wait = base_wait * 2 ** attempt
        print(f"  … server is rate-limiting ({fn.__name__}), waiting {wait} s "
              f"(attempt {attempt + 1}/{retries})", flush=True)
        time.sleep(wait)


def write(df: pd.DataFrame, path: Path, as_csv: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if as_csv:
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)


def read(path: Path, as_csv: bool) -> pd.DataFrame:
    return pd.read_csv(path) if as_csv else pd.read_parquet(path)


def save(df, name: str, as_csv: bool) -> None:
    if df is None or df.empty:
        print(f"  (empty) {name}")
        return
    path = OUT / (name + (".csv" if as_csv else ".parquet"))
    write(df, path, as_csv)
    print(f"  ✓ {path}  {df.shape[0]} rows × {df.shape[1]} cols")


def played_gamecodes(client, season: int) -> list[int]:
    games = fetch(client.get_gamecodes_season, season)
    # the package converts the 'played' column unreliably, so check the score instead
    played = games[(games["homescore"] + games["awayscore"]) > 0]
    return sorted(played["gameCode"].unique().tolist())


def download_games(kind: str, fn, client, season: int, delay: float, as_csv: bool):
    """Download every game separately (cached) and return the combined season."""
    ext = ".csv" if as_csv else ".parquet"
    folder = RAW / kind / str(season)
    folder.mkdir(parents=True, exist_ok=True)

    try:
        codes = played_gamecodes(client, season)
    except GameMissing:
        print(f"  season {season} not available")
        return None

    todo = [c for c in codes
            if not (folder / f"{c}{ext}").exists() and not (folder / f"{c}.{NO_DATA}").exists()]
    print(f"{kind} {season}: {len(codes)} games, {len(todo)} left to download", flush=True)

    for i, code in enumerate(todo, 1):
        try:
            df = fetch(fn, season, code)
            if df is None or df.empty:
                (folder / f"{code}.{NO_DATA}").touch()
            else:
                write(df, folder / f"{code}{ext}", as_csv)
        except GameMissing:
            (folder / f"{code}.{NO_DATA}").touch()
        except Exception as e:
            # no marker is written, so the game is retried next run
            print(f"  ! game {code}: {e}")
        if i % 25 == 0:
            print(f"    {i}/{len(todo)}", flush=True)
        time.sleep(delay)

    files = sorted(folder.glob(f"*{ext}"))
    if not files:
        return None
    return pd.concat([read(f, as_csv) for f in files], ignore_index=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--start", type=int, default=2021)
    p.add_argument("--end", type=int, default=2026)
    p.add_argument("--shots", action="store_true", help="shot coordinates for every attempt")
    p.add_argument("--pbp", action="store_true", help="play-by-play with on-court lineups")
    p.add_argument("--no-season", action="store_true", help="skip season-level player stats")
    p.add_argument("--no-boxscore", action="store_true", help="skip box scores")
    p.add_argument("--delay", type=float, default=1.5, help="pause between requests, seconds")
    p.add_argument("--csv", action="store_true")
    a, _ = p.parse_known_args()  # also works inside Jupyter (ignores -f kernel.json)

    seasons = range(a.start, a.end + 1)
    ps, bx, sh, pbp = PlayerStats(COMP), BoxScoreData(COMP), ShotData(COMP), PlayByPlay(COMP)

    # 1) Season-level player stats: 4 categories × 3 aggregation modes
    if not a.no_season:
        for ep in ENDPOINTS:
            for mode in MODES:
                print(f"Season stats: {ep} / {mode}", flush=True)
                frames = []
                for season in seasons:
                    try:
                        df = fetch(ps.get_player_stats_single_season, ep, season,
                                   statistic_mode=mode)
                    except Exception as e:
                        print(f"  ! {season}: {e}")
                        df = None
                    if df is not None and not df.empty:
                        df.insert(0, "season", season)
                        frames.append(df)
                    time.sleep(a.delay)
                if frames:
                    save(pd.concat(frames, ignore_index=True), f"player_{ep}_{mode}", a.csv)

    # 2-4) Per-game data
    jobs = []
    if not a.no_boxscore:
        jobs.append(("boxscore", bx.get_players_boxscore_stats, bx))
    if a.shots:
        jobs.append(("shots", sh.get_game_shot_data, sh))
    if a.pbp:
        jobs.append(("pbp_lineups", pbp.get_game_pbp_data_lineups, pbp))

    for kind, fn, client in jobs:
        for season in seasons:
            df = download_games(kind, fn, client, season, a.delay, a.csv)
            save(df, f"{kind}_{season}", a.csv)

    print("\nDone. If you saw errors, run it again: only missing games will be fetched.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Running it again resumes where it left off.")
        sys.exit(1)
