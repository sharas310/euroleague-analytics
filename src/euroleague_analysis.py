"""
EuroLeague player analysis
==========================

Part 1 - PLAYER IMPORTANCE / ABSENCES
  For every player who averages >= 20 minutes:
  * team results WITH and WITHOUT him (games he missed while on the roster,
    usually through injury);
  * results adjusted for opponent strength and home court
    (otherwise a missed game against Real Madrid and one against ALBA would look the same);
  * on/off from plus-minus: team +/- per 40 minutes with him on the floor and off it.

Part 2 - MATCHUP MODEL (how many points a player will score against a given team)
  * Every player gets a role PG/SG/SF/PF/C from his statistical profile
    (the EuroLeague API only provides Guard/Forward/Center).
    Real positions can be supplied in data/positions.csv.
  * "Defense vs position": how much each team allows to each role,
    compared with what the same players usually score.
  * Individual defender effect (e.g. Milutinov on the floor): from play-by-play
    with lineups - how many points opposing C score per 40 minutes with him on and off.
    (Needs pbp_lineups files: euroleague_download.py --pbp)
  * Test: every season separately, first 80 % of games (chronologically) for
    training, last 20 % for testing. Models compared:
      M0  player average (points per minute × expected minutes)
      M1  M0 × opponent defense against that role
      M2  M1 × home/away factor
      M3  Poisson regression: player + team + opponent×role + home,
          all effects estimated jointly and shrunk (regularisation)

Run (from the repository root, where data/ lives):
  python src/euroleague_analysis.py
  python src/euroleague_analysis.py --min-minutes 20 --train-frac 0.8
  python src/euroleague_analysis.py --predict "sloukas" --vs OLY --home
  python src/euroleague_analysis.py --only 1          # part 1 only

Results: output/ folder (CSV and, if openpyxl is installed, results.xlsx)

Requires: pandas numpy scikit-learn scipy (pyarrow for .parquet data)
"""
import argparse
import ast
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

DATA = Path("data")
OUT = Path("output")
ROLES = ["PG", "SG", "SF", "PF", "C"]
NOT_PLAYERS = {"Team", "Total", "TEAM", "TOTAL"}

STAT_COLS = [
    "Points", "FieldGoalsMade2", "FieldGoalsAttempted2", "FieldGoalsMade3",
    "FieldGoalsAttempted3", "FreeThrowsMade", "FreeThrowsAttempted",
    "OffensiveRebounds", "DefensiveRebounds", "TotalRebounds", "Assistances",
    "Steals", "Turnovers", "BlocksFavour", "BlocksAgainst", "FoulsCommited",
    "FoulsReceived", "Valuation",
]

RESULTS: dict[str, pd.DataFrame] = {}  # everything that goes into the xlsx


# ───────────────────────────── helpers ─────────────────────────────
def season_files(prefix: str) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for ext in (".csv", ".parquet"):  # parquet overrides csv if both exist
        for f in DATA.glob(f"{prefix}_*{ext}"):
            try:
                files[int(f.stem.split("_")[-1])] = f
            except ValueError:
                pass
    return dict(sorted(files.items()))


def read_any(f: Path) -> pd.DataFrame:
    return pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f, low_memory=False)


def to_minutes(x) -> float:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return 0.0
    if isinstance(x, (int, float, np.integer, np.floating)):
        return float(x)
    s = str(x).strip()
    if ":" in s:
        try:
            m, sec = s.split(":")[:2]
            return int(m) + int(sec) / 60
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0  # "DNP"


def clock_to_sec(x) -> float:
    if not isinstance(x, str) or ":" not in x:
        return np.nan
    try:
        m, s = x.strip().split(":")[:2]
        return int(m) * 60 + int(s)
    except ValueError:
        return np.nan


def save(df: pd.DataFrame, name: str, sheet: str | None = None) -> None:
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    RESULTS[sheet or name[:31]] = df
    print(f"  ✓ output/{name}.csv  ({len(df)} rows)")


def write_xlsx() -> None:
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        print("  (openpyxl not installed - results saved as CSV only)")
        return
    with pd.ExcelWriter(OUT / "results.xlsx") as xw:
        for sheet, df in RESULTS.items():
            df.head(100_000).to_excel(xw, sheet_name=sheet[:31], index=False)
    print("  ✓ output/results.xlsx")


# ───────────────────────────── data ─────────────────────────────
def load_boxscores(seasons=None) -> pd.DataFrame:
    files = season_files("boxscore")
    if seasons:
        files = {s: f for s, f in files.items() if s in seasons}
    if not files:
        sys.exit("No data/boxscore_<season>.parquet or .csv found. "
                 "Run the script from the folder that contains 'data'.")
    df = pd.concat([read_any(f) for f in files.values()], ignore_index=True)
    need = ["Season", "Gamecode", "Home", "Player_ID", "Team", "Player", "Minutes", "Points"]
    miss = [c for c in need if c not in df.columns]
    if miss:
        sys.exit(f"Box score files are missing columns {miss}.\nAvailable: {list(df.columns)}")

    df["Player_ID"] = df["Player_ID"].astype(str).str.strip()
    df["Team"] = df["Team"].astype(str).str.strip()
    df["Player"] = df["Player"].astype(str).str.strip()
    df["Min"] = df["Minutes"].map(to_minutes)
    for c in STAT_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0) if c in df.columns else 0.0
    df["Plusminus"] = pd.to_numeric(df.get("Plusminus", np.nan), errors="coerce")
    for c in ("Season", "Gamecode", "Home"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype(int)
    df = df.drop_duplicates(["Season", "Gamecode", "Team", "Player_ID"])
    print(f"Box scores: seasons {sorted(files)}, {df.groupby('Season').Gamecode.nunique().sum()} games")
    return df


def team_games(box: pd.DataFrame) -> pd.DataFrame:
    """One row = one team in one game."""
    tot = box[box["Player_ID"] == "Total"].copy()
    if tot.empty:  # fallback: sum the players
        pl = box[~box["Player_ID"].isin(NOT_PLAYERS)]
        tot = pl.groupby(["Season", "Gamecode", "Team", "Home"], as_index=False)[STAT_COLS + ["Min"]].sum()
    tot["poss"] = (tot["FieldGoalsAttempted2"] + tot["FieldGoalsAttempted3"]
                   + 0.44 * tot["FreeThrowsAttempted"] - tot["OffensiveRebounds"] + tot["Turnovers"])
    tot["GameMin"] = np.where(tot["Min"] > 150, (tot["Min"] / 5).round(), 40.0)
    g = tot[["Season", "Gamecode", "Team", "Home", "Points", "poss", "GameMin"]]
    g = g[g.groupby(["Season", "Gamecode"])["Team"].transform("size") == 2]
    opp = g[["Season", "Gamecode", "Team", "Points", "poss"]].rename(
        columns={"Team": "Opp", "Points": "OppPoints", "poss": "OppPoss"})
    m = g.merge(opp, on=["Season", "Gamecode"])
    m = m[m["Team"] != m["Opp"]].copy()
    m["Margin"] = m["Points"] - m["OppPoints"]
    m["Win"] = (m["Margin"] > 0).astype(int)
    m["Poss"] = (m["poss"] + m["OppPoss"]) / 2
    return m.drop(columns=["poss", "OppPoss"]).reset_index(drop=True)


def fit_ratings(tg: pd.DataFrame, ridge: float = 2.0) -> dict:
    """Margin = home advantage + rating(team) − rating(opponent), ridge-regularised."""
    out = {}
    for s, g in tg[tg["Home"] == 1].groupby("Season"):
        teams = sorted(set(g["Team"]) | set(g["Opp"]))
        idx = {t: i for i, t in enumerate(teams)}
        n, k = len(g), len(teams)
        X = np.zeros((n, k + 1))
        X[:, -1] = 1
        X[np.arange(n), g["Team"].map(idx).values] = 1
        X[np.arange(n), g["Opp"].map(idx).values] = -1
        A = X.T @ X + ridge * np.diag([1.0] * k + [0.0])
        w = np.linalg.solve(A, X.T @ g["Margin"].values)
        out[s] = (dict(zip(teams, w[:-1])), w[-1])
    return out


def add_expected_margin(tg: pd.DataFrame, ratings: dict) -> pd.DataFrame:
    tg = tg.copy()
    r_t = [ratings[s][0].get(t, 0) for s, t in zip(tg["Season"], tg["Team"])]
    r_o = [ratings[s][0].get(o, 0) for s, o in zip(tg["Season"], tg["Opp"])]
    hca = np.array([ratings[s][1] for s in tg["Season"]])
    tg["ExpMargin"] = np.array(r_t) - np.array(r_o) + np.where(tg["Home"] == 1, hca, -hca)
    tg["Resid"] = tg["Margin"] - tg["ExpMargin"]
    return tg


# ───────────────────────────── PART 1 ─────────────────────────────
def _impact_stats(g: pd.DataFrame) -> pd.Series:
    w, wo = g[g["Played"]], g[~g["Played"]]

    def rtg(d, col):
        return d[col].sum() / d["Poss"].sum() * 100 if d["Poss"].sum() > 0 else np.nan

    r = {
        "player": g["Player"].iloc[-1],
        "team": g["_team"].iloc[-1],
        "MPG": g["Min"].mean(),
        "games_with": len(w), "games_without": len(wo),
        "win_pct_with": w["Win"].mean() * 100,
        "win_pct_without": wo["Win"].mean() * 100 if len(wo) else np.nan,
        "ortg_with": rtg(w, "Points"), "ortg_without": rtg(wo, "Points"),
        "drtg_with": rtg(w, "OppPoints"), "drtg_without": rtg(wo, "OppPoints"),
        "adj_margin_with": w["Resid"].mean(),
        "adj_margin_without": wo["Resid"].mean() if len(wo) else np.nan,
    }
    r["impact_raw"] = r["adj_margin_with"] - r["adj_margin_without"]
    # small samples are very noisy -> shrink toward 0
    r["impact_shrunk"] = r["impact_raw"] * len(wo) / (len(wo) + 6) if len(wo) else np.nan
    r["p_value"] = np.nan
    if len(w) >= 3 and len(wo) >= 3:
        try:
            from scipy.stats import ttest_ind
            r["p_value"] = ttest_ind(w["Resid"], wo["Resid"], equal_var=False).pvalue
        except ImportError:
            pass
    # on/off from plus-minus (games played only)
    pm = w.dropna(subset=["Plusminus"])
    on_min = pm["Min"].sum()
    off_min = (pm["GameMin"] - pm["Min"]).sum()
    r["on_per40"] = pm["Plusminus"].sum() / on_min * 40 if on_min > 0 else np.nan
    r["off_per40"] = (pm["Margin"] - pm["Plusminus"]).sum() / off_min * 40 if off_min > 0 else np.nan
    r["on_off_per40"] = r["on_per40"] - r["off_per40"]
    return pd.Series(r)


def part1_impact(box, tg, min_minutes, min_games) -> pd.DataFrame:
    print("\n═══ PART 1: player importance / absences ═══")
    tg = add_expected_margin(tg, fit_ratings(tg))
    pl = box[~box["Player_ID"].isin(NOT_PLAYERS)]
    played = pl[pl["Min"] > 0]

    summ = (played.groupby(["Season", "Team", "Player_ID"])
            .agg(GP=("Gamecode", "nunique"), MPG=("Min", "mean"),
                 first=("Gamecode", "min"), last=("Gamecode", "max"))
            .reset_index())
    reg = summ[(summ["MPG"] >= min_minutes) & (summ["GP"] >= min_games)]
    print(f"Player-seasons with >= {min_minutes} min and >= {min_games} games: {len(reg)}")

    # all team games between the player's first and last game for this team
    # (so time before he arrived / after he left is not counted)
    long = reg.merge(tg, on=["Season", "Team"])
    long = long[(long["Gamecode"] >= long["first"]) & (long["Gamecode"] <= long["last"])]
    long = long.drop(columns=["MPG", "GP", "first", "last"]).merge(
        played[["Season", "Team", "Player_ID", "Gamecode", "Min", "Plusminus"]],
        on=["Season", "Team", "Player_ID", "Gamecode"], how="left")
    long["Played"] = long["Min"].notna()
    names = pl.groupby("Player_ID")["Player"].last()
    long["Player"] = long["Player_ID"].map(names)
    long["_team"] = long["Team"]  # copy, because groupby hides grouping columns in some pandas versions

    per_season = (long.groupby(["Season", "Team", "Player_ID"]).apply(_impact_stats)
                  .reset_index().drop(columns=["Team"], errors="ignore"))
    per_season = per_season.sort_values(["Season", "impact_shrunk"], ascending=[True, False])

    pooled = long.groupby("Player_ID").apply(_impact_stats).reset_index()
    pooled["team"] = long.groupby("Player_ID")["Team"].agg(lambda s: "/".join(dict.fromkeys(s)))\
        .reindex(pooled["Player_ID"]).values
    pooled = pooled.sort_values("impact_shrunk", ascending=False)

    rnd = lambda d: d.round(2)
    save(rnd(per_season), "1_impact_by_season", "1_impact_seasons")
    save(rnd(pooled), "1_impact_all_seasons", "1_impact_all")

    last = per_season["Season"].max()
    show = per_season[(per_season["Season"] == last) & (per_season["games_without"] >= 3)]
    cols = ["player", "team", "MPG", "games_with", "games_without", "win_pct_with", "win_pct_without",
            "impact_raw", "impact_shrunk", "on_off_per40"]
    print(f"\nSeason {last}: players the team missed most (>= 3 games missed)")
    print(show[cols].head(15).round(1).to_string(index=False))
    print("\nTop 15 by on/off (all seasons, >= 40 games):")
    print(pooled[pooled["games_with"] >= 40].sort_values("on_off_per40", ascending=False)
          [["player", "team", "MPG", "games_with", "on_per40", "off_per40", "on_off_per40"]]
          .head(15).round(1).to_string(index=False))
    return per_season


# ───────────────────────────── PART 2 ─────────────────────────────
FEATS = ["ast40", "trb40", "oreb40", "blk40", "stl40", "tpar", "ftr"]


def player_profiles(pg: pd.DataFrame) -> pd.DataFrame:
    a = pg.groupby(["Season", "Player_ID"]).agg(
        Min=("Min", "sum"), ast=("Assistances", "sum"), trb=("TotalRebounds", "sum"),
        oreb=("OffensiveRebounds", "sum"), blk=("BlocksFavour", "sum"), stl=("Steals", "sum"),
        fga2=("FieldGoalsAttempted2", "sum"), fga3=("FieldGoalsAttempted3", "sum"),
        fta=("FreeThrowsAttempted", "sum")).reset_index()
    m = a["Min"].clip(lower=1)
    for c in ("ast", "trb", "oreb", "blk", "stl"):
        a[f"{c}40"] = a[c] / m * 40
    fga = (a["fga2"] + a["fga3"]).replace(0, np.nan)
    a["tpar"] = (a["fga3"] / fga).fillna(0)
    a["ftr"] = (a["fta"] / fga).fillna(0)
    return a


def assign_roles(prof: pd.DataFrame) -> pd.DataFrame:
    """K-means into 5 groups; groups are named by "size" (rebounds + blocks)
    and assists. This is an approximation - real positions can be given in positions.csv."""
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler

    prof = prof.copy()
    fit = prof[prof["Min"] >= 150]
    if len(fit) < 25:
        fit = prof[prof["Min"] > 0]
    sc = StandardScaler().fit(fit[FEATS])
    km = KMeans(5, n_init=20, random_state=0).fit(sc.transform(fit[FEATS]))
    c = pd.DataFrame(km.cluster_centers_, columns=FEATS)
    size = c["trb40"] + c["blk40"] + c["oreb40"] - c["tpar"]
    order = size.sort_values().index.tolist()  # smallest -> biggest
    guards = sorted(order[:2], key=lambda i: -c.loc[i, "ast40"])  # more assists = PG
    labels = {guards[0]: "PG", guards[1]: "SG", order[2]: "SF", order[3]: "PF", order[4]: "C"}
    prof["Role"] = [labels[k] for k in km.predict(sc.transform(prof[FEATS].fillna(0)))]

    pos_file = DATA / "positions.csv"
    if pos_file.exists():
        pos = pd.read_csv(pos_file)
        pos.columns = [x.strip() for x in pos.columns]
        key = "Player_ID" if "Player_ID" in pos.columns else None
        if key and "Role" in pos.columns:
            pos[key] = pos[key].astype(str).str.strip()
            m = prof["Player_ID"].map(pos.set_index(key)["Role"])
            prof["Role"] = m.fillna(prof["Role"])
            print(f"  positions.csv: {m.notna().sum()} roles overridden")
    return prof


def build_player_games(box, tg) -> pd.DataFrame:
    pl = box[(~box["Player_ID"].isin(NOT_PLAYERS)) & (box["Min"] > 0)].copy()
    pl = pl.merge(tg[["Season", "Gamecode", "Team", "Opp"]], on=["Season", "Gamecode", "Team"])
    return pl


def season_cuts(tg, frac) -> dict:
    cuts = {}
    for s, g in tg.groupby("Season"):
        codes = np.sort(g["Gamecode"].unique())
        cuts[s] = codes[max(int(len(codes) * frac) - 1, 0)]
    return cuts


def _val_split(tr: pd.DataFrame, frac=0.85):
    """Split training data chronologically: the last part is used to tune parameters."""
    codes = np.sort(tr["Gamecode"].unique())
    cut = codes[int(len(codes) * frac)]
    return tr[tr["Gamecode"] <= cut], tr[tr["Gamecode"] > cut]


class RatioModel:
    """M0-M2: player points/min × expected minutes × opponent-role factor × home factor.
    K is the shrinkage strength (in points): the larger it is, the more the factors
    are pulled toward 1.0. It is chosen automatically on the end of the training data."""

    K_GRID = [60, 250, 1000, 3000, 10000]

    def __init__(self, K=None, prior_min=100.0, last_n=10):
        self.K, self.prior_min, self.last_n = K, prior_min, last_n

    def _fit(self, tr, K):
        role_ppm = tr.groupby("Role").apply(lambda d: d["Points"].sum() / d["Min"].sum())
        p = tr.groupby("Player_ID").agg(pts=("Points", "sum"), mins=("Min", "sum"), Role=("Role", "last"))
        p["ppm"] = (p["pts"] + self.prior_min * p["Role"].map(role_ppm)) / (p["mins"] + self.prior_min)
        p["exp_min"] = tr.sort_values("Gamecode").groupby("Player_ID")["Min"].apply(
            lambda s: s.tail(self.last_n).mean())
        p["games"] = tr.groupby("Player_ID").size()
        self.players = p
        tr = tr.assign(exp_pts=tr["Player_ID"].map(p["ppm"]) * tr["Min"])
        d = tr.groupby(["Opp", "Role"]).agg(act=("Points", "sum"), exp=("exp_pts", "sum"),
                                            n=("Points", "size"))
        d["factor"] = (d["act"] + K) / (d["exp"] + K)
        self.dvp = d
        h = tr.groupby("Home").agg(act=("Points", "sum"), exp=("exp_pts", "sum"))
        self.home = (h["act"] / h["exp"]).to_dict()
        return self

    def fit(self, tr: pd.DataFrame):
        if self.K is None:
            a, b = _val_split(tr)
            b = b[b["Player_ID"].isin(a["Player_ID"])]
            errs = {}
            for K in self.K_GRID:
                pr = self._fit(a, K).predict(b, minutes=b["Min"])["M2"]
                errs[K] = ((pr - b["Points"]) ** 2).mean()
            self.K = min(errs, key=errs.get)
        return self._fit(tr, self.K)

    def predict(self, te: pd.DataFrame, minutes=None) -> pd.DataFrame:
        p = self.players
        mins = te["Player_ID"].map(p["exp_min"]) if minutes is None else minutes
        base_rate = te["Player_ID"].map(p["ppm"])
        f_dvp = pd.Series([self.dvp["factor"].get((o, r), 1.0) for o, r in zip(te["Opp"], te["Role"])],
                          index=te.index)
        f_home = te["Home"].map(self.home).fillna(1.0)
        return pd.DataFrame({
            "M0": base_rate * mins,
            "M1": base_rate * mins * f_dvp,
            "M2": base_rate * mins * f_dvp * f_home,
            "dvp_factor": f_dvp,
        }, index=te.index)


class GLMModel:
    """M3: Poisson regression on scoring rate (points/min), weighted by minutes.
    log(pts/min) = player + his team + opponent×role + home.
    Opponent×role columns are multiplied by 'scale' < 1, which shrinks them harder
    than the player effects. alpha and scale are chosen automatically."""

    GRID = [(a, s) for a in (1e-4, 3e-4, 1e-3) for s in (1.0, 0.3, 0.1)]

    def __init__(self, alpha=None, scale=None):
        self.alpha, self.scale = alpha, scale

    @staticmethod
    def _X(d):
        return pd.DataFrame({"p": d["Player_ID"].astype(str), "t": d["Team"].astype(str),
                             "or": d["Opp"].astype(str) + "_" + d["Role"].astype(str),
                             "h": d["Home"].astype(str)})

    def _transform(self, enc, d, scale):
        X = enc.transform(self._X(d)).tocsc().astype(float)
        X[:, self._or_cols(enc)] *= scale
        return X.tocsr()

    @staticmethod
    def _or_cols(enc):
        return np.where(pd.Index(enc.get_feature_names_out(["p", "t", "or", "h"])).str.startswith("or_"))[0]

    def _fit(self, tr, alpha, scale):
        from sklearn.linear_model import PoissonRegressor
        from sklearn.preprocessing import OneHotEncoder
        tr = tr[tr["Min"] >= 3]
        enc = OneHotEncoder(handle_unknown="ignore").fit(self._X(tr))
        m = PoissonRegressor(alpha=alpha, max_iter=1000)
        m.fit(self._transform(enc, tr, scale), tr["Points"] / tr["Min"], sample_weight=tr["Min"])
        return enc, m

    def fit(self, tr: pd.DataFrame):
        if self.alpha is None or self.scale is None:
            a, b = _val_split(tr)
            b = b[b["Player_ID"].isin(a["Player_ID"])]
            errs = {}
            for al, sc in self.GRID:
                enc, m = self._fit(a, al, sc)
                pred = m.predict(self._transform(enc, b, sc)) * b["Min"]
                errs[(al, sc)] = ((pred - b["Points"]) ** 2).mean()
            self.alpha, self.scale = min(errs, key=errs.get)
        self.enc, self.m = self._fit(tr, self.alpha, self.scale)
        return self

    def rate(self, d):
        return pd.Series(self.m.predict(self._transform(self.enc, d, self.scale)), index=d.index)

    def opp_role_factors(self) -> pd.DataFrame:
        names = self.enc.get_feature_names_out(["p", "t", "or", "h"])
        coef = pd.Series(self.m.coef_, index=names)
        c = coef[coef.index.str.startswith("or_")] * self.scale
        df = c.rename("coef").reset_index()
        df[["Opp", "Role"]] = df["index"].str[3:].str.rsplit("_", n=1, expand=True)
        # centre per role, so 1.0 = league average
        df["glm_factor"] = np.exp(df["coef"] - df.groupby("Role")["coef"].transform("mean"))
        return df[["Opp", "Role", "glm_factor"]]


def evaluate(df: pd.DataFrame, models: list[str]) -> dict:
    r = {"n": len(df)}
    for m in models:
        e = df[m] - df["Points"]
        r[f"{m}_MAE"] = e.abs().mean()
        r[f"{m}_RMSE"] = np.sqrt((e ** 2).mean())
        r[f"{m}_bias"] = e.mean()
        if f"{m}_true_min" in df:
            r[f"{m}_MAE_true_min"] = (df[f"{m}_true_min"] - df["Points"]).abs().mean()
        if m != "M0":  # does the model's deviation from the average point the right way (over/under)?
            dev = df[m] - df["M0"]
            sel = dev.abs() >= 1.5
            hit = np.sign(df.loc[sel, "Points"] - df.loc[sel, "M0"]) == np.sign(dev[sel])
            r[f"{m}_direction_%"] = hit.mean() * 100 if sel.any() else np.nan
            r[f"{m}_direction_n"] = int(sel.sum())
    return r


def part2_matchups(box, tg, frac) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    print("\n═══ PART 2: matchup model ═══")
    pg = build_player_games(box, tg)
    cuts = season_cuts(tg, frac)

    # roles from the FULL season profile (used for tables and the defender analysis)
    roles_full = assign_roles(player_profiles(pg))
    role_counts = roles_full[roles_full["Min"] >= 150].groupby("Role").size()
    print("Role distribution (player-seasons, >= 150 min):", role_counts.to_dict())
    names = pg.groupby("Player_ID")["Player"].last()
    rf = roles_full.assign(player=roles_full["Player_ID"].map(names))
    save(rf[["Season", "Player_ID", "player", "Role", "Min"] + FEATS].round(2),
         "2_player_roles", "2_roles")

    all_test, metrics, dvp_tables, glm_tables = [], [], [], []
    for s in sorted(pg["Season"].unique()):
        d = pg[pg["Season"] == s]
        tr, te = d[d["Gamecode"] <= cuts[s]].copy(), d[d["Gamecode"] > cuts[s]].copy()
        if tr["Gamecode"].nunique() < 40 or te.empty:
            print(f"  season {s}: too few games for a test - skipping")
            continue
        # roles for the test come from the training part only (no information from the future)
        r_tr = assign_roles(player_profiles(tr)).set_index("Player_ID")["Role"]
        tr["Role"] = tr["Player_ID"].map(r_tr)
        te["Role"] = te["Player_ID"].map(r_tr)

        rm = RatioModel().fit(tr)
        glm = GLMModel().fit(tr)

        ok = te["Player_ID"].map(rm.players["games"]).fillna(0) >= 3
        te = te[ok & te["Role"].notna()].copy()
        pr = rm.predict(te)
        pr_true = rm.predict(te, minutes=te["Min"])
        te = te.join(pr)
        for m in ("M0", "M1", "M2"):
            te[f"{m}_true_min"] = pr_true[m]
        rate = glm.rate(te)
        te["M3"] = rate * te["Player_ID"].map(rm.players["exp_min"])
        te["M3_true_min"] = rate * te["Min"]

        met = evaluate(te, ["M0", "M1", "M2", "M3"])
        met.update(Season=s, train_games=tr["Gamecode"].nunique(),
                   test_games=te["Gamecode"].nunique(), glm_alpha=glm.alpha,
                   glm_scale=glm.scale, ratio_K=rm.K)
        metrics.append(met)
        all_test.append(te)

        # tables from the FULL season (for interpretation, not for the test)
        d = d.assign(Role=d["Player_ID"].map(roles_full[roles_full["Season"] == s]
                                             .set_index("Player_ID")["Role"]))
        rm_full = RatioModel(K=rm.K).fit(d)
        t = rm_full.dvp.reset_index().assign(Season=s)
        dvp_tables.append(t)
        g = GLMModel(alpha=glm.alpha, scale=glm.scale).fit(d).opp_role_factors().assign(Season=s)
        glm_tables.append(g)
        print(f"  season {s}: train {met['train_games']} games, test {met['test_games']} games, "
              f"MAE M0={met['M0_MAE']:.2f} M2={met['M2_MAE']:.2f} M3={met['M3_MAE']:.2f}")

    if not metrics:
        print("Not enough data for part 2.")
        return pd.DataFrame(), None

    met = pd.DataFrame(metrics)
    test = pd.concat(all_test)
    overall = evaluate(test, ["M0", "M1", "M2", "M3"])
    overall.update(Season="ALL")
    met = pd.concat([met, pd.DataFrame([overall])], ignore_index=True)
    first = ["Season", "n", "train_games", "test_games", "ratio_K", "glm_alpha", "glm_scale"]
    met = met[[c for c in first if c in met] + [c for c in met if c not in first]]
    save(met.round(3), "2_model_test", "2_test")

    keep = ["Season", "Gamecode", "Player_ID", "Player", "Team", "Opp", "Home", "Role", "Min",
            "Points", "M0", "M1", "M2", "M3", "dvp_factor"]
    save(test[keep].round(2), "2_test_predictions", "2_predictions")

    dvp = pd.concat(dvp_tables).merge(pd.concat(glm_tables), on=["Season", "Opp", "Role"], how="left")
    dvp = dvp.rename(columns={"Opp": "defending_team", "act": "pts_allowed", "exp": "pts_expected",
                              "n": "player_games", "factor": "dvp_factor"})
    dvp["Role"] = pd.Categorical(dvp["Role"], ROLES, ordered=True)
    dvp = dvp.sort_values(["Season", "defending_team", "Role"])
    save(dvp.round(3), "2_defense_vs_role", "2_defense_vs_role")

    last = dvp["Season"].max()
    piv = dvp[dvp["Season"] == last].pivot(index="defending_team", columns="Role", values="glm_factor")
    print(f"\nSeason {last}: how much each team allows to each role "
          f"(1.00 = league average, 1.15 = +15 %, GLM factor)")
    print(piv.round(2).to_string())
    print("\nModel comparison on the test set (MAE - mean absolute error in points, lower = better):")
    cols = ["Season", "n"] + [f"M{i}_MAE" for i in range(4)] + ["M2_direction_%", "M3_direction_%", "M3_direction_n"]
    print(met[[c for c in cols if c in met]].round(2).to_string(index=False))
    print("Same error if minutes were known in advance (shows how much error comes from minutes):")
    cols = ["Season"] + [f"M{i}_MAE_true_min" for i in range(4)]
    print(met[[c for c in cols if c in met]].round(2).to_string(index=False))
    return roles_full, dvp


# ──────────────── defender on/off from play-by-play with lineups ────────────────
def parse_lineup(v) -> list[str]:
    if isinstance(v, (list, tuple, np.ndarray)):
        return [str(x).strip() for x in v]
    if isinstance(v, str) and v.strip().startswith("["):
        try:
            return [str(x).strip() for x in ast.literal_eval(v)]
        except (ValueError, SyntaxError):
            return []
    return []


def defender_onoff(box, roles_full, min_on=150.0) -> pd.DataFrame | None:
    files = season_files("pbp_lineups")
    if not files:
        print("\n(No data/pbp_lineups_*.* - individual defender analysis skipped. "
              "Download with: python src/euroleague_download.py --no-season --no-boxscore --pbp)")
        return None
    print("\n═══ Individual defenders: opponent points by role with the defender on / off ═══")
    pl = box[~box["Player_ID"].isin(NOT_PLAYERS)]
    ids_by_name = pl.groupby(["Season", "Player"])["Player_ID"].first().to_dict()
    home_team = box[box["Home"] == 1].groupby(["Season", "Gamecode"])["Team"].first().to_dict()
    away_team = box[box["Home"] == 0].groupby(["Season", "Gamecode"])["Team"].first().to_dict()
    role_map = roles_full.set_index(["Season", "Player_ID"])["Role"].to_dict()

    time_parts, pts_parts = [], []
    for s, f in files.items():
        p = read_any(f)
        need = ["Gamecode", "PERIOD", "MARKERTIME", "PLAYTYPE", "CODETEAM", "Lineup_A", "Lineup_B"]
        miss = [c for c in need if c not in p.columns]
        if miss:
            print(f"  ! {f.name}: missing {miss}, skipping")
            continue
        p = p.reset_index(drop=True)
        p["Season"] = s
        p["Gamecode"] = pd.to_numeric(p["Gamecode"], errors="coerce").astype(int)
        p["PERIOD"] = pd.to_numeric(p["PERIOD"], errors="coerce").fillna(1).astype(int)
        p["CODETEAM"] = p["CODETEAM"].fillna("").astype(str).str.strip()
        p["PLAYTYPE"] = p["PLAYTYPE"].fillna("").astype(str).str.strip()
        plen = np.where(p["PERIOD"] <= 4, 600, 300)
        rem = p["MARKERTIME"].map(clock_to_sec)
        rem = rem.groupby([p["Gamecode"], p["PERIOD"]]).transform(lambda x: x.ffill().fillna(x.max()))
        p["el"] = (plen - rem.fillna(pd.Series(plen))).clip(0, plen)
        nxt = p.groupby(["Gamecode", "PERIOD"])["el"].shift(-1)
        p["dur"] = (nxt.fillna(pd.Series(plen, index=p.index)) - p["el"]).clip(lower=0)
        p["A"] = p["Lineup_A"].map(parse_lineup)
        p["B"] = p["Lineup_B"].map(parse_lineup)
        p["home"] = [home_team.get((s, g)) for g in p["Gamecode"]]
        p["away"] = [away_team.get((s, g)) for g in p["Gamecode"]]
        p = p.dropna(subset=["home", "away"])

        # time on court
        for side, team_col in (("A", "home"), ("B", "away")):
            t = p[["Season", "Gamecode", team_col, side, "dur"]].rename(columns={team_col: "Team", side: "nm"})
            t = t.explode("nm").dropna(subset=["nm"])
            time_parts.append(t.groupby(["Season", "Gamecode", "Team", "nm"], as_index=False)["dur"].sum())

        # points scored: who scored (role) and who was defending at the time
        p["pts"] = p["PLAYTYPE"].map({"2FGM": 2, "3FGM": 3, "FTM": 1}).fillna(0)
        sc = p[p["pts"] > 0].copy()
        if "PLAYER_ID" in sc.columns:
            sc["sid"] = sc["PLAYER_ID"].astype(str).str.strip()
        else:
            sc["sid"] = [ids_by_name.get((s, str(n).strip())) for n in sc["PLAYER"]]
        sc["role"] = [role_map.get((s, i)) for i in sc["sid"]]
        sc = sc.dropna(subset=["role"])
        home_scored = sc["CODETEAM"] == sc["home"]
        sc["def_team"] = np.where(home_scored, sc["away"], sc["home"])
        sc["def_lineup"] = np.where(home_scored, sc["B"], sc["A"])
        e = sc[["Season", "Gamecode", "def_team", "def_lineup", "role", "pts"]].explode("def_lineup")
        e = e.rename(columns={"def_team": "Team", "def_lineup": "nm"}).dropna(subset=["nm"])
        pts_parts.append(e.groupby(["Season", "Gamecode", "Team", "nm", "role"], as_index=False)["pts"].sum())
        # team total in those games (all points against the team, by role)
        tt = sc.groupby(["Season", "Gamecode", "def_team", "role"], as_index=False)["pts"].sum()
        tt = tt.rename(columns={"def_team": "Team"})
        tt["nm"] = "__TEAM__"
        pts_parts.append(tt)
        print(f"  {f.name}: {p['Gamecode'].nunique()} games")

    if not time_parts:
        return None
    T = pd.concat(time_parts)
    P = pd.concat(pts_parts)
    team_time = (T.groupby(["Season", "Gamecode", "Team"])["dur"].sum() / 5).rename("team_dur")
    team_pts = P[P["nm"] == "__TEAM__"].set_index(["Season", "Gamecode", "Team", "role"])["pts"]
    P = P[P["nm"] != "__TEAM__"]

    on_t = T.groupby(["Season", "Gamecode", "Team", "nm"])["dur"].sum().rename("on_dur").reset_index()
    on_t = on_t[on_t["on_dur"] > 0].merge(team_time.reset_index(), on=["Season", "Gamecode", "Team"])
    rows = []
    for role in ROLES:
        on_p = P[P["role"] == role].set_index(["Season", "Gamecode", "Team", "nm"])["pts"]
        x = on_t.copy()
        x["role"] = role
        x["on_pts"] = on_p.reindex(pd.MultiIndex.from_frame(x[["Season", "Gamecode", "Team", "nm"]])).fillna(0).values
        x["team_pts"] = team_pts.reindex(pd.MultiIndex.from_frame(
            x[["Season", "Gamecode", "Team", "role"]])).fillna(0).values
        rows.append(x)
    X = pd.concat(rows)
    agg = X.groupby(["Season", "Team", "nm", "role"]).agg(
        on_sec=("on_dur", "sum"), team_sec=("team_dur", "sum"),
        on_pts=("on_pts", "sum"), team_pts=("team_pts", "sum"),
        games=("Gamecode", "nunique")).reset_index()
    agg["off_sec"] = agg["team_sec"] - agg["on_sec"]
    agg["off_pts"] = agg["team_pts"] - agg["on_pts"]
    agg["on_min"] = agg["on_sec"] / 60
    agg["off_min"] = agg["off_sec"] / 60
    agg = agg[(agg["on_min"] >= min_on) & (agg["off_min"] >= 60)].copy()
    agg["per40_on"] = agg["on_pts"] / agg["on_sec"] * 2400
    agg["per40_off"] = agg["off_pts"] / agg["off_sec"] * 2400
    agg["diff"] = agg["per40_on"] - agg["per40_off"]
    # approximate z: how many standard deviations away from what the "off" rate predicts
    # (÷1.5 because points come in 2s and 3s, not 1s - more spread than Poisson)
    exp_on = agg["off_pts"] / agg["off_sec"] * agg["on_sec"]
    agg["z"] = (agg["on_pts"] - exp_on) / np.sqrt(exp_on.clip(lower=1)) / 1.5
    # shrinkage: the fewer "off" minutes, the less we trust the difference
    agg["diff_shrunk"] = agg["diff"] * agg["off_min"] / (agg["off_min"] + 300)
    agg["Player_ID"] = [ids_by_name.get((s, n)) for s, n in zip(agg["Season"], agg["nm"])]
    agg["defender_role"] = [role_map.get((s, i)) for s, i in zip(agg["Season"], agg["Player_ID"])]
    out = agg.rename(columns={"nm": "defender", "role": "opponent_role", "Team": "team"})[
        ["Season", "team", "defender", "defender_role", "opponent_role", "games", "on_min", "off_min",
         "on_pts", "off_pts", "per40_on", "per40_off", "diff", "diff_shrunk", "z"]]
    out = out.sort_values(["Season", "opponent_role", "diff_shrunk"])
    save(out.round(2), "2_defenders_on_off", "2_defenders")

    last = out["Season"].max()
    for role in ("C", "SF", "PG"):
        sub = out[(out["Season"] == last) & (out["opponent_role"] == role)]
        print(f"\nSeason {last}: defenders with whom opposing {role} score the LEAST (per 40 min)")
        print(sub.head(8)[["defender", "team", "defender_role", "on_min", "per40_on",
                           "per40_off", "diff_shrunk", "z"]].round(1).to_string(index=False))
    print("(The table has thousands of pairs, so a few z ≈ -3 appear by chance.\n"
          " Trust defenders whose z < -2 repeats across several seasons.)")
    return out


# ───────────────────────────── prediction ─────────────────────────────
def predict_cli(box, tg, name, opp, home, season, defenders):
    pg = build_player_games(box, tg)
    if season is None:
        counts = pg.groupby("Season")["Gamecode"].nunique()
        ok = counts[counts >= 60]
        season = int(ok.index.max() if len(ok) else counts.index.max())
    d = pg[pg["Season"] == season].copy()
    roles = assign_roles(player_profiles(d)).set_index("Player_ID")["Role"]
    d["Role"] = d["Player_ID"].map(roles)

    cand = d[d["Player"].str.contains(name, case=False, na=False)]["Player"].unique()
    if len(cand) == 0:
        sys.exit(f"Player '{name}' not found in season {season}.")
    if len(cand) > 1:
        print(f"Several found: {list(cand)} - using the first. Narrow down --predict.")
    player = cand[0]
    pid = d.loc[d["Player"] == player, "Player_ID"].iloc[0]
    team = d.loc[d["Player_ID"] == pid].sort_values("Gamecode")["Team"].iloc[-1]
    opp = opp.upper()
    teams = sorted(d["Team"].unique())
    if opp not in teams:
        sys.exit(f"Team code '{opp}' not found. Available: {teams}")

    # parameters from the part 2 test (if already run), so tables and prediction agree
    K = alpha = scale = None
    tf = OUT / "2_model_test.csv"
    if tf.exists():
        t = pd.read_csv(tf)
        t = t[t["Season"].astype(str) != "ALL"]
        if {"ratio_K", "glm_alpha", "glm_scale"} <= set(t.columns) and len(t):
            K, alpha, scale = t["ratio_K"].median(), t["glm_alpha"].median(), t["glm_scale"].median()
    rm = RatioModel(K=K).fit(d)
    glm = GLMModel(alpha=alpha, scale=scale).fit(d)
    role = roles[pid]
    row = pd.DataFrame({"Player_ID": [pid], "Team": [team], "Opp": [opp], "Role": [role],
                        "Home": [1 if home else 0]})
    pr = rm.predict(row).iloc[0]
    rate = glm.rate(row).iloc[0]
    exp_min = rm.players.loc[pid, "exp_min"]
    hist = d[(d["Player_ID"] == pid) & (d["Opp"] == opp)][["Gamecode", "Home", "Min", "Points"]]

    print(f"\n═══ {player} ({team}, role {role}) vs {opp}, {'home' if home else 'away'}, season {season} ═══")
    print(f"Season average: {d.loc[d.Player_ID == pid, 'Points'].mean():.1f} pts, "
          f"expected minutes (last 10 games): {exp_min:.1f}")
    print(f"M0 average from minutes: {pr['M0']:.1f}")
    print(f"M2 with {opp} defense vs {role} (factor {pr['dvp_factor']:.2f}) and home court: {pr['M2']:.1f}")
    print(f"M3 Poisson model: {rate * exp_min:.1f}")
    gf = glm.opp_role_factors()
    gf = gf[gf["Role"] == role].set_index("Opp")["glm_factor"].sort_values()
    print(f"{opp} defense vs {role} (GLM factor): {gf.get(opp, 1.0):.2f}")
    print(f"\nTeams by how much they allow the {role} role (1.00 = average, lower = tougher):")
    print(gf.round(2).to_string())
    if not hist.empty:
        print(f"\nThis season's games vs {opp}:\n{hist.round(1).to_string(index=False)}")
    if defenders is not None and not defenders.empty:
        dd = defenders[(defenders["Season"] == season) & (defenders["team"] == opp)
                       & (defenders["opponent_role"] == role)]
        if not dd.empty:
            print(f"\n{opp} defenders vs {role} (negative difference = {role} scores less with him on court):")
            print(dd.sort_values("on_min", ascending=False)[
                ["defender", "defender_role", "on_min", "per40_on", "per40_off",
                 "diff_shrunk", "z"]].head(8).round(1).to_string(index=False))


# ───────────────────────────── main ─────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-minutes", type=float, default=20, help="minimum average minutes (part 1)")
    ap.add_argument("--min-games", type=int, default=8, help="minimum games played (part 1)")
    ap.add_argument("--train-frac", type=float, default=0.8, help="training share within each season")
    ap.add_argument("--seasons", type=int, nargs="*", help="e.g. --seasons 2023 2024 2025")
    ap.add_argument("--only", choices=["1", "2"], help="run one part only")
    ap.add_argument("--predict", help="part of a player's name, e.g. sloukas")
    ap.add_argument("--vs", help="opponent code, e.g. OLY")
    ap.add_argument("--home", action="store_true", help="the player's team plays at home")
    ap.add_argument("--season", type=int, help="season for the prediction (default: last full season)")
    a, _ = ap.parse_known_args()

    box = load_boxscores(a.seasons)
    tg = team_games(box)

    if a.predict:
        if not a.vs:
            sys.exit("Give an opponent: --vs OLY")
        defenders = None
        f = OUT / "2_defenders_on_off.csv"
        if f.exists():
            defenders = pd.read_csv(f)
        predict_cli(box, tg, a.predict, a.vs, a.home, a.season, defenders)
        return

    if a.only in (None, "1"):
        part1_impact(box, tg, a.min_minutes, a.min_games)
    if a.only in (None, "2"):
        roles_full, _ = part2_matchups(box, tg, a.train_frac)
        if roles_full is not None and len(roles_full):
            defender_onoff(box, roles_full)
    write_xlsx()
    print("\nDone. Results are in output/.")


if __name__ == "__main__":
    main()
