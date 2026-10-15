"""
Defense rating by position + 5 tiers + out-of-sample test
=========================================================

1. For every team and every position (PG/SG/SF/PF/C) a defense rating is computed:
   how much players of that position scored against the team, compared with
   what the same players usually score (in the same minutes).
   1.10 = allows 10 % more than usual, 0.90 = 10 % less.
2. For every position the teams are split into 5 tiers:
   1 = toughest defense ... 5 = softest.
   Method: --method quantiles (equal number of teams) or kmeans (natural groups).
3. Test, each season separately:
   tiers and player averages come from the first 80 % of games,
   they are checked on the last 20 %.
   For every tier: how much players scored compared with their own average,
   with a 95 % confidence interval, and how often they beat their average.
   The same is shown on the training data - the difference between the two shows
   how much of the "effect" is real and how much is noise.

Run (from the repository root):
  python src/euroleague_tiers.py
  python src/euroleague_tiers.py --method kmeans --train-frac 0.8
  python src/euroleague_tiers.py --ranking-season 2025    # rating table for a given season

Results: output/3_*.csv, output/3_tiers.xlsx, output/3_tiers.png (if matplotlib is installed)
"""
import argparse
import sys

import numpy as np
import pandas as pd

import euroleague_analysis as ea  # reuse the same data and role functions

ROLES = ea.ROLES
OUT = ea.OUT
K_PRIOR = 60.0     # shrinkage of the rating (in points), so 2-3 games do not distort it
PRIOR_MIN = 100.0  # shrinkage of player points/min toward the role average
SHEETS: dict[str, pd.DataFrame] = {}


def save(df, name):
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8-sig")
    SHEETS[name.replace("3_", "")[:31]] = df
    print(f"  ✓ output/{name}.csv  ({len(df)} rows)")


# ───────────────────────── rating and tiers ─────────────────────────
def player_rates(d: pd.DataFrame) -> pd.DataFrame:
    """Player points per minute (shrunk toward the role average) and points average."""
    role_ppm = d.groupby("Role")["Points"].sum() / d.groupby("Role")["Min"].sum()
    p = d.groupby("Player_ID").agg(pts=("Points", "sum"), mins=("Min", "sum"),
                                   games=("Points", "size"), avg_pts=("Points", "mean"),
                                   Role=("Role", "last"))
    p["ppm"] = (p["pts"] + PRIOR_MIN * p["Role"].map(role_ppm)) / (p["mins"] + PRIOR_MIN)
    return p


def defense_ratings(d: pd.DataFrame, rates: pd.DataFrame) -> pd.DataFrame:
    d = d.assign(exp=d["Player_ID"].map(rates["ppm"]) * d["Min"])
    r = d.groupby(["Opp", "Role"]).agg(allowed=("Points", "sum"), expected=("exp", "sum"),
                                       player_games=("Points", "size"),
                                       games=("Gamecode", "nunique")).reset_index()
    r["rating"] = (r["allowed"] + K_PRIOR) / (r["expected"] + K_PRIOR)
    r["pts_per_game_diff"] = (r["allowed"] - r["expected"]) / r["games"]
    return r


def make_tiers(r: pd.DataFrame, method: str) -> pd.DataFrame:
    """For every role: 1 = allows the least ... 5 = allows the most."""
    r = r.copy()
    r["tier"] = 0
    for role, g in r.groupby("Role"):
        x = g["rating"]
        if method == "kmeans" and len(g) >= 5:
            from sklearn.cluster import KMeans
            km = KMeans(5, n_init=20, random_state=0).fit(x.values.reshape(-1, 1))
            order = np.argsort(km.cluster_centers_.ravel())
            rank = {c: i + 1 for i, c in enumerate(order)}
            r.loc[g.index, "tier"] = [rank[c] for c in km.labels_]
        else:
            r.loc[g.index, "tier"] = pd.qcut(x.rank(method="first"), 5, labels=False) + 1
    return r


# ───────────────────────── evaluation by tier ─────────────────────────
def tier_summary(rows: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """rows contain: Points, exp_pts (from real minutes), avg_pts (player average), tier."""
    def f(g):
        resid = g["Points"] - g["exp_pts"]
        n = len(g)
        se = resid.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
        return pd.Series({
            "n": n,
            "scored_avg": g["Points"].mean(),
            "expected_avg": g["exp_pts"].mean(),
            "ratio": g["Points"].sum() / g["exp_pts"].sum(),
            "diff_pts": resid.mean(),
            "ci95_lo": resid.mean() - 1.96 * se,
            "ci95_hi": resid.mean() + 1.96 * se,
            "above_own_avg_pct": (g["Points"] > g["avg_pts"]).mean() * 100,
        })
    return rows.groupby(by).apply(f).reset_index()


def trend_test(rows: pd.DataFrame) -> dict:
    """Do points (relative to expectation) rise from tier 1 to tier 5?"""
    resid = rows["Points"] - rows["exp_pts"]
    x = rows["tier"].astype(float)
    slope, se = np.nan, np.nan
    if x.nunique() > 1:
        X = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(X, resid, rcond=None)
        e = resid - X @ beta
        cov = np.linalg.inv(X.T @ X) * (e @ e) / (len(x) - 2)
        slope, se = beta[1], np.sqrt(cov[1, 1])
    out = {"pts_per_tier": slope, "z": slope / se if se else np.nan}
    try:
        from scipy.stats import norm, kruskal
        out["p_trend"] = 2 * (1 - norm.cdf(abs(out["z"])))
        groups = [resid[rows["tier"] == t] for t in range(1, 6) if (rows["tier"] == t).any()]
        out["p_kruskal"] = kruskal(*groups).pvalue if len(groups) > 1 else np.nan
    except ImportError:
        pass
    return out


def run(box, tg, frac, method, ranking_season):
    pg = ea.build_player_games(box, tg)
    cuts = ea.season_cuts(tg, frac)
    test_rows, train_rows = [], []

    for s in sorted(pg["Season"].unique()):
        d = pg[pg["Season"] == s]
        tr, te = d[d["Gamecode"] <= cuts[s]].copy(), d[d["Gamecode"] > cuts[s]].copy()
        if tr["Gamecode"].nunique() < 40 or te.empty:
            print(f"  season {s}: too few games - skipping the test")
            continue
        roles = ea.assign_roles(ea.player_profiles(tr)).set_index("Player_ID")["Role"]
        tr["Role"], te["Role"] = tr["Player_ID"].map(roles), te["Player_ID"].map(roles)
        rates = player_rates(tr)
        tiers = make_tiers(defense_ratings(tr, rates), method)
        tmap = tiers.set_index(["Opp", "Role"])["tier"]

        for part, df in (("train", tr), ("test", te)):
            df = df[df["Player_ID"].map(rates["games"]).fillna(0) >= 5].copy()
            df["exp_pts"] = df["Player_ID"].map(rates["ppm"]) * df["Min"]
            df["avg_pts"] = df["Player_ID"].map(rates["avg_pts"])
            df["tier"] = [tmap.get((o, r)) for o, r in zip(df["Opp"], df["Role"])]
            df = df.dropna(subset=["tier", "exp_pts"])
            df["tier"] = df["tier"].astype(int)
            (test_rows if part == "test" else train_rows).append(df)

    if not test_rows:
        sys.exit("Not enough data for the test.")
    test, train = pd.concat(test_rows), pd.concat(train_rows)

    # 1) Results by tier
    print("\n═══ Do players score differently against different defensive tiers? ═══")
    print("diff_pts = scored more(+)/less(-) on average than expected from his average and minutes")
    res = []
    for name, rows in (("TRAIN (80 %, same data the tiers were built on)", train),
                       ("TEST (20 %, new games)", test)):
        t = tier_summary(rows, ["tier"])
        tt = trend_test(rows)
        print(f"\n{name} - all seasons, all positions")
        print(t.round(2).to_string(index=False))
        p = tt.get("p_trend", np.nan)
        print("Trend: {:+.2f} pts per tier, p {}".format(
            tt["pts_per_tier"], "< 0.001" if p < 0.001 else f"= {p:.3f}"))
        res.append(t.assign(split=name.split()[0].lower()))
    save(pd.concat(res).round(3), "3_tiers_overall")

    t_role = tier_summary(test, ["Role", "tier"])
    t_role["Role"] = pd.Categorical(t_role["Role"], ROLES, ordered=True)
    t_role = t_role.sort_values(["Role", "tier"])
    save(t_role.round(3), "3_tiers_by_role_test")
    print("\nTEST by position (diff_pts):")
    piv = t_role.pivot(index="Role", columns="tier", values="diff_pts").round(2)
    piv["trend"] = [trend_test(test[test["Role"] == r])["pts_per_tier"] for r in piv.index]
    piv["p"] = [trend_test(test[test["Role"] == r]).get("p_trend", np.nan) for r in piv.index]
    print(piv.round(3).to_string())

    t_season = tier_summary(test, ["Season", "tier"])
    save(t_season.round(3), "3_tiers_by_season_test")
    print("\nTEST by season (diff_pts):")
    print(t_season.pivot(index="Season", columns="tier", values="diff_pts").round(2).to_string())

    # 2) Current rating table - from the whole selected season
    counts = pg.groupby("Season")["Gamecode"].nunique()
    if ranking_season is None:
        ok = counts[counts >= 60]
        ranking_season = int(ok.index.max() if len(ok) else counts.index.max())
    d = pg[pg["Season"] == ranking_season].copy()
    roles = ea.assign_roles(ea.player_profiles(d)).set_index("Player_ID")["Role"]
    d["Role"] = d["Player_ID"].map(roles)
    rt = make_tiers(defense_ratings(d, player_rates(d)), method)
    rt = rt.rename(columns={"Opp": "team"})
    rt["Role"] = pd.Categorical(rt["Role"], ROLES, ordered=True)
    rt = rt.sort_values(["Role", "rating"])
    save(rt.assign(Season=ranking_season).round(3), "3_defense_rating")
    print(f"\n═══ Defense rating, season {ranking_season} ({counts[ranking_season]} games) ═══")
    print("rating (tier): 1 = toughest defense, 5 = softest")
    lbl = rt.assign(x=rt["rating"].map("{:.2f}".format) + " (" + rt["tier"].astype(str) + ")")
    print(lbl.pivot(index="team", columns="Role", values="x").to_string())

    plot(test, train)
    write_xlsx()


def plot(test, train):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for rows, label, color in ((train, "Train (80 %)", "#9aa5b1"), (test, "Test (20 %)", "#1f5fbf")):
        t = tier_summary(rows, ["tier"])
        off = -0.08 if color == "#9aa5b1" else 0.08
        ax.errorbar(t["tier"] + off, t["diff_pts"],
                    yerr=[t["diff_pts"] - t["ci95_lo"], t["ci95_hi"] - t["diff_pts"]],
                    fmt="o-", capsize=4, label=label, color=color)
    ax.axhline(0, color="#444", lw=0.8)
    ax.set_xticks(range(1, 6))
    ax.set_xticklabels(["1\ntoughest", "2", "3", "4", "5\nsoftest"])
    ax.set_xlabel("Opponent defensive tier against the player's position")
    ax.set_ylabel("Scored more (+) / less (−) than expected, pts")
    ax.set_title("Does the opponent's defensive tier change a player's points?")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "3_tiers.png", dpi=150)
    print("  ✓ output/3_tiers.png")


def write_xlsx():
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return
    with pd.ExcelWriter(OUT / "3_tiers.xlsx") as xw:
        for name, df in SHEETS.items():
            df.to_excel(xw, sheet_name=name, index=False)
    print("  ✓ output/3_tiers.xlsx")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--method", choices=["quantiles", "kmeans"], default="quantiles")
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--ranking-season", type=int, help="season for the rating table (default: last full season)")
    ap.add_argument("--seasons", type=int, nargs="*", help="which seasons to use, e.g. 2022 2023 2024")
    a, _ = ap.parse_known_args()
    box = ea.load_boxscores(a.seasons)
    tg = ea.team_games(box)
    run(box, tg, a.train_frac, a.method, a.ranking_season)
    print("\nDone.")


if __name__ == "__main__":
    main()
