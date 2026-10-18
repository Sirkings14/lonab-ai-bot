"""
Honest evaluation. Run any time:   python backtest.py

What it answers: does anything beat "just trust the odds"?

Method (chosen to avoid the traps that fooled us earlier):
- Whole races are held out together (a random row split leaks information
  between horses of the same race and produced a fake ~0.99 AUC once).
- One consistent target for every race: finished in the top 3.
- Repeated over several different shuffles of the races, so one lucky split
  can't decide the result.
- Differences are reported with a standard error. With few races, a gap
  smaller than ~2 standard errors is noise, however big it looks.
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from main import (FEATURE_COLS, CHAMPION_FEATURES, CHALLENGER_FEATURES,
                  HISTORICAL_DB_PATH)

warnings.filterwarnings("ignore")
N_SHUFFLES = 8
MIN_RACES = 8


def logistic(C=0.3):
    return make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))


def evaluate(D, candidates):
    """candidates: {name: feature list, or None for 'rank by raw odds'}.
    Returns {name: {'race_hits': Series (mean over shuffles, per race), 'auc': float}}."""
    races = D["Race_Date"].unique()
    n_splits = min(6, len(races))
    hits = {n: {r: [] for r in races} for n in candidates}
    aucs = {n: [] for n in candidates}
    for seed in range(N_SHUFFLES):
        order = {r: i for i, r in enumerate(np.random.RandomState(seed).permutation(races))}
        groups = D["Race_Date"].map(order)
        for tr, te in GroupKFold(n_splits=n_splits).split(D, groups=groups):
            train, test = D.iloc[tr], D.iloc[te].copy()
            for name, feats in candidates.items():
                if feats is None:
                    test["p"] = test["Market_Prob"]
                else:
                    model = logistic().fit(train[feats].astype(float), train["y"])
                    test["p"] = model.predict_proba(test[feats].astype(float))[:, 1]
                aucs[name].append(roc_auc_score(test["y"], test["p"]))
                for race, g in test.groupby("Race_Date"):
                    hits[name][race].append(g.nlargest(3, "p")["y"].sum())
    return {n: {"race_hits": pd.Series({r: np.mean(v) for r, v in hits[n].items()}),
                "auc": float(np.mean(aucs[n]))} for n in candidates}


def paired(a, b):
    """Mean and standard error of (a - b) across races."""
    d = a - b
    return d.mean(), d.std(ddof=1) / np.sqrt(len(d))


def main():
    db = pd.read_csv(HISTORICAL_DB_PATH)
    D = db[db["Race_Date"].notna() & (db["Finish_Position"] >= 0)].reset_index(drop=True)
    D["y"] = D["Finish_Position"].between(1, 3).astype(int)
    n_races = D["Race_Date"].nunique()
    print(f"{n_races} real races, {len(D)} runners (target: finished top 3)")
    if n_races < MIN_RACES:
        print(f"Need at least {MIN_RACES} races for this to mean anything. Run the rebuild/backfill first.")
        return
    print(f"(a random pick of 3 horses gets about {3 * 3 / D.groupby('Race_Date').size().mean():.2f} of 3 right)\n")

    named = {
        "raw odds ranking (no model)": None,
        "CHAMPION  (calibrated odds)": CHAMPION_FEATURES,
        "CHALLENGER (odds + extras)": CHALLENGER_FEATURES,
        "all 16 features": FEATURE_COLS,
    }
    named_res = evaluate(D, named)
    champ = named_res["CHAMPION  (calibrated odds)"]["race_hits"]

    print(f"{'candidate':30}{'top-3 hits/3':>14}{'AUC':>8}   vs champion (+/- std err)")
    for name, r in named_res.items():
        diff, se = paired(r["race_hits"], champ)
        tail = "" if name.startswith("CHAMPION") else f"{diff:+.3f} +/- {se:.3f}"
        print(f"{name:30}{r['race_hits'].mean():>14.3f}{r['auc']:>8.3f}   {tail}")

    # Does any single feature add anything on top of the odds?
    ablation = {f: ["Market_Prob", f] for f in FEATURE_COLS if f != "Market_Prob"}
    abl = evaluate(D, ablation)
    base_auc = named_res["CHAMPION  (calibrated odds)"]["auc"]
    print(f"\nOdds + one feature at a time (change vs odds alone; real only if > 2 std err):")
    print(f"{'feature':26}{'d hits':>9}{'+/- se':>9}{'d AUC':>9}")
    rows = []
    for f, r in abl.items():
        diff, se = paired(r["race_hits"], champ)
        rows.append((f, diff, se, r["auc"] - base_auc))
    for f, diff, se, dauc in sorted(rows, key=lambda x: -x[1]):
        flag = "  <-- adds signal" if diff > 2 * se else ("  <-- HURTS" if diff < -2 * se else "")
        print(f"{f:26}{diff:>+9.3f}{se:>9.3f}{dauc:>+9.4f}{flag}")

    diff, se = paired(named_res["CHALLENGER (odds + extras)"]["race_hits"], champ)
    print("\nVERDICT")
    if diff > 2 * se:
        print(f"Challenger beats the champion ({diff:+.3f} hits/race, > 2 std err). Worth promoting.")
    elif diff < -2 * se:
        print(f"Challenger is clearly WORSE than the champion ({diff:+.3f}). Extra features are adding noise.")
    else:
        print(f"No real difference yet ({diff:+.3f} +/- {se:.3f}). Keep the champion; the extra features have not earned a place.")
    print(f"Based on {n_races} races. More races shrink the error bars; this is the number to watch.")


if __name__ == "__main__":
    main()
