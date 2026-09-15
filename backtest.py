"""
Training/backtest mode: reports honest, race-grouped performance instead
of the misleading ~0.99 AUC a naive random k-fold split produces (horses
from the same race leaking between train and test).

Run this any time to check where the model actually stands:
    python backtest.py

Only rows with a real Race_Date (added going forward, from this point in
the project onward) are used for the grouped comparison — older rows
predate that column and are reported separately, not silently mixed in
with a fake grouping.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

from main import FEATURE_COLS, HISTORICAL_DB_PATH


def make_model():
    # Same regularized config as main.py's live predictions, so backtest
    # numbers actually reflect what gets deployed.
    return RandomForestClassifier(
        n_estimators=300, random_state=42, max_depth=5,
        min_samples_leaf=8, max_features='sqrt'
    )


def main():
    db = pd.read_csv(HISTORICAL_DB_PATH)
    print(f"Total rows in database: {len(db)}")

    has_race_date = db["Race_Date"].notna() & (db["Race_Date"] != "")
    dated = db[has_race_date]
    undated = db[~has_race_date]
    print(f"Rows with a real Race_Date (usable for grouped CV): {len(dated)}")
    print(f"Rows without one (older data, reported separately, not mixed in): {len(undated)}")

    if dated["Race_Date"].nunique() < 5:
        print("\nNot enough distinct dated races yet for a meaningful grouped "
              "backtest (need several dozen races minimum). Re-run this once "
              "the weekly backfill and daily results collection have built "
              "up more Race_Date-tagged rows.")
        return

    X = dated[FEATURE_COLS]
    y = dated["Is_Winner"]
    groups = dated["Race_Date"]
    n_splits = min(5, dated["Race_Date"].nunique())

    gkf = GroupKFold(n_splits=n_splits)
    model_aucs, baseline_aucs = [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        model = make_model()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict_proba(X.iloc[test_idx])[:, 1]
        model_aucs.append(roc_auc_score(y.iloc[test_idx], preds))
        baseline_aucs.append(roc_auc_score(y.iloc[test_idx], X.iloc[test_idx]["Market_Prob"]))

    print(f"\n--- Race-grouped {n_splits}-fold backtest (dated rows only, "
          f"{dated['Race_Date'].nunique()} distinct races) ---")
    print(f"Model ROC-AUC:         {[round(a,3) for a in model_aucs]} | mean: {round(np.mean(model_aucs),3)}")
    print(f"Market-only baseline:  {[round(a,3) for a in baseline_aucs]} | mean: {round(np.mean(baseline_aucs),3)}")

    diff = np.mean(model_aucs) - np.mean(baseline_aucs)
    if diff > 0.02:
        print(f"\nModel beats the market-odds baseline by {round(diff,3)} AUC on this sample.")
    elif diff < -0.02:
        print(f"\nModel is BEHIND the market-odds baseline by {round(-diff,3)} AUC on this sample.")
    else:
        print(f"\nModel is roughly at parity with the market-odds baseline (diff: {round(diff,3)}). "
              f"No demonstrated edge yet on this sample size.")

    print("\nReminder: with a modest number of distinct races, these numbers "
          "carry real uncertainty. Treat direction (better/worse/parity) as "
          "more informative than the exact decimal, and re-run periodically "
          "as more real race data accumulates.")


if __name__ == "__main__":
    main()
