"""
Self-knowledge for the bot: how good it really is, and how sure it is about TODAY's race.
Everything is computed from the repository's own data every run.
`python assessment.py` prints the full self-report.
"""
import os
import numpy as np
import pandas as pd

from ticket_math import (race_weights, top_k_marginals, prob_overlap_at_least,
                         prob_exact_set, prob_exact_order, break_even_multiple, POWER)

SCORECARD_PATH = os.path.join("predictions", "scorecard.csv")
DB_PATH = "real_history_db.csv"


def labeled_races(db):
    D = db[db["Race_Date"].notna() & (db["Finish_Position"] >= 0)].copy()
    D["Market_Prob"] = D["Market_Prob"].astype(float)
    return D


def calibration_report(db):
    """For every stored race: rank by odds, take the top 3 as 'the picks', predict how likely each
    ticket type was, and compare with what really happened."""
    D = labeled_races(db)
    rows = []
    for rd, g in D.groupby("Race_Date"):
        g = g.sort_values("Market_Prob", ascending=False).reset_index(drop=True)
        w = race_weights(g["Market_Prob"].values)
        fp = g["Finish_Position"].values
        in_top3 = (fp[:3] > 0) & (fp[:3] <= 3)
        rows.append({
            "Race_Date": rd,
            "n": len(g),
            "attele": int(g["Discipline_Attele"].iloc[0]),
            "pred_ge2": prob_overlap_at_least(w, [0, 1, 2], 3, 2),
            "pred_set3": prob_exact_set(w, [0, 1, 2]),
            "pred_exact3": prob_exact_order(w, [0, 1, 2]),
            "act_ge2": int(in_top3.sum() >= 2),
            "act_set3": int(in_top3.sum() == 3),
            "act_exact3": int((fp[:3] == np.array([1, 2, 3])).all()),
            "act_hits": int(in_top3.sum()),
        })
    R = pd.DataFrame(rows)
    out = {"n_races": len(R), "races": R}
    if R.empty:
        return out
    R["bucket"] = pd.qcut(R["pred_ge2"], 4, labels=False, duplicates="drop")
    out["buckets"] = R.groupby("bucket").agg(
        n=("Race_Date", "size"), pred_ge2=("pred_ge2", "mean"), act_ge2=("act_ge2", "mean"),
        pred_set3=("pred_set3", "mean"), act_set3=("act_set3", "mean"), avg_hits=("act_hits", "mean"))
    out["cuts"] = np.quantile(R["pred_ge2"], [0.25, 0.75])
    out["summary"] = {k: float(R[k].mean()) for k in
                      ["pred_ge2", "act_ge2", "pred_set3", "act_set3", "pred_exact3", "act_exact3"]}
    out["summary"]["avg_hits"] = float(R["act_hits"].mean())
    return out


def track_record(path=SCORECARD_PATH, last=30):
    """Live record of published picks vs raw odds vs the experimental rule-based method."""
    if not os.path.exists(path):
        return None
    sc = pd.read_csv(path).tail(last)
    if sc.empty:
        return None
    baseline = float(np.mean(9.0 / sc["runners"]))   # expected hits of 3 random picks: 3*3/n
    rec = {"n": len(sc), "random": baseline}
    for k in ["model", "market", "benter", "challenger"]:
        col = f"{k}_top3_hits"
        if col in sc.columns and sc[col].notna().any():
            rec[k] = float(sc[col].mean())
    return rec


def data_health(df, header_n):
    """Sanity checks on today's parsed race. Returns a list of warnings (empty = clean)."""
    warns = []
    n = len(df)
    if header_n and header_n - n > 3:
        warns.append(f"parsed {n} runners but the programme header says {header_n}")
    over = float((1.0 / df["Odds"].astype(float)).sum())
    if not (0.75 <= over <= 1.40):
        warns.append(f"odds look wrong (implied probabilities add to {over:.2f}, normal is ~0.8-1.3)")
    if n < 6:
        warns.append(f"only {n} runners parsed")
    return warns


def assess_today(df, header_n, calib):
    """df: today's runners sorted best-first by odds-based strength. Returns an assessment dict."""
    w = race_weights(df["Market_Prob"].values)
    picks = [0, 1, 2]
    a = {
        "n": len(df),
        "warnings": data_health(df, header_n),
        "marginals": top_k_marginals(w, 3),
        "p_ge2": prob_overlap_at_least(w, picks, 3, 2),
        "p_set3": prob_exact_set(w, picks),
        "p_exact3": prob_exact_order(w, picks),
    }
    if len(df) >= 4:
        a["p_set4"] = prob_exact_set(w, [0, 1, 2, 3])
    cuts = calib.get("cuts") if calib else None
    if cuts is not None:
        lo, hi = cuts
        a["verdict"] = "STRONGER" if a["p_ge2"] >= hi else ("WEAKER" if a["p_ge2"] <= lo else "NORMAL")
        R = calib["races"]
        a["percentile"] = float((R["pred_ge2"] < a["p_ge2"]).mean() * 100)
    else:
        a["verdict"] = "UNKNOWN"
    return a


STAKE_MULT = {"STRONGER": "x1.0", "NORMAL": "x0.5", "WEAKER": "x0 (pass) or minimum", "UNKNOWN": "x0.5"}


def format_assessment(a, calib, record):
    L = []
    if a["warnings"]:
        L.append("DATA WARNING - do not trust this card until checked:")
        for w_ in a["warnings"]:
            L.append(f"  - {w_}")
        L.append("")
    v = a["verdict"]
    L.append(f"RACE QUALITY: {v}" + (f" (more predictable than {a['percentile']:.0f}% of the {calib['n_races']} races I have on file)"
                                   if "percentile" in a else ""))
    L.append(f"Suggested stake vs your normal unit: {STAKE_MULT[v]}  (risk control, not a proven edge)")
    L.append("")
    L.append("WHAT MY TICKETS ARE WORTH (honest odds):")
    L.append(f"- Tierce desordre on my top 3: {a['p_set3']*100:.1f}% -> needs a rapport of at least {break_even_multiple(a['p_set3']):.0f}x the stake to break even")
    L.append(f"- At least 2 of my top 3 in the first 3: {a['p_ge2']*100:.0f}%")
    L.append(f"- Exact order of my top 3: {a['p_exact3']*100:.2f}% (about 1 time in {1/max(a['p_exact3'],1e-9):.0f})")
    if "p_set4" in a:
        L.append(f"- Quarte desordre on my top 4: {a['p_set4']*100:.1f}% -> break-even rapport {break_even_multiple(a['p_set4']):.0f}x")
    L.append("")
    L.append("WHAT I AM: the odds ranked and turned into real ticket probabilities. On every stored race, no extra signal")
    L.append("(form, draw, shoeing, driver, trainer, age) has beaten the odds beyond noise, so I do not pretend to.")
    if calib and calib.get("summary"):
        s = calib["summary"]
        L.append(f"Reality check over {calib['n_races']} races: top-3 picks averaged {s['avg_hits']:.2f}/3 correct; "
                 f"tierce desordre hit {s['act_set3']*100:.1f}% (I predicted {s['pred_set3']*100:.1f}%); "
                 f"exact order {s['act_exact3']*100:.1f}%.")
    if record:
        r = record
        parts = [f"picks {r.get('model', float('nan')):.2f}", f"raw odds {r.get('market', float('nan')):.2f}"]
        if "benter" in r:
            parts.append(f"rule-based Benter {r['benter']:.2f}")
        L.append(f"Live record, last {r['n']} published races (top-3 hits out of 3; pure luck = {r['random']:.2f}): " + ", ".join(parts) + ".")
    return "\n".join(L)


def full_report(db):
    calib = calibration_report(db)
    lines = [f"=== SELF-REPORT: {calib['n_races']} stored races (Plackett-Luce, power={POWER}) ==="]
    if calib["n_races"]:
        s = calib["summary"]
        lines += [
            "Predicted vs actual, top-3 picks by odds:",
            f"  >=2 of 3 in top-3 : predicted {s['pred_ge2']*100:5.1f}%   actual {s['act_ge2']*100:5.1f}%",
            f"  tierce desordre   : predicted {s['pred_set3']*100:5.1f}%   actual {s['act_set3']*100:5.1f}%",
            f"  exact order top-3 : predicted {s['pred_exact3']*100:5.2f}%   actual {s['act_exact3']*100:5.2f}%",
            "By race-quality quartile:",
            calib["buckets"].round(3).to_string(),
        ]
    rec = track_record()
    if rec:
        lines.append(f"Live scorecard (last {rec['n']}): " + ", ".join(f"{k}={v:.2f}" for k, v in rec.items() if k != "n"))
    return "\n".join(lines)


if __name__ == "__main__":
    print(full_report(pd.read_csv(DB_PATH)))
