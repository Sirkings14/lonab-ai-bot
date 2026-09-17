"""
Independent "second opinion" scorer, implementing the Kingsley-Benter
method's eight weighted factors directly (rule-based), NOT via the
ML ensemble. This is deliberately a different lens on the same race:
where the ML model learns patterns from historical outcomes, this
scores each horse against Kingsley's own explicit heuristics. Showing
both side by side lets you see where they agree (higher confidence)
and where they diverge (worth a manual look before staking).

Factor weights (100pts total): form trajectory (25), course form (20),
trainer confidence signal (15), shoeing/equipment change (15),
draw/echelon position (10), jockey quality (10), Benter value gap (5),
surface fit (5).

Honesty note: course form and surface fit rely on keyword-matching the
horse's prose comment and the race's track name — there's no structured
per-track history in the data yet, so these two factors are necessarily
coarser than the others. They degrade to a neutral score rather than a
wrong one when no clear signal is found.
"""
import re


TRAINER_CONFIDENCE_MAX = [
    "tous les feux sont au vert", "c'est son année", "confiant", "confiante",
]
TRAINER_CONFIDENCE_ELIMINATE = [
    "c'est un test", "pas encore à 100%", "pas encore a 100%",
]

COURSE_POSITIVE = [
    "sur ce parcours", "sur cette piste", "remporté sur ce parcours",
    "recordman", "lauréat de la course clé",
]
COURSE_NEGATIVE = [
    "découvre", "decouvre", "n'a jamais couru", "n'a encore jamais couru",
]

# Track/distance-specific patterns from Kingsley's own post-mortems.
# Each entry: (track keyword, distance test, adjustment, note)
TRACK_PATTERNS = [
    ("LONGCHAMP", lambda d: d is not None and d >= 2400,
     -3, "Longchamp 2400m+: front-runners historically underperform"),
    ("DEAUVILLE", lambda d: d is not None and 1800 <= d <= 2000,
     2, "Deauville ~1900m PSF: rewards proven PSF/right-handed track form"),
]


def _parse_perf_positions(perf_str):
    """
    "3.T.2.8.5" -> [3, None, 2, 8, 5], most recent race first.
    Non-numeric codes (T=tombé/fell, D=disqualifié, A=arrêté) become None
    and are tracked separately for the steeplechase elimination filter.
    """
    if not perf_str:
        return [], 0
    parts = perf_str.split('.')
    positions = []
    fall_count = 0
    for p in parts:
        if p.isdigit():
            positions.append(int(p))
        else:
            positions.append(None)
            if p.upper() == 'T':
                fall_count += 1
    return positions, fall_count


def _form_trajectory_score(perf_str):
    """25pts max. Recency-weighted: most recent finishes matter more.
    A finish of 1st-3rd scores well; higher numbers or DNF codes score low."""
    positions, _ = _parse_perf_positions(perf_str)
    if not positions:
        return 12.5  # neutral — no form data available
    weights = [0.35, 0.25, 0.20, 0.12, 0.08][:len(positions)]
    total_weight = sum(weights)
    score = 0.0
    for pos, w in zip(positions, weights):
        if pos is None:
            continue  # fall/DNF contributes 0 to this race's slot
        # position 1 -> full marks, position 10+ -> near zero
        pos_score = max(0.0, 1.0 - (pos - 1) / 10.0)
        score += pos_score * w
    return round((score / total_weight) * 25, 2) if total_weight else 12.5


def _course_form_score(comment):
    """20pts max. Keyword-based — coarse by necessity (see module docstring)."""
    if not comment:
        return 10.0
    c = comment.lower()
    if any(kw in c for kw in COURSE_POSITIVE):
        return 20.0
    if any(kw in c for kw in COURSE_NEGATIVE):
        return 8.0
    return 12.0  # neutral — no clear course-specific signal


def _trainer_confidence_score(comment):
    """15pts max. Returns (score, eliminated: bool)."""
    if not comment:
        return 7.5, False
    c = comment.lower()
    if any(kw in c for kw in TRAINER_CONFIDENCE_ELIMINATE):
        return 0.0, True
    if any(kw in c for kw in TRAINER_CONFIDENCE_MAX):
        return 15.0, False
    return 7.5, False


def _shoeing_score(shoe_status):
    """15pts max. Shoe_Status: 0=no change, 1=partial change, 2=full change (D4)."""
    return {0: 5.0, 1: 10.0, 2: 15.0}.get(shoe_status, 5.0)


def _draw_score(discipline_attele, draw, autostart_pos, n_runners):
    """10pts max."""
    if discipline_attele:
        return 10.0 if autostart_pos == 1 else 3.0
    if draw is None or not n_runners:
        return 5.0
    return 10.0 if draw <= max(1, n_runners // 2) else 5.0


def _jockey_score(driver_rank):
    """10pts max. Driver_Rank is on a 1-3 scale (1.5 = unranked default)."""
    return round(min(driver_rank, 3.0) / 3.0 * 10, 2)


def _surface_fit_score(track_name, distance_m):
    """5pts max, adjusted by known track/distance patterns. Starts neutral (2.5)."""
    score = 2.5
    notes = []
    if track_name:
        for keyword, dist_test, adjustment, note in TRACK_PATTERNS:
            if keyword in track_name.upper() and dist_test(distance_m):
                score += adjustment
                notes.append(note)
    return max(0.0, min(5.0, score)), notes


def score_race(df, comments, track_name=None, distance_m=None):
    """
    df: the race's DataFrame from build_todays_dataframe (must include
        Num, Perf isn't currently carried through — see note below).
    comments: {horse_num: full_comment_text} from extract_horse_comments.
    Returns a new DataFrame with Benter_Score, Benter_Rank, Eliminated,
    and Elimination_Reason columns, sorted best-first.
    """
    n_runners = len(df)
    rows = []
    for _, r in df.iterrows():
        comment = comments.get(r["Num"], "")
        perf_str = r.get("Perf")  # populated if caller merged it in; see main.py wiring

        form = _form_trajectory_score(perf_str)
        course = _course_form_score(comment)
        confidence, confidence_eliminated = _trainer_confidence_score(comment)
        shoeing = _shoeing_score(r["Shoe_Status"])
        draw = _draw_score(r["Discipline_Attele"], r.get("Draw"), r["Autostart_Pos"], n_runners)
        jockey = _jockey_score(r["Driver_Rank"])
        surface, surface_notes = _surface_fit_score(track_name, distance_m)

        _, fall_count = _parse_perf_positions(perf_str)
        steeple_eliminated = fall_count >= 2

        raw_total = form + course + confidence + shoeing + draw + jockey + surface

        eliminated = confidence_eliminated or steeple_eliminated
        elimination_reason = (
            "Trainer signals elimination phrase" if confidence_eliminated else
            "2+ falls in recent form (steeplechase filter)" if steeple_eliminated else
            None
        )

        rows.append({
            "Num": r["Num"],
            "Horse": r["Horse"],
            "Form_Score": form,
            "Course_Score": course,
            "Confidence_Score": confidence,
            "Shoeing_Score": shoeing,
            "Draw_Score": draw,
            "Jockey_Score": jockey,
            "Surface_Score": surface,
            "Surface_Notes": "; ".join(surface_notes),
            "Benter_Score": 0.0 if eliminated else raw_total,
            "Eliminated": eliminated,
            "Elimination_Reason": elimination_reason,
        })

    import pandas as pd
    result = pd.DataFrame(rows)

    # Benter value gap (5pts): reward horses this method ranks meaningfully
    # better than the market does — capped so it can't rescue an eliminated
    # horse or dominate the rest of the score.
    if "Market_Rank_In_Race" in df.columns:
        result = result.merge(df[["Num", "Market_Rank_In_Race"]], on="Num", how="left")
        method_rank = result["Benter_Score"].rank(ascending=False, method="average") / n_runners
        value_gap = (result["Market_Rank_In_Race"] - method_rank).clip(lower=0)
        result["Value_Gap_Score"] = (value_gap * 5).round(2)
        result.loc[result["Eliminated"], "Value_Gap_Score"] = 0.0
        result["Benter_Score"] = result["Benter_Score"] + result["Value_Gap_Score"]

    result = result.sort_values("Benter_Score", ascending=False).reset_index(drop=True)
    result["Benter_Rank"] = range(1, len(result) + 1)
    return result
