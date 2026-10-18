import os
import re
import urllib.parse
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import pdfplumber
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier

from pdf_parser import (parse_race_card, extract_horse_comments, chrono_to_speed_index,
                         extract_pdf_text_multi_strategy, extract_program_own_date, detect_discipline)
from benter_score import score_race

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

URL_PROGRAMS = "https://lonab.bf/programme-pmub"
URL_RESULTS = "https://lonab.bf/resultats-gains-pmub"
RESULTS_FILES_BASE = "https://lonab.bf/sites/default/files"
PREDICTIONS_DIR = "predictions"
BACKFILLED_DATES_PATH = "backfilled_dates.txt"
LOCAL_PDF_PATH = "todays_active_program.pdf"
HISTORICAL_DB_PATH = "real_history_db.csv"
STATS_DB_PATH = "driver_trainer_stats.csv"

FEATURE_COLS = [
    "Earnings", "Age", "Shoe_Status", "Driver_Rank", "Trainer_Rank",
    "Days_Rest", "DQ_Rate", "Speed_Index", "Autostart_Pos", "Market_Prob",
    "Discipline_Attele", "Draw", "Weight_KG",
    "Market_Rank_In_Race", "Earnings_Rank_In_Race", "Speed_Rank_In_Race"
]


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or Chat ID missing. Skipping alert.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        res = requests.post(url, json=payload)
        if res.status_code == 200:
            print("Telegram message sent successfully!")
        else:
            print(f"Telegram Error: {res.text}")
    except Exception as e:
        print(f"Telegram Connection Error: {e}")


def get_person_rank(name, role="Driver"):
    if not os.path.exists(STATS_DB_PATH):
        return 1.5
    try:
        df = pd.read_csv(STATS_DB_PATH)
        clean_search = re.sub(r'[^A-Z\s\.]', '', str(name).upper()).strip()
        match = df[(df["Type"] == role) & (df["Name"].str.contains(clean_search, regex=False, na=False))]
        if not match.empty:
            return float(match.iloc[0]["Skill_Rank"])
    except Exception:
        pass
    return 1.5


def parse_rest_days(comment_text):
    if "rentree" in comment_text.lower() or "rentrée" in comment_text.lower() or "absent" in comment_text.lower():
        return 60.0
    elif "recent" in comment_text.lower() or "récent" in comment_text.lower() or "en forme" in comment_text.lower():
        return 14.0
    return 21.0


def parse_autostart_position(horse_num, full_text):
    is_autostart = "autostart" in full_text.lower() or "a l'attele" in full_text.lower()
    if is_autostart:
        try:
            num = int(horse_num)
            return 1 if 1 <= num <= 8 else 0
        except ValueError:
            pass
    return 1


def get_n_runners(full_text, fallback_count):
    """Reads the '16 CONCURRENTS' style header. Falls back to the count
    of horse numbers we actually found if the header isn't matched."""
    m = re.search(r'(\d+)\s*CONCURRENTS', full_text, re.IGNORECASE)
    if m:
        return int(m.group(1))
    return fallback_count


def load_real_dataset():
    if os.path.exists(HISTORICAL_DB_PATH):
        df = pd.read_csv(HISTORICAL_DB_PATH)
        print(f"Loaded {len(df)} real historical race records.")
        return df
    print("Error: real_history_db.csv not found in repository.")
    return None


def softmax_probabilities(logits, temperature=1.0):
    exp_scores = np.exp((logits - np.max(logits)) / temperature)
    return exp_scores / np.sum(exp_scores)


def calculate_kelly_stake(ai_prob, decimal_odds, fraction=0.25):
    p = ai_prob
    b = decimal_odds - 1.0
    if b <= 0 or p <= 0:
        return 0.0
    kelly_f = ((p * b) - (1.0 - p)) / b
    if kelly_f <= 0:
        return 0.0
    recommended_stake = min(kelly_f * fraction * 100, 5.0)
    return round(recommended_stake, 2)


def build_todays_dataframe(full_text):
    """
    Parses the day's PDF text into one row per horse using REAL table data
    (Age, Earnings, Chrono, Driver, Trainer, Odds) merged with signals
    pulled from each horse's full prose paragraph (shoe status, rest,
    disqualification history). No feature here is a hardcoded constant
    shared across every horse in the race.
    """
    fallback_n = len(re.findall(r'^\s*(\d{1,2})\s*[-\xe2\x80\x93.]?\s*[A-Z][A-Z\s\'\.]{2,30}\s*:', full_text, re.MULTILINE))
    n_runners = get_n_runners(full_text, fallback_n)

    table_rows = parse_race_card(full_text, n_runners)
    if not table_rows:
        return None

    comments = extract_horse_comments(full_text, n_runners)

    records = []
    for row in table_rows:
        comment = comments.get(row["Num"], "").lower()

        shoe = 2 if ("d4" in comment or "deferre des 4" in comment or "déferré des 4" in comment) else \
            (1 if ("dp" in comment or "da" in comment or "deferre" in comment or "déferré" in comment) else 0)
        driver_rank = get_person_rank(row["Driver"], "Driver")
        trainer_rank = get_person_rank(row["Trainer"], "Trainer")
        is_attele = 1 if row["Discipline"] == "ATTELE" else 0
        # Speed_Index only has real meaning for ATTELE (trot) races, where a
        # genuine per-km chrono is published. Non-trot disciplines get 0
        # ("not applicable") rather than a plausible-looking fake value —
        # the Discipline_Attele flag lets the model learn to ignore it there.
        speed_idx = chrono_to_speed_index(row["Chrono"]) if is_attele else 0.0
        draw = row["Draw"] if row["Draw"] is not None else 0
        weight_kg = row["Weight_KG"] if row["Weight_KG"] is not None else 0.0
        days_rest = parse_rest_days(comment)
        autostart_pos = parse_autostart_position(row["Num"], full_text)
        dq_rate = 0.2 if ("da" in comment or "disqualification" in comment) else 0.05

        dec_odds = row["Decimal_Odds"] if row["Decimal_Odds"] else 10.0
        market_prob = round(1.0 / dec_odds, 3) if dec_odds > 0 else 0.05

        records.append({
            "Num": row["Num"],
            "Horse": f"{row['Num']} - {row['Horse']}",
            "Driver": row["Driver"],
            "Trainer": row["Trainer"],
            "Comment": comment[:300],
            "Earnings": np.log1p(row["Earnings"]),
            "Age": row["Age"],
            "Shoe_Status": shoe,
            "Driver_Rank": driver_rank,
            "Trainer_Rank": trainer_rank,
            "Days_Rest": days_rest,
            "DQ_Rate": dq_rate,
            "Speed_Index": speed_idx,
            "Autostart_Pos": autostart_pos,
            "Market_Prob": market_prob,
            "Odds": dec_odds,
            "Raw_Earnings": row["Earnings"],
            "Discipline": row["Discipline"],
            "Discipline_Attele": is_attele,
            "Draw": draw,
            "Weight_KG": weight_kg,
            "Perf": row["Perf"],
        })

    df = pd.DataFrame(records)
    if df.empty:
        return df

    # Race-relative features: a horse's raw Speed_Index/Market_Prob/Earnings
    # only means something compared to the OTHER horses in the SAME race —
    # 12.5 speed units is fast or slow depending entirely on the field. Rank
    # each horse against its own race's field (normalized to [0,1], lower =
    # better/faster/more fancied) instead of relying on raw values alone.
    # Speed rank is only meaningful within ATTELE races (see Speed_Index
    # note above); non-trot rows get a neutral 0.5.
    n = len(df)
    df["Market_Rank_In_Race"] = df["Market_Prob"].rank(ascending=False, method="average") / n
    df["Earnings_Rank_In_Race"] = df["Earnings"].rank(ascending=False, method="average") / n
    if df["Discipline_Attele"].iloc[0] == 1:
        df["Speed_Rank_In_Race"] = df["Speed_Index"].rank(ascending=True, method="average") / n
    else:
        df["Speed_Rank_In_Race"] = 0.5

    return df


KNOWN_TRACKS = [
    "PARISLONGCHAMP", "LONGCHAMP", "PARIS-VINCENNES", "VINCENNES", "AUTEUIL",
    "DEAUVILLE", "SAINT-CLOUD", "VICHY", "MARSEILLE-BORELY", "MARSEILLE-BORÉLY",
    "CABOURG", "ENGHIEN", "CAGNES-SUR-MER", "DIEPPE", "CLAIREFONTAINE",
]


def extract_track_and_distance(full_text):
    """
    Pulls the track name and race distance from the header, e.g.
    'AUTEUIL - PRIX JEAN BART - HAIES' and
    '98 000 EUROS (ENV. 64 000 000 F CFA) - 3 600 METRES'
    -> ('AUTEUIL', 3600.0)
    Matches against a whitelist of known tracks rather than a generic
    "first line shaped like X - Y" heuristic — the document's edition
    line ('AN XXV - N° 44 462 - GRATUIT') has that same shape and was
    being matched instead of the real track name.
    Used only for the Benter-method surface-fit factor's known
    track/distance patterns; returns (None, None) if not found rather
    than guessing.
    """
    track = None
    text_upper = full_text.upper()
    for candidate in KNOWN_TRACKS:
        if candidate in text_upper:
            track = candidate
            break
    distance = None
    m_dist = re.search(r'(\d[\d\s]*)\s*METRES', full_text)
    if m_dist:
        try:
            distance = float(m_dist.group(1).replace(' ', ''))
        except ValueError:
            pass
    return track, distance


def run_predictions():
    print("=== MORNING WORKFLOW: QUANTITATIVE KELLY AI PREDICTIONS ===")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL_PROGRAMS, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Only real race-card programs, not result-recap PDFs (LONAB's site also
    # lists filenames like "Res_08_09_2026_QUARTE.pdf" among the .pdf links,
    # which are results summaries, not race cards — grabbing one of those
    # by taking the first ".pdf" link on the page was the cause of predicting
    # on the wrong date/document).
    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf") and "JH_PMU" in href.upper():
            pdf_links.append(urllib.parse.urljoin(URL_PROGRAMS, href))

    if not pdf_links:
        send_telegram_message("LONAB AI Alert: No active PDF program found today.")
        return

    # The expected race date is tomorrow, local (Burkina Faso is UTC+0 all
    # year, so "today" server-side is the correct reference with no offset).
    expected_date = (datetime.utcnow() + timedelta(days=1)).strftime("%d-%m-%Y")

    todays_df = None
    full_text = None
    matched_url = None
    for candidate_url in pdf_links:
        res = requests.get(candidate_url, headers=headers)
        with open(LOCAL_PDF_PATH, "wb") as f:
            f.write(res.content)

        with pdfplumber.open(LOCAL_PDF_PATH) as pdf:
            candidate_text = extract_pdf_text_multi_strategy(pdf)

        program_date = extract_program_own_date(candidate_text)
        if program_date == expected_date:
            full_text = candidate_text
            matched_url = candidate_url
            break
        print(f"Skipping {candidate_url}: program date is {program_date}, expected {expected_date}")

    if full_text is None:
        send_telegram_message(
            f"LONAB AI Alert: Could not find a program PDF dated {expected_date} "
            f"(checked {len(pdf_links)} candidates). The site's listing may not "
            f"have tomorrow's race up yet, or its layout changed."
        )
        return

    print(f"-> Program downloaded successfully: {matched_url} (date {expected_date})")

    todays_df = build_todays_dataframe(full_text)
    if todays_df is None or todays_df.empty:
        # Rich diagnostics printed to the Action log (not the Telegram
        # message, which stays short) — this is what was missing last
        # time: the generic "layout may have changed" message gave no
        # way to diagnose the ACTUAL cause without fetching the real PDF
        # separately, which cost real back-and-forth to resolve before.
        n_runners_match = re.search(r'(\d+)\s*CONCURRENTS', full_text, re.IGNORECASE)
        detected_n = n_runners_match.group(1) if n_runners_match else None
        disc = detect_discipline(full_text)
        marker_count = full_text.count('N°')
        print("=== PARSE FAILURE DIAGNOSTICS ===")
        print(f"Discipline detected: {disc}")
        print(f"CONCURRENTS match: {detected_n}")
        print(f"'N°' occurrences in text: {marker_count}")
        lines = [l.strip() for l in full_text.split('\n') if l.strip()]
        marker_indices = [i for i, l in enumerate(lines) if l == 'N°']
        if marker_indices:
            for idx in marker_indices[:2]:
                print(f"--- Lines around 'N°' at index {idx} ---")
                for l in lines[max(0, idx - 2):idx + 20]:
                    print(f"  {l!r}")
        else:
            print("No exact 'N°' line found. First 30 non-empty lines:")
            for l in lines[:30]:
                print(f"  {l!r}")
        print("=== END DIAGNOSTICS ===")

        send_telegram_message(
            "LONAB AI Alert: Could not parse today's race table. "
            "The PDF layout may have changed - needs a manual check. "
            "(Full diagnostics printed in the Action run log.)"
        )
        return

    db = load_real_dataset()
    if db is None:
        send_telegram_message("LONAB AI Error: Missing historical database.")
        return

    X = db[FEATURE_COLS]
    y = db["Is_Winner"]

    # Regularized deliberately for a still-modest dataset (~400 rows as of
    # this writing): the earlier config (max_depth=10, no min_samples_leaf)
    # produced a suspicious ~0.99 cross-validated AUC on 66 rows — a
    # textbook overfitting signal, not real skill. Shallower trees and a
    # minimum leaf size keep the model from memorizing individual races.
    rf_model = RandomForestClassifier(
        n_estimators=300, random_state=42, max_depth=5,
        min_samples_leaf=8, max_features='sqrt'
    )
    rf_model.fit(X, y)

    lgb_model = LGBMClassifier(
        n_estimators=150, learning_rate=0.03, max_depth=4,
        min_child_samples=15, reg_alpha=0.1, reg_lambda=0.1,
        random_state=42, verbose=-1
    )
    lgb_model.fit(X, y)

    X_today = todays_df[FEATURE_COLS].astype(np.float64)

    rf_probs = rf_model.predict_proba(X_today)[:, 1]
    lgb_probs = lgb_model.predict_proba(X_today)[:, 1]
    raw_ensemble_scores = (0.50 * rf_probs) + (0.50 * lgb_probs)

    calibrated_probs = softmax_probabilities(raw_ensemble_scores, temperature=0.35)

    todays_df["Prob_Val"] = calibrated_probs
    todays_df["Prob"] = np.round(calibrated_probs * 100, 1)

    # Second, independent opinion: Kingsley-Benter rule-based scoring,
    # computed directly from the eight weighted factors rather than
    # learned from data. Deliberately a different lens on the same race —
    # agreement between the two systems is a stronger signal than either
    # one alone; disagreement is worth a manual look before staking.
    comments_by_num = {r["Num"]: r["Comment"] for _, r in todays_df.iterrows()}
    track_name, distance_m = extract_track_and_distance(full_text)
    benter_df = score_race(todays_df, comments_by_num, track_name, distance_m)

    # Stake sizing and the "value overlay" are deliberately OFF. Audit of 30
    # real races showed the model's probabilities are flatter than the odds,
    # so longshots (25x+) were flagged as "value" 73% of the time purely as a
    # calibration artifact, and Kelly sizing assumes calibrated probabilities.
    # Re-enable only after the forward scorecard (predictions/scorecard.csv)
    # shows real, calibrated edge. calculate_kelly_stake() is kept for then.
    todays_df = todays_df.sort_values(by="Prob", ascending=False)

    # Persist this race's predictions under the RACE date (the program's own
    # date, = tomorrow when this runs in the morning), so the evening job can
    # score them. Previously these were written to a temp GitHub machine and
    # lost, which is why the evening job never had anything to learn from.
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)
    todays_df.to_csv(os.path.join(PREDICTIONS_DIR, f"runners_{expected_date}.csv"), index=False)
    benter_df.to_csv(os.path.join(PREDICTIONS_DIR, f"benter_{expected_date}.csv"), index=False)

    elim_nums = set(benter_df.loc[benter_df["Eliminated"] == True, "Num"])

    def tag(row):
        return row["Horse"] + (" (BENTER: ELIMINATE)" if row["Num"] in elim_nums else "")

    top_list = [tag(r) for _, r in todays_df.iterrows()]
    top_3 = " - ".join(top_list[:3])
    top_4 = " - ".join(top_list[:4])
    top_5 = " - ".join(top_list[:5])

    msg = f"LONAB QUANT AI PREDICTIONS\n"
    msg += f"Engine: RF + LightGBM | Runners: {len(todays_df)}\n\n"
    msg += f"TOP 3 (TIERCE):\n{top_3}\n\n"
    msg += f"TOP 4 (QUARTE):\n{top_4}\n\n"
    msg += f"TOP 5 (QUINTE):\n{top_5}\n\n"

    # Second opinion: Kingsley-Benter rule-based ranking, shown alongside
    # the ML model's order rather than blended into it, so agreement and
    # disagreement between the two systems stay visible.
    benter_top5 = benter_df.head(5)
    ml_top5_nums = set(todays_df.head(5)["Num"])
    benter_top5_nums = set(benter_top5["Num"])
    consensus_nums = ml_top5_nums & benter_top5_nums

    msg += "KINGSLEY-BENTER METHOD (independent second opinion):\n"
    for _, b in benter_top5.iterrows():
        flag = " [CONSENSUS]" if b["Num"] in consensus_nums else ""
        elim_note = f" [WOULD ELIMINATE: {b['Elimination_Reason']}]" if b["Eliminated"] else ""
        msg += f"{b['Benter_Rank']}. {b['Horse']} — {round(b['Benter_Score'], 1)}/100{flag}{elim_note}\n"
    msg += "\n"

    eliminated = benter_df[benter_df["Eliminated"] == True]
    if not eliminated.empty:
        msg += "BENTER METHOD ELIMINATIONS (in either system's top 8):\n"
        top8_nums = set(todays_df.head(8)["Num"]) | set(benter_df.head(8)["Num"])
        for _, e in eliminated[eliminated["Num"].isin(top8_nums)].iterrows():
            msg += f"- {e['Horse']}: {e['Elimination_Reason']}\n"
        msg += "\n"

    agreement_count = len(consensus_nums)
    msg += f"Model/Method agreement: {agreement_count}/5 horses appear in both top-5 lists.\n\n"

    msg += "Top probabilities:\n"
    for _, r in todays_df.head(8).iterrows():
        msg += f"- {r['Horse']} ({r['Driver']}): {r['Prob']}%\n"
    msg += "\nStake sizing is OFF until the model's probabilities are calibrated against the odds.\n"

    send_telegram_message(msg)


FR_MONTH_NAMES = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                  "août", "septembre", "octobre", "novembre", "décembre"]


def parse_arrival_text(text):
    """
    Extracts the official finishing order from a LONAB results sheet.
    Verified against real sheets of both formats:
      QUARTE: "ARR : 8 - 14 - 9 - 4 NPO :00 NP: 00"        (4 numbers)
      4+1:    "ARR : 12-11-14-16-7 NPO : 17 NP:00"         (5 numbers)
    Returns zero-padded strings in finishing order, or [].
    """
    m = re.search(r'ARR\s*:\s*([\d\s\-\u2013]+)', text)
    if not m:
        return []
    return [n.zfill(2) for n in re.findall(r'\d{1,2}', m.group(1))]


def results_candidate_urls(race_dt):
    """Direct file URLs, pattern confirmed on real files
    (Res41_21_12_2025.pdf, Res_29_06_2026_QUARTE.pdf)."""
    dd, mm, yyyy = race_dt.strftime("%d"), race_dt.strftime("%m"), race_dt.strftime("%Y")
    folder = f"{RESULTS_FILES_BASE}/{yyyy}-{mm}"
    return [
        f"{folder}/Res41_{dd}_{mm}_{yyyy}.pdf",
        f"{folder}/Res_{dd}_{mm}_{yyyy}_QUARTE.pdf",
    ]


def find_results_pdf_via_listing(race_dt, headers):
    """Fallback if the direct filename guess misses (e.g. a new naming
    variant): find the day's results page on the listing, then its PDF.
    NOTE: written from the observed page structure; not yet exercised
    against the live listing HTML."""
    try:
        listing = requests.get(URL_RESULTS, headers=headers, timeout=30)
        soup = BeautifulSoup(listing.text, "html.parser")
        date_str = f"{race_dt.strftime('%d')} {FR_MONTH_NAMES[race_dt.month - 1]} {race_dt.year}".lower()
        for a in soup.find_all("a", href=True):
            if date_str in a.get_text(" ", strip=True).lower():
                node_url = urllib.parse.urljoin(URL_RESULTS, a["href"])
                node = requests.get(node_url, headers=headers, timeout=30)
                node_soup = BeautifulSoup(node.text, "html.parser")
                for link in node_soup.find_all("a", href=True):
                    if link["href"].lower().endswith(".pdf") and "res" in link["href"].lower():
                        return urllib.parse.urljoin(node_url, link["href"])
    except Exception as e:
        print(f"Listing fallback failed: {e}")
    return None


def fetch_official_arrival(race_dt, headers):
    """Returns (arrival_numbers, source_url) or ([], None) if not published yet."""
    urls = results_candidate_urls(race_dt)
    fallback = None
    for attempt in range(2):
        for url in urls:
            try:
                res = requests.get(url, headers=headers, timeout=30)
            except Exception as e:
                print(f"Could not fetch {url}: {e}")
                continue
            if res.status_code != 200 or not res.content.startswith(b"%PDF"):
                continue
            with open("tmp_results.pdf", "wb") as f:
                f.write(res.content)
            with pdfplumber.open("tmp_results.pdf") as pdf:
                text = "\n".join((p.extract_text() or "") for p in pdf.pages)
            os.remove("tmp_results.pdf")
            arrival = parse_arrival_text(text)
            if len(arrival) >= 3:
                return arrival, url
        if attempt == 0:
            fallback = find_results_pdf_via_listing(race_dt, headers)
            if not fallback:
                break
            urls = [fallback]
    return [], None


def score_forward_test(runners_df, benter_df, arrival):
    """
    One scorecard row: how did each system's picks do on this race, out of
    sample (predictions were made the morning before, with this race not yet
    in the training data)?
    """
    n_arr = len(arrival)
    top3, topn = set(arrival[:3]), set(arrival)
    model_order = runners_df.sort_values("Prob_Val", ascending=False)["Num"].tolist()
    market_order = runners_df.sort_values("Market_Prob", ascending=False)["Num"].tolist()
    benter_order = benter_df.sort_values("Benter_Rank")["Num"].tolist()

    def hits(order, k, target):
        return len(set(order[:k]) & target)

    winner = arrival[0]
    return {
        "arrival": "-".join(arrival),
        "positions_scored": n_arr,
        "runners": len(runners_df),
        "discipline": runners_df["Discipline"].iloc[0],
        "model_top3_hits": hits(model_order, 3, top3),
        "market_top3_hits": hits(market_order, 3, top3),
        "benter_top3_hits": hits(benter_order, 3, top3),
        "model_topN_hits": hits(model_order, n_arr, topn),
        "market_topN_hits": hits(market_order, n_arr, topn),
        "benter_topN_hits": hits(benter_order, n_arr, topn),
        "winner_model_rank": model_order.index(winner) + 1 if winner in model_order else None,
        "winner_market_rank": market_order.index(winner) + 1 if winner in market_order else None,
        "winner_benter_rank": benter_order.index(winner) + 1 if winner in benter_order else None,
    }


def collect_daily_results(race_date=None):
    """
    Evening job. Scores the race run TODAY (predicted yesterday morning), adds
    its labeled rows to the training database, and appends a scorecard row.

    Fixes vs the original (which never produced a single commit):
    - reads predictions saved by date (the morning job now commits them),
    - fetches the real results PDFs (not a page that doesn't exist),
    - stamps the true race date, and is idempotent (safe to run twice).
    """
    print("=== EVENING WORKFLOW: SCORING TODAY'S RACE ===")
    race_dt = race_date or datetime.utcnow()
    race_str = race_dt.strftime("%d-%m-%Y")

    runners_path = os.path.join(PREDICTIONS_DIR, f"runners_{race_str}.csv")
    benter_path = os.path.join(PREDICTIONS_DIR, f"benter_{race_str}.csv")
    if not os.path.exists(runners_path):
        print(f"No saved predictions for {race_str} (no race today, or none were made). Nothing to score.")
        return

    db = load_real_dataset()
    if db is None:
        return
    if (db["Race_Date"].astype(str) == race_str).any():
        print(f"{race_str} is already in the database. Nothing to do.")
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    arrival, source_url = fetch_official_arrival(race_dt, headers)
    if not arrival:
        print(f"Official results for {race_str} not published yet. A later scheduled run will retry.")
        return
    print(f"Official arrival for {race_str}: {arrival} (from {source_url})")

    runners = pd.read_csv(runners_path, dtype={"Num": str})
    runners["Num"] = runners["Num"].str.zfill(2)
    benter = pd.read_csv(benter_path, dtype={"Num": str})
    benter["Num"] = benter["Num"].str.zfill(2)

    runners["Finish_Position"] = runners["Num"].apply(lambda n: arrival.index(n) + 1 if n in arrival else 0)
    runners["Is_Winner"] = (runners["Finish_Position"] > 0).astype(int)
    runners["Race_Date"] = race_str

    # Same feature columns and order as the database, so rows match exactly.
    new_rows = runners[[c for c in db.columns if c in runners.columns]]
    updated_db = pd.concat([db, new_rows], ignore_index=True)
    updated_db.to_csv(HISTORICAL_DB_PATH, index=False)
    with open(BACKFILLED_DATES_PATH, "a") as f:   # so the weekly backfill doesn't add it twice
        f.write(race_str + "\n")

    # Forward scorecard: the only genuinely out-of-sample record of how the
    # system performs. Review this, not the backtest, before trusting stakes.
    card = score_forward_test(runners, benter, arrival)
    card_row = pd.DataFrame([{"race_date": race_str, **card}])
    card_path = os.path.join(PREDICTIONS_DIR, "scorecard.csv")
    if os.path.exists(card_path):
        card_row = pd.concat([pd.read_csv(card_path), card_row], ignore_index=True)
    card_row.to_csv(card_path, index=False)

    n_races = len(card_row)
    msg = f"LONAB RESULT & SCORECARD - {race_str}\n\n"
    msg += f"Official arrival: {' - '.join(arrival)}\n\n"
    msg += f"Top-3 picks that finished top-3 (out of 3):\n"
    msg += f"- ML model: {card['model_top3_hits']}\n- Shortest odds: {card['market_top3_hits']}\n- Benter method: {card['benter_top3_hits']}\n\n"
    msg += (f"Where the winner ranked in each system's order: model #{card['winner_model_rank']}, "
            f"odds #{card['winner_market_rank']}, Benter #{card['winner_benter_rank']}\n\n")
    msg += f"Races scored so far: {n_races}. Database: {len(updated_db)} rows."
    send_telegram_message(msg)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--evening":
        collect_daily_results()
    else:
        run_predictions()
