import os
import glob
import re
import urllib.parse
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import pdfplumber
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pdf_parser import (parse_race_card, extract_horse_comments, chrono_to_speed_index,
                         extract_pdf_text_multi_strategy, extract_program_own_date, detect_discipline)
from benter_score import score_race, BENTER_COLS
from ticket_math import race_weights, top_k_marginals
import assessment

# Stored with every new database row from now on, so form (Perf), both odds columns and the
# runner identity can be backtested later. Old rows simply have these blank.
EXTRA_COLS = ["Num", "Perf", "Odds", "Odds2", "Driver", "Trainer", "Horse"]

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

# Every row in the database stores all FEATURE_COLS (so features can be tested
# as data grows), but production only USES features that have earned it.
#   champion   = the odds, turned into real top-3 probabilities (Plackett-Luce, see ticket_math.py)
#   challenger = odds + a few extra features (logged only)
CHAMPION_FEATURES = ["Market_Prob"]
CHALLENGER_FEATURES = ["Market_Prob", "Market_Rank_In_Race", "Earnings_Rank_In_Race",
                       "Driver_Rank", "Trainer_Rank", "Age"]


def fit_top3_model(train_df, features, C=0.3):
    """Regularized logistic regression for P(finishes top 3)."""
    y = train_df["Finish_Position"].between(1, 3).astype(int)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    model.fit(train_df[features].astype(float), y)
    return model


def http_get(url, headers=None, retries=3, timeout=45):
    """GET with timeout + retries. A single slow response from lonab.bf used to kill the whole run."""
    import time
    last = None
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            if r.status_code < 500:
                return r
            last = RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise last


def send_telegram_message(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram token or Chat ID missing. Skipping alert.")
        print(message)
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": message[:4000],
    }
    try:
        res = requests.post(url, json=payload, timeout=30)
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
    """Reads the '16 CONCURRENTS' style header. Falls back to the count of horse numbers found."""
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
    Parses the day's PDF text into one row per horse using the real table data
    merged with signals from each horse's prose paragraph.
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
            "Odds2": row.get("Decimal_Odds2"),
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
    """Pulls the track name and race distance from the header, e.g. ('AUTEUIL', 3600.0)."""
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
    print("=== MORNING WORKFLOW: PREDICTIONS ===")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = http_get(URL_PROGRAMS, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    # Only real race-card programs, not result-recap PDFs.
    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf") and "JH_PMU" in href.upper():
            pdf_links.append(urllib.parse.urljoin(URL_PROGRAMS, href))

    if not pdf_links:
        send_telegram_message("LONAB AI Alert: No active PDF program found today.")
        return

    # Expected race date is tomorrow (Burkina Faso is UTC+0 all year).
    # Set TARGET_DATE (e.g. 22-11-2026) to predict a specific day instead.
    expected_date = (os.environ.get("TARGET_DATE", "").strip()
                     or (datetime.utcnow() + timedelta(days=1)).strftime("%d-%m-%Y"))

    if (os.path.exists(os.path.join(PREDICTIONS_DIR, f"runners_{expected_date}.csv"))
            and os.environ.get("FORCE") != "1"):
        print(f"Predictions for {expected_date} already exist - nothing to do (set FORCE=1 to redo).")
        return

    todays_df = None
    full_text = None
    matched_url = None
    for candidate_url in pdf_links:
        res = http_get(candidate_url, headers=headers)
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

    train = db[db["Finish_Position"] >= 0]
    n_races_train = train["Race_Date"].nunique()
    if len(train) >= 60 and n_races_train >= 5:
        challenger = fit_top3_model(train, CHALLENGER_FEATURES)
        todays_df["Prob_Val"] = top_k_marginals(race_weights(todays_df["Market_Prob"].values), 3)
        todays_df["Challenger_Prob"] = challenger.predict_proba(todays_df[CHALLENGER_FEATURES].astype(float))[:, 1]
    else:
        fallback = 3 * todays_df["Market_Prob"] / todays_df["Market_Prob"].sum()
        todays_df["Prob_Val"] = fallback
        todays_df["Challenger_Prob"] = fallback
    todays_df["Prob"] = np.round(todays_df["Prob_Val"] * 100, 1)

    # Rule-based Kingsley-Benter score: computed and logged for the record only.
    comments_by_num = {r["Num"]: r["Comment"] for _, r in todays_df.iterrows()}
    track_name, distance_m = extract_track_and_distance(full_text)
    benter_df = score_race(todays_df, comments_by_num, track_name, distance_m)

    todays_df = todays_df.sort_values(by="Prob", ascending=False)

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

    calib = assessment.calibration_report(db)
    header_n = None
    m_hdr = re.search(r'(\d{1,2})\s+CONCURRENTS', full_text, re.I)
    if m_hdr:
        header_n = int(m_hdr.group(1))
    a = assessment.assess_today(todays_df.reset_index(drop=True), header_n, calib)
    record = assessment.track_record()

    msg = f"LONAB AI - race of {expected_date} | {len(todays_df)} runners\n\n"
    import pdf_parser as _pp
    if _pp.LAST_PARSE_MISSING:
        msg += (f"WARNING: could not read horse(s) {', '.join(_pp.LAST_PARSE_MISSING)} from the programme - "
                f"they are NOT in the ranking below. Check them by hand.\n\n")
    msg += f"TOP 3 (TIERCE):\n{top_3}\n\nTOP 4 (QUARTE):\n{top_4}\n\nTOP 5 (QUINTE):\n{top_5}\n\n"
    msg += assessment.format_assessment(a, calib, record) + "\n\n"

    msg += "Chance of finishing top-3:\n"
    for _, r in todays_df.head(8).iterrows():
        msg += f"- {r['Horse']} ({r['Driver']}): {r['Prob']}%\n"

    eliminated = benter_df[benter_df["Eliminated"] == True]
    top8_nums = set(todays_df.head(8)["Num"])
    warn_rows = eliminated[eliminated["Num"].isin(top8_nums)]
    if not warn_rows.empty:
        msg += "\nCAUTION flags on top-8 horses (your own rules):\n"
        for _, e in warn_rows.iterrows():
            msg += f"- {e['Horse']}: {e['Elimination_Reason']}\n"

    # The rule-based Benter score is kept for the record, but it is NOT advice: on 11 published
    # races it averaged 0.55/3 top-3 hits vs 1.36 for the odds (pure luck = 0.58).
    msg += "\n(Benter rule-based score: running for the record only, not used for these picks.)\n"
    msg += "No ticket here is a sure thing. Never stake money you need.\n"

    send_telegram_message(msg)


FR_MONTH_NAMES = ["janvier", "février", "mars", "avril", "mai", "juin", "juillet",
                  "août", "septembre", "octobre", "novembre", "décembre"]


def parse_arrival_text(text):
    """
    Extracts the official finishing order from a LONAB results sheet.
      QUARTE: "ARR : 8 - 14 - 9 - 4 NPO :00 NP: 00"        (4 numbers)
      4+1:    "ARR : 12-11-14-16-7 NPO : 17 NP:00"         (5 numbers)
    Returns zero-padded strings in finishing order, or [].
    """
    m = re.search(r'ARR\s*:\s*([\d\s\-\u2013]+)', text)
    if not m:
        return []
    return [n.zfill(2) for n in re.findall(r'\d{1,2}', m.group(1))]


def results_candidate_urls(race_dt):
    """Direct file URLs, pattern confirmed on real files."""
    dd, mm, yyyy = race_dt.strftime("%d"), race_dt.strftime("%m"), race_dt.strftime("%Y")
    folder = f"{RESULTS_FILES_BASE}/{yyyy}-{mm}"
    return [
        f"{folder}/Res41_{dd}_{mm}_{yyyy}.pdf",
        f"{folder}/Res_{dd}_{mm}_{yyyy}_QUARTE.pdf",
    ]


def find_results_pdf_via_listing(race_dt, headers):
    """Fallback if the direct filename guess misses."""
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
    """One scorecard row: how did each system's picks do on this race, out of sample?"""
    n_arr = len(arrival)
    top3, topn = set(arrival[:3]), set(arrival)
    model_order = runners_df.sort_values("Prob_Val", ascending=False)["Num"].tolist()
    market_order = runners_df.sort_values("Market_Prob", ascending=False)["Num"].tolist()
    benter_order = benter_df.sort_values("Benter_Rank")["Num"].tolist()

    def hits(order, k, target):
        return len(set(order[:k]) & target)

    winner = arrival[0]
    card = {
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
    if "Challenger_Prob" in runners_df.columns:
        chall_order = runners_df.sort_values("Challenger_Prob", ascending=False)["Num"].tolist()
        card["challenger_top3_hits"] = hits(chall_order, 3, top3)
        card["challenger_topN_hits"] = hits(chall_order, n_arr, topn)
        card["winner_challenger_rank"] = chall_order.index(winner) + 1 if winner in chall_order else None
    return card


def heal_scorecard():
    """Add scorecard rows for every saved prediction whose race is already in the database but missing
    from the scorecard. Horses are matched by Num when the DB has it (new rows) or by odds+age+earnings."""
    db = load_real_dataset()
    if db is None:
        return 0
    card_path = os.path.join(PREDICTIONS_DIR, "scorecard.csv")
    done = set(pd.read_csv(card_path)["race_date"].astype(str)) if os.path.exists(card_path) else set()
    added = []
    for f in sorted(glob.glob(os.path.join(PREDICTIONS_DIR, "runners_*.csv"))):
        rd = os.path.basename(f)[len("runners_"):-4]
        if rd in done:
            continue
        g = db[(db["Race_Date"].astype(str) == rd) & (db["Finish_Position"] >= 0)]
        bpath = os.path.join(PREDICTIONS_DIR, f"benter_{rd}.csv")
        if g.empty or not os.path.exists(bpath):
            continue
        runners = pd.read_csv(f, dtype={"Num": str}); runners["Num"] = runners["Num"].str.zfill(2)
        benter = pd.read_csv(bpath, dtype={"Num": str}); benter["Num"] = benter["Num"].str.zfill(2)
        if "Num" in g.columns and g["Num"].notna().all():
            g = g.assign(Num=g["Num"].astype(str).str.zfill(2))
            pos = dict(zip(g["Num"], g["Finish_Position"]))
        else:
            key = lambda d: (d["Market_Prob"].round(3).astype(str) + "_" + d["Age"].astype(str) + "_"
                             + d["Earnings"].round(3).astype(str))
            kp = dict(zip(key(g), g["Finish_Position"]))
            pos = {n: kp.get(k, 0) for n, k in zip(runners["Num"], key(runners))}
            if sum(1 for v in pos.values() if v > 0) < 3:
                continue
        arrival = [n for n, p in sorted(pos.items(), key=lambda kv: kv[1]) if p > 0]
        runners["Finish_Position"] = runners["Num"].map(lambda n: pos.get(n, 0))
        card = score_forward_test(runners, benter, arrival)
        added.append({"race_date": rd, **card})
    if added:
        base = pd.read_csv(card_path) if os.path.exists(card_path) else pd.DataFrame()
        out = pd.concat([base, pd.DataFrame(added)], ignore_index=True)
        out["_d"] = pd.to_datetime(out["race_date"], format="%d-%m-%Y", errors="coerce")
        out.sort_values("_d").drop(columns="_d").to_csv(card_path, index=False)
    print(f"Scorecard healed: {len(added)} missing race(s) added.")
    return len(added)


def collect_daily_results(race_date=None):
    """
    Evening job. Scores the race run TODAY (predicted yesterday morning), adds its labeled rows to
    the training database, and appends a scorecard row. Safe to run twice.
    """
    print("=== EVENING WORKFLOW: SCORING TODAY'S RACE ===")
    race_dt = race_date or datetime.utcnow()
    race_str = race_dt.strftime("%d-%m-%Y")
    try:
        heal_scorecard()
    except Exception as e:                      # never let bookkeeping block today's scoring
        print(f"heal_scorecard skipped: {e}")

    runners_path = os.path.join(PREDICTIONS_DIR, f"runners_{race_str}.csv")
    benter_path = os.path.join(PREDICTIONS_DIR, f"benter_{race_str}.csv")
    if not os.path.exists(runners_path):
        print(f"No saved predictions for {race_str} (no race today, or none were made). Nothing to score.")
        return

    db = load_real_dataset()
    if db is None:
        return
    card_path = os.path.join(PREDICTIONS_DIR, "scorecard.csv")
    in_db = (db["Race_Date"].astype(str) == race_str).any()
    scored = os.path.exists(card_path) and (pd.read_csv(card_path)["race_date"].astype(str) == race_str).any()
    # The weekly backfill often adds a race to the database first. The old code then quit, so the
    # forward scorecard silently stopped growing. DB and scorecard are now independent.
    if in_db and scored:
        print(f"{race_str} is already in the database and the scorecard. Nothing to do.")
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

    benter["Benter_Eliminated"] = benter["Eliminated"].astype(int)
    keep = ["Num"] + [c for c in BENTER_COLS if c in benter.columns]
    runners = runners.drop(columns=[c for c in keep if c != "Num" and c in runners.columns]).merge(
        benter[keep], on="Num", how="left")

    runners["Finish_Position"] = runners["Num"].apply(lambda n: arrival.index(n) + 1 if n in arrival else 0)
    runners["Is_Winner"] = (runners["Finish_Position"] > 0).astype(int)
    runners["Race_Date"] = race_str
    runners["Positions_Known"] = len(arrival)

    if not in_db:
        cols = list(db.columns) + [c for c in EXTRA_COLS if c not in db.columns]
        new_rows = runners[[c for c in cols if c in runners.columns]]
        updated_db = pd.concat([db, new_rows], ignore_index=True)
        updated_db.to_csv(HISTORICAL_DB_PATH, index=False)
        with open(BACKFILLED_DATES_PATH, "a") as f:   # so the weekly backfill doesn't add it twice
            f.write(race_str + "\n")
    else:
        updated_db = db

    card = score_forward_test(runners, benter, arrival)
    card_row = pd.DataFrame([{"race_date": race_str, **card}])
    if os.path.exists(card_path):
        card_row = pd.concat([pd.read_csv(card_path), card_row], ignore_index=True)
    card_row.to_csv(card_path, index=False)

    n_races = len(card_row)
    msg = f"LONAB RESULT & SCORECARD - {race_str}\n\n"
    msg += f"Official arrival: {' - '.join(arrival)}\n\n"
    msg += "Top-3 picks that finished top-3 (out of 3; pure luck averages ~0.6):\n"
    msg += f"- My picks (odds-based): {card['model_top3_hits']}\n- Challenger (extra features): {card.get('challenger_top3_hits', 'n/a')}\n- Raw shortest odds: {card['market_top3_hits']}\n- Benter method: {card['benter_top3_hits']}\n\n"
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
