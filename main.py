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
                         extract_pdf_text_multi_strategy, extract_program_own_date)

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

URL_PROGRAMS = "https://lonab.bf/programme-pmub"
URL_RESULTS = "https://lonab.bf/resultats-et-rapports"
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
        send_telegram_message(
            "LONAB AI Alert: Could not parse today's race table. "
            "The PDF layout may have changed - needs a manual check."
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

    kelly_stakes = []
    is_value_list = []

    for idx, r in todays_df.iterrows():
        ai_p = r["Prob_Val"]
        odds = r["Odds"]
        market_p = 1.0 / odds if odds > 0 else 0.05

        stake = calculate_kelly_stake(ai_p, odds, fraction=0.25)
        kelly_stakes.append(stake)

        is_value = (ai_p > (market_p * 1.25)) and stake > 0
        is_value_list.append(is_value)

    todays_df["Kelly_Stake"] = kelly_stakes
    todays_df["Is_Value"] = is_value_list

    todays_df = todays_df.sort_values(by="Prob", ascending=False)
    todays_df.to_csv("todays_active_runners.csv", index=False)

    top_list = todays_df["Horse"].tolist()
    top_3 = " - ".join(top_list[:3])
    top_4 = " - ".join(top_list[:4])
    top_5 = " - ".join(top_list[:5])

    msg = f"LONAB QUANT AI PREDICTIONS\n"
    msg += f"Engine: RF + LightGBM | Runners: {len(todays_df)}\n\n"
    msg += f"TOP 3 (TIERCE):\n{top_3}\n\n"
    msg += f"TOP 4 (QUARTE):\n{top_4}\n\n"
    msg += f"TOP 5 (QUINTE):\n{top_5}\n\n"

    value_bets = todays_df[todays_df["Is_Value"] == True]
    if not value_bets.empty:
        msg += "VALUE OVERLAY BETS IDENTIFIED:\n"
        for _, v in value_bets.iterrows():
            msg += f"- {v['Horse']} | Odds: {v['Odds']}x | AI Prob: {v['Prob']}% | Stake: {v['Kelly_Stake']}% Bankroll\n"
        msg += "\n"
    else:
        msg += "Market Value Check: No major market mispricings detected today.\n\n"

    msg += "Full Probabilities & Kelly Allocations:\n"
    for _, r in todays_df.head(8).iterrows():
        stake_str = f" | Stake: {r['Kelly_Stake']}%" if r['Kelly_Stake'] > 0 else ""
        msg += f"- {r['Horse']} ({r['Driver']}): {r['Prob']}%{stake_str}\n"

    send_telegram_message(msg)


def collect_daily_results():
    print("=== EVENING WORKFLOW: SCRAPING OFFICIAL RESULTS ===")
    if not os.path.exists("todays_active_runners.csv"):
        print("No active runner data found to match results.")
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(URL_RESULTS, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        text_content = soup.get_text()

        arrival_match = re.search(r'(?:Arrivee|Arrivée|ARRIVEE)\s*:?\s*([\d\s\-\xe2\x80\x93]+)', text_content)
        if not arrival_match:
            print("Official results not yet published on LONAB.")
            return

        winning_numbers = re.findall(r'\b\d{1,2}\b', arrival_match.group(1))[:5]
        print(f"Official Winning Numbers Scraped: {winning_numbers}")

        if not winning_numbers:
            return

        todays_df = pd.read_csv("todays_active_runners.csv")
        new_records = []

        for idx, row in todays_df.iterrows():
            num = str(row["Num"]).zfill(2)
            is_win = 1 if (num in winning_numbers or num.lstrip("0") in winning_numbers) else 0

            # Reuse the SAME real per-horse features used for the morning
            # prediction (not hardcoded constants), so the database that
            # trains tomorrow's model matches what the model actually saw.
            new_records.append({
                "Earnings": row["Earnings"],
                "Age": row["Age"],
                "Shoe_Status": row["Shoe_Status"],
                "Driver_Rank": row["Driver_Rank"],
                "Trainer_Rank": row["Trainer_Rank"],
                "Days_Rest": row["Days_Rest"],
                "DQ_Rate": row["DQ_Rate"],
                "Speed_Index": row["Speed_Index"],
                "Autostart_Pos": row["Autostart_Pos"],
                "Market_Prob": row["Market_Prob"],
                "Discipline_Attele": row["Discipline_Attele"],
                "Draw": row["Draw"],
                "Weight_KG": row["Weight_KG"],
                "Market_Rank_In_Race": row["Market_Rank_In_Race"],
                "Earnings_Rank_In_Race": row["Earnings_Rank_In_Race"],
                "Speed_Rank_In_Race": row["Speed_Rank_In_Race"],
                "Race_Date": datetime.utcnow().strftime("%d-%m-%Y"),
                "Is_Winner": is_win
            })

        if new_records:
            new_df = pd.DataFrame(new_records)
            db = load_real_dataset()
            if db is not None:
                updated_db = pd.concat([db, new_df], ignore_index=True)
                updated_db.to_csv(HISTORICAL_DB_PATH, index=False)

                print(f"Added {len(new_records)} real outcomes to historical database!")

                msg = f"LONAB AI AUTO-LEARNING UPDATE\n\n"
                msg += f"Official Arrivee: {' - '.join(winning_numbers)}\n"
                msg += f"Added today's race matrix to AI memory. Total database size: {len(updated_db)} records."
                send_telegram_message(msg)

    except Exception as e:
        print(f"Error collecting results: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--evening":
        collect_daily_results()
    else:
        run_predictions()
