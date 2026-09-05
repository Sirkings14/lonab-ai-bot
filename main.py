import os
import re
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import pdfplumber
from sklearn.ensemble import RandomForestClassifier

# --- ENVIRONMENT VARIABLES ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

URL_PROGRAMS = "https://lonab.bf/programme-pmub"
URL_RESULTS = "https://lonab.bf/resultats-et-rapports"
LOCAL_PDF_PATH = "todays_active_program.pdf"
HISTORICAL_DB_PATH = "real_history_db.csv"
STATS_DB_PATH = "driver_trainer_stats.csv"


def send_telegram_message(message):
    """Sends prediction results directly to your Telegram chat."""
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Telegram token or Chat ID missing. Skipping alert.")
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
            print("📱 Telegram message sent successfully!")
        else:
            print(f"❌ Telegram Error: {res.text}")
    except Exception as e:
        print(f"❌ Telegram Connection Error: {e}")


def get_person_rank(name, role="Driver"):
    """Looks up real driver/trainer skill ranks from stats database."""
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
    return 1.5  # Standard baseline rank for unknown competitors


def parse_chrono_speed(comment_text):
    """Parses Kilometric Reduction (Speed Chrono e.g. 1'12"5 -> 12.5)."""
    match = re.search(r"1'(\d{2})\"(\d{1,2})", comment_text)
    if match:
        seconds = float(match.group(1))
        tenths = float(match.group(2)) / 10.0
        return seconds + tenths
    return 13.0  # Average default speed index


def parse_rest_days(comment_text):
    """Parses fitness & recency from comment text or past performance."""
    if "rentrée" in comment_text.lower() or "absent" in comment_text.lower():
        return 60.0  # Returning from long break (Red flag)
    elif "récent" in comment_text.lower() or "en forme" in comment_text.lower():
        return 14.0  # Peak freshness
    return 21.0  # Standard rest window


def load_real_dataset():
    """Loads the real historical database from GitHub repository."""
    if os.path.exists(HISTORICAL_DB_PATH):
        df = pd.read_csv(HISTORICAL_DB_PATH)
        print(f"✔️ Loaded {len(df)} real historical race records.")
        return df
    print("❌ Error: real_history_db.csv not found in repository.")
    return None


def run_predictions():
    print("=== MORNING WORKFLOW: PRO AI PREDICTIONS ===")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL_PROGRAMS, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    pdf_links = []
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if href.endswith(".pdf"):
            pdf_links.append(urllib.parse.urljoin(URL_PROGRAMS, href))

    if not pdf_links:
        send_telegram_message("❌ *LONAB AI Alert:* No active PDF program found today.")
        return

    today_pdf_url = pdf_links[0]
    res = requests.get(today_pdf_url, headers=headers)
    with open(LOCAL_PDF_PATH, "wb") as f:
        f.write(res.content)
    print("-> Program downloaded successfully.")

    full_text = ""
    with pdfplumber.open(LOCAL_PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1)
            if text:
                full_text += "\n" + text

    runner_pattern = re.compile(
        r'^\s*(\d{1,2})\s*[-–.]?\s*([A-Z\s\']{3,25})\s*:\s*(.*)$',
        re.MULTILINE
    )
    matches = runner_pattern.findall(full_text)
    extracted_runners = []

    if matches:
        for num, name, comment in matches:
            clean_name = name.strip()
            if any(hw in clean_name for hw in ["MEILLEURS", "SEMAINE", "ARRIVEE", "PRIX", "COURSE"]):
                continue
            driver_match = re.search(r'\b([A-Z]\.\s*[A-Z]+|\b[A-Z]{3,}\b)', comment)
            driver = driver_match.group(0) if driver_match else "Unknown Driver"
            
            trainer_match = re.search(r'pensionnaire de ([A-Z][a-z\s\'\.-]+)', comment, re.IGNORECASE)
            trainer = trainer_match.group(1).upper() if trainer_match else "Unknown Trainer"

            extracted_runners.append({
                "Horse": f"{num} - {clean_name}",
                "Driver": driver,
                "Trainer": trainer,
                "Comment": comment.strip()[:150]
            })

    if not extracted_runners:
        for line in full_text.split('\n'):
            m = re.match(r'^\s*(\d{1,2})\s+([A-Z\s\']{4,20})', line)
            if m:
                num, name = m.group(1), m.group(2).strip()
                if not any(hw in name for hw in ["MEILLEURS", "SEMAINE", "ARRIVEE", "PRIX", "COURSE"]):
                    extracted_runners.append({
                        "Horse": f"{num} - {name}",
                        "Driver": "Assigned Pilot",
                        "Trainer": "Unknown Trainer",
                        "Comment": line
                    })

    if not extracted_runners:
        send_telegram_message("⚠️ *LONAB AI Alert:* Could not parse runners from today's PDF.")
        return

    todays_df = pd.DataFrame(extracted_runners)

    # Load REAL dataset and train ML model
    db = load_real_dataset()
    if db is None:
        send_telegram_message("❌ *LONAB AI Error:* Missing real historical database.")
        return

    # Advanced Feature Set
    X = db[["Earnings", "Age", "Shoe_Status", "Driver_Rank", "Trainer_Rank", "Days_Rest", "DQ_Rate", "Speed_Index"]]
    y = db["Is_Winner"]
    
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X, y)

    processed_today = []
    for idx, row in todays_df.iterrows():
        comment = str(row.get("Comment", "")).lower()
        
        # Pro Feature 1: Shoe Status (D4 Barefoot)
        shoe = 2 if ("d4" in comment or "déferré des 4" in comment) else (1 if ("dp" in comment or "da" in comment or "déferré" in comment) else 0)
        
        # Pro Feature 2: Driver & Trainer Real Skill Lookup
        driver_rank = get_person_rank(row["Driver"], "Driver")
        trainer_rank = get_person_rank(row["Trainer"], "Trainer")
        
        # Pro Feature 3: Speed Index (Chrono)
        speed_idx = parse_chrono_speed(comment)
        
        # Pro Feature 4: Fitness & Recency
        days_rest = parse_rest_days(comment)
        
        dq_rate = 0.2 if "da" in comment or "disqualification" in comment else 0.05
        earnings = np.log1p(25000.0)
        age = 5.0

        processed_today.append([
            earnings,
            age,
            shoe,
            driver_rank,
            trainer_rank,
            days_rest,
            dq_rate,
            speed_idx
        ])

    X_today = pd.DataFrame(
        processed_today,
        columns=["Earnings", "Age", "Shoe_Status", "Driver_Rank", "Trainer_Rank", "Days_Rest", "DQ_Rate", "Speed_Index"],
        dtype=np.float64
    )
    
    todays_df["Prob"] = np.round(model.predict_proba(X_today)[:, 1] * 100, 1)
    todays_df = todays_df.sort_values(by="Prob", ascending=False)

    todays_df.to_csv("todays_active_runners.csv", index=False)

    top_list = todays_df["Horse"].tolist()
    top_3 = " - ".join(top_list[:3])
    top_4 = " - ".join(top_list[:4])
    top_5 = " - ".join(top_list[:5])

    msg = f"🐎 *PRO LONAB AI PREDICTIONS* 🐎\n\n"
    msg += f"🥇 *TOP 3 (TIERCÉ):*\n`{top_3}`\n\n"
    msg += f"🥈 *TOP 4 (QUARTÉ):*\n`{top_4}`\n\n"
    msg += f"🥉 *TOP 5 (QUINTÉ):*\n`{top_5}`\n\n"
    msg += "📊 *Full Machine Rankings:*\n"
    for _, r in todays_df.head(8).iterrows():
        msg += f"• {r['Horse']} ({r['Driver']}): *{r['Prob']}%*\n"

    send_telegram_message(msg)


def collect_daily_results():
    print("=== EVENING WORKFLOW: SCRAPING OFFICIAL RESULTS ===")
    if not os.path.exists("todays_active_runners.csv"):
        print("❌ No active runner data found to match results.")
        return

    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(URL_RESULTS, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        text_content = soup.get_text()

        arrival_match = re.search(r'(?:Arrivée|ARRIVEE)\s*:?\s*([\d\s\-–]+)', text_content)
        if not arrival_match:
            print("⚠️ Official results not yet published on LONAB.")
            return

        winning_numbers = re.findall(r'\b\d{1,2}\b', arrival_match.group(1))[:5]
        print(f"🎯 Official Winning Numbers Scraped: {winning_numbers}")

        if not winning_numbers:
            return

        todays_df = pd.read_csv("todays_active_runners.csv")
        new_records = []

        for idx, row in todays_df.iterrows():
            horse_num_match = re.search(r'^\s*(\d{1,2})', str(row["Horse"]))
            if horse_num_match:
                num = horse_num_match.group(1)
                is_win = 1 if num in winning_numbers else 0
                comment = str(row.get("Comment", "")).lower()
                shoe = 2 if ("d4" in comment or "déferré des 4" in comment) else (1 if ("dp" in comment or "da" in comment or "déferré" in comment) else 0)
                driver_rank = get_person_rank(row.get("Driver", ""), "Driver")
                trainer_rank = get_person_rank(row.get("Trainer", ""), "Trainer")

                new_records.append({
                    "Earnings": 25000.0,
                    "Age": 5.0,
                    "Shoe_Status": shoe,
                    "Driver_Rank": driver_rank,
                    "Trainer_Rank": trainer_rank,
                    "Days_Rest": parse_rest_days(comment),
                    "DQ_Rate": 0.2 if "da" in comment or "disqualification" in comment else 0.05,
                    "Speed_Index": parse_chrono_speed(comment),
                    "Is_Winner": is_win
                })

        if new_records:
            new_df = pd.DataFrame(new_records)
            db = load_real_dataset()
            if db is not None:
                updated_db = pd.concat([db, new_df], ignore_index=True)
                updated_db.to_csv(HISTORICAL_DB_PATH, index=False)

                print(f"✔️ Added {len(new_records)} real outcomes to historical database!")

                msg = f"📊 *LONAB AI AUTO-LEARNING UPDATE*\n\n"
                msg += f"🏁 *Official Arrivée:* `{' - '.join(winning_numbers)}`\n"
                msg += f"🧠 Memory updated with real Pro features. Database size: *{len(updated_db)} records*."
                send_telegram_message(msg)

    except Exception as e:
        print(f"❌ Error collecting results: {e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--evening":
        collect_daily_results()
    else:
        run_predictions()
