import os
import re
import urllib.parse
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import requests
import pdfplumber
from sklearn.ensemble import RandomForestClassifier

# --- ENVIRONMENT VARIABLES (Loaded from GitHub Secrets) ---
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

URL_PROGRAMS = "https://lonab.bf/programme-pmub"
LOCAL_PDF_PATH = "todays_active_program.pdf"

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

def run_pipeline():
    print("=== STEP 1: DOWNLOADING TODAY'S PROGRAM PDF ===")
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

    print("\n=== STEP 2: PARSING RUNNERS ===")
    full_text = ""
    with pdfplumber.open(LOCAL_PDF_PATH) as pdf:
        for page in pdf.pages:
            text = page.extract_text(x_tolerance=1)
            if text:
                full_text += "\n" + text

    runner_pattern = re.compile(r'^\s*(\d{1,2})\s*[-–.]?\s*([A-Z\s\']{3,25})\s*:\s*(.*)$', re.MULTILINE)
    matches = runner_pattern.findall(full_text)
    extracted_runners = []

    if matches:
        for num, name, comment in matches:
            clean_name = name.strip()
            if any(hw in clean_name for hw in ["MEILLEURS", "SEMAINE", "ARRIVEE", "PRIX", "COURSE"]):
                continue
            driver_match =re.search(r'\b([A-Z]\.\s*[A-Z]+|\b[A-Z]{3,}\b)', comment)
            driver = driver_match.group(0) if driver_match else "Driver Info"
            extracted_runners.append({
                "Horse": f"{num} - {clean_name}",
                "Driver": driver,
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
                        "Comment": line
                    })

    if not extracted_runners:
        send_telegram_message("⚠️ *LONAB AI Alert:* Could not parse runners from today's PDF.")
        return

    todays_df = pd.DataFrame(extracted_runners)

    print("\n=== STEP 3: TRAINING MODEL & PREDICTING ===")
    # Train synthetic baseline model
    np.random.seed(42)
    num_runners = 6000
    gains = np.clip(np.random.exponential(scale=50000, size=num_runners) + np.random.randint(5000, 500000, size=num_runners), 0, 1e7)
    age = np.random.randint(3, 11, size=num_runners)
    shoe_status = np.random.choice([0, 1, 2], size=num_runners, p=[0.4, 0.2, 0.4])
    driver_rating = np.random.choice([1, 2, 3], size=num_runners, p=[0.2, 0.6, 0.2])
    trainer_rating = np.random.choice([1, 2, 3], size=num_runners, p=[0.2, 0.6, 0.2])
    days_rest = np.random.randint(5, 90, size=num_runners)
    dq_rate = np.random.uniform(0.0, 0.6, size=num_runners)

    winning_score = (shoe_status * 2.5) + (driver_rating * 2.0) + (trainer_rating * 1.5) + (np.log1p(gains) * 1.2) - (days_rest * 0.05) - (dq_rate * 4.0) + np.random.normal(0, 2, num_runners)
    is_winner = np.zeros(num_runners)
    for race_id in np.unique(np.arange(num_runners) // 12):
        idx = np.where(np.arange(num_runners) // 12 == race_id)[0]
        if len(idx) > 0:
            is_winner[idx[np.argmax(winning_score[idx])]] = 1

    db = pd.DataFrame({"Earnings": gains, "Age": age, "Shoe_Status": shoe_status, "Driver_Rank": driver_rating, "Trainer_Rank": trainer_rating, "Days_Rest": days_rest, "DQ_Rate": dq_rate, "Is_Winner": is_winner.astype(int)})
    
    X = db[["Earnings", "Age", "Shoe_Status", "Driver_Rank", "Trainer_Rank", "Days_Rest", "DQ_Rate"]]
    y = db["Is_Winner"]
    model = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=8)
    model.fit(X, y)

    # Process today's data
    processed_today = []
    for idx, row in todays_df.iterrows():
        comment = str(row.get("Comment", "")).lower()
        shoe = 2 if ("d4" in comment or "déferré des 4" in comment) else (1 if ("dp" in comment or "da" in comment or "déferré" in comment) else 0)
        processed_today.append([np.log1p(25000.0), 5.0, shoe, np.random.choice([1.0, 2.0, 3.0]), np.random.choice([1.0, 2.0, 3.0]), 20.0, 0.1 if "da" in comment else 0.05])

    X_today = pd.DataFrame(processed_today, columns=["Earnings", "Age", "Shoe_Status", "Driver_Rank", "Trainer_Rank", "Days_Rest", "DQ_Rate"], dtype=np.float64)
    todays_df["Prob"] = np.round(model.predict_proba(X_today)[:, 1] * 100, 1)
    todays_df = todays_df.sort_values(by="Prob", ascending=False)

    # Pick top horses
    top_list = todays_df["Horse"].tolist()
    top_3 = " - ".join(top_list[:3])
    top_4 = " - ".join(top_list[:4])
    top_5 = " - ".join(top_list[:5])

    # Format Telegram Message
    msg = f"🐎 *LONAB PMU DAILY AI PREDICTIONS* 🐎\n\n"
    msg += f"🥇 *TOP 3 (TIERCÉ):*\n`{top_3}`\n\n"
    msg += f"🥈 *TOP 4 (QUARTÉ):*\n`{top_4}`\n\n"
    msg += f"🥉 *TOP 5 (QUINTÉ):*\n`{top_5}`\n\n"
    msg += "📊 *Full Probabilities:*\n"
    for _, r in todays_df.head(8).iterrows():
        msg += f"• {r['Horse']} ({r['Driver']}): *{r['Prob']}%*\n"

    send_telegram_message(msg)

if __name__ == "__main__":
    run_pipeline()
