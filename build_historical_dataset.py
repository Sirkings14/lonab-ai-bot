"""
Backfills real_history_db.csv with genuine, labeled historical races.

How LONAB's PMU'B document works, which this script relies on:
- Each day's "Journal Hippique" PDF contains tomorrow's race PROGRAM
  (the horse table we parse for features).
- It ALSO contains a "RESULTATS DES COURSES" recap of the most recent
  "4+1" race's official result (the Arrivee line) -- but that recap is
  for a PAST race date, not the program printed in the same PDF.

So building one labeled training row requires TWO documents:
  1. The program PDF for date D (the horse table / features)
  2. A LATER PDF (any date after D) whose embedded recap names date D
     and gives its official Arrivee (top-5 finish order)

This script scans a range of program PDFs, extracts any embedded
recaps it finds, and once it has both a program and its matching
recap, produces labeled rows using the exact same feature-extraction
code path as live predictions (main.build_todays_dataframe), so
backfilled rows are never computed differently than live ones.

Run this from GitHub Actions (unrestricted network) to actually reach
back through LONAB's archive at scale -- a sandboxed assistant session
can only reach a small, recent slice of it.
"""
import os
import re
import sys
import time
import urllib.parse
from datetime import datetime, timedelta

import requests
import pdfplumber
import pandas as pd

from main import build_todays_dataframe, FEATURE_COLS

BASE_LIST_URL = "https://lonab.bf/programme-pmub"
HISTORICAL_DB_PATH = "real_history_db.csv"
CACHE_DIR = "backfill_cache"
RECAP_RE = re.compile(
    r'"?4\+1"?\s+DU\s+\w+\s+(\d{2})[/\.](\d{2})[/\.](\d{4}).*?'
    r'Arriv[ée]e\s*:\s*([\d\s\-\u2013]+)',
    re.IGNORECASE | re.DOTALL
)


def fetch_pdf_text(url, headers):
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, urllib.parse.quote(url, safe=""))
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return f.read()

    res = requests.get(url, headers=headers, timeout=30)
    res.raise_for_status()
    tmp_path = "tmp_backfill.pdf"
    with open(tmp_path, "wb") as f:
        f.write(res.content)

    full_text = ""
    with pdfplumber.open(tmp_path) as pdf:
        for page in pdf.pages:
            t = page.extract_text(layout=True)
            if t:
                full_text += "\n" + t
    os.remove(tmp_path)

    with open(cache_path, "w", encoding="utf-8") as f:
        f.write(full_text)
    return full_text


FR_MONTHS = {
    'JANVIER': '01', 'FEVRIER': '02', 'FÉVRIER': '02', 'MARS': '03', 'AVRIL': '04',
    'MAI': '05', 'JUIN': '06', 'JUILLET': '07', 'AOUT': '08', 'AOÛT': '08',
    'SEPTEMBRE': '09', 'OCTOBRE': '10', 'NOVEMBRE': '11', 'DECEMBRE': '12', 'DÉCEMBRE': '12',
}

HEADER_DATE_RE = re.compile(
    r'"?4\+1"?\s+DU\s+\w+\s+(\d{1,2})\s+([A-ZÉÛ]+)\s+(\d{4})', re.IGNORECASE
)


def extract_program_own_date(full_text):
    """
    Reads the program's own date from its printed header, e.g.
    '"4+1" DU DIMANCHE 06 SEPTEMBRE 2026' -> '06-09-2026'.
    This is far more reliable than the URL filename, which LONAB
    formats inconsistently (hyphens vs underscores, encoding bugs).
    Returns 'DD-MM-YYYY' or None.
    """
    m = HEADER_DATE_RE.search(full_text)
    if not m:
        return None
    dd = m.group(1).zfill(2)
    month_name = m.group(2).upper()
    mm = FR_MONTHS.get(month_name)
    if not mm:
        return None
    yyyy = m.group(3)
    return f"{dd}-{mm}-{yyyy}"


def extract_embedded_recap(full_text):
    """Returns (date_str 'DD-MM-YYYY', [top5 horse numbers as zero-padded strings]) or None."""
    m = RECAP_RE.search(full_text)
    if not m:
        return None
    dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
    numbers = re.findall(r'\d{1,2}', m.group(4))[:5]
    numbers = [n.zfill(2) for n in numbers]
    if len(numbers) < 3:  # too little to trust as a real Arrivee line
        return None
    return f"{dd}-{mm}-{yyyy}", numbers


def program_url_for_date(dt):
    return f"https://lonab.bf/sites/default/files/{dt.strftime('%Y-%m')}/JH_PMUB_DU_{dt.strftime('%d-%m-%Y')}.pdf"


def list_recent_program_urls(headers, max_pages=5):
    """Walks the paginated program listing to discover recent program URLs."""
    urls = []
    for page in range(max_pages):
        url = BASE_LIST_URL if page == 0 else f"{BASE_LIST_URL}?page={page}"
        try:
            res = requests.get(url, headers=headers, timeout=30)
            found = re.findall(r'href="([^"]+\.pdf)"', res.text)
            if not found:
                break
            urls.extend(urllib.parse.urljoin(url, f) for f in found)
        except Exception as e:
            print(f"Could not list page {page}: {e}")
            break
    return list(dict.fromkeys(urls))  # dedupe, keep order


def backfill(days_back=60):
    headers = {"User-Agent": "Mozilla/5.0"}
    db = pd.read_csv(HISTORICAL_DB_PATH) if os.path.exists(HISTORICAL_DB_PATH) else pd.DataFrame(columns=FEATURE_COLS + ["Is_Winner"])

    # Track which race-dates we've already labeled, to avoid duplicate rows
    # on repeated runs. Stored as a companion file since real_history_db.csv
    # itself has no date column (kept minimal to match the live schema).
    seen_path = "backfilled_dates.txt"
    seen_dates = set()
    if os.path.exists(seen_path):
        with open(seen_path) as f:
            seen_dates = set(line.strip() for line in f if line.strip())

    recaps_found = {}  # date_str -> [top5 numbers]
    program_texts = {}  # date_str -> full_text

    program_urls = list_recent_program_urls(headers)
    print(f"Discovered {len(program_urls)} program URLs from the listing pages.")

    # Also probe further back by date, in case older ones aren't listed
    # on the paginated page anymore but are still hosted.
    today = datetime.utcnow()
    for i in range(days_back):
        dt = today - timedelta(days=i)
        program_urls.append(program_url_for_date(dt))
    program_urls = list(dict.fromkeys(program_urls))

    for url in program_urls:
        try:
            text = fetch_pdf_text(url, headers)
        except Exception:
            continue  # 404s expected for guessed dates; skip quietly

        date_key = extract_program_own_date(text)
        if not date_key:
            # Fallback for the rare case the header text didn't extract cleanly
            mm_date = re.search(r'JH_PMU\S*?B?_DU[_-](\d{2})-(\d{2})-(\d{4})', url)
            if mm_date:
                date_key = f"{mm_date.group(1)}-{mm_date.group(2)}-{mm_date.group(3)}"
        if date_key:
            program_texts[date_key] = text

        recap = extract_embedded_recap(text)
        if recap:
            recap_date, top5 = recap
            recaps_found[recap_date] = top5

        time.sleep(0.3)  # polite pacing

    print(f"Found {len(program_texts)} programs and {len(recaps_found)} embedded recaps.")
    print(f"Program dates available: {sorted(program_texts.keys())}")
    print(f"Recap dates needed:      {sorted(recaps_found.keys())}")
    missing = sorted(set(recaps_found.keys()) - set(program_texts.keys()))
    if missing:
        print(f"Recap dates with NO matching program fetched: {missing}")

    new_rows = []
    newly_labeled_dates = []
    skipped_already_seen = []
    skipped_no_program = []
    for date_key, top5 in recaps_found.items():
        if date_key in seen_dates:
            skipped_already_seen.append(date_key)
            continue
        if date_key not in program_texts:
            skipped_no_program.append(date_key)
            continue  # have the result, but not that day's own program text
        df = build_todays_dataframe(program_texts[date_key])
        if df is None or df.empty:
            if not globals().get('_debug_dumped'):
                globals()['_debug_dumped'] = True
                from pdf_parser import detect_discipline
                text = program_texts[date_key]
                disc = detect_discipline(text)
                n_match = re.search(r'(\d+)\s*CONCURRENTS', text, re.IGNORECASE)
                print(f"  --- DEBUG for {date_key} (discipline={disc}, "
                      f"CONCURRENTS match={n_match.group(1) if n_match else None}) ---")
                lines = [l.strip() for l in text.split('\n') if l.strip()]
                try:
                    marker_idx = lines.index('N°')
                    print(f"  Lines around 'N°' marker (idx {marker_idx}):")
                    for l in lines[max(0, marker_idx-3):marker_idx+15]:
                        print(f"    {repr(l)}")
                except ValueError:
                    print("  'N\u00b0' marker not found in text at all.")
                    print(f"  First 25 non-empty lines: {[repr(l) for l in lines[:25]]}")
                print(f"  --- END DEBUG ---")
            print(f"  {date_key}: program text found but failed to parse into rows")
            continue
        df["Is_Winner"] = df["Num"].apply(lambda n: 1 if n in top5 else 0)
        new_rows.append(df[FEATURE_COLS + ["Is_Winner"]])
        newly_labeled_dates.append(date_key)

    if skipped_already_seen:
        print(f"Skipped (already backfilled): {skipped_already_seen}")
    if skipped_no_program:
        print(f"Skipped (no matching program fetched): {skipped_no_program}")

    if new_rows:
        combined_new = pd.concat(new_rows, ignore_index=True)
        db = pd.concat([db, combined_new], ignore_index=True)
        db.to_csv(HISTORICAL_DB_PATH, index=False)
        with open(seen_path, "a") as f:
            for d in newly_labeled_dates:
                f.write(d + "\n")
        print(f"Added {len(combined_new)} real labeled rows from {len(newly_labeled_dates)} races.")
        print(f"Database now has {len(db)} total rows.")
    else:
        print("No new labeled races found this run (either already backfilled, or no matching program+recap pairs reachable).")


if __name__ == "__main__":
    days_back = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    backfill(days_back=days_back)
