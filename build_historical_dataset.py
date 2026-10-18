"""
Builds real_history_db.csv from LONAB's own published documents.

How the source documents work (verified on real files):
- Each day's "Journal Hippique" PDF holds tomorrow's race PROGRAM (the horse
  table we turn into features).
- Each PDF ALSO embeds one or two recaps of PAST races' official arrival.
So a labeled training race needs two documents: the program for race date D,
and any later PDF whose embedded recap names D and gives its arrival.

Two modes (both use the exact feature code path as live predictions):
  python build_historical_dataset.py [days_back]            incremental backfill
  python build_historical_dataset.py [days_back] --rebuild  one-time clean rebuild

--rebuild exists because older rows were built before the schema was settled:
they lack race IDs and finish positions, carry placeholder rank features, mix
top-3/4/5 labels, and the first 50 have unknown provenance. A rebuild
regenerates every reachable race from source with ONE consistent layout, keeps
any already-dated race it can't reproduce, archives (never deletes) the legacy
rows, and refuses to touch the database if the result looks wrong.

Run it from GitHub Actions (full internet access).
"""
import os
import re
import sys
import urllib.parse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta

import requests
import pdfplumber
import pandas as pd

from pdf_parser import (extract_pdf_text_multi_strategy, extract_program_own_date,
                        FR_MONTHS, BET_LABEL)
from main import build_todays_dataframe, FEATURE_COLS

BASE_LIST_URL = "https://lonab.bf/programme-pmub"
FILES_BASE = "https://lonab.bf/sites/default/files"
HISTORICAL_DB_PATH = "real_history_db.csv"
SEEN_DATES_PATH = "backfilled_dates.txt"
ARCHIVE_DIR = "archive"
CACHE_DIR = "backfill_cache"
HEADERS = {"User-Agent": "Mozilla/5.0"}

DB_COLUMNS = FEATURE_COLS + ["Race_Date", "Positions_Known", "Finish_Position", "Is_Winner"]

# Filename variants actually seen on lonab.bf for program PDFs. LONAB is not
# consistent (hyphen vs underscore, a missing "B", a missing "DU"), which is
# why guessing a single pattern by date silently missed programs.
NAME_VARIANTS = [
    "JH_PMUB_DU_{d}.pdf", "JH_PMUB_DU-{d}.pdf", "JH_PMU_DU_{d}.pdf",
    "JH_PMUB_{d}.pdf", "JH_PMU_DU-{d}.pdf", "JH_PMUB_DU_{d}_0.pdf",
]

RECAP_RE = re.compile(
    rf'{BET_LABEL}\s+DU\s+\w+\s+(\d{{2}})[/\.](\d{{2}})[/\.](\d{{4}}).*?'
    r'Arriv[ée]e\s*:\s*([\d\s\-\u2013]+)',
    re.IGNORECASE | re.DOTALL
)
# A second, differently-formatted recap that appears near the top of each
# document (textual month, "ARRIVEE DU" prefix) recapping a DIFFERENT race
# than the one in the "RESULTATS DES COURSES" section below it — verified
# on a real document where both appeared for different dates. Capturing
# both roughly doubles real backfill yield per document.
RECAP_RE_HEADER = re.compile(
    rf'ARRIV[ÉE]E?\s+DU\s+{BET_LABEL}\s+DU\s+\w+\s+(\d{{1,2}})\s+([A-ZÉÛ]+)\s+(\d{{4}})\s*:\s*([\d\s\-\u2013]+)',
    re.IGNORECASE
)


def extract_embedded_recaps(full_text):
    """
    Returns a list of (date_str 'DD-MM-YYYY', [top5 horse numbers]) tuples —
    a single document can contain TWO differently-formatted recaps for
    TWO different past races (verified on a real document): one in the
    "RESULTATS DES COURSES" section (numeric DD/MM/YYYY date) and one in
    a prominent "ARRIVEE DU ..." line nearer the top (textual month name).
    Capturing both roughly doubles real backfill yield per document.
    """
    results = []

    for m in RECAP_RE.finditer(full_text):
        dd, mm, yyyy = m.group(1), m.group(2), m.group(3)
        numbers = re.findall(r'\d{1,2}', m.group(4))[:5]
        numbers = [n.zfill(2) for n in numbers]
        if len(numbers) >= 3:
            results.append((f"{dd}-{mm}-{yyyy}", numbers))

    for m in RECAP_RE_HEADER.finditer(full_text):
        dd = m.group(1).zfill(2)
        month_name = m.group(2).upper()
        mm = FR_MONTHS.get(month_name)
        if not mm:
            continue
        yyyy = m.group(3)
        numbers = re.findall(r'\d{1,2}', m.group(4))[:5]
        numbers = [n.zfill(2) for n in numbers]
        if len(numbers) >= 3:
            results.append((f"{dd}-{mm}-{yyyy}", numbers))

    return results


def candidate_program_urls(dt):
    """Every plausible URL for the program of date dt. Files normally sit in
    the race month's folder, but a program published on the last day of the
    previous month sits in that month's folder, so try both."""
    d = dt.strftime("%d-%m-%Y")
    folders = [dt.strftime("%Y-%m"), (dt.replace(day=1) - timedelta(days=1)).strftime("%Y-%m")]
    return [f"{FILES_BASE}/{folder}/{name.format(d=d)}" for folder in folders for name in NAME_VARIANTS]


def pdf_to_text(content):
    tmp = f"tmp_{os.getpid()}_{abs(hash(content[:200]))}.pdf"
    with open(tmp, "wb") as f:
        f.write(content)
    try:
        with pdfplumber.open(tmp) as pdf:
            return extract_pdf_text_multi_strategy(pdf)
    finally:
        os.remove(tmp)


def try_fetch_program_text(url):
    """Program text, or None if the URL isn't a real PDF (404s are expected
    for guessed names and non-race days)."""
    try:
        res = requests.get(url, headers=HEADERS, timeout=30)
    except Exception:
        return None
    if res.status_code != 200 or not res.content.startswith(b"%PDF"):
        return None
    try:
        return pdf_to_text(res.content)
    except Exception as e:
        print(f"  could not read {url}: {e}")
        return None


def find_program_for_date(dt):
    for url in candidate_program_urls(dt):
        text = try_fetch_program_text(url)
        if text:
            return url, text
    return None, None


def list_listing_urls(max_pages=6):
    urls = []
    for page in range(max_pages):
        url = BASE_LIST_URL if page == 0 else f"{BASE_LIST_URL}?page={page}"
        try:
            html = requests.get(url, headers=HEADERS, timeout=30).text
        except Exception:
            break
        found = re.findall(r'href="([^"]+\.pdf)"', html)
        if not found:
            break
        urls.extend(urllib.parse.urljoin(url, f) for f in found if "JH_PMU" in f.upper())
    return list(dict.fromkeys(urls))


def resolve_recaps(candidates):
    """All recap lists seen for one date must agree on their common prefix
    (e.g. a 4-number QUARTE sheet vs a 5-number 4+1 line). Returns the longest
    consistent list, or None if two sources contradict each other."""
    ordered = sorted(candidates, key=len, reverse=True)
    longest = ordered[0]
    for other in ordered[1:]:
        if other != longest[:len(other)]:
            return None
    return longest


def gather_sources(days_back):
    """Fetch programs and collect embedded recaps. Returns
    (program_texts: {DD-MM-YYYY: text}, recaps: {DD-MM-YYYY: [lists]})."""
    program_texts, recaps = {}, defaultdict(list)

    def absorb(text):
        date_key = extract_program_own_date(text)
        if date_key:
            program_texts[date_key] = text
        for recap_date, numbers in extract_embedded_recaps(text):
            recaps[recap_date].append(numbers)

    for url in list_listing_urls():
        text = try_fetch_program_text(url)
        if text:
            absorb(text)
    print(f"Listing pages gave {len(program_texts)} programs.")

    today = datetime.utcnow()
    dates = [today + timedelta(days=1) - timedelta(days=i) for i in range(days_back + 1)]
    have = set(program_texts)
    todo = [d for d in dates if d.strftime("%d-%m-%Y") not in have]
    with ThreadPoolExecutor(max_workers=6) as pool:
        for url, text in pool.map(find_program_for_date, todo):
            if text:
                absorb(text)
    print(f"Probed {len(todo)} dates by filename; now {len(program_texts)} programs, "
          f"{len(recaps)} dates with an embedded recap.")
    return program_texts, recaps


def label_races(program_texts, recaps, skip_dates=()):
    """Turn every date that has BOTH a program and a consistent recap into
    labeled rows. Returns (DataFrame, {date: reason} for skipped races)."""
    frames, skipped = [], {}
    for date_key, lists in sorted(recaps.items(), key=lambda kv: datetime.strptime(kv[0], "%d-%m-%Y")):
        if date_key in skip_dates:
            continue
        if date_key not in program_texts:
            skipped[date_key] = "no program fetched for that date"
            continue
        arrival = resolve_recaps(lists)
        if arrival is None:
            skipped[date_key] = f"recaps contradict each other: {lists}"
            continue
        if len(arrival) < 3:
            skipped[date_key] = f"only {len(arrival)} positions in the recap"
            continue
        df = build_todays_dataframe(program_texts[date_key])
        if df is None or df.empty:
            skipped[date_key] = "program found but could not be parsed"
            continue
        # Integrity check: every horse in the official arrival must be a runner
        # we parsed. If not, the program and recap are misaligned (or a horse
        # was mis-parsed) and the labels would be wrong, so skip, never guess.
        if not set(arrival) <= set(df["Num"]):
            skipped[date_key] = f"arrival {arrival} contains numbers not in parsed field"
            continue
        df["Finish_Position"] = df["Num"].apply(lambda n: arrival.index(n) + 1 if n in arrival else 0)
        df["Is_Winner"] = (df["Finish_Position"] > 0).astype(int)
        df["Positions_Known"] = len(arrival)
        df["Race_Date"] = date_key
        frames.append(df[DB_COLUMNS])
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=DB_COLUMNS)
    return out, skipped


def report_skipped(skipped):
    if skipped:
        print(f"Skipped {len(skipped)} race(s):")
        for d, why in sorted(skipped.items()):
            print(f"  {d}: {why}")


def read_seen_dates():
    if os.path.exists(SEEN_DATES_PATH):
        return {l.strip() for l in open(SEEN_DATES_PATH) if l.strip()}
    return set()


def backfill(days_back=60):
    """Incremental mode: add races not already in the database."""
    db = pd.read_csv(HISTORICAL_DB_PATH) if os.path.exists(HISTORICAL_DB_PATH) else pd.DataFrame(columns=DB_COLUMNS)
    seen = read_seen_dates() | set(db["Race_Date"].dropna().astype(str))
    program_texts, recaps = gather_sources(days_back)
    new_rows, skipped = label_races(program_texts, recaps, skip_dates=seen)
    report_skipped(skipped)
    if new_rows.empty:
        print("No new labeled races this run.")
        return
    for col in DB_COLUMNS:
        if col not in db.columns:
            db[col] = None
    db = pd.concat([db[DB_COLUMNS], new_rows], ignore_index=True)
    db.to_csv(HISTORICAL_DB_PATH, index=False)
    with open(SEEN_DATES_PATH, "a") as f:
        for d in new_rows["Race_Date"].unique():
            f.write(d + "\n")
    print(f"Added {len(new_rows)} rows from {new_rows['Race_Date'].nunique()} races. Database: {len(db)} rows.")


def rebuild(days_back=400, min_races=10):
    """One-time clean rebuild. Safe by construction:
    - legacy undated rows are ARCHIVED to archive/, not deleted
    - already-dated races the rebuild can't reproduce are KEPT
    - if fewer than min_races come out, the database is left untouched."""
    program_texts, recaps = gather_sources(days_back)
    rebuilt, skipped = label_races(program_texts, recaps)
    report_skipped(skipped)
    n_rebuilt = rebuilt["Race_Date"].nunique() if not rebuilt.empty else 0
    print(f"Rebuilt {n_rebuilt} races ({len(rebuilt)} rows) from source documents.")
    if n_rebuilt < min_races:
        print(f"ABORTING: fewer than {min_races} races rebuilt, which suggests a fetch problem "
              f"rather than a real result. Database left untouched.")
        return

    existing = pd.read_csv(HISTORICAL_DB_PATH) if os.path.exists(HISTORICAL_DB_PATH) else pd.DataFrame(columns=DB_COLUMNS)
    dated_mask = existing["Race_Date"].notna() & (existing["Race_Date"].astype(str).str.len() > 0)
    legacy, existing_dated = existing[~dated_mask], existing[dated_mask].copy()

    kept = existing_dated[~existing_dated["Race_Date"].isin(set(rebuilt["Race_Date"]))].copy()
    if "Positions_Known" not in kept.columns or kept["Positions_Known"].isna().any():
        kept["Positions_Known"] = kept.groupby("Race_Date")["Finish_Position"].transform("max")
    kept = kept[DB_COLUMNS]

    if len(legacy):
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d")
        legacy.to_csv(os.path.join(ARCHIVE_DIR, f"legacy_undated_rows_{stamp}.csv"), index=False)

    new_db = pd.concat([rebuilt, kept], ignore_index=True)
    new_db.to_csv(HISTORICAL_DB_PATH, index=False)
    with open(SEEN_DATES_PATH, "w") as f:
        for d in sorted(new_db["Race_Date"].unique(), key=lambda x: datetime.strptime(x, "%d-%m-%Y")):
            f.write(d + "\n")
    print(f"Archived {len(legacy)} legacy undated rows; kept {kept['Race_Date'].nunique()} existing dated races "
          f"the rebuild couldn't reproduce.")
    print(f"Database is now {len(new_db)} rows across {new_db['Race_Date'].nunique()} dated races, "
          f"every row with a race date and finish position.")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    days = int(args[0]) if args else None
    if "--rebuild" in sys.argv:
        rebuild(days_back=days or 400)
    else:
        backfill(days_back=days or 60)
