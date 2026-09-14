"""
Reconstructs per-horse rows from the LONAB "Journal Hippique" PMU'B PDF text.

pdfplumber extracts this document's table column-by-column instead of
row-by-row, so all N horses' odds come first, then all N horse numbers,
then all N sex/age codes, etc. This walks the text in that known column
order and zips the columns back into rows.
"""
import re


SEX_AGE_RE = re.compile(r'^[HMF]\.\d{1,2}$')
DIST_RE = re.compile(r'^\d[\d\s]*\.M$')
CHRONO_RE = re.compile(r'^\d\.\d{2}\.\d{2}$')
CORDE_RE = re.compile(r'^\d{1,2}$')
POIDS_RE = re.compile(r'^\d{1,3}(\.\d)?\.KG$')
PERF_RE = re.compile(r'^[0-9A-Za-z]+(\.[0-9A-Za-z]+){2,6}$')
GAINS_RE = re.compile(r'^\d{1,3}(\s\d{3})+$|^\d{4,7}$')
ODDS_RE = re.compile(r'^\d{1,3}/1$')
HORSE_NUM_RE = re.compile(r'^\d{1,2}$')


def detect_discipline(full_text):
    """
    The PDF uses a different table schema per discipline:
    ATTELE (trot) publishes DIST./CHRONO per horse.
    PLAT (flat) and others publish CORDE/POIDS (draw/weight) instead —
    no per-horse speed figure exists in that layout.
    """
    m = re.search(r'CONCURRENTS.*?-\s*([A-ZÀ-Üa-zà-ü\-]+)\s*(?:\n|$)', full_text)
    if m and 'ATTELE' in m.group(1).upper():
        return 'ATTELE'
    if re.search(r'\bATTELE\b', full_text, re.IGNORECASE):
        return 'ATTELE'
    return 'PLAT'  # default assumption for CORDE/POIDS-style layouts (flat, obstacle, etc.)


def parse_race_card(full_text, n_runners, discipline=None):
    """
    full_text: raw text extracted from one race's block of the PDF.
    n_runners: number of horses in the race (from "N CONCURRENTS").
    discipline: 'ATTELE' or 'PLAT'. Auto-detected from full_text if omitted.
    Returns a list of dicts, one per horse, or [] if the shape doesn't match.

    ATTELE (trot) rows include real Distance_M/Chrono (a genuine speed figure).
    PLAT/other rows include Draw/Weight_KG instead — this document format
    doesn't publish a per-horse speed figure for non-trot disciplines, so
    Chrono is None and callers should fall back to a neutral Speed_Index.
    """
    if discipline is None:
        discipline = detect_discipline(full_text)

    lines = [l.strip() for l in full_text.split('\n') if l.strip()]
    n = n_runners

    # Try every occurrence of "N°" in the text — the document mixes
    # full-width tables with two-column prose, and depending on which
    # text-extraction pass produced this text, the table may appear
    # cleanly at one occurrence but garbled at another. Use whichever
    # occurrence actually yields a fully valid block of rows.
    marker_indices = [i for i, l in enumerate(lines) if l == 'N°']
    for marker_idx in marker_indices:
        result = _try_parse_from_marker(lines, marker_idx, n, discipline)
        if result:
            return result
    return []


def _try_parse_from_marker(lines, marker_idx, n, discipline):
    def take_block(start_idx, regex=None, size=n):
        """Take `size` consecutive lines from start_idx, optionally validating each against regex."""
        block = lines[start_idx:start_idx + size]
        if len(block) < size:
            return None, start_idx
        if regex and not all(regex.match(x) for x in block):
            return None, start_idx
        return block, start_idx + size

    # The first odds column ("PARIS TURF" pronostic) sits directly above
    # the "N°" marker, in the same horse order.
    odds_start = marker_idx - n
    odds_block = None
    if odds_start >= 0:
        candidate = lines[odds_start:marker_idx]
        if all(ODDS_RE.match(x) for x in candidate):
            odds_block = candidate

    idx = marker_idx + 1
    horse_nums, idx = take_block(idx, HORSE_NUM_RE)
    if horse_nums is None:
        return None

    sex_age, idx = take_block(idx, SEX_AGE_RE)
    if sex_age is None:
        return None

    distance, chrono, draw, weight = None, None, None, None

    if discipline == 'ATTELE':
        distance, idx = take_block(idx, DIST_RE)
        if distance is None:
            return None
        chrono, idx = take_block(idx, CHRONO_RE)
        if chrono is None:
            return None
    else:
        draw, idx = take_block(idx, CORDE_RE)
        if draw is None:
            return None
        weight, idx = take_block(idx, POIDS_RE)
        if weight is None:
            return None

    perf, idx = take_block(idx, PERF_RE)
    if perf is None:
        return None

    gains, idx = take_block(idx, GAINS_RE)
    if gains is None:
        return None

    # Names/drivers-or-jockeys/trainers don't have a reliable regex
    # signature, so just take them positionally.
    names, idx = take_block(idx)
    if names is None:
        return None
    drivers, idx = take_block(idx)
    if drivers is None:
        return None
    trainers, idx = take_block(idx)
    if trainers is None:
        return None

    rows = []
    for i in range(n):
        sex, age = sex_age[i].split('.')
        gains_val = float(gains[i].replace(' ', ''))
        # Convert decimal odds fraction "X/1" to real decimal odds (X + 1.0)
        odds_val = (float(odds_block[i].split('/')[0]) + 1.0) if odds_block else None
        rows.append({
            "Num": horse_nums[i].zfill(2),
            "Horse": names[i],
            "Driver": drivers[i],
            "Trainer": trainers[i],
            "Sex": sex,
            "Age": float(age),
            "Distance_M": float(distance[i].replace(' ', '').replace('.M', '')) if distance else None,
            "Chrono": chrono[i] if chrono else None,
            "Draw": int(draw[i]) if draw else None,
            "Weight_KG": float(weight[i].replace('.KG', '')) if weight else None,
            "Perf": perf[i],
            "Earnings": gains_val,
            "Decimal_Odds": odds_val,
            "Discipline": discipline,
        })
    return rows


def extract_pdf_text_multi_strategy(pdf):
    """
    Runs several pdfplumber extraction strategies per page and concatenates
    all of them. This document mixes full-width tables with two-column
    prose, and no single strategy reliably handles both:
    - plain extract_text(): works for full-width tables, merges 2-column
      prose text together mid-line.
    - layout=True: preserves intra-line spacing, doesn't fix cross-column
      line merging by itself.
    - left/right half crop: separates 2-column prose properly, but can
      chop a full-width table in half.
    Since parse_race_card tries every "N°" occurrence in the combined
    text and only accepts one that yields a fully valid block, whichever
    strategy happens to produce a clean table for a given page wins,
    without needing to know in advance which one that'll be.
    """
    full_text = ""
    for page in pdf.pages:
        t1 = page.extract_text(x_tolerance=1) or ""
        t2 = page.extract_text(layout=True) or ""
        full_text += "\n" + t1 + "\n" + t2

        try:
            width = page.width
            left = page.within_bbox((0, 0, width / 2, page.height)).extract_text() or ""
            right = page.within_bbox((width / 2, 0, width, page.height)).extract_text() or ""
            full_text += "\n" + left + "\n" + right
        except Exception:
            pass  # cropping can fail on unusual page geometry; other strategies still apply

    return full_text


def chrono_to_speed_index(chrono_str, default=13.0):
    """
    Converts the table's 'M.SS.HH' chrono (e.g. "1.12.50" = 1'12"50/km)
    to the same seconds(+fraction) scale used historically:
    "1.12.50" -> 12.50
    """
    if not chrono_str:
        return default
    parts = chrono_str.split('.')
    if len(parts) != 3:
        return default
    try:
        return float(f"{parts[1]}.{parts[2]}")
    except ValueError:
        return default


def extract_horse_comments(full_text, n_runners):
    """
    Extracts the FULL multi-line prose paragraph for each horse
    (e.g. "1 - HAND FULL : Après quatre podiums ... D'autres protagonistes
    lui sont préférables."), not just its first line. Shoe-status and
    trainer-confidence phrases can appear anywhere in the paragraph, so
    truncating to the first line (as the old regex did) silently drops
    signal on any horse whose keyword phrase isn't in line one.
    Returns {horse_num_str: full_comment_text}.
    """
    pattern = re.compile(
        r'^\s*(\d{1,2})\s*[-–.]?\s*([A-Z][A-Z\s\'\.]{2,30})\s*:\s*(.*)$'
    )
    lines = full_text.split('\n')
    comments = {}
    current_num = None
    buffer = []

    for line in lines:
        m = pattern.match(line)
        if m and int(m.group(1)) <= n_runners:
            if current_num is not None:
                comments[current_num] = " ".join(buffer).strip()
            current_num = m.group(1).zfill(2)
            buffer = [m.group(3).strip()]
        elif current_num is not None:
            # Stop buffering once we hit an all-caps section header/footer
            if line.strip() and line.strip().isupper() and len(line.strip()) > 15:
                comments[current_num] = " ".join(buffer).strip()
                current_num = None
                buffer = []
            else:
                buffer.append(line.strip())

    if current_num is not None:
        comments[current_num] = " ".join(buffer).strip()

    return comments


if __name__ == "__main__":
    with open("/home/claude/lonab-ai-bot/fix/sample_real_sept11.txt") as f:
        text = f.read()
    rows = parse_race_card(text, 14)
    print(f"[ATTELE, real 11-Sep-2026] Parsed {len(rows)} rows")
    assert len(rows) == 14
    assert rows[0]["Num"] == "01"
    assert rows[0]["Horse"] == "OH CEAN"
    assert rows[13]["Num"] == "14"
    assert rows[13]["Horse"] == "JAZZ DE PADD"
    assert rows[8]["Horse"] == "HERMES PAT"
    print("[ATTELE] All sanity checks passed against real unpadded-horse-number document.\n")

    with open("/home/claude/lonab-ai-bot/fix/sample_plat.txt") as f:
        text2 = f.read()
    rows2 = parse_race_card(text2, 18)
    print(f"[PLAT] Parsed {len(rows2)} rows")
    assert len(rows2) == 18, "Expected 18 horses"
    assert rows2[0]["Horse"] == "WAPI"
    assert rows2[0]["Discipline"] == "PLAT"
    assert rows2[0]["Chrono"] is None
    assert rows2[0]["Draw"] == 8
    assert rows2[0]["Weight_KG"] == 60.0
    assert rows2[3]["Horse"] == "BIG LOG"
    assert rows2[3]["Earnings"] == 199990.0
    assert rows2[13]["Horse"] == "JUST JIM"
    assert rows2[13]["Weight_KG"] == 54.0
    print("[PLAT] All sanity checks passed against real 13-Sep-2026 ParisLongchamp card.")

