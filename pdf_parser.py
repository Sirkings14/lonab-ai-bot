"""
Reconstructs per-horse rows from the LONAB "Journal Hippique" PMU'B PDF text.

IMPORTANT HISTORY: this module was originally built against text produced
by a different (smarter) PDF-to-text tool, which reshaped the table into
column-major blocks (all N horses' odds together, then all N horse
numbers, etc). Raw pdfplumber extraction of the ACTUAL production PDF
does NOT produce that shape — verified directly against a real uploaded
document (JH_PMUB_DU_08-09-2026.pdf). The real table extracts as clean
ROW-MAJOR text: one full horse record per line, e.g.

  "01 HAMANDIO J.REVELEY N. GEORGE & A. ZETTERHOLM C.ALEXANDER H.9 72.KG 3.T.2.8.5 347 012 7/1 6/1"

This version parses that real row-major shape directly. The two-column
prose commentary section (horse-by-horse analysis) DOES get merged
left/right column text onto the same line by plain extract_text() — that
part of the original diagnosis was correct — but it doesn't block the
table, which is what actually matters for the model's features.
"""
import re


SEX_AGE_RE = re.compile(r'^[HMF]\.(\d{1,2})$')
ODDS_RE = re.compile(r'^(\d{1,3})/1$')
FIELD_START_RE = re.compile(r'^[A-Z]{1,4}\.')
ORG_STARTERS = {'GAEC', 'TERRAINS', 'EC.', 'SANG.', 'MP.'}

# Tail-field patterns, tried in order. Each captures, from the point right
# after the name-blob: sex/age, then discipline-specific fields, then perf,
# gains, and the two odds columns. Only the ATTELE (trot) shape has been
# verified against a raw sample; the CORDE-included PLAT variant is
# extrapolated from the column-major PLAT sample and not yet confirmed
# against a raw pdfplumber PLAT document.
TAIL_PATTERNS = [
    ("ATTELE", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'([\d\s]+\.M)\s+'
        r'(\d\.\d{2}\.\d{2})\s+'
        r'([0-9A-Za-z]+(?:\.[0-9A-Za-z]+){2,6})\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        r'(\d{1,3}/1)\s+(\d{1,3}/1)\s*$'
    )),
    ("PLAT_WITH_DRAW", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'(\d{1,2})\s+'
        r'(\d{1,3}(?:\.\d)?\.KG)\s+'
        r'([0-9A-Za-z]+(?:\.[0-9A-Za-z]+){2,6})\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        r'(\d{1,3}/1)\s+(\d{1,3}/1)\s*$'
    )),
    ("PLAT_NO_DRAW", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'(\d{1,3}(?:\.\d)?\.KG)\s+'
        r'([0-9A-Za-z]+(?:\.[0-9A-Za-z]+){2,6})\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        r'(\d{1,3}/1)\s+(\d{1,3}/1)\s*$'
    )),
]

ROW_START_RE = re.compile(r'^(\d{1,2})\s+(.+)$')


def detect_discipline(full_text):
    """
    The PDF uses a different table schema per discipline. ATTELE (trot)
    publishes Distance/Chrono per horse. Others (PLAT, HAIES/obstacle)
    publish Weight instead, sometimes with a draw/Corde number too.
    """
    m = re.search(r'CONCURRENTS.*?-\s*([A-ZÀ-Üa-zà-ü\-]+)\s*(?:\n|$)', full_text)
    if m and 'ATTELE' in m.group(1).upper():
        return 'ATTELE'
    if re.search(r'\bATTELE\b', full_text, re.IGNORECASE):
        return 'ATTELE'
    return 'PLAT'  # covers PLAT, HAIES, and other non-trot disciplines


def _split_names(blob):
    """
    Splits the "HORSE JOCKEY TRAINER OWNER" blob into its four fields.
    Horse name = leading tokens with no period, until the first token
    that looks like a person's initials ("J.REVELEY") or a known
    organization-name starter ("GAEC", "EC.", ...). Remaining tokens are
    split into up to 3 more fields the same way, with "&" treated as
    joining two names into the same field (joint jockeys/trainers).
    Verified against 15/16 real rows; the one miss (a joint-trainer name
    with no period on its first word) falls back gracefully rather than
    crashing, since an unmatched name just means a neutral rank default
    downstream, not an error.
    """
    tokens = blob.split(' ')
    horse_tokens = []
    i = 0
    while i < len(tokens):
        t = tokens[i]
        if FIELD_START_RE.match(t) or t in ORG_STARTERS:
            break
        horse_tokens.append(t)
        i += 1
    remainder = tokens[i:]
    if not remainder:
        return ' '.join(horse_tokens), '', '', ''

    fields = []
    current = [remainder[0]]
    prev = remainder[0]
    for tok in remainder[1:]:
        if prev == '&':
            current.append(tok)
        elif FIELD_START_RE.match(tok) or tok in ORG_STARTERS:
            fields.append(' '.join(current))
            current = [tok]
        else:
            current.append(tok)
        prev = tok
    fields.append(' '.join(current))

    while len(fields) < 3:
        fields.append('')
    if len(fields) > 3:
        fields = fields[:2] + [' '.join(fields[2:])]

    return ' '.join(horse_tokens), fields[0], fields[1], fields[2]


def parse_race_card(full_text, n_runners, discipline=None):
    """
    full_text: raw text extracted from the PDF (plain extract_text() is
    sufficient — the table itself extracts cleanly without needing any
    column-separation trick).
    n_runners: number of horses in the race (from "N CONCURRENTS").
    discipline: 'ATTELE' or 'PLAT'. Auto-detected from full_text if omitted.
    Returns a list of dicts, one per horse, or [] if no matching rows found.
    """
    if discipline is None:
        discipline = detect_discipline(full_text)

    ordered_patterns = TAIL_PATTERNS if discipline == 'ATTELE' else (
        [p for p in TAIL_PATTERNS if p[0] != 'ATTELE'] +
        [p for p in TAIL_PATTERNS if p[0] == 'ATTELE']
    )

    rows_by_num = {}
    for line in full_text.split('\n'):
        line = line.strip()
        m = ROW_START_RE.match(line)
        if not m:
            continue
        num_str, rest = m.group(1), m.group(2)
        try:
            num_int = int(num_str)
        except ValueError:
            continue
        if not (1 <= num_int <= n_runners):
            continue

        for tail_name, pattern in ordered_patterns:
            tail_match = pattern.search(rest)
            if not tail_match:
                continue
            names_blob = rest[:tail_match.start()].strip()
            if not names_blob:
                continue
            groups = tail_match.groups()

            if tail_name == 'ATTELE':
                sex_age, dist, chrono, perf, gains, odds1, odds2 = groups
                draw, weight = None, None
                dist_val = float(dist.replace(' ', '').replace('.M', ''))
            elif tail_name == 'PLAT_WITH_DRAW':
                sex_age, draw, weight, perf, gains, odds1, odds2 = groups
                chrono, dist_val = None, None
                weight = float(weight.replace('.KG', ''))
                draw = int(draw)
            else:  # PLAT_NO_DRAW
                sex_age, weight, perf, gains, odds1, odds2 = groups
                chrono, dist_val, draw = None, None, None
                weight = float(weight.replace('.KG', ''))

            sex, age = sex_age.split('.')
            horse, jockey, trainer, owner = _split_names(names_blob)
            gains_val = float(gains.replace(' ', ''))
            odds_val = float(odds1.split('/')[0]) + 1.0

            rows_by_num[num_str.zfill(2)] = {
                "Num": num_str.zfill(2),
                "Horse": horse,
                "Driver": jockey,
                "Trainer": trainer,
                "Owner": owner,
                "Sex": sex,
                "Age": float(age),
                "Distance_M": dist_val,
                "Chrono": chrono,
                "Draw": draw,
                "Weight_KG": weight,
                "Perf": perf,
                "Earnings": gains_val,
                "Decimal_Odds": odds_val,
                "Discipline": discipline,
            }
            break  # stop trying other tail patterns for this line

    if len(rows_by_num) < n_runners:
        return []  # incomplete — don't return a partial, possibly-wrong race

    return [rows_by_num[str(i).zfill(2)] for i in range(1, n_runners + 1)
            if str(i).zfill(2) in rows_by_num]


def extract_pdf_text_multi_strategy(pdf):
    """
    The data table extracts cleanly with plain extract_text() — no column
    trickery needed there. The two-column horse-by-horse prose commentary
    DOES get its left/right columns merged onto the same line by plain
    extraction, which can corrupt the comment-based signals (shoe status,
    rest, confidence phrases). We concatenate a left/right-cropped pass
    too so extract_horse_comments has a chance at a cleaner version of
    that section; the table parser only needs the plain pass and ignores
    the rest.
    """
    full_text = ""
    for page in pdf.pages:
        full_text += "\n" + (page.extract_text(x_tolerance=1) or "")
        try:
            width = page.width
            left = page.within_bbox((0, 0, width / 2, page.height)).extract_text() or ""
            right = page.within_bbox((width / 2, 0, width, page.height)).extract_text() or ""
            full_text += "\n" + left + "\n" + right
        except Exception:
            pass
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

    Known limitation: when the source text has two prose columns merged
    onto the same line (see extract_pdf_text_multi_strategy), a comment
    can come out with another horse's sentence fragments mixed in. This
    degrades keyword detection accuracy for shoe/rest signals on some
    horses but doesn't crash — an unmatched keyword just falls back to
    the neutral default, same as today.
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
    with open("/home/claude/page1_default.txt") as f:
        text = f.read()
    rows = parse_race_card(text, 16, discipline='HAIES_LIKE')
    print(f"[HAIES, real 08-Sep-2026, raw pdfplumber text] Parsed {len(rows)} rows")
    assert len(rows) == 16
    assert rows[0]["Num"] == "01"
    assert rows[0]["Horse"] == "HAMANDIO"
    assert rows[0]["Driver"] == "J.REVELEY"
    assert rows[0]["Trainer"] == "N. GEORGE & A. ZETTERHOLM"
    assert rows[0]["Owner"] == "C.ALEXANDER"
    assert rows[0]["Weight_KG"] == 72.0
    assert rows[0]["Earnings"] == 347012.0
    assert rows[0]["Decimal_Odds"] == 8.0
    assert rows[15]["Horse"] == "LORELEY DES PLACES"
    assert rows[15]["Owner"] == "MP.LEJEUNE/L.AUBANEL"
    print("All checks passed against the exact raw pdfplumber output of a real uploaded PDF.")
