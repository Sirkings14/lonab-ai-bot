"""
Reconstructs per-horse rows from the LONAB "Journal Hippique" PMU'B PDF text.
The real table extracts as clean ROW-MAJOR text: one full horse record per line, e.g.
  "01 HAMANDIO J.REVELEY N. GEORGE & A. ZETTERHOLM C.ALEXANDER H.9 72.KG 3.T.2.8.5 347 012 7/1 6/1"
"""
import re


SEX_AGE_RE = re.compile(r'^[HMF]\.(\d{1,2})$')
ODDS_RE = re.compile(r'^(\d{1,3})/1$')
FIELD_START_RE = re.compile(r'^[A-Z]{1,4}\.')
ORG_STARTERS = {'GAEC', 'TERRAINS', 'EC.', 'SANG.', 'MP.'}

PERF = r'([0-9A-Za-z]+(?:\.[0-9A-Za-z]*){2,6})'     # "3.T.2.8.5" and also "0..0.3.3" (empty slot)
ODDS2 = r'(\d{1,3}/1?)\s+(\d{1,3}/1?)(?=\s|$)'      # "13/1 10/1" and also "60/ 63/1"

TAIL_PATTERNS = [
    ("ATTELE", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'([\d\s]+\.M)\s+'
        r'(\d{1,2}\.\d{2}\.\d{1,2})\s+'
        + PERF + r'\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        + ODDS2, re.IGNORECASE
    )),
    ("PLAT_WITH_DRAW", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'(\d{1,2})\s+'
        r'(\d{1,3}(?:[.,]\d)?\.KG)\s+'
        + PERF + r'\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        + ODDS2, re.IGNORECASE
    )),
    ("PLAT_NO_DRAW", re.compile(
        r'([HMF]\.\d{1,2})\s+'
        r'(\d{1,3}(?:[.,]\d)?\.KG)\s+'
        + PERF + r'\s+'
        r'(\d{1,3}(?:\s\d{3})*)\s+'
        + ODDS2, re.IGNORECASE
    )),
]

ROW_START_RE = re.compile(r'^(\d{1,2})\s+(.+)$')


FR_MONTHS = {
    'JANVIER': '01', 'FEVRIER': '02', 'FÉVRIER': '02', 'MARS': '03', 'AVRIL': '04',
    'MAI': '05', 'JUIN': '06', 'JUILLET': '07', 'AOUT': '08', 'AOÛT': '08',
    'SEPTEMBRE': '09', 'OCTOBRE': '10', 'NOVEMBRE': '11', 'DECEMBRE': '12', 'DÉCEMBRE': '12',
}

BET_LABEL = r'(?:"4\+1"|4\+1|"QUARTE"|QUARTE|"QUINTE\+"|QUINTE\+|"TIERCE"|TIERCE)'

HEADER_DATE_RE = re.compile(
    rf'{BET_LABEL}\s+DU\s+\w+\s+(\d{{1,2}})\s+([A-ZÉÛ]+)\s+(\d{{4}})', re.IGNORECASE
)


def extract_program_own_date(full_text):
    """Reads the program's own date from its header, e.g. '"QUARTE" DU MARDI 08 SEPTEMBRE 2026' -> '08-09-2026'.
    Skips 'ARRIVEE DU ...' recap lines. Returns 'DD-MM-YYYY' or None."""
    for m in HEADER_DATE_RE.finditer(full_text):
        preceding = full_text[max(0, m.start() - 20):m.start()]
        if re.search(r'ARRIV[ÉE]E?\s*$', preceding, re.IGNORECASE):
            continue
        dd = m.group(1).zfill(2)
        month_name = m.group(2).upper()
        mm = FR_MONTHS.get(month_name)
        if not mm:
            continue
        yyyy = m.group(3)
        return f"{dd}-{mm}-{yyyy}"
    return None


def detect_discipline(full_text):
    """ATTELE (trot) publishes Distance/Chrono per horse. Others (PLAT, HAIES) publish Weight."""
    m = re.search(r'CONCURRENTS.*?-\s*([A-ZÀ-Üa-zà-ü\-]+)\s*(?:\n|$)', full_text)
    if m and 'ATTELE' in m.group(1).upper():
        return 'ATTELE'
    if re.search(r'\bATTELE\b', full_text, re.IGNORECASE):
        return 'ATTELE'
    return 'PLAT'


def _split_names(blob):
    """Splits the 'HORSE JOCKEY TRAINER OWNER' blob into its four fields."""
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


LAST_PARSE_MISSING = []


def parse_race_card(full_text, n_runners, discipline=None):
    """Returns a list of dicts, one per horse, or [] if the race could not be read."""
    if discipline is None:
        discipline = detect_discipline(full_text)

    ordered_patterns = TAIL_PATTERNS if discipline == 'ATTELE' else (
        [p for p in TAIL_PATTERNS if p[0] != 'ATTELE'] +
        [p for p in TAIL_PATTERNS if p[0] == 'ATTELE']
    )

    rows_by_num = {}
    scratched_nums = set()
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

        # Scratched horses print "N.P." in every column.
        if rest.count('N.P.') >= 3 or 'NON PARTANT' in rest:
            scratched_nums.add(num_str.zfill(2))
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
                weight = float(re.sub(r'\.kg$', '', weight, flags=re.I).replace(',', '.'))
                draw = int(draw)
            else:  # PLAT_NO_DRAW
                sex_age, weight, perf, gains, odds1, odds2 = groups
                chrono, dist_val, draw = None, None, None
                weight = float(re.sub(r'\.kg$', '', weight, flags=re.I).replace(',', '.'))

            sex, age = sex_age.split('.')
            horse, jockey, trainer, owner = _split_names(names_blob)
            gains_val = float(gains.replace(' ', ''))
            odds_val = float(odds1.split('/')[0]) + 1.0
            odds2_val = float(odds2.split('/')[0]) + 1.0

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
                "Decimal_Odds2": odds2_val,
                "Discipline": discipline,
            }
            break

    expected_nums = {str(i).zfill(2) for i in range(1, n_runners + 1)}
    missing = expected_nums - set(rows_by_num.keys()) - scratched_nums
    global LAST_PARSE_MISSING
    LAST_PARSE_MISSING = sorted(missing)
    # Up to 2 unreadable rows on a field of 8+ no longer cancel the whole race.
    if missing and not (n_runners >= 8 and len(missing) <= 2
                        and len(rows_by_num) >= n_runners - 2 - len(scratched_nums)):
        return []

    return [rows_by_num[str(i).zfill(2)] for i in range(1, n_runners + 1)
            if str(i).zfill(2) in rows_by_num]


def extract_pdf_text_multi_strategy(pdf):
    """Plain extraction for the table, plus left/right-cropped passes for the two-column prose comments."""
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
    """Converts 'M.SS.HH' chrono (e.g. "1.12.50") to the seconds scale used historically: 12.50"""
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
    """Extracts the full multi-line prose paragraph for each horse. Returns {horse_num_str: text}."""
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
