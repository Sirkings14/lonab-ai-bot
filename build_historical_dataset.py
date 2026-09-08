import os
import re
import csv
import json
import hashlib
from pathlib import Path
from collections import defaultdict

import pandas as pd
import pdfplumber


# ============================================================
# LONAB HISTORICAL DATASET BUILDER
# ============================================================

OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HISTORICAL_OUTPUT = OUTPUT_DIR / "historical_races.csv"
MATCH_REPORT_OUTPUT = OUTPUT_DIR / "match_report.csv"
UNMATCHED_OUTPUT = OUTPUT_DIR / "unmatched_documents.csv"
SUMMARY_OUTPUT = OUTPUT_DIR / "dataset_summary.json"


# ============================================================
# FILE DISCOVERY
# ============================================================

def find_first_existing(paths):
    for path in paths:
        p = Path(path)
        if p.exists():
            return p
    return None


def find_manifest():
    candidates = [
        "data/download_manifest.csv",
        "archive/download_manifest.csv",
        "download_manifest.csv",
    ]

    found = find_first_existing(candidates)

    if found:
        return found

    for p in Path(".").rglob("download_manifest.csv"):
        return p

    return None


def find_pdf_directories():
    programme_candidates = [
        "data/raw_programmes",
        "raw_programmes",
    ]

    result_candidates = [
        "data/raw_results",
        "raw_results",
    ]

    programme_dir = find_first_existing(programme_candidates)
    result_dir = find_first_existing(result_candidates)

    return programme_dir, result_dir


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_pdf_text(pdf_path):
    try:
        pages = []

        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(
                    x_tolerance=1,
                    y_tolerance=3
                )

                if text:
                    pages.append(text)

        return "\n".join(pages)

    except Exception as e:
        print(f"❌ Could not read PDF: {pdf_path} -> {e}")
        return ""


# ============================================================
# DATE EXTRACTION
# ============================================================

def normalize_date(day, month, year):
    try:
        return f"{int(year):04d}-{int(month):02d}-{int(day):02d}"
    except Exception:
        return None


def extract_date_from_string(text):
    if not text:
        return None

    patterns = [
        r"(\d{2})[-_/](\d{2})[-_/](\d{4})",
        r"(\d{2})(\d{2})(\d{4})",
    ]

    for pattern in patterns:
        match = re.search(pattern, text)

        if match:
            return normalize_date(
                match.group(1),
                match.group(2),
                match.group(3)
            )

    return None


def extract_date_from_url(url):
    if not url:
        return None

    return extract_date_from_string(url)


def extract_date_from_text(text):
    if not text:
        return None

    patterns = [
        r"(?:LUNDI|MARDI|MERCREDI|JEUDI|VENDREDI|SAMEDI|DIMANCHE)"
        r"\s+(\d{1,2})[ /-]+"
        r"(?:JANVIER|FEVRIER|FÉVRIER|MARS|AVRIL|MAI|JUIN|"
        r"JUILLET|AOUT|AOÛT|SEPTEMBRE|OCTOBRE|NOVEMBRE|DECEMBRE|DÉCEMBRE)"
        r"\s+(\d{4})",

        r"(\d{1,2})/(\d{1,2})/(\d{4})",
        r"(\d{1,2})-(\d{1,2})-(\d{4})",
    ]

    month_map = {
        "JANVIER": 1,
        "FEVRIER": 2,
        "FÉVRIER": 2,
        "MARS": 3,
        "AVRIL": 4,
        "MAI": 5,
        "JUIN": 6,
        "JUILLET": 7,
        "AOUT": 8,
        "AOÛT": 8,
        "SEPTEMBRE": 9,
        "OCTOBRE": 10,
        "NOVEMBRE": 11,
        "DECEMBRE": 12,
        "DÉCEMBRE": 12,
    }

    match = re.search(
        patterns[0],
        text,
        re.IGNORECASE
    )

    if match:
        day = int(match.group(1))

        month_search = re.search(
            r"(JANVIER|FEVRIER|FÉVRIER|MARS|AVRIL|MAI|JUIN|"
            r"JUILLET|AOUT|AOÛT|SEPTEMBRE|OCTOBRE|NOVEMBRE|"
            r"DECEMBRE|DÉCEMBRE)",
            match.group(0),
            re.IGNORECASE
        )

        if month_search:
            month_name = month_search.group(1).upper()
            month = month_map[month_name]
            year = int(match.group(2))

            return normalize_date(day, month, year)

    for pattern in patterns[1:]:
        match = re.search(pattern, text)

        if match:
            return normalize_date(
                match.group(1),
                match.group(2),
                match.group(3)
            )

    return None


# ============================================================
# RACE TYPE
# ============================================================

def detect_race_type(text):
    text_upper = str(text).upper()

    if "QUINTE" in text_upper or "QUINTÉ" in text_upper:
        return "QUINTE"

    if "QUARTE" in text_upper or "QUARTÉ" in text_upper:
        return "QUARTE"

    if "TIERCE" in text_upper or "TIERCÉ" in text_upper:
        return "TIERCE"

    if "4+1" in text_upper:
        return "4+1"

    if "PMUB" in text_upper:
        return "PMUB"

    return "UNKNOWN"


# ============================================================
# PROGRAMME PARSING
# ============================================================

def parse_race_metadata(text, fallback_date=None):
    race_date = extract_date_from_text(text)

    if not race_date:
        race_date = fallback_date

    race_type = detect_race_type(text)

    distance = None

    distance_match = re.search(
        r"(\d[\d\s]{2,6})\s*(?:METRES|MÈTRES|M)",
        text,
        re.IGNORECASE
    )

    if distance_match:
        distance_str = re.sub(
            r"\D",
            "",
            distance_match.group(1)
        )

        try:
            distance = int(distance_str)
        except ValueError:
            distance = None

    competitors = None

    competitor_match = re.search(
        r"(\d{1,2})\s+CONCURRENTS",
        text,
        re.IGNORECASE
    )

    if competitor_match:
        competitors = int(
            competitor_match.group(1)
        )

    racecourse = None

    course_patterns = [
        r"([A-ZÀ-ÖØ-Ý' -]{3,40})\s*-\s*PRIX",
        r"([A-ZÀ-ÖØ-Ý' -]{3,40})\s*-\s*[A-ZÀ-ÖØ-Ý' -]+\s*-\s*(?:HAIES|PLAT|ATTELÉ|ATTELE|STEEPLE)",
    ]

    for pattern in course_patterns:
        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:
            candidate = match.group(1).strip()

            if len(candidate) >= 3:
                racecourse = candidate
                break

    race_name = None

    name_match = re.search(
        r"([A-ZÀ-ÖØ-Ý' -]{3,40})\s*-\s*(PRIX\s+[A-ZÀ-ÖØ-Ý' -]{3,80})",
        text,
        re.IGNORECASE
    )

    if name_match:
        race_name = (
            name_match.group(1).strip()
            + " - "
            + name_match.group(2).strip()
        )

    return {
        "Race_Date": race_date,
        "Race_Type": race_type,
        "Racecourse": racecourse,
        "Race_Name": race_name,
        "Distance": distance,
        "Declared_Competitors": competitors,
    }


def normalize_horse_name(name):
    name = re.sub(
        r"\s+",
        " ",
        str(name).strip().upper()
    )

    return name


def parse_programme_runners(text):
    runners = []

    # Main format found in LONAB programme PDFs:
    #
    # 1 - HORSE NAME : comment
    #
    pattern = re.compile(
        r"(?m)^\s*(\d{1,2})\s*[-–]\s*"
        r"([A-ZÀ-ÖØ-Ý0-9' .-]{3,60}?)\s*:"
    )

    matches = list(
        pattern.finditer(text)
    )

    seen_numbers = set()

    for index, match in enumerate(matches):

        horse_number = int(
            match.group(1)
        )

        horse_name = normalize_horse_name(
            match.group(2)
        )

        # Avoid obvious non-runner text.
        blocked_words = [
            "ARRIVEE",
            "ARRIVÉE",
            "RESULTATS",
            "RÉSULTATS",
            "COURSE",
            "PRIX",
            "MEILLEURS",
            "GAINS",
        ]

        if any(
            word in horse_name
            for word in blocked_words
        ):
            continue

        if horse_number in seen_numbers:
            continue

        seen_numbers.add(
            horse_number
        )

        comment_start = match.end()

        if index + 1 < len(matches):
            comment_end = matches[
                index + 1
            ].start()
        else:
            comment_end = min(
                len(text),
                comment_start + 2000
            )

        comment = text[
            comment_start:comment_end
        ]

        comment = re.sub(
            r"\s+",
            " ",
            comment
        ).strip()

        runners.append({
            "Horse_Number": horse_number,
            "Horse_Name": horse_name,
            "Programme_Comment": comment[:1500],
        })

    return runners


# ============================================================
# RESULT PARSING
# ============================================================

def parse_result_arrival(text):
    patterns = [
        r"ARR(?:IVEE|IVÉE|IV)?\s*[:\-]?\s*"
        r"(\d{1,2}(?:\s*[-–]\s*\d{1,2}){1,9})",

        r"ARR\s*[:\-]?\s*"
        r"(\d{1,2}(?:\s*[-–]\s*\d{1,2}){1,9})",

        r"ARRIVEE\s*DU.*?[:\-]\s*"
        r"(\d{1,2}(?:\s*[-–]\s*\d{1,2}){1,9})",
    ]

    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            numbers = re.findall(
                r"\d{1,2}",
                match.group(1)
            )

            numbers = [
                int(number)
                for number in numbers
            ]

            if len(numbers) >= 2:
                return numbers

    return []


# ============================================================
# MANIFEST
# ============================================================

def load_manifest(manifest_path):
    records = []

    if not manifest_path:
        return records

    try:
        with open(
            manifest_path,
            "r",
            encoding="utf-8-sig",
            newline=""
        ) as file:

            reader = csv.DictReader(file)

            for row in reader:

                document_type = (
                    row.get(
                        "document_type",
                        ""
                    )
                    .strip()
                    .lower()
                )

                original_file = row.get(
                    "file",
                    ""
                )

                filename = Path(
                    original_file
                ).name

                records.append({
                    "document_type": document_type,
                    "url": row.get(
                        "url",
                        ""
                    ),
                    "filename": filename,
                    "date": extract_date_from_url(
                        row.get(
                            "url",
                            ""
                        )
                    ),
                })

    except Exception as e:
        print(
            f"❌ Could not read manifest: {e}"
        )

    return records


# ============================================================
# FILE INDEXING
# ============================================================

def build_file_index(directory):
    index = {}

    if not directory:
        return index

    for pdf_path in Path(
        directory
    ).rglob("*.pdf"):

        index[pdf_path.name] = pdf_path

    return index


# ============================================================
# RESULT MATCHING
# ============================================================

def result_match_score(
    programme_type,
    programme_text,
    result_record,
    result_text
):
    score = 0

    result_type = detect_race_type(
        result_text
        + " "
        + result_record.get(
            "url",
            ""
        )
    )

    if programme_type != "UNKNOWN":
        if programme_type == result_type:
            score += 100

    programme_upper = programme_text.upper()
    result_upper = (
        result_text.upper()
        + " "
        + result_record.get(
            "url",
            ""
        ).upper()
    )

    keywords = [
        "QUINTE",
        "QUINTÉ",
        "QUARTE",
        "QUARTÉ",
        "TIERCE",
        "TIERCÉ",
        "4+1",
    ]

    for keyword in keywords:

        if (
            keyword in programme_upper
            and keyword in result_upper
        ):
            score += 10

    arrival = parse_result_arrival(
        result_text
    )

    if len(arrival) >= 2:
        score += 20

    return score


def select_best_result(
    programme_record,
    programme_text,
    results_by_date,
    result_file_index
):
    race_date = programme_record.get(
        "date"
    )

    if not race_date:
        return None, None, []

    candidates = results_by_date.get(
        race_date,
        []
    )

    if not candidates:
        return None, None, []

    programme_type = detect_race_type(
        programme_text
    )

    scored = []

    for candidate in candidates:

        filename = candidate.get(
            "filename"
        )

        pdf_path = result_file_index.get(
            filename
        )

        if not pdf_path:
            continue

        result_text = extract_pdf_text(
            pdf_path
        )

        arrival = parse_result_arrival(
            result_text
        )

        score = result_match_score(
            programme_type,
            programme_text,
            candidate,
            result_text
        )

        scored.append({
            "record": candidate,
            "path": pdf_path,
            "text": result_text,
            "arrival": arrival,
            "score": score,
        })

    if not scored:
        return None, None, []

    scored.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    best = scored[0]

    return (
        best["record"],
        best,
        scored
    )


# ============================================================
# RACE ID
# ============================================================

def create_race_id(
    race_date,
    programme_filename
):
    raw = (
        str(race_date)
        + "|"
        + str(programme_filename)
    )

    digest = hashlib.sha1(
        raw.encode(
            "utf-8"
        )
    ).hexdigest()[:12]

    return (
        f"{race_date}_{digest}"
    )


# ============================================================
# MAIN BUILDER
# ============================================================

def build_dataset():

    print("\n")
    print("=" * 65)
    print("LONAB HISTORICAL DATASET BUILDER")
    print("=" * 65)

    manifest_path = find_manifest()

    programme_dir, result_dir = (
        find_pdf_directories()
    )

    print(
        f"Manifest: {manifest_path}"
    )

    print(
        f"Programme directory: {programme_dir}"
    )

    print(
        f"Result directory: {result_dir}"
    )

    if not programme_dir:
        raise FileNotFoundError(
            "Programme PDF directory not found."
        )

    if not result_dir:
        raise FileNotFoundError(
            "Result PDF directory not found."
        )

    manifest = load_manifest(
        manifest_path
    )

    programme_file_index = (
        build_file_index(
            programme_dir
        )
    )

    result_file_index = (
        build_file_index(
            result_dir
        )
    )

    print(
        f"\nProgramme PDFs found: "
        f"{len(programme_file_index)}"
    )

    print(
        f"Result PDFs found: "
        f"{len(result_file_index)}"
    )

    programme_records = []

    result_records = []

    for record in manifest:

        if (
            record["document_type"]
            == "programme"
        ):
            if (
                record["filename"]
                in programme_file_index
            ):
                programme_records.append(
                    record
                )

        elif (
            record["document_type"]
            == "result"
        ):
            if (
                record["filename"]
                in result_file_index
            ):
                result_records.append(
                    record
                )

    # Fallback if manifest is unavailable.
    if not programme_records:

        print(
            "\n⚠️ No programme records found "
            "in manifest. Using filenames."
        )

        for filename, path in (
            programme_file_index.items()
        ):

            programme_records.append({
                "document_type": "programme",
                "url": "",
                "filename": filename,
                "date": extract_date_from_string(
                    filename
                ),
            })

    if not result_records:

        print(
            "\n⚠️ No result records found "
            "in manifest. Using filenames."
        )

        for filename, path in (
            result_file_index.items()
        ):

            result_records.append({
                "document_type": "result",
                "url": "",
                "filename": filename,
                "date": extract_date_from_string(
                    filename
                ),
            })

    results_by_date = defaultdict(
        list
    )

    for result in result_records:

        if result.get("date"):

            results_by_date[
                result["date"]
            ].append(
                result
            )

    dataset_rows = []
    match_rows = []
    unmatched_rows = []

    total_programmes = len(
        programme_records
    )

    print(
        f"\nProcessing "
        f"{total_programmes} programme documents..."
    )

    for index, programme_record in enumerate(
        programme_records,
        start=1
    ):

        filename = programme_record[
            "filename"
        ]

        pdf_path = programme_file_index.get(
            filename
        )

        if not pdf_path:
            continue

        print(
            f"[{index}/{total_programmes}] "
            f"{filename}"
        )

        programme_text = extract_pdf_text(
            pdf_path
        )

        if not programme_text:

            unmatched_rows.append({
                "Document": filename,
                "Type": "programme",
                "Reason": "EMPTY_OR_UNREADABLE_PDF",
            })

            continue

        fallback_date = programme_record.get(
            "date"
        )

        metadata = parse_race_metadata(
            programme_text,
            fallback_date
        )

        race_date = (
            metadata["Race_Date"]
            or fallback_date
        )

        runners = parse_programme_runners(
            programme_text
        )

        if not runners:

            unmatched_rows.append({
                "Document": filename,
                "Type": "programme",
                "Reason": "NO_RUNNERS_PARSED",
            })

            continue

        best_record, best_result, all_candidates = (
            select_best_result(
                programme_record,
                programme_text,
                results_by_date,
                result_file_index
            )
        )

        if not best_result:

            unmatched_rows.append({
                "Document": filename,
                "Type": "programme",
                "Reason": "NO_MATCHING_RESULT",
            })

            match_rows.append({
                "Programme_File": filename,
                "Race_Date": race_date,
                "Race_Type": metadata[
                    "Race_Type"
                ],
                "Runners_Parsed": len(
                    runners
                ),
                "Result_File": "",
                "Arrival": "",
                "Match_Score": 0,
                "Status": "UNMATCHED",
            })

            continue

        arrival = best_result[
            "arrival"
        ]

        if len(arrival) < 2:

            unmatched_rows.append({
                "Document": filename,
                "Type": "programme",
                "Reason": "RESULT_HAS_NO_VALID_ARRIVAL",
            })

            match_rows.append({
                "Programme_File": filename,
                "Race_Date": race_date,
                "Race_Type": metadata[
                    "Race_Type"
                ],
                "Runners_Parsed": len(
                    runners
                ),
                "Result_File": best_record[
                    "filename"
                ],
                "Arrival": "",
                "Match_Score": best_result[
                    "score"
                ],
                "Status": "INVALID_RESULT",
            })

            continue

        race_id = create_race_id(
            race_date,
            filename
        )

        arrival_positions = {
            horse_number: position
            for position, horse_number
            in enumerate(
                arrival,
                start=1
            )
        }

        for runner in runners:

            horse_number = runner[
                "Horse_Number"
            ]

            finish_position = (
                arrival_positions.get(
                    horse_number
                )
            )

            dataset_rows.append({

                # RACE IDENTIFICATION

                "Race_ID": race_id,
                "Race_Date": race_date,
                "Race_Type": metadata[
                    "Race_Type"
                ],
                "Racecourse": metadata[
                    "Racecourse"
                ],
                "Race_Name": metadata[
                    "Race_Name"
                ],
                "Distance": metadata[
                    "Distance"
                ],
                "Declared_Competitors": metadata[
                    "Declared_Competitors"
                ],

                # RUNNER

                "Horse_Number": horse_number,
                "Horse_Name": runner[
                    "Horse_Name"
                ],
                "Programme_Comment": runner[
                    "Programme_Comment"
                ],

                # OFFICIAL RESULT

                "Finish_Position": (
                    finish_position
                    if finish_position
                    else pd.NA
                ),

                "Is_Winner": int(
                    finish_position == 1
                ),

                "Is_Top_3": int(
                    finish_position is not None
                    and finish_position <= 3
                ),

                "Is_Top_4": int(
                    finish_position is not None
                    and finish_position <= 4
                ),

                "Is_Top_5": int(
                    finish_position is not None
                    and finish_position <= 5
                ),

                "Is_Returned": int(
                    finish_position is not None
                ),

                # SOURCE TRACKING

                "Programme_File": filename,

                "Result_File": best_record[
                    "filename"
                ],

                "Programme_URL": programme_record.get(
                    "url",
                    ""
                ),

                "Result_URL": best_record.get(
                    "url",
                    ""
                ),

                "Official_Arrival": "-".join(
                    map(
                        str,
                        arrival
                    )
                ),

                "Match_Score": best_result[
                    "score"
                ],
            })

        match_rows.append({

            "Programme_File": filename,

            "Race_Date": race_date,

            "Race_Type": metadata[
                "Race_Type"
            ],

            "Runners_Parsed": len(
                runners
            ),

            "Result_File": best_record[
                "filename"
            ],

            "Arrival": "-".join(
                map(
                    str,
                    arrival
                )
            ),

            "Match_Score": best_result[
                "score"
            ],

            "Status": "MATCHED",
        })

    # ========================================================
    # SAVE DATA
    # ========================================================

    dataset = pd.DataFrame(
        dataset_rows
    )

    match_report = pd.DataFrame(
        match_rows
    )

    unmatched = pd.DataFrame(
        unmatched_rows
    )

    if not dataset.empty:

        dataset = dataset.drop_duplicates(
            subset=[
                "Race_ID",
                "Horse_Number"
            ]
        )

        dataset = dataset.sort_values(
            by=[
                "Race_Date",
                "Race_ID",
                "Horse_Number"
            ],
            na_position="last"
        )

        dataset.to_csv(
            HISTORICAL_OUTPUT,
            index=False,
            encoding="utf-8-sig"
        )

    match_report.to_csv(
        MATCH_REPORT_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    unmatched.to_csv(
        UNMATCHED_OUTPUT,
        index=False,
        encoding="utf-8-sig"
    )

    summary = {

        "programme_records": len(
            programme_records
        ),

        "result_records": len(
            result_records
        ),

        "dataset_rows": len(
            dataset
        ),

        "unique_races": (
            int(
                dataset["Race_ID"].nunique()
            )
            if not dataset.empty
            else 0
        ),

        "unique_horses": (
            int(
                dataset["Horse_Name"].nunique()
            )
            if not dataset.empty
            else 0
        ),

        "winners": (
            int(
                dataset["Is_Winner"].sum()
            )
            if not dataset.empty
            else 0
        ),

        "top_3": (
            int(
                dataset["Is_Top_3"].sum()
            )
            if not dataset.empty
            else 0
        ),

        "unmatched_documents": len(
            unmatched
        ),
    }

    with open(
        SUMMARY_OUTPUT,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4,
            ensure_ascii=False
        )

    print("\n")
    print("=" * 65)
    print("DATASET BUILD COMPLETE")
    print("=" * 65)

    for key, value in summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        f"\nDataset saved to: "
        f"{HISTORICAL_OUTPUT}"
    )

    print(
        f"Match report saved to: "
        f"{MATCH_REPORT_OUTPUT}"
    )

    print(
        f"Unmatched report saved to: "
        f"{UNMATCHED_OUTPUT}"
    )


if __name__ == "__main__":
    build_dataset()
