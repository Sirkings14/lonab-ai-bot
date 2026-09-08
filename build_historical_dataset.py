import os
import re
import csv
import requests
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime


URL_RESULTS = "https://lonab.bf/resultats-et-rapports"

OUTPUT_FILE = "historical_races.csv"


def clean_text(value):
    if value is None:
        return ""

    value = str(value)
    value = value.replace("\n", " ")
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def extract_numbers(text):

    if not text:
        return []

    numbers = re.findall(r"\b\d{1,2}\b", str(text))

    return numbers


def create_empty_dataset():

    columns = [

        "Race_ID",
        "Race_Date",
        "Track",
        "Race_Name",
        "Race_Type",
        "Distance",
        "Surface",
        "Autostart",

        "Horse_Number",
        "Horse_Name",

        "Driver",
        "Trainer",

        "Age",
        "Sex",
        "Earnings",

        "Shoe_Status",
        "Days_Rest",
        "Recent_Form",
        "DQ_History",
        "Speed_Index",

        "Market_Odds",

        "Finish_Position",

        "Is_Winner",
        "Is_Top3",
        "Is_Top5"
    ]

    return pd.DataFrame(columns=columns)


def calculate_targets(finish_position):

    if finish_position is None:
        return 0, 0, 0

    try:

        position = int(finish_position)

    except:

        return 0, 0, 0


    is_winner = 1 if position == 1 else 0

    is_top3 = 1 if position <= 3 else 0

    is_top5 = 1 if position <= 5 else 0


    return is_winner, is_top3, is_top5


def parse_arrival(text):

    if not text:
        return []

    patterns = [

        r"ARRIVEE\s*:?\s*([0-9\s\-–]+)",

        r"Arrivée\s*:?\s*([0-9\s\-–]+)",

        r"RESULTAT\s*:?\s*([0-9\s\-–]+)"
    ]


    for pattern in patterns:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE
        )

        if match:

            result = extract_numbers(
                match.group(1)
            )

            if result:

                return result


    return []


def scrape_results_page():

    headers = {

        "User-Agent":
        "Mozilla/5.0"
    }


    print(
        "Downloading historical results page..."
    )


    response = requests.get(

        URL_RESULTS,

        headers=headers,

        timeout=30
    )


    response.raise_for_status()


    soup = BeautifulSoup(

        response.text,

        "html.parser"
    )


    page_text = clean_text(

        soup.get_text(
            " "
        )
    )


    return page_text


def build_dataset():

    print(
        "=" * 60
    )

    print(
        "BUILDING HISTORICAL RACE DATASET"
    )

    print(
        "=" * 60
    )


    if os.path.exists(
        OUTPUT_FILE
    ):

        historical_df = pd.read_csv(
            OUTPUT_FILE
        )

        print(

            f"Existing historical records: "
            f"{len(historical_df)}"

        )

    else:

        historical_df = create_empty_dataset()

        print(
            "Creating new historical database..."
        )


    try:

        page_text = scrape_results_page()

    except Exception as e:

        print(
            f"ERROR DOWNLOADING RESULTS: {e}"
        )

        return


    arrival = parse_arrival(
        page_text
    )


    if not arrival:

        print(
            "No race arrival found."
        )

        print(
            "The scraper structure works, "
            "but we need to inspect the exact "
            "LONAB results page format."
        )

        return


    print(
        "Latest arrival found:"
    )

    print(
        arrival
    )


    print(
        "Dataset builder completed."
    )


    historical_df.to_csv(

        OUTPUT_FILE,

        index=False,

        encoding="utf-8"
    )


if __name__ == "__main__":

    build_dataset()
