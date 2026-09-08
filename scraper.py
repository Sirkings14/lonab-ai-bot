import os
import re
import time
import hashlib
import urllib.parse
from datetime import datetime

import requests
from bs4 import BeautifulSoup


BASE_URL = "https://lonab.bf"

PROGRAMME_URL = "https://lonab.bf/fr/programme-pmub"
RESULTS_URL = "https://lonab.bf/fr/resultats-gains-pmub"

DATA_DIR = "data"
PROGRAMME_DIR = os.path.join(DATA_DIR, "raw_programmes")
RESULTS_DIR = os.path.join(DATA_DIR, "raw_results")
ARCHIVE_DIR = os.path.join(DATA_DIR, "archive")


HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}


def ensure_directories():
    directories = [
        DATA_DIR,
        PROGRAMME_DIR,
        RESULTS_DIR,
        ARCHIVE_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def clean_filename(text):
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"\s+", "_", text)
    return text[:150]


def download_file(url, folder, prefix="file"):
    try:
        response = requests.get(
            url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        content_type = response.headers.get(
            "Content-Type",
            ""
        ).lower()

        if "pdf" not in content_type and not url.lower().endswith(".pdf"):
            print(f"Skipping non-PDF: {url}")
            return None

        file_hash = hashlib.md5(
            response.content
        ).hexdigest()

        filename = f"{prefix}_{file_hash[:12]}.pdf"

        file_path = os.path.join(
            folder,
            filename
        )

        if os.path.exists(file_path):
            print(
                f"Already downloaded: {filename}"
            )
            return file_path

        with open(file_path, "wb") as file:
            file.write(response.content)

        print(
            f"Downloaded: {filename}"
        )

        return file_path

    except Exception as error:
        print(
            f"Download error: {url} -> {error}"
        )

        return None


def extract_pdf_links(page_url):
    pdf_links = []

    try:
        response = requests.get(
            page_url,
            headers=HEADERS,
            timeout=30
        )

        response.raise_for_status()

        soup = BeautifulSoup(
            response.text,
            "html.parser"
        )

        for link in soup.find_all(
            "a",
            href=True
        ):
            href = link["href"]

            if ".pdf" in href.lower():
                full_url = urllib.parse.urljoin(
                    page_url,
                    href
                )

                title = link.get_text(
                    " ",
                    strip=True
                )

                pdf_links.append(
                    {
                        "title": title,
                        "url": full_url
                    }
                )

        return pdf_links

    except Exception as error:

        print(
            f"Page error: {page_url} -> {error}"
        )

        return []


def collect_paginated_pdfs(
    base_url,
    folder,
    prefix,
    max_pages=50
):

    collected = []

    for page in range(max_pages):

        if page == 0:
            page_url = base_url
        else:
            page_url = (
                f"{base_url}?page={page}"
            )

        print(
            f"\nScanning page {page}:"
        )
        print(page_url)

        links = extract_pdf_links(
            page_url
        )

        if not links:

            print(
                "No PDF links found."
            )

            continue

        for item in links:

            title = item["title"]
            url = item["url"]

            print(
                f"Found: {title}"
            )

            file_path = download_file(
                url=url,
                folder=folder,
                prefix=prefix
            )

            if file_path:

                collected.append(
                    {
                        "title": title,
                        "url": url,
                        "file": file_path
                    }
                )

            time.sleep(1)

        time.sleep(2)

    return collected


def save_manifest(
    programmes,
    results
):

    manifest_path = os.path.join(
        ARCHIVE_DIR,
        "download_manifest.csv"
    )

    rows = []

    for item in programmes:

        rows.append(
            {
                "document_type": "programme",
                "title": item["title"],
                "url": item["url"],
                "file": item["file"],
                "downloaded_at": datetime.now().isoformat()
            }
        )

    for item in results:

        rows.append(
            {
                "document_type": "result",
                "title": item["title"],
                "url": item["url"],
                "file": item["file"],
                "downloaded_at": datetime.now().isoformat()
            }
        )

    import pandas as pd

    dataframe = pd.DataFrame(rows)

    dataframe.to_csv(
        manifest_path,
        index=False
    )

    print(
        f"\nManifest saved:"
    )

    print(
        manifest_path
    )


def run_historical_collection():

    ensure_directories()

    print(
        "\n=============================="
    )

    print(
        "LONAB HISTORICAL DATA COLLECTOR"
    )

    print(
        "==============================\n"
    )

    print(
        "COLLECTING PROGRAMMES..."
    )

    programmes = collect_paginated_pdfs(
        base_url=PROGRAMME_URL,
        folder=PROGRAMME_DIR,
        prefix="programme",
        max_pages=50
    )

    print(
        "\nCOLLECTING RESULTS..."
    )

    results = collect_paginated_pdfs(
        base_url=RESULTS_URL,
        folder=RESULTS_DIR,
        prefix="result",
        max_pages=50
    )

    save_manifest(
        programmes,
        results
    )

    print(
        "\n=============================="
    )

    print(
        "COLLECTION COMPLETE"
    )

    print(
        f"Programmes: {len(programmes)}"
    )

    print(
        f"Results: {len(results)}"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":

    run_historical_collection()
