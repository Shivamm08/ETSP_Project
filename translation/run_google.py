"""
Translate the benchmark's English sentences into Hindi with Google Translate.
"""
import csv
import os

import requests

from config import KEEP_COLUMNS, OUTPUT_FIELDS, PATHS, load_env

SYSTEM = "google-translate"
MODEL = "nmt"      # Google's normal neural translation model
BATCH_SIZE = 100   # the API accepts at most 128 sentences per request
OUTPUT_FILE = PATHS["translations"] / "google_translate.csv"
API_URL = "https://translation.googleapis.com/language/translate/v2"


def main():
    # pull the API key out of .env first, otherwise the line below can't find it
    load_env()
    api_key = os.environ["GOOGLE_TRANSLATE_API_KEY"]

    # one row per English sentence -> 150 rows in total
    with PATHS["benchmark"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    outputs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = [r["english"] for r in rows[start:start + BATCH_SIZE]]
        # format "text" matters: without it Google returns HTML, e.g. ' comes back as &#39;
        response = requests.post(API_URL, params={"key": api_key}, json={
            "q": batch, "source": "en", "target": "hi", "format": "text", "model": MODEL,
        })
        # stop with Google's error message
        if not response.ok:
            raise SystemExit(f"Google returned {response.status_code}: {response.text}")
        outputs += [t["translatedText"] for t in response.json()["data"]["translations"]]

    # make outputs/translations/ if it isn't there yet
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        # Google gives the translations back in the same order we sent them,
        # so zip just pairs each sentence with its Hindi output
        for r, hindi in zip(rows, outputs):
            # copy id, profile, cue etc. from the benchmark so we can group results later,
            # then add what this run produced
            writer.writerow({**{k: r[k] for k in KEEP_COLUMNS}, "system": SYSTEM, "model": MODEL,
                             "hindi_output": hindi})

    print(f"Translated {len(rows)} sentences -> {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
