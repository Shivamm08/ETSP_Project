"""
Translate the benchmark's English sentences into Hindi with Google Translate.
Sentences already in the output file are kept, so a rerun only translates new ones.
Needs GOOGLE_TRANSLATE_API_KEY in .env.
"""
import requests

from config import PATHS, api_key, load_benchmark, pending, write_outputs

SYSTEM = "google-translate"
MODEL = "nmt"      # Google's normal neural translation model
BATCH_SIZE = 100   # the API accepts at most 128 sentences per request
OUTPUT_FILE = PATHS["translations"] / "google_translate.csv"
API_URL = "https://translation.googleapis.com/language/translate/v2"


def main():
    rows = load_benchmark()
    done, todo = pending(rows, OUTPUT_FILE)
    if not todo:
        print(f"Nothing to do: all {len(rows)} sentences are already in {OUTPUT_FILE}")
        return
    key = api_key("GOOGLE_TRANSLATE_API_KEY")

    outputs = []
    for start in range(0, len(todo), BATCH_SIZE):
        batch = [r["english"] for r in todo[start:start + BATCH_SIZE]]
        # format "text" matters: without it Google returns HTML, e.g. ' comes back as &#39;
        response = requests.post(API_URL, params={"key": key}, json={
            "q": batch, "source": "en", "target": "hi", "format": "text", "model": MODEL,
        })
        # stop with Google's error message
        if not response.ok:
            raise SystemExit(f"Google returned {response.status_code}: {response.text}")
        outputs += [t["translatedText"] for t in response.json()["data"]["translations"]]

    # Google gives the translations back in the same order we sent them
    new = {r["id"]: hindi for r, hindi in zip(todo, outputs)}
    write_outputs(OUTPUT_FILE, rows, done, new, SYSTEM, MODEL)
    print(f"Translated {len(new)} new sentences ({len(done)} were already done) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
