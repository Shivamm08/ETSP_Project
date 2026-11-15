"""
Translate the benchmark's English sentences into Hindi with Meta's NLLB-200 (distilled 600M)
the model (~2.5 GB) downloads from Hugging Face the first time.
"""
import csv

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import KEEP_COLUMNS, OUTPUT_FIELDS, PATHS

SYSTEM = "nllb-200"
MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"   # NLLB's code for English
TARGET_LANG = "hin_Deva"   # Hindi in Devanagari script
BATCH_SIZE = 8             # small batches
NUM_BEAMS = 5
OUTPUT_FILE = PATHS["translations"] / "nllb_600m.csv"


def main():
    # use the Mac's GPU if it's there, otherwise the CPU (slower, but works)
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()
    # NLLB needs to be told which language to write in; this is the token that starts a Hindi output
    hindi_token = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    with PATHS["benchmark"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    outputs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = [r["english"] for r in rows[start:start + BATCH_SIZE]]
        # padding=True because sentences in a batch have different lengths
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():   # just translating, no training, so skip the gradient bookkeeping
            generated = model.generate(**inputs, forced_bos_token_id=hindi_token,
                                       num_beams=NUM_BEAMS, max_new_tokens=128)
        outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(f"  {len(outputs)}/{len(rows)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        # same columns as the Google file, so the two can be compared row by row
        for r, hindi in zip(rows, outputs):
            writer.writerow({**{k: r[k] for k in KEEP_COLUMNS}, "system": SYSTEM, "model": MODEL,
                             "hindi_output": hindi})

    print(f"Translated {len(rows)} sentences -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
