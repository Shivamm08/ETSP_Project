"""
Translate the benchmark's English sentences into Hindi with AI4Bharat's IndicTrans2,
a model trained specifically for Indian languages (distilled 200M version).
Runs locally; the model (~1 GB) downloads from Hugging Face the first time.
"""
import csv

import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import KEEP_COLUMNS, OUTPUT_FIELDS, PATHS

SYSTEM = "indictrans2"
MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"   # "en-indic" = English -> Indian languages
SOURCE_LANG = "eng_Latn"   # same language codes as NLLB
TARGET_LANG = "hin_Deva"
BATCH_SIZE = 8             
NUM_BEAMS = 5              # tries 5 candidate translations, keeps the best
OUTPUT_FILE = PATHS["translations"] / "indictrans2_200m.csv"


def main():
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    # IndicTrans2 ships its own model code on Hugging Face, hence trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(device)
    model.eval()   # we're only using it, not training
    # the processor adds the language tags before translating and tidies the Hindi afterwards
    processor = IndicProcessor(inference=True)

    with PATHS["benchmark"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    outputs = []
    for start in range(0, len(rows), BATCH_SIZE):
        batch = [r["english"] for r in rows[start:start + BATCH_SIZE]]
        # tells the model "English in, Hindi out" - it knows 22 Indian languages, so it needs this
        batch = processor.preprocess_batch(batch, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG)
        # padding lines up sentences of different lengths; the mask marks which bits are filler
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                           return_attention_mask=True).to(device)
        with torch.no_grad():   # saves memory
            # use_cache=False: breaks with the cache for some reason
            generated = model.generate(**inputs, num_beams=NUM_BEAMS, max_length=256, use_cache=False)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
        outputs += processor.postprocess_batch(decoded, lang=TARGET_LANG)
        print(f"  {len(outputs)}/{len(rows)}")

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_FILE.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        # same columns as Google and NLLB, so all three files line up
        for r, hindi in zip(rows, outputs):
            writer.writerow({**{k: r[k] for k in KEEP_COLUMNS}, "system": SYSTEM, "model": MODEL,
                             "hindi_output": hindi})

    print(f"Translated {len(rows)} sentences -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
