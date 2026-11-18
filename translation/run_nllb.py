"""
Translate the benchmark's English sentences into Hindi with Meta's NLLB-200 (distilled 600M).
The model (~2.5 GB) downloads from Hugging Face the first time.
Sentences already in the output file are kept, so a rerun only translates new ones.
"""
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import PATHS, load_benchmark, load_env, pending, pick_device, write_outputs

SYSTEM = "nllb-200"
MODEL = "facebook/nllb-200-distilled-600M"
SOURCE_LANG = "eng_Latn"   # NLLB's code for English
TARGET_LANG = "hin_Deva"   # Hindi in Devanagari script
BATCH_SIZE = 8             # small batches
NUM_BEAMS = 5
OUTPUT_FILE = PATHS["translations"] / "nllb_600m.csv"


def main():
    rows = load_benchmark()
    done, todo = pending(rows, OUTPUT_FILE)
    if not todo:
        print(f"Nothing to do: all {len(rows)} sentences are already in {OUTPUT_FILE}")
        return

    load_env()   # HF_TOKEN in .env (if set) is used to download the model
    device = pick_device(torch)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, src_lang=SOURCE_LANG)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL).to(device)
    model.eval()
    # NLLB needs to be told which language to write in; this is the token that starts a Hindi output
    hindi_token = tokenizer.convert_tokens_to_ids(TARGET_LANG)

    outputs = []
    for start in range(0, len(todo), BATCH_SIZE):
        batch = [r["english"] for r in todo[start:start + BATCH_SIZE]]
        # padding=True because sentences in a batch have different lengths
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():   # just translating, no training, so skip the gradient bookkeeping
            generated = model.generate(**inputs, forced_bos_token_id=hindi_token,
                                       num_beams=NUM_BEAMS, max_new_tokens=128)
        outputs += tokenizer.batch_decode(generated, skip_special_tokens=True)
        print(f"  {len(outputs)}/{len(todo)}")

    new = {r["id"]: hindi for r, hindi in zip(todo, outputs)}
    write_outputs(OUTPUT_FILE, rows, done, new, SYSTEM, MODEL)
    print(f"Translated {len(new)} new sentences ({len(done)} were already done) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
