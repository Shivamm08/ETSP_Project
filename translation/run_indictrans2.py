"""
Translate the benchmark's English sentences into Hindi with AI4Bharat's IndicTrans2,
a model trained specifically for Indian languages (distilled 200M version).
Runs locally; the model (~1 GB) downloads from Hugging Face the first time.
Sentences already in the output file are kept, so a rerun only translates new ones.
"""
import torch
from IndicTransToolkit.processor import IndicProcessor
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from config import PATHS, load_benchmark, load_env, pending, pick_device, write_outputs

SYSTEM = "indictrans2"
MODEL = "ai4bharat/indictrans2-en-indic-dist-200M"   # "en-indic" = English -> Indian languages
SOURCE_LANG = "eng_Latn"   # same language codes as NLLB
TARGET_LANG = "hin_Deva"
BATCH_SIZE = 8
NUM_BEAMS = 5              # tries 5 candidate translations, keeps the best
OUTPUT_FILE = PATHS["translations"] / "indictrans2_200m.csv"


def main() -> None:
    rows = load_benchmark()
    done, todo = pending(rows, OUTPUT_FILE)
    if not todo:
        print(f"Nothing to do: all {len(rows)} sentences are already in {OUTPUT_FILE}")
        return

    load_env()   # HF_TOKEN in .env (if set) is used to download the model
    device = pick_device(torch)
    # IndicTrans2 ships its own model code on Hugging Face, hence trust_remote_code
    tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL, trust_remote_code=True).to(device)
    model.eval()   # we're only using it, not training
    # the processor adds the language tags before translating and tidies the Hindi afterwards
    processor = IndicProcessor(inference=True)

    outputs = []
    for start in range(0, len(todo), BATCH_SIZE):
        batch = [r["english"] for r in todo[start:start + BATCH_SIZE]]
        # tells the model "English in, Hindi out" - it knows 22 Indian languages, so it needs this
        batch = processor.preprocess_batch(batch, src_lang=SOURCE_LANG, tgt_lang=TARGET_LANG)
        # padding lines up sentences of different lengths; the mask marks which bits are filler
        inputs = tokenizer(batch, truncation=True, padding="longest", return_tensors="pt",
                           return_attention_mask=True).to(device)
        with torch.no_grad():   # saves memory
            # use_cache=False: the model's own code breaks with the cache in newer transformers
            generated = model.generate(**inputs, num_beams=NUM_BEAMS, max_length=256, use_cache=False)
        decoded = tokenizer.batch_decode(generated, skip_special_tokens=True,
                                         clean_up_tokenization_spaces=True)
        outputs += processor.postprocess_batch(decoded, lang=TARGET_LANG)
        print(f"  {len(outputs)}/{len(todo)}")

    new = {r["id"]: hindi for r, hindi in zip(todo, outputs)}
    write_outputs(OUTPUT_FILE, rows, done, new, SYSTEM, MODEL)
    print(f"Translated {len(new)} new sentences ({len(done)} were already done) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
