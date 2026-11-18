"""Settings shared by all translation systems.

API keys do NOT go here: they go in .env (see .env.example), which is never pushed.

"""
import csv
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]   # config.py is in translation/, so go one folder up to ETSP_Project/

PATHS = {
    "benchmark": ROOT / "dataset" / "benchmark_v1.csv",            # input: one row per English sentence
    "benchmark_long": ROOT / "dataset" / "benchmark_v1_long.csv",  # scoring: one row per Hindi reference
    "translations": ROOT / "outputs" / "translations",             # one CSV per system goes here
    "env_file": ROOT / ".env",
}

# Which sentences to translate
DEFAULT_SPLIT = "all"          # "all" (150), "core" (138) or "h2_probe" (12)

# Output format
# Benchmark columns copied into every output row, so results can be grouped later
KEEP_COLUMNS = ["id", "split", "profile", "construction", "frame_id", "cue", "english"]
# Full column list of every output CSV
OUTPUT_FIELDS = KEEP_COLUMNS + ["system", "model", "hindi_output"]


# LLM settings shared by every LLM (the model-specific bits live in run_llm.py)
# extra columns only the LLM files have; empty where they don't apply
LLM_OUTPUT_FIELDS = OUTPUT_FIELDS + ["condition", "requested_register", "requested_gender",
                                     "target_cell", "raw_output"]

REGISTERS = ["tu", "tum", "aap"]   # same codes as the benchmark's register column
GENDERS = ["M", "F"]               # same codes as the benchmark's gender column

# what goes into the prompt for each code
PRONOUNS = {"tu": "तू", "tum": "तुम", "aap": "आप"}
GENDER_WORDS = {"M": "a man", "F": "a woman"}

# one prompt for every LLM and every condition; only the instruction lines change
PROMPT = (
    "Translate the following English sentence into Hindi.{instructions}\n"
    "Reply with the Hindi translation only.\n\n"
    "Sentence: {english}"
)
REGISTER_INSTRUCTION = " Address the person with the pronoun {pronoun}."
GENDER_INSTRUCTION = " The person you are talking to is {gender_word}."


# Helpers
def load_env(path=PATHS["env_file"]):
    """Read KEY=value lines from .env into the environment (no extra package needed).
    Variables already set in the shell take priority."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def api_key(name):
    """The key from .env (or the shell), with a clear message if it is missing."""
    load_env()
    key = os.environ.get(name, "").strip()
    if not key:
        raise SystemExit(f"{name} is missing: copy .env.example to .env and fill it in")
    return key


def load_benchmark():
    """One row per English sentence."""
    with PATHS["benchmark"].open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pending(rows, output_file):
    """Split the benchmark into sentences already in output_file and sentences still to translate,
    so a rerun after the benchmark grows only translates the new ones."""
    done = {}
    if output_file.exists():
        with output_file.open(encoding="utf-8", newline="") as f:
            done = {r["id"]: r for r in csv.DictReader(f)}
    return done, [r for r in rows if r["id"] not in done]


def write_outputs(output_file, rows, done, new, system, model):
    """One row per benchmark sentence, in benchmark order: earlier outputs unchanged, new ones added.
    new maps an item id to its Hindi output."""
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for r in rows:
            if r["id"] in done:
                writer.writerow(done[r["id"]])
            elif r["id"] in new:
                # copy id, profile, cue etc. from the benchmark so results can be grouped later
                writer.writerow({**{k: r[k] for k in KEEP_COLUMNS}, "system": system, "model": model,
                                 "hindi_output": new[r["id"]]})


def pick_device(torch):
    """A CUDA GPU (e.g. Colab), else the Mac's GPU, else the CPU (slower, but works)."""
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"
