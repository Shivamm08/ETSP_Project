"""
Translate the benchmark with an LLM, with and without instructions about register and gender.

Conditions:
    none      all 150 sentences, plain prompt (H1, H2)
    register  138 core sentences x tu / tum / aap
    gender    138 core sentences x man / woman
    both      138 core sentences x all 6 combinations (H3)

Usage:
    python run_llm.py --model gpt-oss-120b --limit 2      # quick test
    python run_llm.py --model gpt-oss-120b                # everything, resumes where it stopped
    python run_llm.py --model sarvam-105b
    python run_llm.py --model qwen3.8-27b --conditions none both register gender

Models: gpt-oss-120b, gpt-oss-20b, qwen3.8-27b (Groq) and sarvam-105b (Sarvam).
Needs GROQ_API_KEY and SARVAM_API_KEY in .env. Groq's free plan allows 1,000 requests a day per
model and refills them gradually, so a full run (1,668 requests) takes about a day; when the limit
is hit, the script waits for the refill and carries on.
"""
from __future__ import annotations

import argparse
import csv
import re
import time

import requests

from config import (GENDER_INSTRUCTION, GENDER_WORDS, GENDERS, KEEP_COLUMNS, LLM_OUTPUT_FIELDS,
                    PATHS, PROMPT, PRONOUNS, REGISTER_INSTRUCTION, REGISTERS, api_key as read_key)

# Model-specific settings. Both providers speak the same "chat completions" format,
# so the rest of the script doesn't care which one it's talking to.
LLMS = {
    # general open-weight model, served free by Groq
    "gpt-oss-120b": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "openai/gpt-oss-120b",
        "key_env": "GROQ_API_KEY",
        "requests_per_minute": 30,     # Groq free plan
        # it's a reasoning model; low keeps the hidden thinking short so we don't burn the daily token limit
        "extra": {"reasoning_effort": "low"},
    },
    # its smaller sibling (20B), to see what model size changes
    "gpt-oss-20b": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "openai/gpt-oss-20b",
        "key_env": "GROQ_API_KEY",
        "requests_per_minute": 30,
        "extra": {"reasoning_effort": "low"},
    },
    # Alibaba's Qwen, a different model family; thinking switched off, it answers directly
    "qwen3.8-27b": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "model": "qwen/qwen3.8-27b",
        "key_env": "GROQ_API_KEY",
        "requests_per_minute": 30,
        "extra": {"reasoning_effort": "none"},
    },
    # built for Indian languages; runs on Sarvam's free signup credits
    "sarvam-105b": {
        "url": "https://api.sarvam.ai/v1/chat/completions",
        "model": "sarvam-105b",
        "key_env": "SARVAM_API_KEY",
        "key_header": "api-subscription-key",   # Sarvam's own way of sending the key
        "requests_per_minute": 20,     # the big model has lower limits, so go easy
        # lowest setting Sarvam accepts; keeps the thinking short and saves credits
        "extra": {"reasoning_effort": "low"},
    },
}

CONDITIONS = ["none", "register", "gender", "both"]
DEVANAGARI = re.compile(r"[ऀ-ॿ]")


Job = tuple[dict[str, str], str, str, str]   # (benchmark row, condition, register, gender)


def build_jobs(rows: list[dict[str, str]], conditions: list[str]) -> list[Job]:
    """Every (sentence, condition, register, gender) combination we want to ask for."""
    jobs = []
    for cond in conditions:
        for r in rows:
            if cond == "none":
                jobs.append((r, cond, "", ""))
                continue
            if r["split"] != "core":   # the steering conditions only use the 138 core sentences
                continue
            if cond == "register":
                jobs += [(r, cond, reg, "") for reg in REGISTERS]
            elif cond == "gender":
                jobs += [(r, cond, "", gen) for gen in GENDERS]
            elif cond == "both":
                jobs += [(r, cond, reg, gen) for reg in REGISTERS for gen in GENDERS]
    return jobs


def make_prompt(english: str, register: str, gender: str) -> str:
    instructions = ""
    if register:
        instructions += REGISTER_INSTRUCTION.format(pronoun=PRONOUNS[register])
    if gender:
        instructions += GENDER_INSTRUCTION.format(gender_word=GENDER_WORDS[gender])
    return PROMPT.format(instructions=instructions, english=english)


def clean(raw: str) -> str:
    """Keep just the Hindi sentence: first line with Devanagari in it, minus labels and quotes."""
    lines = [line.strip() for line in raw.splitlines() if DEVANAGARI.search(line)]
    if not lines:
        return raw.strip()
    line = re.sub(r"^(translation|hindi|अनुवाद)\s*[:：]\s*", "", lines[0], flags=re.I)
    return line.strip().strip('"“”\'*').strip()


def ask(llm: dict, api_key: str, prompt: str) -> str:
    """One request. Waits and retries on short rate limits, stops on daily ones."""
    body = {"model": llm["model"], "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, **llm["extra"]}
    for _ in range(5):
        headers = {"Authorization": f"Bearer {api_key}"}
        if "key_header" in llm:   # some providers want the key under their own header name
            headers[llm["key_header"]] = api_key
        r = requests.post(llm["url"], headers=headers, json=body, timeout=60)
        if r.status_code == 429:
            wait = float(r.headers.get("retry-after", 10))
            if wait > 120:   # that's the daily limit, not the per-minute one
                raise SystemExit(f"Daily limit reached ({r.text[:200]}). Run again later, "
                                 "it will continue where it stopped.")
            time.sleep(wait)
            continue
        if r.status_code >= 500:   # provider hiccup, just try again
            time.sleep(5)
            continue
        if not r.ok:
            raise SystemExit(f"{r.status_code}: {r.text}")
        content = (r.json()["choices"][0]["message"]["content"] or "").strip()
        if content:
            return content
        # Sarvam sometimes thinks and then answers nothing; asking again usually works
        time.sleep(2)
    return ""   # still empty after 5 tries: main() leaves it out, so the next run asks again


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=LLMS)
    ap.add_argument("--conditions", nargs="+", default=CONDITIONS, choices=CONDITIONS)
    ap.add_argument("--limit", type=int, help="only the first N sentences, for testing")
    args = ap.parse_args()

    llm = LLMS[args.model]
    api_key = read_key(llm["key_env"])
    output_file = PATHS["translations"] / f"llm_{args.model}.csv"

    with PATHS["benchmark"].open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    # anything already in the output file is skipped, so a stopped run just picks up again
    done = set()
    if output_file.exists():
        with output_file.open(encoding="utf-8", newline="") as f:
            done = {(d["id"], d["condition"], d["requested_register"], d["requested_gender"])
                    for d in csv.DictReader(f)}
    jobs = [j for j in build_jobs(rows, args.conditions)
            if (j[0]["id"], j[1], j[2], j[3]) not in done]
    print(f"{len(jobs)} requests to go ({len(done)} already done)")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    new_file = not output_file.exists()
    with output_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LLM_OUTPUT_FIELDS)
        if new_file:
            writer.writeheader()

        empty = 0
        for i, (r, cond, reg, gen) in enumerate(jobs, 1):
            raw = ask(llm, api_key, make_prompt(r["english"], reg, gen))
            if not raw:
                empty += 1   # not saved, so it gets another go next time you run the script
                continue
            writer.writerow({
                **{k: r[k] for k in KEEP_COLUMNS},
                "system": args.model, "model": llm["model"],
                "hindi_output": clean(raw),
                "condition": cond, "requested_register": reg, "requested_gender": gen,
                # only "both" has one single right answer, e.g. tum_f
                "target_cell": f"{reg}_{gen.lower()}" if cond == "both" else "",
                "raw_output": raw,
            })
            f.flush()   # write it to disk now, not at the end
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}")
            time.sleep(60 / llm["requests_per_minute"])   # stay under the per-minute limit

    print(f"Done -> {output_file}")
    if empty:
        print(f"{empty} came back empty and weren't saved. Run the same command again to retry them.")


if __name__ == "__main__":
    main()
