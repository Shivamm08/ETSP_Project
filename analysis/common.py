"""Shared loading for the analysis scripts: the benchmark, every MT output with the extractor's labels,
and small statistics helpers.

Every label in the analysis comes from dataset/scripts/extractor.py, the same tool whose accuracy
dataset/scripts/evaluate_extractor.py measures on hand-labelled outputs.
"""
from __future__ import annotations

import csv
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
TRANSLATIONS = ROOT / "outputs" / "translations"
RESULTS = ROOT / "outputs" / "results"

sys.path.insert(0, str(DATASET / "scripts"))
from evaluate_extractor import wilson  # noqa: E402  (95% Wilson score interval, already tested)
from extractor import extract  # noqa: E402

# Display names, in the order tables and figures list them: translation systems, then LLMs.
SYSTEMS = {
    "google-translate": "Google Translate",
    "indictrans2": "IndicTrans2-200M",
    "nllb-200": "NLLB-200-600M",
    "gpt-oss-120b": "GPT-OSS-120B",
    "gpt-oss-20b": "GPT-OSS-20B",
    "qwen3.8-27b": "Qwen3.8-27B",
    "sarvam-105b": "Sarvam-105B",
}
LLMS = ["gpt-oss-120b", "gpt-oss-20b", "qwen3.8-27b", "sarvam-105b"]
REGISTERS = ("tu", "tum", "aap")
HINDI = {"tu": "तू", "tum": "तुम", "aap": "आप"}
PROFILES = ("FUT", "PAST", "PRES")
CELLS = tuple(f"{r}_{g}" for r in REGISTERS for g in ("m", "f"))


def load_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter=delimiter))


def benchmark() -> dict[str, dict[str, str]]:
    return {r["id"]: r for r in load_csv(DATASET / "benchmark_v1.csv")}


def labelled_outputs(items: dict[str, dict[str, str]]) -> list[dict[str, str]]:
    """Every MT output, with the item's metadata and the extractor's register and gender."""
    rows = []
    for path in sorted(TRANSLATIONS.glob("*.csv")):
        for r in load_csv(path):
            if r["system"] not in SYSTEMS or r["id"] not in items:
                continue
            item = items[r["id"]]
            ex = extract(r["hindi_output"])
            rows.append({**r, "condition": r.get("condition") or "none", "split": item["split"],
                         "profile": item["profile"], "frame_id": item.get("frame_id", ""), "cue": item["cue"],
                         "tags": item.get("tags", ""), "register": ex.register, "gender": ex.gender})
    return rows


def human_baseline() -> list[dict[str, str]]:
    """The extractor's labels for the first human translation on Tatoeba of each candidate sentence that has one
    (330 of the 38,658 sentences in dataset/sources/tatoeba_candidate_pool.tsv)."""
    pool = load_csv(DATASET / "sources" / "tatoeba_candidate_pool.tsv", delimiter="\t")
    out = []
    for r in pool:
        if r["existing_hindi"].strip():
            ex = extract(r["existing_hindi"].split(" | ")[0])
            out.append({"system": "tatoeba-human", "profile": r["bucket"], "register": ex.register, "gender": ex.gender})
    return out


def register_group(label: str) -> str:
    """tu / tum / aap, 'ambiguous' for a set such as tum|aap (pronoun dropped), else mixed / unmarked."""
    if label in REGISTERS or label in ("mixed", "unmarked"):
        return label
    return "ambiguous"


def share(k: int, n: int) -> str:
    """'61.2% [52.9-68.9]': a percentage with its 95% Wilson interval."""
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{100 * k / n:.1f}% [{100 * max(0.0, lo):.1f}–{100 * min(1.0, hi):.1f}]"


def pct(k: int, n: int) -> str:
    return "n/a" if n == 0 else f"{100 * k / n:.1f}%"


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """Difference p1 - p2 of two independent proportions with Newcombe's 95% hybrid score interval."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def sign_test(b: int, c: int) -> float:
    """Exact two-sided McNemar (sign) test on the discordant pairs b and c."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def error_type(ref_cell: str, hyp_cell: str) -> str:
    """Which attribute separates two cells: 'gender' (tu_m vs tu_f), 'register' (tu_m vs aap_m) or 'both'."""
    (rr, rg), (hr, hg) = ref_cell.split("_"), hyp_cell.split("_")
    if rr == hr:
        return "gender"
    return "register" if rg == hg else "both"


def auc(pos: list[float], neg: list[float]) -> float:
    """Probability that a random positive scores higher than a random negative (ties count half)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > q) + 0.5 * (p == q) for p in pos for q in neg)
    return wins / (len(pos) * len(neg))


def counts(rows: list[dict[str, str]], key: str) -> Counter:
    return Counter(r[key] for r in rows)


def md_table(header: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return "\n".join(out)
