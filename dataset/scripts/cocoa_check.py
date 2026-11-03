"""Literature check: how CoCoA-MT (Nadejde et al., 2022) covers register and gender for English-Hindi.

Answers three questions on the en-hi test split (3 domains x 200 segments):
  1. Which pronoun does the "informal" reference use: तुम or तू?
  2. How many segments have distinct masculine / feminine references (the .feminine files)?
  3. Do the feminine references change the ADDRESSEE's gender, the SPEAKER's, or both?
     (heuristic: a changed span next to हूँ / -ऊँगा is first person; next to तुम / हो / हैं is second person)

Usage:
  git clone https://github.com/amazon-science/contrastive-controlled-mt
  python scripts/cocoa_check.py --repo contrastive-controlled-mt
"""
from __future__ import annotations

import argparse
import difflib
import re
import sys
from pathlib import Path

W = r"(?<![ऀ-ॿ]){}(?![ऀ-ॿ])"
TU = re.compile(W.format("(तू|तूने|तुझे|तुझसे|तेरा|तेरी|तेरे|तुझको)"))
TUM = re.compile(W.format("(तुम|तुमने|तुम्हें|तुम्हारा|तुम्हारी|तुम्हारे|तुमसे|तुमको)"))
AAP = re.compile(W.format("(आप|आपने|आपको|आपका|आपकी|आपके|आपसे)"))
FIRST = re.compile(r"(हूँ|हूं|ूँगा|ूँगी|ूंगा|ूंगी|ऊँगा|ऊँगी|ऊंगा|ऊंगी)")
SECOND = re.compile(W.format("(हो|हैं|सकते|सकती|ओगे|ओगी)"))
DOMAINS = ("call_center", "telephony", "topical_chat")


def lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def who_changed(masc: str, fem: str) -> str:
    """Classify a masculine->feminine edit as 'addressee', 'speaker', 'both' or 'unclear'."""
    a, b = masc.split(), fem.split()
    speaker = addressee = False
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag == "equal":
            continue
        window = " ".join(b[j1:min(j2 + 1, len(b))])
        speaker |= bool(FIRST.search(window))
        addressee |= bool(SECOND.search(window) or TUM.search(" ".join(b[max(0, j1 - 4):j2])))
    return {(True, True): "both", (True, False): "speaker", (False, True): "addressee"}.get((speaker, addressee),
                                                                                           "unclear")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", type=Path, required=True, help="clone of amazon-science/contrastive-controlled-mt")
    args = ap.parse_args(argv)
    base = args.repo / "CoCoA-MT" / "test" / "en-hi"

    rows = []
    for d in DOMAINS:
        stem = base / f"formality-control.test.{d}.en-hi"
        files = {k: lines(Path(f"{stem}{suf}")) for k, suf in [
            ("en", ""), ("fm", ".formal"), ("ff", ".formal.feminine"),
            ("im", ".informal"), ("if", ".informal.feminine")]}
        rows += [dict(zip(files, vals)) for vals in zip(*files.values())]

    n = len(rows)
    print(f"CoCoA-MT en-hi test segments: {n}")
    print(f"1. informal refs with तुम-forms: {sum(bool(TUM.search(r['im'])) for r in rows)}"
          f" | with तू-forms (any ref): {sum(bool(TU.search(r['im'] + r['if'] + r['fm'])) for r in rows)}"
          f" | formal refs with आप-forms: {sum(bool(AAP.search(r['fm'])) for r in rows)}")
    marked = [r for r in rows if r["im"] != r["if"]]
    print(f"2. segments with distinct masculine/feminine references: {len(marked)} ({len(marked) / n:.0%})")
    kinds: dict[str, int] = {}
    for r in marked:
        k = who_changed(r["im"], r["if"])
        kinds[k] = kinds.get(k, 0) + 1
    print("3. what the feminine variant changes (heuristic): "
          + ", ".join(f"{k}={kinds.get(k, 0)}" for k in ("addressee", "speaker", "both", "unclear")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
