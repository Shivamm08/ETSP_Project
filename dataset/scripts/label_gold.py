"""Hand-label MT outputs for the extractor's held-out accuracy (fills validation/extractor_gold.csv).

Run from the dataset/ folder:

    python3 scripts/label_gold.py                  # rows 1-300 and 2119-2193 (the default sample)
    python3 scripts/label_gold.py --rows 1-100     # a different range; several: 1-100,2119-2193

Rows 1-300 are a random sample of the first four systems' outputs (the sheet is shuffled); rows
2119-2193 are 25 random outputs from each of the three LLMs added later.

Each output is shown with its English source, never with the extractor's prediction, the system or
the requested register and gender, so none of them can bias the label. Press one key for the register
and one for the gender. The file is saved after every row, so you can quit with q and continue later;
rows that already have both labels are skipped.

Label what the Hindi itself marks, not what the English intended, and count only the addressee's
agreement: in क्या तुम उस लड़की से मिले जो कल आई थी? the label is tum, M (आई थी is the girl's).
The full rules are in the docstring of evaluate_extractor.py.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SHEET = ROOT / "validation" / "extractor_gold.csv"

REGISTER_KEYS = {"1": "tu", "2": "tum", "3": "aap", "4": "tu|tum", "5": "tum|aap", "6": "unmarked", "7": "mixed"}
GENDER_KEYS = {"m": "M", "f": "F", "u": "unmarked", "x": "mixed"}
REGISTER_MENU = "[1] तू  [2] तुम  [3] आप  [4] तू|तुम  [5] तुम|आप  [6] unmarked  [7] mixed"
GENDER_MENU = "[m] M  [f] F  [u] unmarked  [x] mixed"
COMMANDS = "b = back one row, s = skip, c = add a comment, q = save and quit"

BOLD, DIM, RESET = ("\033[1m", "\033[2m", "\033[0m") if sys.stdout.isatty() else ("", "", "")


def read_key(prompt: str) -> str:
    """One keypress without Enter in a terminal; a typed line otherwise."""
    print(prompt, end=" ", flush=True)
    if not sys.stdin.isatty():
        line = sys.stdin.readline()
        return line.strip()[:1].lower() if line else "q"   # end of input = quit
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        key = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    print(key)
    if key == "\x03":   # Ctrl-C
        raise KeyboardInterrupt
    return key.lower()


def save(rows: list[dict[str, str]], fields: list[str], path: Path) -> None:
    """Write to a temporary file first, so an interruption can never leave a half-written sheet."""
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def parse_ranges(text: str) -> list[tuple[int, int]]:
    """'1-300,2119-2193' -> [(1, 300), (2119, 2193)]"""
    out = []
    for part in text.split(","):
        first, _, last = part.strip().partition("-")
        out.append((int(first), int(last or first)))
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rows", default="1-300,2119-2193", help="row numbers of the sheet, e.g. 1-300,2119-2193")
    ap.add_argument("--sheet", type=Path, default=SHEET)
    args = ap.parse_args(argv)

    with args.sheet.open(encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    ranges = parse_ranges(args.rows)
    todo = [r for r in rows if any(a <= int(r["row"]) <= b for a, b in ranges)]

    def done(r: dict[str, str]) -> bool:
        return bool(r["gold_register"].strip() and r["gold_gender"].strip())

    print(f"{BOLD}Rows {args.rows}{RESET}: {sum(map(done, todo))} of {len(todo)} already labelled.")
    print(f"{DIM}Register {REGISTER_MENU}\nGender   {GENDER_MENU}\n{COMMANDS}{RESET}")

    i = 0
    while i < len(todo):
        r = todo[i]
        if done(r):
            i += 1
            continue
        print(f"\n{DIM}row {r['row']}  ({sum(map(done, todo))}/{len(todo)} labelled){RESET}")
        print(f"EN  {r['english']}")
        print(f"HI  {BOLD}{r['hindi']}{RESET}")

        key = read_key("register?")
        while key not in REGISTER_KEYS and key not in {"b", "s", "c", "q"}:
            key = read_key("  one of 1-7, b, s, c or q. register?")
        if key == "q":
            break
        if key == "b":   # reopen the previous row in the range
            if i > 0:
                i -= 1
                todo[i]["gold_register"] = todo[i]["gold_gender"] = ""
            continue
        if key == "s":
            i += 1
            continue
        if key == "c":
            r["comment"] = input("  comment: ").strip()
            continue
        register = REGISTER_KEYS[key]

        key = read_key("gender?  ")
        while key not in GENDER_KEYS and key != "q":
            key = read_key("  one of m, f, u, x or q. gender?")
        if key == "q":
            break
        r["gold_register"], r["gold_gender"] = register, GENDER_KEYS[key]
        save(rows, fields, args.sheet)
        i += 1

    save(rows, fields, args.sheet)
    print(f"\nSaved. {sum(map(done, todo))} of {len(todo)} rows ({args.rows}) are labelled.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped; every row labelled so far is saved.")
