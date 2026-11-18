"""Measure how accurately extractor.py reads register and gender.

Three modes (run from the dataset/ folder):

  1. Development check on the benchmark's own Hindi (no MT needed):
         python scripts/evaluate_extractor.py --benchmark
     Scores the 900 references, the 52 prescriptive तुम · F alternatives and the 900 pro-drop
     variants. A pro-drop variant is scored against the set of cells it is compatible with:
     क्या ठीक से सोए? must come out as register tum|aap, gender M. These sentences were used to
     build the rules, so this number is an upper bound and not the accuracy to report.

  2. Make a sheet for hand-labelling MT outputs:
         python scripts/evaluate_extractor.py --make-sheet --out validation/extractor_gold.csv
     With no argument it reads every CSV in ../outputs/translations/ (the machine translations);
     you can also pass CSV files or folders: --make-sheet ../outputs/translations/nllb_600m.csv.
     Each input needs columns `id` (benchmark item id), `system` and the MT output in `hindi_output`
     (or `hindi`); `english` and `condition` (the LLM prompting condition) are carried over when
     present. The sheet comes shuffled, with the English source next to each output and empty gold
     columns. The extractor's predictions and the requested cell are left out so that they cannot
     bias the annotator.

  3. The held-out accuracy, on hand-labelled MT outputs (the number to report):
         python scripts/evaluate_extractor.py --labelled validation/extractor_gold.csv
     Reports register, gender and joint accuracy with 95% confidence intervals, confusion matrices,
     a breakdown by system, by condition and by profile, and every error.

Gold labels (columns gold_register, gold_gender) take the same values the extractor outputs:

  gold_register  tu, tum, aap; a set like tum|aap when the sentence fits several registers (pro-drop
                 गए थे?); unmarked when nothing marks register; mixed when the marks contradict each
                 other (आप ... हो)
  gold_gender    M, F; unmarked when nothing agrees with the addressee (तुम्हें भूख लगी है?,
                 तुमने खाना खाया?, बैठिए); mixed when the addressee's agreement has both genders

Label what the Hindi itself marks, not what the English intended. Only the addressee's agreement
counts (in क्या तुम उस लड़की से मिले जो कल आई थी?, आई थी is the girl's: tum, M).
"""
from __future__ import annotations

import argparse
import csv
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor import REGISTERS, extract, register_label  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TRANSLATIONS = ROOT.parent / "outputs" / "translations"
GENDERS = ("M", "F", "unmarked", "mixed")


# --------------------------------------------------------------------------- labels
def canonical_register(label: str) -> str:
    """Normalise a gold register label: 'aap|tum' -> 'tum|aap', 'tu|tum|aap' -> 'unmarked'."""
    label = label.strip().lower().replace(" ", "")
    if label in ("mixed", "unmarked", ""):
        return label
    parts = set(label.split("|"))
    unknown = parts - set(REGISTERS)
    if unknown:
        raise ValueError(f"unknown register label {label!r}")
    return register_label(frozenset(parts))


def canonical_gender(label: str) -> str:
    g = label.strip()
    g = g.upper() if g.upper() in ("M", "F") else g.lower()
    if g not in GENDERS and g != "":
        raise ValueError(f"unknown gender label {label!r}")
    return g


# --------------------------------------------------------------------------- statistics
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion k/n."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return centre - half, centre + half


def pct(k: int, n: int, ci: bool = False) -> str:
    if n == 0:
        return "   n/a"
    s = f"{100 * k / n:5.1f}%"
    if ci:
        lo, hi = wilson(k, n)
        s += f" [{100 * lo:4.1f}-{100 * hi:4.1f}]"
    return s


def score(rows: list[dict]) -> dict[str, int]:
    """rows need gold_register, gold_gender, pred_register, pred_gender."""
    reg = sum(r["pred_register"] == r["gold_register"] for r in rows)
    gen = sum(r["pred_gender"] == r["gold_gender"] for r in rows)
    joint = sum(r["pred_register"] == r["gold_register"] and r["pred_gender"] == r["gold_gender"] for r in rows)
    return {"n": len(rows), "register": reg, "gender": gen, "joint": joint}


def table(groups: dict[str, list[dict]], title: str, ci: bool = False) -> None:
    width = max([len(title)] + [len(k) for k in groups]) + 2
    pad = 21 if ci else 8
    print(f"\n{title:<{width}}{'n':>5}  {'register':>{pad}}  {'gender':>{pad}}  {'joint':>{pad}}")
    for name, rows in groups.items():
        s = score(rows)
        print(f"{name:<{width}}{s['n']:>5}  {pct(s['register'], s['n'], ci):>{pad}}  "
              f"{pct(s['gender'], s['n'], ci):>{pad}}  {pct(s['joint'], s['n'], ci):>{pad}}")


def confusion(rows: list[dict], attr: str) -> None:
    gold_key, pred_key = f"gold_{attr}", f"pred_{attr}"
    labels = sorted({r[gold_key] for r in rows} | {r[pred_key] for r in rows},
                    key=lambda x: (x in ("unmarked", "mixed"), x.count("|"), x))
    counts = Counter((r[gold_key], r[pred_key]) for r in rows)
    corner = "gold / pred"
    w = max(len(x) for x in labels + [corner]) + 2
    print(f"\n{attr} confusion (rows = gold, columns = prediction)")
    print(f"{corner:<{w}}" + "".join(f"{x:>{w}}" for x in labels))
    for g in labels:
        print(f"{g:<{w}}" + "".join(f"{counts[(g, p)] or '.':>{w}}" for p in labels))


def show_errors(rows: list[dict], limit: int) -> None:
    errs = [r for r in rows if (r["pred_register"], r["pred_gender"]) != (r["gold_register"], r["gold_gender"])]
    if not errs:
        return
    print(f"\nerrors ({len(errs)}{', first ' + str(limit) if len(errs) > limit else ''}):")
    for r in errs[:limit]:
        tag = f"{r.get('id', '')} {r.get('tier', r.get('system', ''))}".strip()
        print(f"  [{tag}] {r['hindi']}\n      gold {r['gold_register']}/{r['gold_gender']}"
              f"  pred {r['pred_register']}/{r['pred_gender']}")


def predict(rows: list[dict]) -> None:
    for r in rows:
        ex = extract(r["hindi"])
        r["pred_register"], r["pred_gender"] = ex.register, ex.gender


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- mode 1: benchmark
def benchmark_rows(long_path: Path) -> list[dict]:
    rows = []
    for r in load_csv(long_path):
        base = {k: r[k] for k in ("id", "split", "profile", "construction", "tags", "cell")}
        rows.append({**base, "tier": "overt", "hindi": r["hindi"],
                     "gold_register": r["register"], "gold_gender": r["gender"]})
        if r["hindi_alt"]:
            rows.append({**base, "tier": "alt", "hindi": r["hindi_alt"],
                         "gold_register": r["register"], "gold_gender": r["gender"]})
        cells = r["prodrop_compatible_cells"].split("|")
        rows.append({**base, "tier": "prodrop", "hindi": r["hindi_prodrop"],
                     "gold_register": register_label(frozenset(c.split("_")[0] for c in cells)),
                     "gold_gender": r["gender"]})
    return rows


def run_benchmark(long_path: Path, limit: int) -> int:
    rows = benchmark_rows(long_path)
    predict(rows)
    print(f"Development check on {long_path.name}. The rules were written while looking at these "
          f"sentences,\nso this is an upper bound; the held-out accuracy comes from --labelled.")
    by_tier = defaultdict(list)
    for r in rows:
        by_tier[r["tier"]].append(r)
    table(dict(by_tier), "tier")
    for tier in ("overt", "prodrop"):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in by_tier[tier]:
            groups[r["profile"]].append(r)
            for t in filter(None, r["tags"].split("|")):
                groups[f"tag:{t}"].append(r)
        table(dict(sorted(groups.items())), f"{tier} by")
    show_errors(rows, limit)
    return 0


# --------------------------------------------------------------------------- mode 2: annotation sheet
def load_mt(paths: list[Path]) -> list[dict[str, str]]:
    """Read MT outputs from CSV files and/or folders of CSVs, with the output text under `hindi`."""
    files: list[Path] = []
    for p in paths:
        files.extend(sorted(p.glob("*.csv")) if p.is_dir() else [p])
    if not files:
        raise ValueError(f"no CSV files in {', '.join(map(str, paths))}")
    rows = []
    for f in files:
        data = load_csv(f)
        cols = set(data[0]) if data else set()
        text_col = "hindi_output" if "hindi_output" in cols else "hindi"
        missing = {"id", "system", text_col} - cols
        if missing:
            raise ValueError(f"{f} lacks column(s) {sorted(missing)}")
        for r in data:
            rows.append({"id": r["id"], "system": r["system"], "condition": r.get("condition", ""),
                         "english": r.get("english", ""), "hindi": r[text_col], "source": f.name})
    return rows


def make_sheet(mt_paths: list[Path], out: Path, bench_path: Path, seed: int) -> int:
    try:
        mt = load_mt(mt_paths)
    except ValueError as e:
        print(e, file=sys.stderr)
        return 1
    english = {r["id"]: r["english"] for r in load_csv(bench_path)}
    random.Random(seed).shuffle(mt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "id", "english", "hindi", "gold_register", "gold_gender", "comment", "system", "condition"])
        for i, r in enumerate(mt, 1):
            w.writerow([i, r["id"], r["english"] or english.get(r["id"], ""), r["hindi"], "", "", "",
                        r["system"], r["condition"]])
    by_system = Counter(r["system"] for r in mt)
    print(f"wrote {len(mt)} rows to {out} ({', '.join(f'{s}: {n}' for s, n in sorted(by_system.items()))}); "
          f"fill in gold_register and gold_gender (see the docstring)")
    return 0


# --------------------------------------------------------------------------- mode 3: hand-labelled MT
def run_labelled(path: Path, bench_path: Path, limit: int) -> int:
    rows = load_csv(path)
    profile = {r["id"]: r["profile"] for r in load_csv(bench_path)}
    labelled = []
    for i, r in enumerate(rows, 2):
        try:
            gr, gg = canonical_register(r.get("gold_register", "")), canonical_gender(r.get("gold_gender", ""))
        except ValueError as e:
            print(f"{path.name} line {i}: {e}", file=sys.stderr)
            return 1
        if gr and gg:
            labelled.append({**r, "gold_register": gr, "gold_gender": gg})
    if not labelled:
        print(f"no labelled rows in {path}", file=sys.stderr)
        return 1
    print(f"{len(labelled)} of {len(rows)} rows labelled in {path.name}")
    predict(labelled)

    table({"all": labelled}, "held-out", ci=True)
    for key, name in (("system", "by system"), ("condition", "by condition"), ("profile", "by profile")):
        groups: dict[str, list[dict]] = defaultdict(list)
        for r in labelled:
            k = r.get(key) or (profile.get(r.get("id", ""), "") if key == "profile" else "")
            if k:
                groups[k].append(r)
        if len(groups) > 1:
            table(dict(sorted(groups.items())), name, ci=True)
    for attr in ("register", "gender"):
        confusion(labelled, attr)
    show_errors(labelled, limit)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--benchmark", action="store_true", help="development check on the benchmark")
    mode.add_argument("--make-sheet", type=Path, nargs="*", metavar="MT_CSV_OR_DIR",
                      help=f"make a hand-labelling sheet from MT outputs (default: {TRANSLATIONS})")
    mode.add_argument("--labelled", type=Path, metavar="GOLD_CSV", help="score against hand labels")
    ap.add_argument("--out", type=Path, default=ROOT / "validation" / "extractor_gold.csv")
    ap.add_argument("--long", type=Path, default=ROOT / "benchmark_v1_long.csv")
    ap.add_argument("--wide", type=Path, default=ROOT / "benchmark_v1.csv")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--errors", type=int, default=40, help="how many errors to print")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.benchmark:
        return run_benchmark(args.long, args.errors)
    if args.make_sheet is not None:
        if args.out.exists():
            print(f"{args.out} exists; not overwriting hand labels (pass another --out)", file=sys.stderr)
            return 1
        return make_sheet(args.make_sheet or [TRANSLATIONS], args.out, args.wide, args.seed)
    return run_labelled(args.labelled, args.wide, args.errors)


if __name__ == "__main__":
    sys.exit(main())
