"""Check the register x gender benchmark and build its derived files.

The wide CSV (one row per English source, six Hindi cells) is the single source of truth.
This script never edits it; it only checks it and derives:

  * benchmark_<version>_long.csv  one row per (item, register, gender) cell, with the pro-drop
                                  variant, the set of cells that pro-drop variant is compatible with,
                                  and the tokens that carry agreement

Usage (from the dataset/ folder):
    python scripts/build_dataset.py                    # checks + long CSV
    python scripts/build_dataset.py --check-only       # checks only, exit 1 on errors
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_hindi  # noqa: E402

REGISTERS = ("tu", "tum", "aap")
GENDERS = ("m", "f")
CELLS = tuple(f"{r}_{g}" for r in REGISTERS for g in GENDERS)
PRONOUN = {"tu": "तू", "tum": "तुम", "aap": "आप"}

PROFILES = {"FUT", "PAST", "PRES"}
CONSTRUCTIONS = {
    "FUT": {"future"},
    "PAST": {"perfective", "past_perfect", "past_copula", "past_progressive", "past_habitual"},
    "PRES": {"present_progressive", "present_habitual", "present_perfect", "predicate_adjective", "modal_sakna"},
}
CUES = {"none", "neutral", "fem_stereotype", "masc_stereotype"}
REQUIRED = ["id", "split", "profile", "construction", "sentence_type", "polarity", "cue", "english", "source", *CELLS]

PUNCT = "?।!.,"
DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# Agreement endings, longest first, used to reduce an inflected token to its stem.
AGREEMENT_SUFFIXES = (
    "एँगे", "एँगी", "एंगे", "एंगी", "ोंगे", "ोंगी", "ेंगे", "ेंगी",
    "एगा", "एगी", "ेगा", "ेगी", "ओगे", "ओगी", "ोगे", "ोगी", "ोगा",
    "या", "ीं", "ईं", "ैं", "ा", "ी", "े", "ई", "ए", "आ", "ै", "ो",
)

# Expected shape of the finite (rightmost agreeing) token, per profile and cell.
FUT_ENDINGS = {
    "tu_m": ("एगा", "ेगा", "होगा"), "tu_f": ("एगी", "ेगी", "होगी"),
    "tum_m": ("ओगे", "ोगे"), "tum_f": ("ओगी", "ोगी"),
    "aap_m": ("एँगे", "एंगे", "ेंगे", "ोंगे"), "aap_f": ("एँगी", "एंगी", "ेंगी", "ोंगी"),
}
PRES_AUX = {"tu": "है", "tum": "हो", "aap": "हैं"}
# Invariant words that happen to end in a nasalised ī (not agreement).
NASAL_INVARIANT = {"नहीं", "कहीं", "यहीं", "वहीं"}


@dataclass
class Report:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, item_id: str, msg: str) -> None:
        self.errors.append(f"[{item_id}] {msg}")

    def warn(self, item_id: str, msg: str) -> None:
        self.warnings.append(f"[{item_id}] {msg}")


def tokens(sentence: str) -> list[str]:
    """Whitespace tokens with sentence punctuation stripped."""
    return [t.strip(PUNCT) for t in sentence.split() if t.strip(PUNCT)]


def stem(token: str) -> str:
    """Strip one agreement ending (gaya/gayi/gaye -> g, jaaoge -> jaa, hai/ho/hain -> h)."""
    for suf in AGREEMENT_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf):
            return token[: -len(suf)]
    return token


def is_masc_sg(tok: str) -> bool:
    return tok.endswith(("ा", "आ"))


def is_masc_pl(tok: str) -> bool:
    return tok.endswith(("े", "ए"))


def is_fem(tok: str) -> bool:
    return tok.endswith(("ी", "ई"))


def is_fem_pl(tok: str) -> bool:
    return tok.endswith(("ीं", "ईं"))


def load(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- checks
def check_schema(item: dict[str, str], rep: Report) -> None:
    iid = item.get("id", "?")
    for col in REQUIRED:
        if not item.get(col, "").strip():
            rep.error(iid, f"empty required column '{col}'")
    if item["profile"] not in PROFILES:
        rep.error(iid, f"unknown profile {item['profile']}")
    elif item["construction"] not in CONSTRUCTIONS[item["profile"]]:
        rep.error(iid, f"construction {item['construction']} does not belong to profile {item['profile']}")
    if item["cue"] not in CUES:
        rep.error(iid, f"unknown cue {item['cue']}")
    if item["cue"] != "none" and not item.get("frame_id"):
        rep.error(iid, "cued item without frame_id")


def check_pronouns(item: dict[str, str], rep: Report) -> None:
    for cell in CELLS:
        toks = tokens(item[cell])
        own = PRONOUN[cell.split("_")[0]]
        others = [p for p in PRONOUN.values() if p != own and p in toks]
        if toks.count(own) != 1:
            rep.error(item["id"], f"{cell}: expected exactly one overt '{own}', got {toks.count(own)}")
        if others:
            rep.error(item["id"], f"{cell}: contains other register pronoun(s) {others}")


def check_minimal_pairs(item: dict[str, str], rep: Report) -> None:
    """All six cells must have the same length and differ only by pronoun or agreement endings."""
    toks = {c: tokens(item[c]) for c in CELLS}
    lengths = {len(t) for t in toks.values()}
    if len(lengths) != 1:
        rep.error(item["id"], f"cells differ in token count {sorted(lengths)}; not minimal pairs")
        return
    for i in range(lengths.pop()):
        column = {c: toks[c][i] for c in CELLS}
        if len(set(column.values())) == 1:
            continue
        if set(column.values()) <= set(PRONOUN.values()):
            continue
        if len({stem(t) for t in column.values()}) != 1:
            rep.error(item["id"], f"position {i} differs beyond agreement morphology: {sorted(set(column.values()))}")


def check_paradigm(item: dict[str, str], rep: Report) -> None:
    """Check the rightmost agreeing token against the expected paradigm for the profile."""
    prof, iid = item["profile"], item["id"]
    for cell in CELLS:
        reg, gen = cell.split("_")
        toks = tokens(item[cell])
        if not toks:
            continue
        last = toks[-1]
        if prof == "FUT":
            fut = [t for t in toks if t.endswith(("गा", "गी", "गे"))]
            if not fut or not fut[-1].endswith(FUT_ENDINGS[cell]):
                rep.error(iid, f"{cell}: future verb {fut[-1:] or '-'} does not end in {FUT_ENDINGS[cell]}")
        elif prof == "PAST":
            ok = {
                "tu_m": is_masc_sg(last), "tu_f": is_fem(last) and not is_fem_pl(last),
                "tum_m": is_masc_pl(last), "tum_f": is_fem(last) or is_fem_pl(last),
                "aap_m": is_masc_pl(last), "aap_f": is_fem_pl(last),
            }[cell]
            if not ok:
                rep.error(iid, f"{cell}: final agreeing token '{last}' has the wrong ending")
            # Feminine plural nasalisation surfaces once, on the last element only (गई थीं, not गईं थीं).
            if is_fem_pl(last) and len(toks) > 1 and is_fem_pl(toks[-2]) and toks[-2] not in NASAL_INVARIANT:
                rep.error(iid, f"{cell}: nasalisation doubled on '{toks[-2]} {last}'")
        elif prof == "PRES":
            if last != PRES_AUX[reg]:
                rep.error(iid, f"{cell}: auxiliary '{last}' should be '{PRES_AUX[reg]}'")
            if len(toks) >= 2:
                part = toks[-2]
                ok = is_fem(part) if gen == "f" else (is_masc_sg(part) if reg == "tu" else is_masc_pl(part))
                if not ok:
                    rep.error(iid, f"{cell}: participle/adjective '{part}' does not agree")
    alt = item.get("tum_f_alt", "")
    if prof == "PAST":
        if not alt:
            rep.warn(iid, "PAST item without tum_f_alt (prescriptive गईं/थीं form)")
        elif alt.replace("तुम", "आप", 1) != item["aap_f"]:
            rep.error(iid, "tum_f_alt should equal aap_f with आप -> तुम")
    elif alt:
        rep.warn(iid, "tum_f_alt is only expected for PAST items")


def check_orthography(item: dict[str, str], rep: Report) -> None:
    for cell in CELLS:
        s = item[cell]
        if "‌" in s or "‍" in s:
            rep.warn(item["id"], f"{cell}: contains ZWJ/ZWNJ")
        if re.search(r"[A-Za-z]", s):
            rep.error(item["id"], f"{cell}: Latin characters in Hindi")
        if "  " in s or s != s.strip():
            rep.warn(item["id"], f"{cell}: stray whitespace")
        if not s.endswith(("?", "।", "!")):
            rep.warn(item["id"], f"{cell}: no final ? or ।")
        expected = "?" if item["sentence_type"] == "question" else "।"
        if not s.endswith(expected):
            rep.warn(item["id"], f"{cell}: sentence_type is {item['sentence_type']} but ends with '{s[-1:]}'")
        for t in tokens(s):
            if len(t) > 2 and t.endswith(("ये", "यी")) and t not in {"लिये", "किये"}:
                rep.warn(item["id"], f"{cell}: '{t}' uses ये/यी spelling; convention is ए/ई (गए, गई)")


def check_normalisation(item: dict[str, str], rep: Report) -> None:
    """Spelling normalisation (used by the extractor and metrics) must never merge forms that differ."""
    full = [item[c] for c in CELLS] + ([item["tum_f_alt"]] if item.get("tum_f_alt") else [])
    if len({normalize_hindi(s) for s in full}) != len(set(full)):
        rep.error(item["id"], "normalize_hindi merges two different cells")
    drop = [prodrop(item[c], PRONOUN[c.split("_")[0]]) for c in CELLS]
    if len({normalize_hindi(s) for s in drop}) != len(set(drop)):
        rep.error(item["id"], "normalize_hindi merges two different pro-drop forms")


def check_frames(items: list[dict[str, str]], rep: Report) -> None:
    frames: dict[str, list[str]] = defaultdict(list)
    for it in items:
        if it.get("frame_id"):
            frames[it["frame_id"]].append(it["cue"])
    for fid, cues in sorted(frames.items()):
        if sorted(cues) != ["fem_stereotype", "masc_stereotype", "neutral"]:
            rep.error(fid, f"H2 frame must have one neutral, one fem and one masc member, has {sorted(cues)}")


# --------------------------------------------------------------------------- derived data
def prodrop(sentence: str, pronoun: str) -> str:
    """Remove the overt subject pronoun (Hindi is pro-drop)."""
    out = [w for w in sentence.split() if w.strip(PUNCT) != pronoun]
    return " ".join(out)


def agreement_tokens(item: dict[str, str]) -> dict[str, list[str]]:
    """Tokens (excluding the pronoun) that vary across the six cells, i.e. carry agreement."""
    toks = {c: tokens(item[c]) for c in CELLS}
    n = len(toks["tu_m"])
    if any(len(t) != n for t in toks.values()):
        return {c: [] for c in CELLS}
    varying = [i for i in range(n) if len({toks[c][i] for c in CELLS}) > 1 and toks["tu_m"][i] != PRONOUN["tu"]]
    return {c: [toks[c][i] for i in varying] for c in CELLS}


def compatible_cells(item: dict[str, str]) -> dict[str, str]:
    """For each cell, which cells could have produced its pro-drop form (accepted alternatives included)."""
    surface_to_cells: dict[str, set[str]] = defaultdict(set)
    for c in CELLS:
        surface_to_cells[prodrop(item[c], PRONOUN[c.split("_")[0]])].add(c)
    if item.get("tum_f_alt"):
        surface_to_cells[prodrop(item["tum_f_alt"], PRONOUN["tum"])].add("tum_f")
    out = {}
    for c in CELLS:
        surf = prodrop(item[c], PRONOUN[c.split("_")[0]])
        out[c] = "|".join(x for x in CELLS if x in surface_to_cells[surf])
    return out


def write_long(items: list[dict[str, str]], path: Path) -> None:
    meta = ["id", "split", "profile", "construction", "sentence_type", "polarity", "frame_id", "cue", "english"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([*meta, "cell", "register", "gender", "hindi", "hindi_alt", "hindi_prodrop",
                    "prodrop_compatible_cells", "agreement_tokens"])
        for it in items:
            agr, comp = agreement_tokens(it), compatible_cells(it)
            for c in CELLS:
                reg, gen = c.split("_")
                alt = it.get("tum_f_alt", "") if c == "tum_f" else ""
                w.writerow([*(it.get(m, "") for m in meta), c, reg, gen.upper(), it[c], alt,
                            prodrop(it[c], PRONOUN[reg]), comp[c], "|".join(agr[c])])


# --------------------------------------------------------------------------- main
def summarize(items: list[dict[str, str]]) -> str:
    c_split = Counter(it["split"] for it in items)
    c_prof = Counter((it["split"], it["profile"]) for it in items)
    c_cons = Counter(it["construction"] for it in items if it["split"] == "core")
    c_type = Counter(it["sentence_type"] for it in items if it["split"] == "core")
    c_src = Counter(it["source"] for it in items)
    distinct = defaultdict(list)
    for it in items:
        distinct[it["profile"]].append(len({prodrop(it[c], PRONOUN[c.split("_")[0]]) for c in CELLS}))
    lines = [
        f"items: {len(items)} {dict(c_split)}  ->  Hindi cells: {len(items) * 6}",
        "core by profile: " + ", ".join(f"{p}={c_prof[('core', p)]}" for p in ("FUT", "PAST", "PRES")),
        "core constructions: " + ", ".join(f"{k}={v}" for k, v in sorted(c_cons.items())),
        f"core sentence types: {dict(c_type)}   sources: {dict(c_src)}",
        "distinct pro-drop surface forms per item (6 = no syncretism): "
        + ", ".join(f"{p}: {Counter(v).most_common()}" for p, v in sorted(distinct.items())),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent.parent
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, default=root / "benchmark_v0.csv")
    ap.add_argument("--long-output", type=Path, default=None)
    ap.add_argument("--check-only", action="store_true")
    args = ap.parse_args(argv)

    items = load(args.input)
    rep = Report()
    ids = Counter(it["id"] for it in items)
    for iid, n in ids.items():
        if n > 1:
            rep.error(iid, f"duplicate id ({n}x)")
    for it in items:
        check_schema(it, rep)
        check_pronouns(it, rep)
        check_minimal_pairs(it, rep)
        check_paradigm(it, rep)
        check_orthography(it, rep)
        check_normalisation(it, rep)
    check_frames(items, rep)

    print(summarize(items))
    for w in rep.warnings:
        print("WARN ", w)
    for e in rep.errors:
        print("ERROR", e)
    print(f"{len(rep.errors)} error(s), {len(rep.warnings)} warning(s)")
    if rep.errors or args.check_only:
        return 1 if rep.errors else 0

    long_path = args.long_output or args.input.with_name(args.input.stem + "_long.csv")
    write_long(items, long_path)
    print(f"wrote {long_path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
