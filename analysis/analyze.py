"""H1-H3: what register and gender each system produces, and how well LLMs follow instructions.

Run from the repository root:
    python analysis/analyze.py            # tables -> outputs/results/, figures if matplotlib is installed

H1  defaults: register and gender of the plain translations of the 138 core sentences, per system,
    next to a human baseline (the first human translation on Tatoeba of 330 candidate sentences).
H2  stereotypes: the default gender for the neutral, female- and male-stereotyped version of each frame.
H3  control: LLM outputs when asked for a register, a gender, or both; whether the requested attribute
    comes out, and whether steering one attribute moves or weakens the other.
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict

from common import (CELLS, HINDI, LLMS, PROFILES, REGISTERS, RESULTS, SYSTEMS, benchmark, human_baseline,
                    labelled_outputs, md_table, newcombe, pct, register_group, share, sign_test)

REG_GROUPS = ("tu", "tum", "aap", "ambiguous", "mixed", "unmarked")
GENDERS = ("M", "F", "unmarked", "mixed")


# --------------------------------------------------------------------------- H1
def distribution_row(name: str, rows: list[dict]) -> list[str]:
    n = len(rows)
    reg = Counter(register_group(r["register"]) for r in rows)
    gen = Counter(r["gender"] for r in rows)
    return ([name, str(n)] + [share(reg[g], n) if g == "aap" else pct(reg[g], n) for g in REG_GROUPS]
            + [share(gen[g], n) if g == "M" else pct(gen[g], n) for g in GENDERS])


def h1(rows: list[dict], report: list[str], csv_rows: list[dict]) -> None:
    core = [r for r in rows if r["split"] == "core" and r["condition"] == "none"]
    header = ["system", "n"] + [HINDI.get(g, g) for g in REG_GROUPS] + list(GENDERS)
    table = []
    for s, name in SYSTEMS.items():
        mine = [r for r in core if r["system"] == s]
        if mine:
            table.append(distribution_row(name, mine))
            for r in mine:
                csv_rows.append({"analysis": "H1", "system": s, "id": r["id"], "profile": r["profile"],
                                 "register": r["register"], "gender": r["gender"], "hindi": r["hindi_output"]})
    humans = human_baseline()
    table.append(distribution_row("Humans on Tatoeba (first translation)", humans))
    report += ["## H1: defaults without instructions (138 core sentences)",
               "Share of outputs per register and gender label, with 95% Wilson intervals for आप and M. "
               "'ambiguous' = the pronoun was dropped and the verb fits two registers (e.g. गए थे: तुम or आप). "
               "The human row uses the 330 candidate sentences that already have a human translation on Tatoeba.",
               md_table(header, table), ""]

    # by profile: share of आप and of M
    header = ["system"] + [f"{p}: आप" for p in PROFILES] + [f"{p}: M" for p in PROFILES]
    table = []
    for s, name in SYSTEMS.items():
        mine = [r for r in core if r["system"] == s]
        if not mine:
            continue
        cells = []
        for key, want in (("register", "aap"), ("gender", "M")):
            for p in PROFILES:
                sub = [r for r in mine if r["profile"] == p]
                cells.append(pct(sum(r[key] == want for r in sub), len(sub)))
        table.append([name] + cells[:3] + cells[3:])
    report += ["### H1 by agreement profile", md_table(header, table), ""]

    # robustness: without the distractor items (a third person could nudge the gender)
    header = ["system", "n", "आप", "M", "F"]
    table = []
    for s, name in SYSTEMS.items():
        mine = [r for r in core if r["system"] == s and "distractor" not in r["tags"]]
        if mine:
            n = len(mine)
            table.append([name, str(n), pct(sum(r["register"] == "aap" for r in mine), n),
                          pct(sum(r["gender"] == "M" for r in mine), n), pct(sum(r["gender"] == "F" for r in mine), n)])
    report += ["### H1 without the 23 distractor items", md_table(header, table), ""]


# --------------------------------------------------------------------------- H2
def h2(rows: list[dict], items: dict, report: list[str], csv_rows: list[dict]) -> None:
    frames: dict[str, dict[str, str]] = defaultdict(dict)
    for it in items.values():
        if it.get("frame_id"):
            frames[it["frame_id"]][it["cue"]] = it["id"]
    plain = {(r["system"], r["id"]): r for r in rows if r["condition"] == "none"}
    cues = ("neutral", "fem_stereotype", "masc_stereotype")
    header = ["system", "frames", "F: neutral", "F: fem-stereotyped", "F: masc-stereotyped",
              "fem→F, masc not F", "masc→F, fem not F", "sign test p"]
    table = []
    for s, name in SYSTEMS.items():
        got = {c: [plain[(s, f[c])] for f in frames.values() if (s, f.get(c)) in plain] for c in cues}
        if not got["fem_stereotype"]:
            continue
        b = c = 0
        for f in frames.values():
            fem, masc = plain.get((s, f["fem_stereotype"])), plain.get((s, f["masc_stereotype"]))
            if fem and masc:
                b += fem["gender"] == "F" and masc["gender"] != "F"
                c += masc["gender"] == "F" and fem["gender"] != "F"
        table.append([name, str(len(got["fem_stereotype"]))]
                     + [share(sum(r["gender"] == "F" for r in got[cue]), len(got[cue])) for cue in cues]
                     + [str(b), str(c), "<0.001" if sign_test(b, c) < 0.001 else f"{sign_test(b, c):.3f}"])
        for cue in cues:
            for r in got[cue]:
                csv_rows.append({"analysis": "H2", "system": s, "id": r["id"], "profile": r["profile"],
                                 "cue": cue, "register": r["register"], "gender": r["gender"],
                                 "hindi": r["hindi_output"]})
    report += [f"## H2: does the default gender follow a stereotype? ({len(frames)} frames)",
               "Share of feminine outputs for each version of a frame, and the paired comparison of the "
               "female- and male-stereotyped versions (exact sign test on the frames where they differ).",
               md_table(header, table), ""]


# --------------------------------------------------------------------------- H3
def in_profile(rows: list[dict], profile: str) -> list[dict]:
    """The rows of one agreement profile, or all rows for profile 'all'."""
    return [r for r in rows if profile == "all" or r["profile"] == profile]


def register_ok(r: dict) -> bool:
    return r["register"] == r["requested_register"]


def gender_ok(r: dict) -> bool:
    return r["gender"] == r["requested_gender"]


def h3(rows: list[dict], report: list[str], csv_rows: list[dict]) -> None:
    llm_rows = [r for r in rows if r["system"] in LLMS and r["split"] == "core"]
    present = [s for s in LLMS if any(r["system"] == s for r in llm_rows)]
    by = defaultdict(list)
    for r in llm_rows:
        by[(r["system"], r["condition"])].append(r)

    # 1. does the requested attribute come out?
    header = ["LLM", "register asked alone", "gender asked alone", "both: register", "both: gender",
              "both: both right", "both: gender avoided"]
    table = []
    for s in present:
        reg, gen, both = by[(s, "register")], by[(s, "gender")], by[(s, "both")]
        if not (reg or gen or both):
            continue
        table.append([SYSTEMS[s],
                      share(sum(map(register_ok, reg)), len(reg)), share(sum(map(gender_ok, gen)), len(gen)),
                      share(sum(map(register_ok, both)), len(both)), share(sum(map(gender_ok, both)), len(both)),
                      share(sum(register_ok(r) and gender_ok(r) for r in both), len(both)),
                      pct(sum(r["gender"] == "unmarked" for r in both), len(both))])
        for r in reg + gen + both:
            csv_rows.append({"analysis": "H3", "system": s, "id": r["id"], "profile": r["profile"],
                             "condition": r["condition"], "requested_register": r["requested_register"],
                             "requested_gender": r["requested_gender"], "register": r["register"],
                             "gender": r["gender"], "hindi": r["hindi_output"]})
    report += ["## H3: following instructions (138 core sentences, LLMs only)",
               "Share of outputs whose extracted register / gender is the one requested (95% Wilson intervals). "
               "'gender avoided' = the output takes a construction without addressee gender (e.g. तुम्हें भूख लगी होगी).",
               md_table(header, table), ""]

    # 2. interference: is an attribute followed less well when the other one is requested too?
    header = ["LLM", "profile", "gender: alone → with register", "difference [95% CI]",
              "register: alone → with gender", "difference [95% CI]"]
    table = []
    for s in present:
        for p in PROFILES + ("all",):
            g1, g2 = in_profile(by[(s, "gender")], p), in_profile(by[(s, "both")], p)
            r1, r2 = in_profile(by[(s, "register")], p), g2
            if not (g1 and g2 and r1):
                continue
            kg1, kg2 = sum(map(gender_ok, g1)), sum(map(gender_ok, g2))
            kr1, kr2 = sum(map(register_ok, r1)), sum(map(register_ok, r2))
            dg = newcombe(kg2, len(g2), kg1, len(g1))
            dr = newcombe(kr2, len(r2), kr1, len(r1))
            table.append([SYSTEMS[s], p, f"{pct(kg1, len(g1))} → {pct(kg2, len(g2))}",
                          f"{100 * dg[0]:+.1f} [{100 * dg[1]:+.1f}, {100 * dg[2]:+.1f}]",
                          f"{pct(kr1, len(r1))} → {pct(kr2, len(r2))}",
                          f"{100 * dr[0]:+.1f} [{100 * dr[1]:+.1f}, {100 * dr[2]:+.1f}]"])
    report += ["### H3 interference: asked alone vs together with the other attribute",
               "A negative difference means the attribute is followed less often when the other one is "
               "requested as well. The prediction is that this is strongest where one ending carries both "
               "(FUT, PAST) and weakest in PRES.",
               md_table(header, table), ""]

    # 3. which of the six cells are hard?
    header = ["LLM"] + [f"{HINDI[c.split('_')[0]]}·{c.split('_')[1].upper()}" for c in CELLS]
    table = []
    for s in present:
        both = by[(s, "both")]
        if not both:
            continue
        row = [SYSTEMS[s]]
        for c in CELLS:
            sub = [r for r in both if f"{r['requested_register']}_{r['requested_gender'].lower()}" == c]
            row.append(pct(sum(register_ok(r) and gender_ok(r) for r in sub), len(sub)))
        table.append(row)
    report += ["### H3: both attributes right, per requested cell", md_table(header, table), ""]

    # 4. spill-over: what happens to the attribute that was NOT requested?
    header = ["LLM", "asked for", "n", "M", "F", "gender avoided"]
    table = []
    for s in present:
        plain = [r for r in by[(s, "none")]]
        groups = [("nothing (plain)", plain)] + [(HINDI[reg], [r for r in by[(s, "register")]
                                                              if r["requested_register"] == reg]) for reg in REGISTERS]
        for label, sub in groups:
            if sub:
                n = len(sub)
                table.append([SYSTEMS[s], label, str(n), pct(sum(r["gender"] == "M" for r in sub), n),
                              pct(sum(r["gender"] == "F" for r in sub), n),
                              pct(sum(r["gender"] == "unmarked" for r in sub), n)])
    report += ["### H3 spill-over: the gender an LLM picks when only a register is requested",
               md_table(header, table), ""]

    header = ["LLM", "asked for", "n", "तू", "तुम", "आप", "other"]
    table = []
    for s in present:
        groups = [("nothing (plain)", by[(s, "none")])] + [
            (word, [r for r in by[(s, "gender")] if r["requested_gender"] == g])
            for g, word in (("M", "a man"), ("F", "a woman"))]
        for label, sub in groups:
            if sub:
                n = len(sub)
                reg = Counter(register_group(r["register"]) for r in sub)
                table.append([SYSTEMS[s], label, str(n), pct(reg["tu"], n), pct(reg["tum"], n), pct(reg["aap"], n),
                              pct(n - reg["tu"] - reg["tum"] - reg["aap"], n)])
    report += ["### H3 spill-over: the register an LLM picks when only a gender is requested",
               md_table(header, table), ""]

    header = ["LLM", "तुम for 'a man'", "तुम for 'a woman'", "difference [95% CI]"]
    table = []
    for s in present:
        m = [r for r in by[(s, "gender")] if r["requested_gender"] == "M"]
        f = [r for r in by[(s, "gender")] if r["requested_gender"] == "F"]
        if m and f:
            km, kf = sum(r["register"] == "tum" for r in m), sum(r["register"] == "tum" for r in f)
            d = newcombe(km, len(m), kf, len(f))
            table.append([SYSTEMS[s], f"{km}/{len(m)} ({pct(km, len(m))})", f"{kf}/{len(f)} ({pct(kf, len(f))})",
                          f"{100 * d[0]:+.1f} [{100 * d[1]:+.1f}, {100 * d[2]:+.1f}]"])
    report += ["### H3 spill-over: is a man addressed less formally than a woman?",
               "Share of the familiar तुम when only the addressee's gender is given "
               "(Newcombe 95% interval for the difference).",
               md_table(header, table), ""]


def main() -> int:
    items = benchmark()
    rows = labelled_outputs(items)
    RESULTS.mkdir(parents=True, exist_ok=True)
    report: list[str] = ["# Results: H1-H3", "",
                         "Generated by `analysis/analyze.py` from `outputs/translations/` and the extractor's labels.",
                         ""]
    csv_rows: list[dict] = []
    h1(rows, report, csv_rows)
    h2(rows, items, report, csv_rows)
    h3(rows, report, csv_rows)

    (RESULTS / "h1_h3.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    fields = ["analysis", "system", "id", "profile", "cue", "condition", "requested_register", "requested_gender",
              "register", "gender", "hindi"]
    with (RESULTS / "labels.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, restval="")
        w.writeheader()
        w.writerows(csv_rows)
    print("\n".join(report))
    try:
        import figures
        figures.main(rows, items)
    except ImportError as e:
        print(f"(figures skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
