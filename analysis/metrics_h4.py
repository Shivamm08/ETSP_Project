"""H4: do automatic MT metrics notice a wrong register or gender?

Run from the repository root (needs sacrebleu; COMET needs unbabel-comet):
    python analysis/metrics_h4.py            # chrF and BLEU
    python analysis/metrics_h4.py --comet    # also COMET (downloads Unbabel/wmt22-comet-da, 2.3 GB, once)

Three tests, all on the 138 core sentences:
  1. Minimal pairs. Each of the six references is scored against the other five, which are perfect
     translations except for the register, the gender, or both.
  2. Wrong copy vs right paraphrase. For every plain MT output whose register and gender the extractor can
     read, the reference with the same register and gender is the target. The output (right attributes,
     its own wording) competes with the reference's minimal-pair neighbour (the reference's wording, wrong
     attribute). A metric that prefers the neighbour cannot tell a wrong register or gender from a rewording.
  3. LLM outputs asked for both attributes. Their score against the requested cell, for outputs whose
     extracted register and gender are right vs wrong (AUC: how well the score separates the two).

Both texts are spelling-normalised (dataset/scripts/normalize.py) before scoring, so a spelling variant
such as आएंगे / आएँगे is never counted as a difference.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from statistics import mean

import sacrebleu

from common import (CELLS, LLMS, PROFILES, RESULTS, SYSTEMS, auc, benchmark, error_type, labelled_outputs,
                    md_table)
from normalize import normalize_hindi  # dataset/scripts is on the path once common is imported


class Scorer:
    """Sentence-level chrF and BLEU (sacrebleu), and optionally COMET, with a cache keyed on (src, hyp, ref)."""

    def __init__(self, use_comet: bool) -> None:
        self.metrics = ["chrF", "BLEU"] + (["COMET"] if use_comet else [])
        self.pending: set[tuple[str, str, str]] = set()
        self.cache: dict[tuple[str, str, str], dict[str, float]] = {}
        self.comet = None
        if use_comet:
            from comet import download_model, load_from_checkpoint
            self.comet = load_from_checkpoint(download_model("Unbabel/wmt22-comet-da"))

    def want(self, src: str, hyp: str, ref: str) -> tuple[str, str, str]:
        key = (src, normalize_hindi(hyp), normalize_hindi(ref))
        if key not in self.cache:
            self.pending.add(key)
        return key

    def run(self) -> None:
        todo = sorted(self.pending)
        for key in todo:
            _, hyp, ref = key
            self.cache[key] = {"chrF": sacrebleu.sentence_chrf(hyp, [ref]).score,
                               "BLEU": sacrebleu.sentence_bleu(hyp, [ref]).score}
        if self.comet is not None and todo:
            pred = self.comet.predict([{"src": s, "mt": h, "ref": r} for s, h, r in todo], batch_size=32,
                                      gpus=1, progress_bar=False)
            for key, score in zip(todo, pred.scores):
                self.cache[key]["COMET"] = 100 * score
        self.pending.clear()

    def __getitem__(self, key: tuple[str, str, str]) -> dict[str, float]:
        return self.cache[key]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--comet", action="store_true", help="also score with COMET (Unbabel/wmt22-comet-da)")
    args = ap.parse_args(argv)

    items = {k: v for k, v in benchmark().items() if v["split"] == "core"}
    scorer = Scorer(args.comet)

    # 1. minimal pairs
    pairs = []
    for it in items.values():
        for rc in CELLS:
            for hc in CELLS:
                if rc != hc:
                    pairs.append({"id": it["id"], "profile": it["profile"], "ref_cell": rc, "hyp_cell": hc,
                                  "error": error_type(rc, hc), "key": scorer.want(it["english"], it[hc], it[rc])})

    # 2. real outputs vs the minimal-pair neighbours of their reference
    rows = [r for r in labelled_outputs(benchmark()) if r["split"] == "core"]
    plain = [r for r in rows if r["condition"] == "none" and r["register"] in ("tu", "tum", "aap")
             and r["gender"] in ("M", "F")]
    duels = []
    for r in plain:
        it = items[r["id"]]
        cell = f"{r['register']}_{r['gender'].lower()}"
        for hc in CELLS:
            if hc != cell:
                duels.append({"system": r["system"], "id": r["id"], "profile": it["profile"], "error": error_type(cell, hc),
                              "exact": normalize_hindi(r["hindi_output"]) == normalize_hindi(it[cell]),
                              "output": scorer.want(it["english"], r["hindi_output"], it[cell]),
                              "copy": scorer.want(it["english"], it[hc], it[cell])})

    # 3. LLM outputs asked for both attributes
    both = []
    for r in rows:
        if r["system"] in LLMS and r["condition"] == "both" and r.get("target_cell"):
            it = items[r["id"]]
            right = r["register"] == r["requested_register"] and r["gender"] == r["requested_gender"]
            both.append({"system": r["system"], "right": right,
                         "key": scorer.want(it["english"], r["hindi_output"], it[r["target_cell"]])})

    print(f"scoring {len(scorer.pending)} distinct (source, output, reference) triples with {', '.join(scorer.metrics)}")
    scorer.run()
    M = scorer.metrics
    report = ["# Results: H4, metric sensitivity", "",
              "Generated by `analysis/metrics_h4.py`. Sentence-level scores on a 0-100 scale (COMET × 100).", ""]

    # table 1: minimal pairs
    header = ["the hypothesis is the reference with"] + M
    table = []
    for e, label in (("gender", "the wrong gender"), ("register", "the wrong register"), ("both", "both wrong")):
        sub = [p for p in pairs if p["error"] == e]
        table.append([f"{label} (n={len(sub)})"] + [f"{mean(scorer[p['key']][m] for p in sub):.1f}" for m in M])
    sub_out = [d for d in duels if d["error"] == "gender"]
    table.append([f"for comparison: real MT outputs with the right attributes (n={len(sub_out)})"]
                 + [f"{mean(scorer[d['output']][m] for d in sub_out):.1f}" for m in M])
    report += ["## 1. Scores of minimal-pair neighbours", "A perfect translation scores 100. Each row is a "
               "translation that is perfect except for the attribute named.", md_table(header, table), ""]

    header = ["profile"] + [f"{m}: wrong gender" for m in M] + [f"{m}: wrong register" for m in M]
    table = []
    for p in PROFILES:
        row = [p]
        for e in ("gender", "register"):
            sub = [x for x in pairs if x["profile"] == p and x["error"] == e]
            row += [f"{mean(scorer[x['key']][m] for x in sub):.1f}" for m in M]
        table.append(row)
    report += ["### By agreement profile", md_table(header, table), ""]

    # table 2: wrong copy vs right paraphrase
    header = ["outputs", "n"] + [f"{m}: wrong-gender copy wins" for m in M] + [f"{m}: wrong-register copy wins" for m in M]
    table = []
    for label, sub in (("all plain outputs", duels), ("outputs that are not word-for-word the reference",
                                                       [d for d in duels if not d["exact"]])):
        row = [label, str(len({(d['system'], d['id']) for d in sub}))]
        for e in ("gender", "register"):
            dd = [d for d in sub if d["error"] == e]
            row += [f"{100 * mean(scorer[d['copy']][m] > scorer[d['output']][m] for d in dd):.1f}%" for m in M]
        table.append(row)
    by_sys = []
    for s, name in SYSTEMS.items():
        dd = [d for d in duels if d["system"] == s and d["error"] == "gender"]
        if dd:
            by_sys.append([name, str(len(dd))] + [f"{100 * mean(scorer[d['copy']][m] > scorer[d['output']][m] for d in dd):.1f}%" for m in M])
    report += ["## 2. Does the metric prefer a wrong-attribute copy of the reference over a right translation?",
               "Share of comparisons in which the reference's minimal-pair neighbour (same words, wrong gender or "
               "register) scores higher than the system's own output (right register and gender, its own wording).",
               md_table(header, table), "", "### Wrong-gender copy wins, per system",
               md_table(["system", "n"] + M, by_sys), ""]

    # table 3: separability of right and wrong LLM outputs
    if both:
        header = ["LLM", "right", "wrong"] + [f"{m}: mean right / wrong" for m in M] + [f"{m}: AUC" for m in M]
        table = []
        for s in LLMS:
            sub = [b for b in both if b["system"] == s]
            if not sub:
                continue
            pos = [b for b in sub if b["right"]]
            neg = [b for b in sub if not b["right"]]
            table.append([SYSTEMS[s], str(len(pos)), str(len(neg))]
                         + [f"{mean(scorer[b['key']][m] for b in pos):.1f} / {mean(scorer[b['key']][m] for b in neg):.1f}"
                            if pos and neg else "n/a" for m in M]
                         + [f"{auc([scorer[b['key']][m] for b in pos], [scorer[b['key']][m] for b in neg]):.2f}" for m in M])
        report += ["## 3. LLM outputs asked for a register and a gender: do scores separate right from wrong?",
                   "Score against the requested cell's reference, for outputs whose extracted register and gender "
                   "are both right vs not. AUC 0.5 = the score cannot tell them apart; 1.0 = perfectly.",
                   md_table(header, table), ""]

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "h4_metrics.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    with (RESULTS / "h4_scores.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["source", "hypothesis", "reference"] + M)
        for (src, hyp, ref), sc in sorted(scorer.cache.items()):
            w.writerow([src, hyp, ref] + [f"{sc[m]:.2f}" for m in M])
    print("\n".join(report))
    try:
        from figures import fig_h4
        fig_h4(pairs, duels, scorer, M)
    except ImportError as e:
        print(f"(figure skipped: {e})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
