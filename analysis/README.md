# Analysis

These scripts turn the translations in [`../outputs/translations/`](../outputs/translations/) into the
results in [`../outputs/results/`](../outputs/results/). Every register and gender label comes from the
extractor ([`../dataset/scripts/extractor.py`](../dataset/scripts/extractor.py)), whose accuracy is measured
on hand-labelled outputs (see the dataset README).

| file | what it does |
|---|---|
| `analyze.py` | H1–H3: tables (`h1_h3.md`), the label of every output (`labels.csv`) and the figures |
| `metrics_h4.py` | H4: chrF, BLEU and COMET scores (`h4_metrics.md`, `h4_scores.csv`) and the H4 figure |
| `figures.py` | the four figures, as PNG and PDF; called by the two scripts above |
| `common.py` | shared loading, the human baseline and the statistics helpers |
| `tests/` | unit tests for the statistics and label helpers |

## Usage

From the repository root, with the packages in [`../requirements.txt`](../requirements.txt):

```bash
python analysis/analyze.py               # H1-H3
python analysis/metrics_h4.py            # H4 with chrF and BLEU
python analysis/metrics_h4.py --comet    # also COMET (downloads Unbabel/wmt22-comet-da, 2.3 GB, once)
python -m unittest discover -s analysis/tests
```

`analyze.py` needs only the standard library (and matplotlib for the figures, which are skipped without
it); `metrics_h4.py` needs sacrebleu, and unbabel-comet for `--comet`.

## What is measured

- **H1, defaults.** For the plain translations of the 138 core sentences: the share of each register
  (तू, तुम, आप, a pronoun-less form that fits two registers, mixed, unmarked) and gender (M, F, unmarked,
  mixed), per system and per agreement profile, and again without the 23 items that mention a third person.
  The human baseline is the first human translation on Tatoeba of the 330 candidate sentences that have one.
- **H2, stereotypes.** For the 30 frames: the share of feminine outputs for the neutral, female- and
  male-stereotyped version, and an exact sign test on the frames where the two stereotyped versions differ.
- **H3, control.** For the LLMs: how often the requested register or gender appears when asked alone or
  together, the difference with Newcombe's interval, per profile and per requested cell, and the
  spill-over, i.e. which register is chosen when only a gender is requested, and vice versa.
- **H4, metrics.** (1) Each reference is scored against its five minimal-pair neighbours; (2) a system's
  correct output is compared with the reference's wrong-gender or wrong-register neighbour; (3) the scores of
  LLM outputs with right and wrong attributes are compared (AUC). Texts are spelling-normalised first.

Proportions come with 95% Wilson intervals. The figures use a colour-blind-checked palette and do not code
gender as blue and pink.
