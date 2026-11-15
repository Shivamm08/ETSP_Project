# Joint Resolution of Underspecified Attributes in English to Hindi Translation

Course project for *Essentials in Text and Speech Processing* (University of Zurich, fall 2026) by
Shivamm Warambhey, Manasi Agarwal and Manan Tejas Sodha.

English *you* marks neither register nor gender, but a Hindi translation has to mark both: the
pronoun तू, तुम or आप, and a masculine or feminine verb ending. We test which forms MT systems choose
when the English leaves both open, and whether they can produce a requested register and gender at
the same time. The hardest cases are verbs where a single ending carries both attributes, as in
*तुम कहाँ गई थी?* vs *आप कहाँ गई थीं?* ("Where did you go?", female addressee).

## Repository

| folder | contents |
|---|---|
| [`dataset/`](dataset/) | The benchmark (v1): 150 English sentences, each with all six Hindi translations (3 registers × 2 genders), and the scripts that build and check it. See [`dataset/README.md`](dataset/README.md). |

The extractor and the experiments get their own folders when work on them starts.

## Getting the data

Clone the repository once; afterwards, `git pull` inside it fetches updates.

```bash
git clone https://github.com/Shivamm08/ETSP_Project.git
```

The benchmark is two UTF-8 CSV files:

| file | one row per | use it for |
|---|---|---|
| [`dataset/benchmark_v1.csv`](dataset/benchmark_v1.csv) | English sentence (150), with its six Hindi translations in `tu_m` … `aap_f` | MT input: the `english` column, rows with `split` = `core` for the main experiments |
| [`dataset/benchmark_v1_long.csv`](dataset/benchmark_v1_long.csv) | Hindi translation (900), labelled with `register` and `gender` | scoring MT outputs and testing the extractor |

GitHub shows both files as tables, so they can be read in the browser. Don't open and save them in
Excel, which can corrupt the Hindi text. The scripts need only Python 3.9 or newer.

## Plan

**Core**

1. **Benchmark** (Shivamm): 150 English sentences × 6 Hindi translations.
2. **Extractor** (Manan): reads register and gender off a Hindi MT output with rules: pronoun
   detection, participle suffixes (गया / गए / गई / गईं), auxiliaries (है / हो / हैं), future endings
   (-ेगा / -ोगे / -ेंगे) and the past copula (था / थे / थी / थीं), read only in the clause whose
   subject is the addressee. Its accuracy is measured on hand-labelled MT outputs before it is used.
3. **Default behaviour (H1):** translate the core sentences with several MT systems, without
   instructions, and report which register and gender each one picks.
4. **Control (H3):** ask LLMs for a given register, a given gender, or both, and measure how often
   each comes out right. Does steering one attribute disturb the other, and more so where one verb
   ending carries both?

**Optional, if time allows**

- **Stereotype probe (H2):** do the defaults shift with a stereotyped activity (*Will you cook?* vs
  *Will you drive?*)? The sentences are already in the dataset.
- **Metric sensitivity:** do chrF and COMET penalise a wrong register or gender?
