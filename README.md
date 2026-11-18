# Joint Resolution of Underspecified Attributes in English to Hindi Translation

Course project for *Essentials in Text and Speech Processing* (University of Zurich, fall 2026) by
Shivamm Warambhey, Manasi Agarwal and Manan Tejas Sodha.

English *you* marks neither register nor gender, but a Hindi translation has to mark both: the
pronoun तू, तुम or आप, and a masculine or feminine verb ending. We test which forms MT systems choose
when the English leaves both open, and whether they can produce a requested register and gender at
the same time. The hardest cases are verbs where a single ending carries both attributes, as in
*तुम कहाँ गई थी?* vs *आप कहाँ गई थीं?* ("Where did you go?", female addressee).

## Research questions

- **H1, defaults.** Without instructions, which register and gender do MT systems and LLMs produce?
- **H2, stereotypes.** Does the default gender shift with a stereotyped activity (*Will you cook?* vs
  *Will you drive?*), although it says nothing about the person addressed?
- **H3, control and interference.** When an LLM is asked for a register, a gender or both, does the
  requested form come out, and does steering one attribute disturb the other?
- **H4, metric sensitivity.** Do chrF, BLEU and COMET penalise a translation whose only error is the
  register or the gender?

## Repository

| folder | contents |
|---|---|
| [`dataset/`](dataset/) | The benchmark: English sentences with all six Hindi translations (3 registers × 2 genders), the scripts that build and check it, and the **extractor** that reads register and gender off any Hindi sentence, with its evaluation. See [`dataset/README.md`](dataset/README.md). |
| [`translation/`](translation/) | Scripts that translate the benchmark with Google Translate, IndicTrans2, NLLB-200 and four LLMs, with and without instructions. See [`translation/README.md`](translation/README.md). |
| [`outputs/translations/`](outputs/translations/) | The translations, one CSV per system. |
| [`analysis/`](analysis/) | The analysis of H1–H3 (`analyze.py`) and H4 (`metrics_h4.py`), with unit tests. |
| [`outputs/results/`](outputs/results/) | Result tables (`h1_h3.md`, `h4_metrics.md`), the extractor's label for every output (`labels.csv`) and the figures. |

## Getting the data

Clone the repository once; afterwards, `git pull` inside it fetches updates.

```bash
git clone https://github.com/Shivamm08/ETSP_Project.git
```

The benchmark is two UTF-8 CSV files:

| file | one row per | use it for |
|---|---|---|
| [`dataset/benchmark_v1.csv`](dataset/benchmark_v1.csv) | English sentence, with its six Hindi translations in `tu_m` … `aap_f` | MT input: the `english` column; rows with `split` = `core` for H1, H3 and H4, the frames (`frame_id`) for H2 |
| [`dataset/benchmark_v1_long.csv`](dataset/benchmark_v1_long.csv) | Hindi translation, labelled with `register` and `gender` | scoring MT outputs and testing the extractor |

GitHub shows both files as tables. Don't open and save them in Excel, which can corrupt the Hindi text.

## Setup

- The dataset scripts and the extractor need only Python 3.9 or newer.
- Translation, analysis and metrics need Python 3.12 and the packages in [`requirements.txt`](requirements.txt):

  ```bash
  conda create -n etsp python=3.12 && conda activate etsp
  pip install -r requirements.txt
  ```
- API keys go in `.env` (copy [`.env.example`](.env.example)); `.env` is never committed.

## Reproducing the results

```bash
cd dataset
python scripts/build_dataset.py                 # check the benchmark, regenerate the long CSV
python -m unittest discover -s tests            # benchmark checks, normaliser, extractor
python scripts/evaluate_extractor.py --labelled validation/extractor_gold.csv   # extractor accuracy
cd ../translation
python run_llm.py --model gpt-oss-20b           # etc., see translation/README.md
cd ..
python -m unittest discover -s analysis/tests   # analysis helpers
python analysis/analyze.py                      # H1-H3 tables and figures
python analysis/metrics_h4.py --comet           # H4 (COMET downloads a 2.3 GB model once)
```
