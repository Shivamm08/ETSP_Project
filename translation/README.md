# Translation

These scripts translate the benchmark's English sentences (`../dataset/benchmark_v1.csv`) into Hindi. The
outputs are in [`../outputs/translations/`](../outputs/translations/), one CSV per system.

| script | system | where it runs | output |
|---|---|---|---|
| `run_google.py` | Google Translate (Cloud Translation API, `nmt` model) | API | `google_translate.csv` |
| `run_indictrans2.py` | IndicTrans2, distilled 200M (AI4Bharat) | locally, about 1 GB | `indictrans2_200m.csv` |
| `run_nllb.py` | NLLB-200, distilled 600M (Meta) | locally, about 2.5 GB | `nllb_600m.csv` |
| `run_llm.py --model gpt-oss-120b` | GPT-OSS-120B (OpenAI, open weights) | Groq API | `llm_gpt-oss-120b.csv` |
| `run_llm.py --model gpt-oss-20b` | GPT-OSS-20B | Groq API | `llm_gpt-oss-20b.csv` |
| `run_llm.py --model qwen3.8-27b` | Qwen3.8-27B (Alibaba) | Groq API | `llm_qwen3.8-27b.csv` |
| `run_llm.py --model sarvam-105b` | Sarvam-105B (built for Indian languages) | Sarvam API | `llm_sarvam-105b.csv` |

`config.py` holds what all scripts share: paths, the output columns and the prompt.

## Setup

1. Python 3.12 with the packages in [`../requirements.txt`](../requirements.txt):
   `conda create -n etsp python=3.12 && conda activate etsp && pip install -r requirements.txt`.
2. API keys: copy `../.env.example` to `../.env` and fill in the keys you need
   (`GOOGLE_TRANSLATE_API_KEY`, `GROQ_API_KEY`, `SARVAM_API_KEY`). `.env` is in `.gitignore`.
3. IndicTrans2 is gated on Hugging Face: with your own account, accept its terms once on the
   [model page](https://huggingface.co/ai4bharat/indictrans2-en-indic-dist-200M), create a read token
   at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and put it in `.env` as `HF_TOKEN`.

## Usage

Run from this folder:

```bash
python run_google.py
python run_nllb.py
python run_indictrans2.py
python run_llm.py --model gpt-oss-20b --limit 2                         # quick test
python run_llm.py --model gpt-oss-20b --conditions none both register gender
```

Every script keeps the sentences already in its output file and translates only the rest, so after the
benchmark grows, a rerun adds just the new sentences, and a stopped LLM run continues where it stopped.

## How the LLMs are prompted

Temperature 0, one request per sentence and condition. The prompt adds zero, one or two instruction sentences:

```
Translate the following English sentence into Hindi. Address the person with the pronoun तुम. The person you are talking to is a woman.
Reply with the Hindi translation only.

Sentence: Where did you go?
```

| condition | instruction | sentences | requests |
|---|---|---|---|
| `none` | none | all items | 1 per item |
| `register` | the pronoun तू, तुम or आप | 138 core | 3 per item |
| `gender` | "a man" or "a woman" | 138 core | 2 per item |
| `both` | pronoun and gender | 138 core | 6 per item |

GPT-OSS is a reasoning model and runs with `reasoning_effort: low`; Qwen runs with thinking off; Sarvam
with `reasoning_effort: low`. The script keeps the first line of the reply that contains Devanagari and
stores the full reply in `raw_output`.

Groq's free plan allows 1,000 requests a day per model and refills them gradually; a full run of one model
(1,740 requests for the 222 v1.1 items) takes about a day, and the script waits whenever the limit is reached.

Every LLM file is complete except Sarvam-105B, which returned an empty answer for 2 of its 1,740 requests
even after repeated retries (PST-45 asked for a woman; PST-48 asked for तुम and a woman). Rerunning
`run_llm.py --model sarvam-105b` tries them again.

## Output columns

`id`, `split`, `profile`, `construction`, `frame_id`, `cue`, `english` (copied from the benchmark),
`system`, `model`, `hindi_output`. The LLM files add `condition`, `requested_register`,
`requested_gender`, `target_cell` (the one correct cell, in the `both` condition only) and `raw_output`.
