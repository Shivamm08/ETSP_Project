# Outputs

## `translations/`

One UTF-8 CSV per system, made by the scripts in [`../translation/`](../translation/):

| file | system | rows |
|---|---|---:|
| `google_translate.csv` | Google Translate | 222 |
| `indictrans2_200m.csv` | IndicTrans2, distilled 200M | 222 |
| `nllb_600m.csv` | NLLB-200, distilled 600M | 222 |
| `llm_gpt-oss-120b.csv` | GPT-OSS-120B | 1,740 |
| `llm_gpt-oss-20b.csv` | GPT-OSS-20B | 1,740 |
| `llm_qwen3.8-27b.csv` | Qwen3.8-27B | 1,740 |
| `llm_sarvam-105b.csv` | Sarvam-105B | 1,738 |

Each row is one translation: the benchmark item (`id`, `split`, `profile`, `construction`, `frame_id`, `cue`,
`english`), the `system`, the `model` and the `hindi_output`. The LLM files add the prompting `condition`
(`none`, `register`, `gender`, `both`), the `requested_register` and `requested_gender`, the `target_cell`
(the one correct cell in the `both` condition) and the model's full reply in `raw_output`. Sarvam-105B
returned no answer for 2 prompts (see the translation README).

## `results/`

Made by the scripts in [`../analysis/`](../analysis/):

| file | contents |
|---|---|
| `h1_h3.md` | all H1–H3 tables |
| `h4_metrics.md` | all H4 tables |
| `labels.csv` | the extractor's register and gender for every output used in H1–H3 |
| `h4_scores.csv` | chrF, BLEU and COMET for every scored (source, hypothesis, reference) triple |
| `figures/` | the four figures of the report, as PNG and PDF |
