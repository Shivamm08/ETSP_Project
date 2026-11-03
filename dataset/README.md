# Register × Gender benchmark for English→Hindi (v0)

English leaves the addressee's **register** and **gender** unspecified; Hindi must mark both. This
benchmark gives every English source sentence all six Hindi translations on the 3 × 2 grid:

| | male addressee | female addressee |
|---|---|---|
| **तू** (intimate) | तू कहाँ गया था? | तू कहाँ गई थी? |
| **तुम** (familiar) | तुम कहाँ गए थे? | तुम कहाँ गई थी? |
| **आप** (respectful) | आप कहाँ गए थे? | आप कहाँ गई थीं? |

*"Where did you go?" (item PST-05)*

| | core | H2 probe | total |
|---|---:|---:|---:|
| English sources | 50 | 12 | 62 |
| Hindi references (×6) | 300 | 72 | 372 |
| Generated pro-drop variants | 300 | 72 | 372 |

Sources: 57 verbatim Tatoeba sentences, 2 adapted from Tatoeba, 3 authored.

The Hindi references were drafted with an LLM (Claude), and every one is checked by the three
authors, all native Hindi speakers. `scripts/build_dataset.py` also checks every cell automatically:
pronoun, agreement endings, minimal pairs and spelling.

---

## Design

### Three agreement profiles (the main design variable)

The central hypothesis (H3) is that register and gender interfere *because they share a morpheme*.
The benchmark therefore varies **where** Hindi puts the two attributes and keeps the 3 × 2 grid fixed.
The finite element decides the profile:

| profile | core items | register is marked on | gender is marked on | distinct forms without the pronoun | तुम · F example |
|---|---:|---|---|---:|---|
| **FUT** (future) | 16 | the verb ending (-एगा / -ओगे / -एँगे) | the **same** ending (-गा / -गी) | 6 of 6 (5 for होना) | जाओगी |
| **PAST** (perfective, past copula, past progressive/habitual/perfect) | 17 | **number only** (sg vs pl), on the participle | the **same** suffix | **4 of 6** | गई थी |
| **PRES** (present progressive/habitual/perfect, adjective + copula, सकना) | 17 | the auxiliary है / हो / हैं | the participle or adjective | 6 of 6 | जा रही हो |

So the profiles differ in exactly the property H3 is about:

- **FUT**: both attributes sit in one synthetic verb form, and every cell stays distinct.
- **PAST**: both attributes sit in one suffix that carries no person. Without the pronoun, the six
  cells collapse into four: {तू·M}, {तू·F, तुम·F}, {तुम·M, आप·M}, {आप·F}.
- **PRES**: the attributes sit on different words. For a female addressee the participle (रही, जाती,
  भूखी) is identical in all three registers, and only the auxiliary changes.

The prediction H3 makes is that steering register should hurt gender accuracy more in FUT and PAST
than in PRES, and most of all for female addressees in PAST, where गई vs गईं is only a nasal.

Core constructions: future 16; perfective 8, past copula 4, past perfect 2, past progressive 2, past
habitual 1; present progressive 5, present habitual 5, predicate adjective 3, present perfect 2,
modal सकना 2. The core has 37 questions and 13 statements, and 3 items are negated.

### H2 probe: controlled lexical variation

H2 asks whether defaults shift on cues that should not matter. It uses six frames, and each frame
comes in three versions that differ only in a stereotyped activity, place or profession. The neutral
version belongs to the core. The other two make up the probe.

| frame | profile | neutral (core) | fem-stereotyped | masc-stereotyped |
|---|---|---|---|---|
| fr1 | FUT | Will you swim? | Will you cook? | Will you drive? |
| fr2 | PRES | Are you going to the market today? | …to dance class today? | …to the gym today? |
| fr3 | PAST | Were you sleeping? | Were you crying? | Were you drinking? |
| fr4 | PRES | Do you play chess? | Do you knit? | Do you lift weights? |
| fr5 | PAST | Did you go to the post office? | …to the beauty parlour? | …to the barber? |
| fr6 | FUT | Will you work as a cashier? | …as a nurse? | …as a mechanic? |

Every cell stays valid in every frame. A man can knit and a woman can go to a barber, so the English
remains underspecified. To test H2, compare the default gender (and register) across the three versions
of each frame.

### What makes a sentence usable

An English source is kept only if its Hindi translation **must** mark both attributes:

- **"you" is the agreeing subject, and it is singular and unnamed.** Excluded: *you guys*, *both of you*,
  vocatives, names.
- **No cue to gender or register.** Excluded: *sir, madam, mom, bro*, *he/she*, family terms. H2 cues
  appear only in the probe.
- **No first-person subject.** *I went* would add the speaker's gender as a third attribute.
- **Nominative agreement survives in Hindi.** The following are excluded because they carry no
  addressee gender:
  - transitive perfectives with ने: *तुमने खाना खाया* agrees with खाना
  - dative subjects: *तुम्हें भूख लगी है*, *तुम्हें पसंद है*, *तुम्हें जाना चाहिए*
  - imperatives: *जाओ*
  - the present copula with an invariant adjective: *तुम तैयार / खुश / सही हो*
- **Short and conversational:** 2–10 words.

Some items still allow a natural Hindi translation that avoids agreement, such as *Are you hungry?* →
*क्या तुम्हें भूख लगी है?* These items are listed in the `escape_route` column. An MT system that takes
this route has made a real choice (neutralisation), and the extractor should label it `unmarked`.

### Where the English sentences come from

They all come from [Tatoeba](https://tatoeba.org) (CC-BY 2.0 FR), except for one authored frame.

1. `scripts/select_sources.py` filtered all 2,039,120 English sentences in the 2026-10-19 export using
   the criteria above. That produced **38,658 candidates**, sorted into buckets by profile, in
   `sources/tatoeba_candidate_pool.tsv`.
2. Items were then picked by hand from the pool to fill the profile quotas.
3. Of the 62 items, 59 are in the pool: 57 verbatim and 2 adapted H2 versions. The 3 authored items
   are frame fr6, because Tatoeba has no profession sentence with agreement.

---

## Conventions in the Hindi references

- **Minimal pairs.** The six cells differ only in the pronoun and the agreement endings. Vocabulary
  stays identical, so there is no added जी and no politer verb for आप. `build_dataset.py` enforces this.
- **Overt pronoun in every reference.** Pro-drop variants are generated, not written.
- **Yes/no questions start with क्या.**
- **Orthography:**
  - गए / गई, not गये / गयी.
  - Chandrabindu where no vowel sign rises above the line (कहाँ, जाएँगे), and anusvara otherwise
    (हैं, नहीं, थीं, गईं).
  - Nukta only on ज़ and फ़.
  - Feminine plural nasalisation appears once, on the last element: **आप गई थीं**, not गईं थीं.
  - Other spellings, such as आएंगे with a dot, गये, or ज्यादा without a nukta, are equally correct.
    `scripts/normalize.py` treats them as identical, so they never count as errors (see
    [Notes for the extractor](#notes-for-the-extractor)).
- **तुम + female addressee in PAST items is the one real variation point.**
  - The references use the colloquial **गई / थी**. Tatoeba's own human translation of *Were you
    drinking?* is also *तुम पी रही थी?*
  - The prescriptive **गईं / थीं** is kept in `tum_f_alt`. It is always the आप · F form with तुम.
  - Both forms count as correct.
- **Everyday Hindustani.** Use common loanwords where they are the everyday word (ट्रेन, जिम, कैशियर).

---

## Files

| path | what |
|---|---|
| `benchmark_v0.csv` | **Source of truth.** One row per English source with six Hindi cells. Edit this file and nothing else. |
| `benchmark_v0_long.csv` | Generated. One row per (item, cell), with pro-drop form, compatible cells and agreement tokens. |
| `validation/validation_sheet.xlsx` | Shivamm's validation sheet: a verdict for each of the 372 Hindi cells and a source check for each English sentence. |
| `sources/tatoeba_candidate_pool.tsv` | The 38,658 filtered Tatoeba candidates (the seed pool). |
| `scripts/build_dataset.py` | Checks the CSV and generates the long CSV. |
| `scripts/normalize.py` | `normalize_hindi()`: makes spelling variants identical before matching or scoring. |
| `tests/test_normalize.py` | Unit tests for the normaliser. |
| `scripts/select_sources.py` | Builds the candidate pool from a Tatoeba export. |
| `scripts/cocoa_check.py` | Counts CoCoA-MT's register and gender coverage (see [Related data](#related-data)). |

### Schema of `benchmark_v0.csv`

| column | values |
|---|---|
| `id` | `FUT-01`…, `PST-01`…, `PRS-01`… (core); `H2-1F`, `H2-1M`… (probe) |
| `split` | `core`, `h2_probe` |
| `profile` / `construction` | `FUT` / `PAST` / `PRES`, and the construction within it (see counts above) |
| `sentence_type`, `polarity` | `question` / `statement`; `affirmative` / `negative` |
| `frame_id`, `cue` | `fr1`–`fr6`; `none` (not in a frame), `neutral`, `fem_stereotype`, `masc_stereotype` |
| `english`, `source`, `source_id` | the source; `tatoeba` / `tatoeba_adapted` / `authored`; Tatoeba sentence id |
| `tu_m` … `aap_f` | the six Hindi cells |
| `tum_f_alt` | prescriptive तुम · F form (PAST items only) |
| `escape_route` | a natural translation that avoids agreement, if one exists |
| `tatoeba_existing_hi` | human Hindi translations already on Tatoeba (a "human prior"), if any |
| `notes` | remarks on the item, such as equally good alternative wordings |

The long file adds the following columns:

| column | contents |
|---|---|
| `cell`, `register`, `gender` | which of the six cells this row is |
| `hindi` | the reference |
| `hindi_alt` | the accepted alternative, if any |
| `hindi_prodrop` | the reference with the pronoun removed |
| `prodrop_compatible_cells` | every cell that could have produced that pro-drop string. Example: `tu_f\|tum_f` |
| `agreement_tokens` | the tokens that vary across cells. Example: `रही\|हो` |

---

## Changing and checking the data

- Edit `benchmark_v0.csv` in a text editor such as VS Code, not in Excel: Excel can corrupt Hindi
  text when it saves a CSV.
- Then run the checks below from this folder until they report 0 errors. They catch typos and
  regenerate `benchmark_v0_long.csv`. Python 3.9 or newer is enough; there is nothing to install.

```bash
python3 scripts/build_dataset.py                  # check the data + regenerate the long CSV
python3 -m unittest discover -s tests             # test the spelling normaliser
```

---

## Notes for the extractor

- **Free gold labels.**
  - 372 sentences with an overt pronoun.
  - 372 pro-drop variants, each with `prodrop_compatible_cells`.
  - Score the pro-drop tier against that **set**. For a syncretic form such as *क्या ठीक से सोए?*
    (तुम·M or आप·M), the right output is "masculine, register ambiguous between तुम and आप". Guessing
    one register is wrong.
- **Labels beyond the six cells:**
  - `unmarked`: the system took an escape route (dative, genitive or imperative).
  - `mixed`: the register is inconsistent, for example *आप … हो*. This happens in real data. Tatoeba's
    human translations include *आप बाद में समझोगे*.
- **Normalise spelling first.** Run `normalize_hindi()` from `scripts/normalize.py` on **both** the MT
  output and the references before matching or scoring.
  - It makes these identical: dot and chandrabindu (आएंगे / आएँगे), गये / गए, nukta (ज़्यादा / ज्यादा),
    and हिन्दी / हिंदी.
  - It never merges forms that differ in grammar (गई / गईं, है / हैं).
  - `tests/test_normalize.py` tests it, and `build_dataset.py` checks that it never merges two cells
    or two pro-drop forms of any item.
- **External test set.** CoCoA-MT's en-hi test split has 272 segments with masculine and feminine
  references for तुम / आप. In about 4 in 10 of them the feminine version changes the *speaker's*
  gender, so keep only those where it changes the addressee's (see the next section).

---

## Related data

The closest existing resource is the **CoCoA-MT** en-hi test set (Nadejde et al., 2022; 600
segments). `scripts/cocoa_check.py` gives these counts:

- The informal references use **तुम** (487 references contain तुम forms). **तू never appears.**
- **272 of 600** segments (45%) have separate feminine references.
- In those, the feminine version changes the addressee only in 137, the speaker only in 67 and both
  in 38; 30 are unclear (a token-diff heuristic). So about 105 change the **speaker** (मैं समझता हूँ →
  समझती हूँ), and CoCoA-MT cannot isolate addressee gender.

This benchmark adds तू, varies the agreement profile in a controlled way and keeps the speaker out of
every sentence.

**Tatoeba human prior.** 330 sentences in the candidate pool already have a human Hindi translation on
Tatoeba. The first translation uses **तुम** in 205, **आप** in 64 and **तू** in 2, and drops the pronoun
in 59: a human reference point for the default register (H1).

---

## Limitations

- **Small analysis cells.** The 50 core items give 16–17 per profile, which shows only large
  differences between profiles. Most constructions have 1–5 items, so analyse by profile, not by
  construction.
- **Two genders only.** Hindi grammatical gender is binary. A non-binary addressee has no
  grammaticalised form, which is outside the scope of this benchmark.
- **Standard Hindi only.** Register use varies by region. Examples: आप + हो in Mumbai, and तू between
  friends in many places.
- **तू cells are grammatical but socially marked.** Whether they are appropriate depends on context,
  which this benchmark does not model.
- **Stereotype directions in H2** reflect the authors' judgement for an Indian setting.
- **Possible contamination.** Tatoeba sentences may appear in MT training data. That matters more for
  translation quality than for the default choice of attributes.

## Licence and attribution

- **English sources and `tatoeba_existing_hi`:** Tatoeba, CC-BY 2.0 FR. Sentence ids are in
  `source_id`, and any redistribution must credit Tatoeba.
- **CoCoA-MT:** no CoCoA-MT text is included (statistics only), so its CDLA-Sharing licence does not
  apply.
