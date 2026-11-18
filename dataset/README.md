# Register × Gender benchmark for English→Hindi (v1.1)

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
| English sources | 138 | 84 | 222 |
| Hindi references (×6) | 828 | 504 | 1,332 |
| Generated pro-drop variants | 828 | 504 | 1,332 |

Sources: 143 verbatim Tatoeba sentences, 2 adapted from Tatoeba, 77 authored (62 of them in the
templated H2 frames of v1.1).

Every Hindi reference is checked, and corrected where needed, by the three authors, all native
Hindi speakers. `scripts/build_dataset.py` also checks every cell automatically: pronoun, agreement
endings, minimal pairs and spelling.

Version 0 (the first 62 items) is in the git history at commit `209fe84`. Version 1 keeps those items
unchanged and adds 88. Version 1.1 keeps all 150 unchanged and adds 72 items: 24 new H2 frames
(fr7–fr30). The core is the same in v1 and v1.1, and the file keeps its name, `benchmark_v1.csv`.

---

## Design

### Three agreement profiles (the main design variable)

The central hypothesis (H3) is that register and gender interfere *because they share a morpheme*.
The benchmark therefore varies **where** Hindi puts the two attributes and keeps the 3 × 2 grid fixed.
The finite element decides the profile:

| profile | core items | register is marked on | gender is marked on | distinct forms without the pronoun | तुम · F example |
|---|---:|---|---|---:|---|
| **FUT** (future; presumptive *must be* with होगा) | 44 | the verb ending (-एगा / -ओगे / -एँगे) | the **same** ending (-गा / -गी) | 6 of 6 (5 for होना and the presumptive) | जाओगी, थक गई होगी |
| **PAST** (perfective, past copula, past progressive/habitual/perfect, counterfactual) | 48 | **number only** (sg vs pl), on the participle | the **same** suffix | **4 of 6** | गई थी |
| **PRES** (present progressive/habitual/perfect, adjective + copula, सकना, *about to* with -ने वाला) | 46 | the auxiliary है / हो / हैं | the participle or adjective | 6 of 6 | जा रही हो |

So the profiles differ in exactly the property H3 is about:

- **FUT**: both attributes sit in one synthetic verb form, and every cell stays distinct.
- **PAST**: both attributes sit in one suffix that carries no person. Without the pronoun, the six
  cells collapse into four: {तू·M}, {तू·F, तुम·F}, {तुम·M, आप·M}, {आप·F}. Counterfactuals
  (*अगर तुम वहाँ होती*) have no auxiliary but agree the same way.
- **PRES**: the attributes sit on different words. For a female addressee the participle (रही, जाती,
  भूखी) is identical in all three registers, and only the auxiliary changes.

The prediction H3 makes is that steering register should hurt gender accuracy more in FUT and PAST
than in PRES, and most of all for female addressees in PAST, where गई vs गईं is only a nasal.

Core constructions: future 36, presumptive 8; perfective 15, past copula 8, past habitual 8, past
progressive 8, past perfect 7, counterfactual 2; present habitual 9, present progressive 9, predicate
adjective 8, present perfect 8, modal सकना 8, prospective 4. The core has 89 questions and 49
statements, and 16 items are negated.

### Variety beyond the profiles

The items also vary in ways that real MT output does, so that an extractor built on them has to read
agreement rather than rely on surface habits. The `tags` column marks these features:

| tag | core items | what it means | example |
|---|---:|---|---|
| `multi_clause` | 21 | two or more clauses | जब बारिश शुरू हुई, तब क्या तुम घर पर थी? |
| `multi_site` | 16 | the addressee's agreement sits on two or more separate stretches of words | तुम सोमवार को **आई** और अगले ही दिन फिर **चली गई** |
| `distractor` | 23 | another word agrees with someone or something else | क्या तुम उस लड़की से मिले जो कल **आई थी**? |
| `conditional` | 7 | an *if* clause | अगर तुम थकी हुई हो, तो सो सकती हो। |
| `request` | 6 | a request or invitation | क्या तुम थोड़ा धीरे बोल सकती हो? |
| `tag_question` | 4 | a statement turned into a question with न | तुम आज आ रही हो न? |

`build_dataset.py` computes `multi_site` from the cells and rejects a wrong tag. In `distractor` items
a gendered ending that does not belong to the addressee appears in every cell, for example the
feminine आई थी in a masculine cell. Three items (FUT-31, FUT-34, PST-35) also mark register on a
possessive or oblique pronoun (तुम्हारी, तुम्हें), so register survives pro-drop there. English sources
are 2–13 words long (mean 5.4); 26 have 8 words or more.

### H2 probe: controlled lexical variation

H2 asks whether defaults shift on cues that should not matter. It uses 30 frames, and each frame
comes in three versions that differ only in a stereotyped activity, place or profession. In frames
fr1–fr6 the neutral version belongs to the core and the other two make up the probe; frames fr7–fr30
(v1.1) keep all three versions in the probe, so the core stays unchanged.

| frame | profile | neutral (core) | fem-stereotyped | masc-stereotyped |
|---|---|---|---|---|
| fr1 | FUT | Will you swim? | Will you cook? | Will you drive? |
| fr2 | PRES | Are you going to the market today? | …to dance class today? | …to the gym today? |
| fr3 | PAST | Were you sleeping? | Were you crying? | Were you drinking? |
| fr4 | PRES | Do you play chess? | Do you knit? | Do you lift weights? |
| fr5 | PAST | Did you go to the post office? | …to the beauty parlour? | …to the barber? |
| fr6 | FUT | Will you work as a cashier? | …as a nurse? | …as a mechanic? |

The 24 frames of v1.1 follow four templates, six frames each, one per agreement pattern:

| frames | profile | template | neutral / fem-stereotyped / masc-stereotyped |
|---|---|---|---|
| fr7–fr12 | FUT | *Will you work as a …?* | accountant / receptionist / electrician; journalist / beautician / plumber; translator / flight attendant / pilot; librarian / babysitter / security guard; bank clerk / secretary / carpenter; pharmacist / kindergarten teacher / truck driver |
| fr13–fr18 | PRES | *Do you …?* | read novels / sew / play football; do yoga / embroider / smoke; paint / make rangoli / play video games; sing / watch soap operas / wrestle; take photos / wear makeup / fix bikes; go for walks / do the dishes / play cricket |
| fr19–fr24 | PAST | *Did you go to the …?* | bank / boutique / garage; library / sewing class / cricket ground; station / kitty party / bar; museum / cooking class / car showroom; hospital / jewellery shop / hardware store; stationery shop / saree shop / paan shop |
| fr25–fr30 | PAST | *Were you …?* | reading / sewing / smoking; resting / cooking / fighting; studying / putting on makeup / playing cricket; walking / shopping / fixing the car; writing / doing the dishes / watching the match; listening to music / applying mehndi / playing cards |

The templates avoid English words that state a gender (*actress*, *housewife*) and Hindi nouns that
change with gender (लेखक / लेखिका), so the stereotype is the only cue.

Every cell stays valid in every frame. A man can knit and a woman can go to a barber, so the English
remains underspecified. To test H2, compare the default gender (and register) across the three versions
of each frame.

### What makes a sentence usable

An English source is kept only if its Hindi translation **must** mark both attributes:

- **"you" is the agreeing subject, and it is singular and unnamed.** Excluded: *you guys*, *both of you*,
  vocatives, names.
- **No cue to the addressee's gender or register.** Excluded: *sir, madam, mom, bro* and other
  forms of address. H2 cues appear only in the probe. A third person (*he*, *the girl*, *your sister*)
  is allowed, because it says nothing about the addressee.
- **No first-person subject.** *I went* would add the speaker's gender as a third attribute.
  First-person objects (*for me*, *with us*) carry no gender and are allowed.
- **Nominative agreement survives in Hindi.** The following are excluded because they carry no
  addressee gender:
  - transitive perfectives with ने: *तुमने खाना खाया* agrees with खाना
  - dative subjects: *तुम्हें भूख लगी है*, *तुम्हें पसंद है*, *तुम्हें जाना चाहिए*
  - imperatives: *जाओ*
  - the present copula with an invariant adjective: *तुम तैयार / खुश / सही हो*
- **Conversational:** 2–13 words.

Some items still allow a natural Hindi translation that avoids agreement, such as *Are you hungry?* →
*क्या तुम्हें भूख लगी है?* These items are listed in the `escape_route` column. An MT system that takes
this route has made a real choice (neutralisation), and the extractor should label it `unmarked`.

### Where the English sentences come from

They come from [Tatoeba](https://tatoeba.org) (CC-BY 2.0 FR), except for 77 authored items, 62 of
which are the templated H2 frames of v1.1.

1. **v0 (62 items).** `scripts/select_sources.py` filtered all 2,039,120 English sentences in the
   2026-10-19 export using the criteria above. That produced **38,658 candidates**, sorted into
   buckets by profile, in `sources/tatoeba_candidate_pool.tsv`. Items were picked by hand from the
   pool to fill the profile quotas. 59 of the 62 are in the pool (57 verbatim, 2 adapted H2 versions);
   the 3 authored items are frame fr6, because Tatoeba has no profession sentence with agreement.
2. **v1 (88 added items, 2026-11-09).** The pool filter is deliberately strict: it drops sentences with
   *must*, *seem* or *know*, any third person, and anything over 10 words. The new items came from
   targeted searches of the full English export of 2026-11-09 for the constructions and features above.
   76 are verbatim Tatoeba sentences; 12 were authored, mostly to get a third person whose agreement
   differs from the addressee's.
3. **v1.1 (72 added items, 2026-11-18).** The 24 H2 frames fr7–fr30, written from the four templates
   above. Each English sentence was looked up on Tatoeba: the 10 that exist word for word keep their
   id, and the other 62 are authored.

Every Tatoeba item keeps its sentence id in `source_id`.

---

## Conventions in the Hindi references

- **Minimal pairs.** The six cells differ only in forms of the pronoun (तू / तुम / आप, and तेरी /
  तुम्हारी / आपकी where a possessive appears) and the agreement endings. Vocabulary stays identical,
  so there is no added जी and no politer verb for आप. `build_dataset.py` enforces this.
- **Exactly one overt subject pronoun per reference.** A second clause drops it, as Hindi normally
  does (अगर तुम थकी हुई हो, तो सो सकती हो). Pro-drop variants are generated, not written.
- **Yes/no questions start with क्या.**
- **Orthography:**
  - गए / गई, not गये / गयी.
  - Chandrabindu where no vowel sign rises above the line (कहाँ, जाएँगे), and anusvara otherwise
    (हैं, नहीं, थीं, गईं).
  - Nukta only on ज़ and फ़.
  - Feminine plural nasalisation appears once per verb group, on its last element: **आप गई थीं**,
    not गईं थीं; **आप आईं और चली गईं**.
  - Other spellings, such as आएंगे with a dot, गये, or ज्यादा without a nukta, are equally correct.
    `scripts/normalize.py` treats them as identical, so they never count as errors (see
    [Notes for the extractor](#notes-for-the-extractor)).
- **तुम + female addressee in PAST items is the one real variation point.**
  - The references use the colloquial **गई / थी**. Tatoeba's own human translation of *Were you
    drinking?* is also *तुम पी रही थी?*
  - The prescriptive **गईं / थीं** is kept in `tum_f_alt`. It is the तुम · F sentence with the
    आप · F endings.
  - Both forms count as correct.
- **Everyday Hindustani.** Use common loanwords where they are the everyday word (ट्रेन, जिम, कैशियर).

---

## Files

| path | what |
|---|---|
| `benchmark_v1.csv` | **Source of truth.** One row per English source with six Hindi cells. Edit this file and nothing else. |
| `benchmark_v1_long.csv` | Generated. One row per (item, cell), with pro-drop form, compatible cells and agreement tokens. |
| `validation/validation_sheet.xlsx` | Shivamm's validation sheet: a verdict for each of the 1,332 Hindi cells and a source check for each English sentence. Bold marks the words that change across an item's six versions. |
| `sources/tatoeba_candidate_pool.tsv` | The 38,658 filtered Tatoeba candidates (the v0 seed pool). |
| `scripts/build_dataset.py` | Checks the CSV and generates the long CSV. |
| `scripts/normalize.py` | `normalize_hindi()`: makes spelling variants identical before matching or scoring. |
| `scripts/extractor.py` | `extract()`: reads the addressee's register and gender off a Hindi sentence such as an MT output (see [The extractor](#the-extractor)). |
| `scripts/evaluate_extractor.py` | Measures the extractor: a development check on this benchmark, a hand-labelling sheet for MT outputs, and the held-out accuracy on those labels. |
| `scripts/label_gold.py` | A keyboard tool for hand-labelling the MT outputs in `validation/extractor_gold.csv` (it never shows the extractor's prediction or the system). |
| `validation/extractor_gold.csv` | The MT outputs in random order, with the hand labels `gold_register` and `gold_gender`. |
| `tests/` | Unit tests for the normaliser, the checks in `build_dataset.py`, and the extractor and its evaluation. |
| `scripts/select_sources.py` | Builds the candidate pool from a Tatoeba export. |
| `scripts/cocoa_check.py` | Counts CoCoA-MT's register and gender coverage (see [Related data](#related-data)). |

### Schema of `benchmark_v1.csv`

| column | values |
|---|---|
| `id` | `FUT-01`…, `PST-01`…, `PRS-01`… (core); `H2-1F`, `H2-1M`… (probe) |
| `split` | `core`, `h2_probe` |
| `profile` / `construction` | `FUT` / `PAST` / `PRES`, and the construction within it (see counts above) |
| `sentence_type`, `polarity` | `question` / `statement`; `affirmative` / `negative` (of the addressee's predicate) |
| `frame_id`, `cue` | `fr1`–`fr6`; `none` (not in a frame), `neutral`, `fem_stereotype`, `masc_stereotype` |
| `tags` | any of `multi_clause`, `multi_site`, `distractor`, `conditional`, `request`, `tag_question`, separated by `\|` |
| `english`, `source`, `source_id` | the source; `tatoeba` / `tatoeba_adapted` / `authored`; Tatoeba sentence id |
| `tu_m` … `aap_f` | the six Hindi cells |
| `tum_f_alt` | prescriptive तुम · F form (PAST items only) |
| `escape_route` | a natural translation that avoids agreement, if one exists |
| `tatoeba_existing_hi` | human Hindi translations already on Tatoeba (a "human prior"), if any |
| `notes` | remarks on the item, such as equally good alternative wordings |

The long file repeats the item columns up to `english` (including `tags`) and adds:

| column | contents |
|---|---|
| `cell`, `register`, `gender` | which of the six cells this row is |
| `hindi` | the reference |
| `hindi_alt` | the accepted alternative, if any |
| `hindi_prodrop` | the reference with the subject pronoun removed |
| `prodrop_compatible_cells` | every cell that could have produced that pro-drop string. Example: `tu_f\|tum_f` |
| `agreement_tokens` | the tokens that vary across cells, excluding forms of the pronoun. Example: `रही\|हो` |

---

## Changing and checking the data

- Edit `benchmark_v1.csv` in a text editor such as VS Code, not in Excel: Excel can corrupt Hindi
  text when it saves a CSV.
- Then run the checks below from this folder until they report 0 errors. They catch typos and
  regenerate `benchmark_v1_long.csv`. Python 3.9 or newer is enough; there is nothing to install.

```bash
python3 scripts/build_dataset.py                  # check the data + regenerate the long CSV
python3 -m unittest discover -s tests             # test the normaliser, the checks and the extractor
```

---

## Notes for the extractor

- **Free gold labels.**
  - 1,332 sentences with an overt pronoun.
  - 1,332 pro-drop variants, each with `prodrop_compatible_cells`.
  - Score the pro-drop tier against that **set**. For a syncretic form such as *क्या ठीक से सोए?*
    (तुम·M or आप·M), the right output is "masculine, register ambiguous between तुम and आप". Guessing
    one register is wrong.
- **Labels beyond the six cells:**
  - `unmarked`: the system took an escape route (dative, genitive or imperative).
  - `mixed`: the register is inconsistent, for example *आप … हो*. This happens in real data. Tatoeba's
    human translations include *आप बाद में समझोगे*.
- **Only the addressee's agreement counts.** In `distractor` items another word agrees with someone
  else (जो कल आई थी, अपनी बहन, इतनी देर). The tokens in `agreement_tokens` are the addressee's;
  a feminine ending elsewhere in a masculine cell is not an error.
- **Dropped auxiliary.** After नहीं, Hindi (and MT) often drops the present auxiliary
  (… नहीं रुक सकती). Register is then marked only by number, as in PAST. `notes` flags the items
  where this is likely.
- **Normalise spelling first.** Run `normalize_hindi()` from `scripts/normalize.py` on **both** the MT
  output and the references before matching or scoring.
  - It makes these identical: dot and chandrabindu (आएंगे / आएँगे), गये / गए, nukta (ज़्यादा / ज्यादा),
    and हिन्दी / हिंदी.
  - It never merges forms that differ in grammar (गई / गईं, है / हैं).
  - `tests/test_normalize.py` tests it, and `build_dataset.py` checks that it never merges two cells
    or two pro-drop forms of any item.
- **Splitting for training.** Keep all six cells and pro-drop variants of an item on the same side of
  a train/test split; otherwise the test set contains near-copies of training sentences.
- **External test set.** CoCoA-MT's en-hi test split has 272 segments with masculine and feminine
  references for तुम / आप. In about 4 in 10 of them the feminine version changes the *speaker's*
  gender, so keep only those where it changes the addressee's (see the next section).

### The extractor

`scripts/extractor.py` reads register and gender without a reference, from the morphology of the
sentence itself, so it works on any wording an MT system produces:

```bash
python3 scripts/extractor.py --explain "क्या उस लड़की से मिलीं जो कल आई थी?"
# register=tum|aap  gender=F   (pro-drop: the nasal plural fits आप and the prescriptive तुम)
python3 scripts/extractor.py --input ../outputs/translations/nllb_600m.csv   # adds pred_register, pred_gender
```

It is rule-based. Its cues are pronoun forms (तू / तुम / आप, तुम्हें, आपका …), participle and
adjective suffixes (गया / गए / गई / गईं), present auxiliaries (है / हो / हैं), future endings
(-ेगा / -ोगे / -ेंगे …) and the past copula (था / थे / थी / थीं). It reads them only in the clause whose
subject is the addressee, so a relative clause's verb (जो कल आई थी) does not count. Future endings
and the copula carry the gender in every FUT and PAST item of this benchmark.

It outputs the labels described above: one register, a set of registers for a syncretic pro-drop
form (`tum|aap`), `unmarked` or `mixed`, and a gender of `M`, `F`, `unmarked` or `mixed`. The
module docstring explains the rules.

**Measuring its accuracy.** The rules were written while looking at this benchmark, which it reads
without error (1,332 references, 88 prescriptive alternatives, 1,332 pro-drop variants). That is a
development check, not the accuracy to report. The reported number comes from MT outputs labelled
by hand:

```bash
python3 scripts/evaluate_extractor.py --benchmark                       # development check
python3 scripts/evaluate_extractor.py --make-sheet                      # -> validation/extractor_gold.csv
python3 scripts/evaluate_extractor.py --add-sample gpt-oss-20b qwen3.8-27b sarvam-105b   # later systems
python3 scripts/label_gold.py                                           # hand labels, see below
python3 scripts/evaluate_extractor.py --labelled validation/extractor_gold.csv
```

`--make-sheet` reads every CSV in `../outputs/translations/` unless you give it files or folders. Each
needs the columns `id`, `system` and `hindi_output` (or `hindi`). The sheet leaves out the extractor's
predictions and the requested cell, so they cannot bias the annotator. `--add-sample` appends a random
sample of systems that were translated after the sheet was made, without renumbering earlier rows.
`--labelled` reports register, gender and joint accuracy with 95% confidence intervals, by system, by LLM
condition and by profile, with confusion matrices and every error.

#### Labelling guide

The evaluation sample is 375 rows of `validation/extractor_gold.csv`: rows 1–300, a random sample of the
Google Translate, IndicTrans2, NLLB and GPT-OSS-120B outputs (the sheet is shuffled), and rows 2119–2193,
25 random outputs from each of GPT-OSS-20B, Qwen3.8-27B and Sarvam-105B. `python3 scripts/label_gold.py`
shows these rows one at a time, the English source and the Hindi output only, and asks for two keys:

| register key | label | when | example |
|---|---|---|---|
| `1` / `2` / `3` | `tu` / `tum` / `aap` | the Hindi shows one register | तुम कहाँ गए थे? → `tum` |
| `4` | `tu\|tum` | no pronoun, and the verb fits both | कहाँ गई थी? |
| `5` | `tum\|aap` | no pronoun, and the verb fits both | कहाँ गए थे? |
| `6` | `unmarked` | nothing in the sentence shows the register | धन्यवाद। |
| `7` | `mixed` | the marks contradict each other | आप जा सकते हो। |

| gender key | label | when | example |
|---|---|---|---|
| `m` / `f` | `M` / `F` | the addressee's agreement is masculine / feminine | गया / गई |
| `u` | `unmarked` | the sentence avoids addressee agreement | तुम्हें भूख लगी है, तुमने खाना खाया |
| `x` | `mixed` | the addressee's agreement has both genders | |

Two rules decide the hard cases:

- **Label what the Hindi itself marks, not what the English intended.** An ungrammatical output gets the
  labels its forms show.
- **Only the addressee's agreement counts.** In क्या तुम उस लड़की से मिले जो कल आई थी? the answer is
  `tum`, `M`: आई थी belongs to the girl.

`c` adds a comment to a row (for doubts), `b` goes back one row and `q` saves and quits. The sheet is saved
after every row, and the next run continues where the last one stopped. Don't change the extractor's rules
after seeing the labels; that would inflate the accuracy.

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

- **Analysis cells.** The 138 core items give 44–48 per profile, so a proportion measured on one
  profile is accurate to about ±14 points (95% confidence). Constructions have 2–36 items, so analyse
  by profile, not by construction.
- **Distractor items name a third person** (a girl, your sister). That could nudge an MT system's
  choice of addressee gender. The `distractor` tag lets analyses exclude them.
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
