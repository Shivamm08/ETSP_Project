"""Build the candidate pool of English source sentences from a Tatoeba export.

Keeps short sentences whose Hindi translation must mark BOTH the addressee's register and gender:
a single, singular, unnamed "you" is the subject, there is no cue to register or gender, no
first-person subject (which would add speaker gender), and the construction keeps nominative
agreement in Hindi. Each candidate is bucketed by the Hindi agreement profile it will need:

  FUT        future: one synthetic verb carries person + number + gender (जाओगे / जाओगी)
  PAST*      perfective / past copula: participle carries number + gender only (गए / गई)
  PRES_*     present periphrastic: participle carries gender, auxiliary है/हो/हैं carries person

The filter is deliberately conservative and high-recall on the structure, not on meaning; the final
items were picked by hand from this pool to fill the per-profile quotas (see README).

Inputs (Tatoeba per-language exports, CC-BY 2.0 FR, downloaded 2026-10-19):
  https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2
  https://downloads.tatoeba.org/exports/per_language/eng/eng-hin_links.tsv.bz2
  https://downloads.tatoeba.org/exports/per_language/hin/hin_sentences.tsv.bz2

Usage:
  python scripts/select_sources.py --eng eng_sentences.tsv.bz2 --links eng-hin_links.tsv.bz2 \
      --hin hin_sentences.tsv.bz2 --out sources/tatoeba_candidate_pool.tsv
"""
from __future__ import annotations

import argparse
import bz2
import csv
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

FIRST_PERSON_SUBJ = re.compile(r"\b(I|I'm|I'll|I've|I'd|we|we're|we'll|we've|we'd|let's|let us)\b", re.I)
GENDERED = re.compile(
    r"\b(he|she|him|her|his|hers|himself|herself|he's|she's|man|men|woman|women|boy|boys|girl|girls|guy|guys|"
    r"lady|ladies|gentleman|gentlemen|sir|madam|ma'am|mom|mum|mother|dad|father|son|daughter|brother|sister|"
    r"husband|wife|boyfriend|girlfriend|king|queen|prince|princess|uncle|aunt|aunty|grandma|grandpa|grandmother|"
    r"grandfather|mr|mrs|ms|pregnant|beard|bride|groom|actress|waitress|nephew|niece|bro|dude|darling|honey|"
    r"sweetheart|baby|kid|kids|children|child|bachelor|spinster|widow|widower|lesbian|gay|feminine|masculine|"
    r"female|male)\b",
    re.I,
)
PLURAL_YOU = re.compile(
    r"\b(you all|y'all|you two|you both|both of you|all of you|each of you|you people|you guys|you kids|yourselves|"
    r"you three|any of you|none of you|either of you|neither of you|one of you|you lot|you folks|everyone|everybody)\b",
    re.I,
)
# Verbs/phrases whose usual Hindi equivalent has a dative or genitive subject (no agreement with "you").
NO_AGREEMENT = re.compile(
    r"\b(like|love|want|need|have to|has to|had to|must|should|ought|remember|know|knew|think|thought|feel|felt|"
    r"hope|wish|believe|mind|hate|prefer|enjoy|seem|seems|got|get to|have got|have a|has a|had a|own|"
    r"is it|isn't it|was it|it's|that's|there|this|these|those)\b",
    re.I,
)
# Present copula + these adjectives carries no gender in Hindi (तैयार, खुश, सही ...).
INVARIANT_ADJ = re.compile(
    r"\b(ready|happy|sure|right|wrong|late|sick|ill|busy|free|angry|upset|sad|smart|clever|lucky|brave|crazy|mad|"
    r"okay|ok|fine|alright|all right|serious|kidding|joking|welcome|correct|responsible|innocent|guilty|famous|rich|"
    r"poor|safe|honest|stubborn|beautiful|pretty|handsome|cute|nice|kind|rude|weird|strange|normal|special|perfect|"
    r"afraid|scared|nervous|worried|jealous|proud|ashamed|embarrassed|surprised|disappointed|interested|done|"
    r"finished|married|single|awake|home|here|there|out|in|back|on time|early|online)\b",
    re.I,
)
# Adjectives with an inflecting -ā form in Hindi (भूखा, थका, प्यासा, अकेला ...).
VARIABLE_ADJ = re.compile(
    r"\b(hungry|thirsty|tired|exhausted|wet|soaked|young|old|tall|short|fat|thin|slim|big|small|little|dirty|"
    r"naked|blind|deaf|bad|good|great|grown up|asleep|drunk|alone|lonely|scared|frightened|lost|stuck|hurt|"
    r"bored|confused|cold|healthy|weak|strong|slow|fast|mistaken)\b",
    re.I,
)
# Verbs that stay non-ergative in the Hindi perfective, so the participle still agrees with "you".
PFV_OK = re.compile(
    r"\b(go|went|gone|come|came|arrive|arrived|sleep|slept|wake|woke|get up|got up|sit|sat|stay|stayed|return|"
    r"returned|come back|came back|go back|went back|fall|fell|laugh|laughed|cry|cried|dance|danced|swim|swam|"
    r"run|ran|walk|walked|stop|stopped|forget|forgot|forgotten|lose|lost|meet|met|bring|brought|leave|left|"
    r"travel|travelled|traveled|climb|climbed|jump|jumped|hide|hid|grow up|grew up|born|pass|passed|fail|failed|"
    r"get home|got home|get there|got there|get back|got back|show up|showed up|move|moved|drive|drove|fly|flew|"
    r"get lost|got lost|get scared|got scared|get tired|got tired|get bored|got bored|get angry|got angry|"
    r"wait|waited|live|lived|work|worked|study|studied|play|played|sing|sang|fight|fought|win|won|"
    r"sleeping|going|coming|crying|laughing|working|studying|playing|singing|dancing|waiting|living|"
    r"hiding|running|walking|fighting|lying|joking|dreaming|thinking|listening|been|changed|become|grown)\b",
    re.I,
)
NAME_LIKE = re.compile(r"(?<!^)(?<![.?!] )\b(?!I\b)[A-Z][a-z]+")

PATTERNS = {
    "FUT": re.compile(r"\b(will you|you will|you'll|you won't|won't you|are you going to|you're going to|"
                      r"you are going to|aren't you going to)\b", re.I),
    "PAST": re.compile(r"\b(did you|you didn't|didn't you|were you|you were|weren't you|you weren't|had you|you had)\b",
                       re.I),
    "PAST_STMT": re.compile(r"\byou (came|went|lost|won|left|fell|forgot|slept|ran|sat|grew|woke|arrived|returned|"
                           r"stayed|laughed|cried|danced|walked|stopped|moved|travelled|traveled|jumped|climbed|"
                           r"failed|passed)\b", re.I),
    "PRES_PROG": re.compile(r"\b(are you|you're|you are|aren't you|you aren't)\s+(\w+ing)\b", re.I),
    "PRES_HAB": re.compile(r"\b(do you|you don't|don't you)\b", re.I),
    "PRES_MOD": re.compile(r"\b(can you|you can|can't you|you can't|could you)\b", re.I),
    "PRES_PERF": re.compile(r"\b(have you|you've|you have|haven't you|you haven't)\s+(\w+)\b", re.I),
    "PRES_HAB_STMT": re.compile(r"^you (sing|speak|talk|work|live|run|walk|swim|dance|cook|drive|play|study|read|"
                                r"write|eat|drink|sleep|smile|laugh|worry|lie)\b", re.I),
    "PRES_ADJ": re.compile(r"\b(are you|you're|you are|aren't you|you look|you seem)\b", re.I),
}
COPULA_PAST = re.compile(r"\b(were you|you were|weren't you|you weren't)\b", re.I)


def bucket(text: str) -> str | None:
    """Return the first matching agreement bucket, or None if the sentence is not usable."""
    for name, pat in PATTERNS.items():
        if pat.search(text):
            if name == "PAST" and not (PFV_OK.search(text) or COPULA_PAST.search(text)):
                return None
            if name == "PRES_ADJ" and (INVARIANT_ADJ.search(text) or not VARIABLE_ADJ.search(text)):
                return None
            if name == "PRES_PERF" and not PFV_OK.search(text):
                return None
            return name
    return None


def usable(text: str, min_words: int, max_words: int) -> bool:
    if not min_words <= len(text.split()) <= max_words:
        return False
    if not re.search(r"\byou\b|\byou'(re|ll|ve|d)\b", text, re.I):
        return False
    blockers = (FIRST_PERSON_SUBJ, GENDERED, PLURAL_YOU, NAME_LIKE, NO_AGREEMENT)
    return not any(p.search(text) for p in blockers)


def load_hindi(links: Path, hin: Path) -> dict[str, list[str]]:
    hindi: dict[str, str] = {}
    with bz2.open(hin, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) == 3:
                hindi[parts[0]] = parts[2]
    eng2hin: dict[str, list[str]] = defaultdict(list)
    with bz2.open(links, "rt", encoding="utf-8") as fh:
        for line in fh:
            eng_id, hin_id = line.rstrip("\n").split("\t")[:2]
            if hin_id in hindi:
                eng2hin[eng_id].append(hindi[hin_id])
    return eng2hin


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eng", type=Path, required=True)
    ap.add_argument("--links", type=Path, required=True)
    ap.add_argument("--hin", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-words", type=int, default=2)
    ap.add_argument("--max-words", type=int, default=10)
    args = ap.parse_args(argv)

    eng2hin = load_hindi(args.links, args.hin)
    kept, seen, counts = [], set(), Counter()
    with bz2.open(args.eng, "rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sid, _, text = parts
            key = text.lower().strip(" ?.!")
            if key in seen or not usable(text, args.min_words, args.max_words):
                continue
            b = bucket(text)
            if b is None:
                continue
            seen.add(key)
            counts[b] += 1
            kept.append((b, sid, text, len(text.split()), " | ".join(eng2hin.get(sid, []))))

    kept.sort(key=lambda r: (r[0], r[3], r[2]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t")
        w.writerow(["bucket", "tatoeba_id", "english", "n_words", "existing_hindi"])
        w.writerows(kept)
    print("candidates per bucket:", dict(sorted(counts.items())), "| total:", len(kept))
    print("with an existing Tatoeba Hindi translation:", sum(1 for r in kept if r[4]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
