"""Read the addressee's register and gender off a Hindi sentence, such as an MT output for a benchmark item.

The extractor is rule-based and needs no reference: it reads the morphology of the sentence itself,
so it works on any wording an MT system produces. It is meant for sentences like the benchmark's,
with a singular "you" as subject, as described in dataset/README.md.

Output, per sentence:

  register  tu | tum | aap          one register
            tu|tum, tum|aap, ...    the sentence is compatible with several (pro-drop syncretism, e.g.
                                    क्या ठीक से सोए? -> tum|aap)
            unmarked                nothing marks register (no pronoun, no agreement that shows it)
            mixed                   the marks contradict each other (आप ... हो, आप ... तुम्हारी)
  gender    M | F                   the addressee's agreement shows the gender
            unmarked                nothing agrees with the addressee (escape route: तुम्हें भूख लगी है,
                                    तुम्हारा काम क्या है, तुमने खाना खाया, imperative बैठिए)
            mixed                   the addressee's agreement has both genders

How it reads a sentence:

  1. Normalise spelling (normalize.py) and split it into clauses at punctuation and at conjunctions
     and relative pronouns (और, तो, जब, अगर, कि, जो ...).
  2. Decide which clauses have the addressee as subject. A clause with the nominative pronoun
     (तू / तुम / आप, not followed by ने / को / का ...) always counts. A clause without it counts unless
       * it is a relative clause (जो कल आई थी),
       * its subject is someone else: मैं, हम, वह, कोई ..., a dative or ergative "you" (तुम्हें, तुमने), or
         a possessed noun (तुम्हारा काम क्या है),
       * it is a कि clause (… कि क्या हुआ), or its verb cannot agree with the "you" it contains
         (आपसे मिलकर अच्छा लगा).
     Two kinds of clause are used only when no better clause is available: one where a bare noun
     stands right before the verb and so is probably its subject (समारोह हो चुका था), and one that
     follows a comma with no conjunction after a main clause (… सो पाओगे, ठंड है).
     A verb form that only a second person can have (हो, -ओगे, -ओगी, an imperative) overrides these.
  3. In each addressee clause, find the finite element (है / हो / हैं, था / थे / थी / थीं, a future
     form, or a final participle) and the verb chain that leads up to it (रहा, चुका, सकता, गया,
     हुआ, वाला ...) plus the main verb or predicate adjective. Possessives and attributive adjectives
     (अपनी बहन, इतनी लंबी यात्रा) are not part of the chain, so they never count.
  4. Each agreeing form constrains the register and the gender; for example थे = masculine plural =
     तुम or आप, थी = feminine, final, not nasal = तू or तुम, हैं = आप. Pronoun forms anywhere in the
     sentence (तुम्हारी) constrain the register too. The register is what is compatible with every
     constraint; the gender is the union of the genders found.

Usage:
    python scripts/extractor.py "क्या आप कल आएँगी?"                    # one sentence
    python scripts/extractor.py --explain "तुम कहाँ गई थी?"            # show the evidence
    python scripts/extractor.py --input mt.csv --column hindi --output mt_labelled.csv

Out of scope: a plural "you" (आप लोग, तुम सब) is read like a singular one, and an invariant predicate
adjective that ends in -ी or -ा is read as agreeing unless it is listed in INVARIANT (दुखी, शाकाहारी).

    from extractor import extract
    extract("क्या आप कल आएँगी?").register, extract("क्या आप कल आएँगी?").gender   # 'aap', 'F'

Its accuracy is measured with scripts/evaluate_extractor.py.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from normalize import normalize_hindi  # noqa: E402

REGISTERS = ("tu", "tum", "aap")
ALL_REGISTERS = frozenset(REGISTERS)


def _words(text: str) -> frozenset[str]:
    """A set of words in normalised spelling, so the lexicon matches normalised input."""
    return frozenset(normalize_hindi(w) for w in text.split())


# --------------------------------------------------------------------------- lexicon
NOMINATIVE = {"tu": normalize_hindi("तू"), "tum": normalize_hindi("तुम"), "aap": normalize_hindi("आप")}
NOMINATIVE_TO_REGISTER = {v: k for k, v in NOMINATIVE.items()}
APNE = normalize_hindi("अपने")
# Fused forms of the pronoun, by register and by what they do to the clause.
ERGATIVE_DATIVE = {
    "tu": _words("तूने तुझे तुझको"),
    "tum": _words("तुमने तुम्हें तुम्हे तुमको"),
    "aap": _words("आपने आपको"),
}
GENITIVE = {
    "tu": _words("तेरा तेरी तेरे"),
    "tum": _words("तुम्हारा तुम्हारी तुम्हारे"),
    "aap": _words("आपका आपकी आपके"),
}
OBLIQUE_OTHER = {
    "tu": _words("तुझसे तुझपर तुझमें"),
    "tum": _words("तुमसे तुमपर तुममें"),
    "aap": _words("आपसे आपपर आपमें"),
}
PRONOUN_FORM_TO_REGISTER = {
    w: r for table in (ERGATIVE_DATIVE, GENITIVE, OBLIQUE_OTHER) for r, ws in table.items() for w in ws
}
# Postpositions written apart from the pronoun (आप ने, आप को, आप का ...).
SUBJECT_BLOCKING_POSTPOSITIONS = _words("ने को")
GENITIVE_POSTPOSITIONS = _words("का की के")
POSTPOSITIONS = _words("ने को का की के से में पर तक लिए लिये साथ बिना जैसा जैसे जैसी तरह बाद पहले पास")

OTHER_SUBJECTS = _words("मैं हम कोई मैंने हमने उसने उन्होंने")
# वह / वो / वे / यह are a subject unless a noun follows, which makes them a demonstrative (वह काम).
THIRD_PERSON = _words("वह वो वे यह")
# Predicate words that end like an agreeing form but never inflect (ज़िंदा, दुखी, शाकाहारी, बिज़ी ...).
INVARIANT = _words("ज़िंदा शादीशुदा ताज़ा बढ़िया घटिया उम्दा जुदा ख़फ़ा फ़िदा गुस्सा दुखी सुखी राज़ी आलसी ज़िद्दी "
                   "लालची ख़ाली भारी शाकाहारी मांसाहारी विदेशी देसी भारतीय जापानी पंजाबी बंगाली गुजराती "
                   "मराठी बिज़ी फ़्री रेडी सॉरी हैप्पी लकी क्रेज़ी सीरियस")
RELATIVE_STARTERS = _words("जो जिसे जिसने जिसका जिसकी जिसके जिससे जिसको जिसमें जिन्हें जिन्होंने जिनका जिनकी "
                           "जिनके जहाँ")
SUBORDINATORS = _words("जब जबकि अगर यदि क्योंकि ताकि")
CLAUSE_STARTERS = _words("और तो लेकिन मगर परंतु किंतु कि तब वरना") | SUBORDINATORS | RELATIVE_STARTERS
KI = normalize_hindi("कि")

NEGATION = _words("नहीं न मत")
TRAILING_PARTICLES = NEGATION | _words("क्या ना जी")
# Words that can stand right before a verb chain without being its subject.
FUNCTION_WORDS = (
    POSTPOSITIONS | NEGATION | TRAILING_PARTICLES | CLAUSE_STARTERS
    | _words("क्यों कब कहाँ कितना कितने कितनी कौन किसे किससे किसको किसके "
             "फिर भी ही तो अभी अब आज कल परसों यहाँ वहाँ इधर उधर पहले बाद बहुत ज़्यादा कम थोड़ा थोड़ी "
             "जल्दी धीरे ठीक वापस साथ सच सचमुच ज़रूर शायद हमेशा कभी कभी-कभी अक्सर रोज़ "
             "सुबह शाम रात दोपहर देर बिल्कुल पूरा सब सबसे ज़रा "
             "इसे उसे इससे उससे इसको उसको इसमें उसमें इन्हें उन्हें मुझे मुझसे हमें हमसे मेरा मेरी मेरे "
             "हमारा हमारी हमारे उसका उसकी उसके इसका इसकी इसके अपना अपनी अपने")
)
# Verb-chain elements after the main verb (auxiliaries, aspectual and modal verbs), every inflection.
CHAIN_FORMS = _words(
    "रहा रहे रही रहीं चुका चुके चुकी चुकीं सकता सकते सकती सकतीं सका सके सकी सकीं "
    "पाया पाए पाई पाईं पाता पाते पाती पातीं वाला वाले वाली वालीं "
    "गया गए गई गईं गये गयी गयीं जाता जाते जाती जातीं हुआ हुए हुई हुईं "
    "लिया लिए ली लीं दिया दिए दी दीं बैठा बैठे बैठी बैठीं"
)
HU_FORMS = _words("हुआ हुए हुई हुईं")
# Future forms of the chain verbs (पहुँच जाओगे, थक गए होगे, आ सकोगे) are chain elements too.
FUTURE_CHAIN_STEMS = tuple(normalize_hindi(s) for s in ("जा", "हो", "सक", "पा", "बैठ"))
PRESENT_AUX = {normalize_hindi("है"): "tu", normalize_hindi("हो"): "tum", normalize_hindi("हैं"): "aap"}
PAST_COPULA = {
    normalize_hindi("था"): ("M", "sg"), normalize_hindi("थे"): ("M", "pl"),
    normalize_hindi("थी"): ("F", "sg"), normalize_hindi("थीं"): ("F", "pl"),
    normalize_hindi("थें"): ("M", "pl"),   # a common misspelling of थे
}
FIRST_PERSON_VERB = re.compile(normalize_hindi("(हूँ|ूँगा|ूँगी|ऊँगा|ऊँगी|ूँगे|ऊँगे)") + "$")
INTENSIFIERS = {normalize_hindi(w): g for w, g in [
    ("इतना", ("M", "sg")), ("इतने", ("M", "pl")), ("इतनी", ("F", None)),
    ("कितना", ("M", "sg")), ("कितने", ("M", "pl")), ("कितनी", ("F", None)),
]}

# Future endings (normalised: chandrabindu -> anusvara) -> (registers, gender).
# होगी is the one syncretic form (तू होगी = तुम होगी); रहोगी is regular.
HOGI = normalize_hindi("होगी")
FUTURE_ENDINGS: list[tuple[str, frozenset[str], str]] = [
    ("ेंगे", frozenset({"aap"}), "M"), ("एंगे", frozenset({"aap"}), "M"), ("ोंगे", frozenset({"aap"}), "M"),
    ("ेंगी", frozenset({"aap"}), "F"), ("एंगी", frozenset({"aap"}), "F"), ("ोंगी", frozenset({"aap"}), "F"),
    ("ोगे", frozenset({"tum"}), "M"), ("ओगे", frozenset({"tum"}), "M"),
    ("ोगी", frozenset({"tum"}), "F"), ("ओगी", frozenset({"tum"}), "F"),
    ("ेगा", frozenset({"tu"}), "M"), ("एगा", frozenset({"tu"}), "M"), ("ोगा", frozenset({"tu"}), "M"),
    ("ेगी", frozenset({"tu"}), "F"), ("एगी", frozenset({"tu"}), "F"),
]
POLITE_IMPERATIVE = re.compile("(िए|इए|िएगा|इएगा|िये|इये)$")
FAMILIAR_IMPERATIVE = re.compile("(ओ|ो)$")

PUNCTUATION = "?।!,;:.\"'()"
_TOKEN = re.compile(rf"[^\s{re.escape(PUNCTUATION)}]+|[{re.escape(PUNCTUATION)}]")


# --------------------------------------------------------------------------- morphology
def future_reading(tok: str) -> tuple[frozenset[str], str] | None:
    if POLITE_IMPERATIVE.search(tok):   # बैठिएगा is a polite imperative, not a तू future
        return None
    if tok == HOGI:
        return frozenset({"tu", "tum"}), "F"
    for ending, regs, gender in FUTURE_ENDINGS:
        if tok.endswith(ending) and len(tok) > len(ending):
            return regs, gender
    return None


def participle_reading(tok: str) -> tuple[str, str] | None:
    """(gender, number) of a participle or adjective ending, or None for an invariant word."""
    if len(tok) < 2 or tok in FUNCTION_WORDS or tok in INVARIANT:
        return None
    if tok.endswith(("ीं", "ईं")):
        return "F", "pl"
    if tok.endswith(("ी", "ई")):
        return "F", "sg"
    if tok.endswith(("ा", "आ")):
        return "M", "sg"
    if tok.endswith(("े", "ए")):
        return "M", "pl"
    return None


def number_to_registers(gender: str, number: str | None, final: bool) -> frozenset[str]:
    """Which registers a gender/number ending allows. Feminine number shows only on the last element
    of the verb group (आप गई थीं), so a non-final feminine form allows every register."""
    if gender == "M":
        return frozenset({"tu"}) if number == "sg" else frozenset({"tum", "aap"})
    if not final or number is None:
        return ALL_REGISTERS
    return frozenset({"tu", "tum"}) if number == "sg" else frozenset({"tum", "aap"})


def is_chain_form(tok: str) -> bool:
    if tok in CHAIN_FORMS:
        return True
    fut = future_reading(tok)
    return fut is not None and tok.startswith(FUTURE_CHAIN_STEMS) and len(tok) <= 6


def is_finite(tok: str) -> bool:
    return tok in PRESENT_AUX or tok in PAST_COPULA or future_reading(tok) is not None


# --------------------------------------------------------------------------- data structures
@dataclass
class Evidence:
    token: str
    registers: frozenset[str]
    gender: str | None
    why: str

    def __str__(self) -> str:
        regs = "|".join(r for r in REGISTERS if r in self.registers) if self.registers != ALL_REGISTERS else "any"
        return f"{self.token}: {self.why} -> register {regs}, gender {self.gender or '-'}"


@dataclass
class ClauseReading:
    tokens: list[str]
    evidence: list[Evidence] = field(default_factory=list)
    has_anchor: bool = False
    second_person_only: bool = False   # a form only a second-person subject can have (हो, -ओगे, imperative)
    likely_other_subject: bool = False  # a bare noun right before the verb chain
    note: str = ""

    @property
    def registers(self) -> frozenset[str]:
        out = ALL_REGISTERS
        for e in self.evidence:
            out = out & e.registers
        return out

    @property
    def genders(self) -> set[str]:
        return {e.gender for e in self.evidence if e.gender}


@dataclass
class Extraction:
    text: str
    register: str            # tu, tum, aap, an ambiguous set like tum|aap, unmarked, mixed
    gender: str              # M, F, unmarked, mixed
    register_set: frozenset[str]
    evidence: list[str] = field(default_factory=list)

    @property
    def cell(self) -> str | None:
        """The benchmark cell (e.g. tum_f), if both attributes are fully determined."""
        if self.register in REGISTERS and self.gender in ("M", "F"):
            return f"{self.register}_{self.gender.lower()}"
        return None


def register_label(regs: frozenset[str]) -> str:
    if not regs:
        return "mixed"
    if regs == ALL_REGISTERS:
        return "unmarked"
    return "|".join(r for r in REGISTERS if r in regs)


# --------------------------------------------------------------------------- clauses
def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(normalize_hindi(text))


def split_clauses(toks: list[str]) -> list[list[str]]:
    clauses: list[list[str]] = [[]]
    for t in toks:
        if t in PUNCTUATION:
            clauses.append([])
            continue
        if t in CLAUSE_STARTERS and clauses[-1] and not (t == normalize_hindi("तो") and _is_emphatic_to(clauses[-1])):
            clauses.append([])
        clauses[-1].append(t)
    return [c for c in clauses if c]


def _is_emphatic_to(clause_so_far: list[str]) -> bool:
    """तो right after the subject pronoun (तुम तो आओगे) is a particle, not "then"."""
    return clause_so_far[-1] in NOMINATIVE_TO_REGISTER


def nominative_positions(clause: list[str]) -> list[int]:
    """Positions of a nominative second-person pronoun (not followed by a postposition, not अपने आप)."""
    out = []
    for i, t in enumerate(clause):
        if t not in NOMINATIVE_TO_REGISTER:
            continue
        nxt = clause[i + 1] if i + 1 < len(clause) else ""
        prev = clause[i - 1] if i > 0 else ""
        if nxt in POSTPOSITIONS or (t == NOMINATIVE["aap"] and prev == APNE):
            continue
        out.append(i)
    return out


def subject_is_someone_else(clause: list[str]) -> str:
    """Return a reason if the clause's subject is evidently not the addressee, else ''."""
    for i, t in enumerate(clause):
        nxt = clause[i + 1] if i + 1 < len(clause) else ""
        after = clause[i + 2] if i + 2 < len(clause) else ""
        if t in OTHER_SUBJECTS or (t in THIRD_PERSON and (not nxt or nxt in FUNCTION_WORDS)):
            return f"subject {t}"
        if FIRST_PERSON_VERB.search(t):
            return f"first-person verb {t}"
        if any(t in ws for ws in ERGATIVE_DATIVE.values()):
            return f"ergative/dative {t}"
        if t in NOMINATIVE_TO_REGISTER and nxt in SUBJECT_BLOCKING_POSTPOSITIONS:
            return f"ergative/dative {t} {nxt}"
        # तुम्हारा काम क्या है: a possessed noun that is not followed by a postposition is the subject.
        is_gen = any(t in ws for ws in GENITIVE.values())
        if t in NOMINATIVE_TO_REGISTER and nxt in GENITIVE_POSTPOSITIONS:
            is_gen, nxt, after = True, after, clause[i + 3] if i + 3 < len(clause) else ""
        if is_gen and nxt and nxt not in FUNCTION_WORDS and after not in POSTPOSITIONS:
            return f"possessed subject {t} {nxt}"
    return ""


# --------------------------------------------------------------------------- reading one clause
def read_clause(clause: list[str]) -> ClauseReading:
    cr = ClauseReading(tokens=clause)
    end = len(clause)
    while end > 0 and clause[end - 1] in TRAILING_PARTICLES:
        end -= 1
    if end == 0:
        return cr

    # 1. the finite element: the rightmost finite form, else a final participle, else an imperative
    anchor = next((i for i in range(end - 1, -1, -1) if is_finite(clause[i])), None)
    if anchor is None:
        last = clause[end - 1]
        if POLITE_IMPERATIVE.search(last) and last not in FUNCTION_WORDS:
            cr.evidence.append(Evidence(last, frozenset({"aap"}), None, "polite imperative"))
            cr.has_anchor = cr.second_person_only = True
            return cr
        if participle_reading(last) and last not in NOMINATIVE_TO_REGISTER and last not in PRONOUN_FORM_TO_REGISTER:
            anchor = end - 1
        elif (FAMILIAR_IMPERATIVE.search(last) and len(last) >= 2 and last not in FUNCTION_WORDS
              and last not in POSTPOSITIONS):
            cr.evidence.append(Evidence(last, frozenset({"tum"}), None, "familiar imperative"))
            cr.has_anchor = cr.second_person_only = True
            return cr
        else:
            return cr
    cr.has_anchor = True

    # 2. walk left from the finite element through the verb chain to the main verb / predicate adjective
    chain = [anchor]
    i = anchor - 1
    if is_chain_form(clause[anchor]) or clause[anchor] in PRESENT_AUX or clause[anchor] in PAST_COPULA:
        while i >= 0 and (is_chain_form(clause[i]) or clause[i] in PAST_COPULA or clause[i] in NEGATION):
            chain.insert(0, i)
            i -= 1
        if i >= 0 and clause[i] not in FUNCTION_WORDS and clause[i] not in NOMINATIVE_TO_REGISTER \
                and clause[i] not in PRONOUN_FORM_TO_REGISTER:
            # The main verb agrees only before हुआ, an auxiliary or a copula (थके हुए, करते हो, भूखे थे).
            # Before रहा, सकता, चुका, पाना, जाना, गया or वाला it is a bare stem or adds nothing
            # (जा रहे, थक गए, बताने वाले).
            nxt = next(clause[p] for p in chain if clause[p] not in NEGATION)
            if nxt in HU_FORMS or nxt in PRESENT_AUX or nxt in PAST_COPULA:
                chain.insert(0, i)
            i -= 1
    before = i  # the word right before the verb chain

    # 3. read every element of the chain
    for pos in chain:
        tok = clause[pos]
        final = pos == anchor
        if tok in NEGATION:
            continue
        if final and tok in PRESENT_AUX:
            reg = PRESENT_AUX[tok]
            cr.evidence.append(Evidence(tok, frozenset({reg}), None, "present auxiliary"))
            cr.second_person_only |= reg == "tum"
            continue
        if tok in PAST_COPULA:
            g, n = PAST_COPULA[tok]
            cr.evidence.append(Evidence(tok, number_to_registers(g, n, final), g, f"past copula {g} {n}"))
            continue
        fut = future_reading(tok)
        if fut:
            regs, g = fut
            cr.evidence.append(Evidence(tok, regs, g, "future"))
            cr.second_person_only |= regs == frozenset({"tum"})
            continue
        pr = participle_reading(tok)
        if pr:
            g, n = pr
            cr.evidence.append(Evidence(tok, number_to_registers(g, n, final), g, f"participle/adjective {g} {n}"))

    # A main word whose gender contradicts every chain element is probably a noun (उसकी बेटी थे?).
    if len(chain) > 1 and clause[chain[0]] not in CHAIN_FORMS and cr.evidence:
        head = [e for e in cr.evidence if e.token == clause[chain[0]]]
        rest = [e for e in cr.evidence if e.token != clause[chain[0]] and e.gender]
        if head and head[0].gender and rest and all(e.gender != head[0].gender for e in rest):
            cr.evidence.remove(head[0])

    # 4. invariant predicate (इतने चुप क्यों हो): fall back on an agreeing intensifier
    if not cr.genders:
        for j in range(before, max(before - 3, -1), -1):
            if clause[j] in INTENSIFIERS:
                g, n = INTENSIFIERS[clause[j]]
                cr.evidence.append(Evidence(clause[j], number_to_registers(g, n, False), g, "agreeing intensifier"))
                break

    cr.likely_other_subject = (
        before >= 0 and clause[before] not in FUNCTION_WORDS and clause[before] not in NOMINATIVE_TO_REGISTER
        and clause[before] not in PRONOUN_FORM_TO_REGISTER and not clause[before].endswith(("ने", "ना"))
    )
    return cr


# --------------------------------------------------------------------------- the sentence
def extract(text: str) -> Extraction:
    toks = tokenize(text)
    clauses = split_clauses(toks)
    evidence: list[str] = []

    primary: list[ClauseReading] = []
    strong: list[ClauseReading] = []
    weak: list[ClauseReading] = []
    for k, c in enumerate(clauses):
        cr = read_clause(c)
        label = " ".join(c)
        if nominative_positions(c):
            primary.append(cr)
            evidence.append(f"clause [{label}]: addressee is the subject (overt pronoun)")
            continue
        if not cr.has_anchor:
            continue
        if c[0] in RELATIVE_STARTERS and not cr.second_person_only:
            evidence.append(f"clause [{label}]: skipped, relative clause")
            continue
        if c[0] == KI and not cr.second_person_only:
            evidence.append(f"clause [{label}]: skipped, complement clause")
            continue
        other = subject_is_someone_else(c)
        if other and not cr.second_person_only:
            evidence.append(f"clause [{label}]: skipped, {other}")
            continue
        # आपसे मिलकर अच्छा लगा: a verb that cannot agree with the "you" in its own clause is someone else's.
        own = {PRONOUN_FORM_TO_REGISTER[t] for t in c if t in PRONOUN_FORM_TO_REGISTER}
        if len(own) == 1 and not (cr.registers & own):
            evidence.append(f"clause [{label}]: skipped, its verb cannot agree with the pronoun")
            continue
        if cr.likely_other_subject and not cr.second_person_only:
            weak.append(cr)
            evidence.append(f"clause [{label}]: used only as a fallback, a bare noun precedes the verb")
            continue
        # A clause after a comma with no conjunction (… सो पाओगे, ठंड है) is linked to the addressee's
        # clause only if that one was subordinate (जब तुम पहुँचे, तब तक …  /  अगर …, …).
        loose = k > 0 and c[0] not in CLAUSE_STARTERS and clauses[k - 1][0] not in SUBORDINATORS
        if loose and not cr.second_person_only:
            weak.append(cr)
            evidence.append(f"clause [{label}]: used only as a fallback, no conjunction links it")
            continue
        strong.append(cr)
        evidence.append(f"clause [{label}]: addressee is the (dropped) subject")

    used = [cr for cr in primary + strong if cr.has_anchor]
    if not used:
        used = weak
    elif weak and not any(cr.evidence for cr in used):
        used = used + weak

    verb_regs = ALL_REGISTERS
    genders: set[str] = set()
    for cr in used:
        for e in cr.evidence:
            evidence.append(f"  {e}")
        verb_regs = verb_regs & cr.registers
        genders |= cr.genders

    # Every form of "you" marks the register: तुम, आप ने, तुम्हारी ... (but not अपने आप, "by oneself").
    pron_regs = set()
    for i, t in enumerate(toks):
        if t in PRONOUN_FORM_TO_REGISTER:
            pron_regs.add(PRONOUN_FORM_TO_REGISTER[t])
        elif t in NOMINATIVE_TO_REGISTER and not (t == NOMINATIVE["aap"] and i > 0 and toks[i - 1] == APNE):
            pron_regs.add(NOMINATIVE_TO_REGISTER[t])
    if pron_regs:
        evidence.append(f"pronoun forms -> register {'|'.join(r for r in REGISTERS if r in pron_regs)}")

    if len(pron_regs) > 1:
        regs: frozenset[str] = frozenset()
    else:
        regs = (frozenset(pron_regs) or ALL_REGISTERS) & verb_regs

    gender = "unmarked" if not genders else ("mixed" if len(genders) > 1 else genders.pop())
    return Extraction(text=text, register=register_label(regs), gender=gender, register_set=regs,
                      evidence=evidence)


# --------------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sentence", nargs="*", help="Hindi sentence(s) to analyse")
    ap.add_argument("--explain", action="store_true", help="print the evidence behind each label")
    ap.add_argument("--input", type=Path, help="CSV file with one sentence per row")
    ap.add_argument("--column", default="hindi", help="column of --input that holds the Hindi text")
    ap.add_argument("--output", type=Path, help="write --input with added columns pred_register, pred_gender")
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    if args.input:
        with args.input.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
        if rows and args.column not in rows[0]:
            ap.error(f"column '{args.column}' not in {args.input}")
        for r in rows:
            ex = extract(r[args.column])
            r["pred_register"], r["pred_gender"] = ex.register, ex.gender
        out = args.output or args.input.with_name(args.input.stem + "_extracted.csv")
        with out.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else [args.column, "pred_register", "pred_gender"])
            w.writeheader()
            w.writerows(rows)
        print(f"labelled {len(rows)} rows -> {out}")
        return 0

    if not args.sentence:
        ap.error("give a sentence or --input")
    for s in args.sentence:
        ex = extract(s)
        print(f"{s}\tregister={ex.register}\tgender={ex.gender}")
        if args.explain:
            for line in ex.evidence:
                print(f"    {line}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
