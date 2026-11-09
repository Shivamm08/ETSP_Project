"""Unit tests for the item checks in scripts/build_dataset.py.  Run from the dataset/ folder:
python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import build_dataset as bd  # noqa: E402

CELLS = ("tu_m", "tu_f", "tum_m", "tum_f", "aap_m", "aap_f")

# "Did you meet the girl who came yesterday?": आई थी belongs to the girl, not the addressee.
GIRL = dict(zip(CELLS, (
    "क्या तू उस लड़की से मिला जो कल आई थी?",
    "क्या तू उस लड़की से मिली जो कल आई थी?",
    "क्या तुम उस लड़की से मिले जो कल आई थी?",
    "क्या तुम उस लड़की से मिली जो कल आई थी?",
    "क्या आप उस लड़की से मिले जो कल आई थी?",
    "क्या आप उस लड़की से मिलीं जो कल आई थी?",
)), tum_f_alt="क्या तुम उस लड़की से मिलीं जो कल आई थी?", tags="multi_clause|distractor")

# "Your mother called while you were sleeping.": तेरी / तुम्हारी / आपकी mark register, not agreement.
MOTHER = dict(zip(CELLS, (
    "जब तू सो रहा था, तब तेरी माँ का फ़ोन आया था।",
    "जब तू सो रही थी, तब तेरी माँ का फ़ोन आया था।",
    "जब तुम सो रहे थे, तब तुम्हारी माँ का फ़ोन आया था।",
    "जब तुम सो रही थी, तब तुम्हारी माँ का फ़ोन आया था।",
    "जब आप सो रहे थे, तब आपकी माँ का फ़ोन आया था।",
    "जब आप सो रही थीं, तब आपकी माँ का फ़ोन आया था।",
)), tum_f_alt="जब तुम सो रही थीं, तब तुम्हारी माँ का फ़ोन आया था।", tags="multi_clause|distractor",
    sentence_type="statement")

# "You came on Monday and left again the next day.": two separate stretches of agreement.
MONDAY = dict(zip(CELLS, (
    "तू सोमवार को आया और अगले ही दिन फिर चला गया।",
    "तू सोमवार को आई और अगले ही दिन फिर चली गई।",
    "तुम सोमवार को आए और अगले ही दिन फिर चले गए।",
    "तुम सोमवार को आई और अगले ही दिन फिर चली गई।",
    "आप सोमवार को आए और अगले ही दिन फिर चले गए।",
    "आप सोमवार को आईं और अगले ही दिन फिर चली गईं।",
)), tum_f_alt="तुम सोमवार को आईं और अगले ही दिन फिर चली गईं।", tags="multi_clause|multi_site",
    sentence_type="statement")


def item(base: dict, **changes: str) -> dict:
    it = {"id": "T-01", "split": "core", "profile": "PAST", "construction": "perfective",
          "sentence_type": "question", "polarity": "affirmative", "frame_id": "", "cue": "none",
          "english": "x", "source": "authored", "source_id": "", **base}
    it.update(changes)
    return it


def errors(it: dict) -> list[str]:
    rep = bd.Report()
    for check in (bd.check_schema, bd.check_pronouns, bd.check_minimal_pairs, bd.check_paradigm,
                  bd.check_agreement, bd.check_tags, bd.check_orthography, bd.check_normalisation):
        check(it, rep)
    return rep.errors


class ValidItemsPass(unittest.TestCase):
    def test_distractor_after_the_addressee(self) -> None:
        self.assertEqual(errors(item(GIRL)), [])

    def test_register_marked_by_a_possessive(self) -> None:
        self.assertEqual(errors(item(MOTHER, construction="past_progressive")), [])

    def test_two_stretches_of_agreement(self) -> None:
        self.assertEqual(errors(item(MONDAY)), [])

    def test_possessives_are_not_agreement_tokens(self) -> None:
        agr = bd.agreement_tokens(item(MOTHER))
        self.assertEqual(agr["tum_m"], ["रहे", "थे"])
        self.assertEqual(agr["aap_f"], ["रही", "थीं"])


class BrokenItemsFail(unittest.TestCase):
    def assertCaught(self, it: dict, fragment: str) -> None:
        errs = errors(it)
        self.assertTrue(any(fragment in e for e in errs), f"expected an error with {fragment!r}, got {errs}")

    def test_masculine_word_in_a_feminine_cell(self) -> None:
        self.assertCaught(item(MONDAY, tu_f="तू सोमवार को आई और अगले ही दिन फिर चला गई।"), "'चला'")

    def test_nasal_plural_outside_aap(self) -> None:
        self.assertCaught(item(GIRL, tu_f="क्या तू उस लड़की से मिलीं जो कल आई थी?"), "'मिलीं'")

    def test_possessive_of_the_wrong_register(self) -> None:
        bad = "जब आप सो रहे थे, तब तुम्हारी माँ का फ़ोन आया था।"
        self.assertCaught(item(MOTHER, construction="past_progressive", aap_m=bad), "another register")

    def test_distractor_changed_in_one_cell(self) -> None:
        self.assertCaught(item(GIRL, tum_m="क्या तुम उस लड़की से मिले जो कल आया था?"), "'आया'")

    def test_missing_multi_site_tag(self) -> None:
        self.assertCaught(item(MONDAY, tags="multi_clause"), "multi_site")

    def test_spurious_multi_site_tag(self) -> None:
        self.assertCaught(item(GIRL, tags="multi_clause|distractor|multi_site"), "multi_site")

    def test_nasalisation_doubled_in_a_verb_group(self) -> None:
        self.assertCaught(item(MONDAY, aap_f="आप सोमवार को आईं और अगले ही दिन फिर चलीं गईं।"), "doubled")

    def test_tum_f_alt_must_take_aap_f_endings(self) -> None:
        self.assertCaught(item(MOTHER, construction="past_progressive",
                               tum_f_alt="जब तुम सो रही थीं, तब आपकी माँ का फ़ोन आया था।"), "tum_f_alt")

    def test_unknown_tag(self) -> None:
        self.assertCaught(item(GIRL, tags="multi_clause|distractor|long"), "unknown tag")

    def test_tatoeba_item_needs_an_id(self) -> None:
        self.assertCaught(item(GIRL, source="tatoeba"), "source_id")


if __name__ == "__main__":
    unittest.main()
