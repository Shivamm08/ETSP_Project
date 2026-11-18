"""Unit tests for scripts/extractor.py and scripts/evaluate_extractor.py.  Run from the dataset/ folder:
python -m unittest discover -s tests"""
import csv
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_extractor as ev  # noqa: E402
from extractor import extract  # noqa: E402


class ExtractorCase(unittest.TestCase):
    def assertReads(self, sentence: str, register: str, gender: str) -> None:
        ex = extract(sentence)
        self.assertEqual((ex.register, ex.gender), (register, gender),
                         f"{sentence}\n" + "\n".join(ex.evidence))


class SixCells(ExtractorCase):
    def test_future(self) -> None:
        for s, r, g in [("तू कब वापस आएगा?", "tu", "M"), ("तू कब वापस आएगी?", "tu", "F"),
                        ("तुम कब वापस आओगे?", "tum", "M"), ("तुम कब वापस आओगी?", "tum", "F"),
                        ("आप कब वापस आएँगे?", "aap", "M"), ("आप कब वापस आएँगी?", "aap", "F")]:
            self.assertReads(s, r, g)

    def test_past_including_the_prescriptive_tum_f(self) -> None:
        for s, r, g in [("तू कहाँ गया था?", "tu", "M"), ("तू कहाँ गई थी?", "tu", "F"),
                        ("तुम कहाँ गए थे?", "tum", "M"), ("तुम कहाँ गई थी?", "tum", "F"),
                        ("तुम कहाँ गई थीं?", "tum", "F"),
                        ("आप कहाँ गए थे?", "aap", "M"), ("आप कहाँ गई थीं?", "aap", "F")]:
            self.assertReads(s, r, g)

    def test_present(self) -> None:
        for s, r, g in [("तू जा रहा है?", "tu", "M"), ("तुम जा रही हो?", "tum", "F"),
                        ("आप जा रही हैं?", "aap", "F"), ("क्या आप भूखे हैं?", "aap", "M")]:
            self.assertReads(s, r, g)


class ProDrop(ExtractorCase):
    """Without the pronoun, a syncretic form must give the whole compatible set, not a guess."""

    def test_past_syncretism(self) -> None:
        self.assertReads("क्या ठीक से सोए?", "tum|aap", "M")
        self.assertReads("कहाँ गई थी?", "tu|tum", "F")
        self.assertReads("कहाँ गई थीं?", "tum|aap", "F")
        self.assertReads("कहाँ गया था?", "tu", "M")

    def test_present_and_future_stay_distinct(self) -> None:
        self.assertReads("कहाँ जा रही हो?", "tum", "F")
        self.assertReads("कब आएँगी?", "aap", "F")
        self.assertReads("क्या आज रात घर पर होगी?", "tu|tum", "F")   # होगी: the one FUT syncretism
        self.assertReads("यहाँ कब तक रहोगी?", "tum", "F")             # …but रहोगी is regular

    def test_dropped_auxiliary_after_negation(self) -> None:
        self.assertReads("तुम कभी थकती नहीं?", "tum", "F")
        self.assertReads("आज रात यहाँ नहीं रुक सकते।", "tum|aap", "M")


class OnlyTheAddresseeCounts(ExtractorCase):
    def test_relative_clause(self) -> None:
        self.assertReads("क्या तुम उस लड़की से मिले जो कल आई थी?", "tum", "M")
        self.assertReads("क्या उस लड़की से मिलीं जो कल आई थी?", "tum|aap", "F")

    def test_possessive_and_attributive_adjectives(self) -> None:
        self.assertReads("क्या तुम अपनी बहन के साथ जाओगे?", "tum", "M")
        self.assertReads("इतनी लंबी यात्रा के बाद तुम थक गए होगे।", "tum", "M")
        self.assertReads("तुम इतनी देर से क्यों आए हो?", "tum", "M")

    def test_another_clause_with_its_own_subject(self) -> None:
        self.assertReads("जब बारिश शुरू हुई, तब क्या तुम घर पर थे?", "tum", "M")
        self.assertReads("जब पहुँचीं, तब तक समारोह हो चुका था।", "tum|aap", "F")
        self.assertReads("जब तक मेहमान पहुँचे, तब तक सो चुकी थी।", "tu|tum", "F")
        self.assertReads("कभी अंदाज़ा नहीं लगा पाओगे कि क्या हुआ।", "tum", "M")
        self.assertReads("बाहर नहीं सो पाओगी, ठंड है।", "tum", "F")

    def test_continuation_clauses_count(self) -> None:
        self.assertReads("अगर तुम वहाँ होते, तो समझ जाते।", "tum", "M")
        self.assertReads("आप सोमवार को आईं और अगले ही दिन फिर चली गईं।", "aap", "F")


class BeyondTheSixCells(ExtractorCase):
    def test_escape_routes_are_unmarked_for_gender(self) -> None:
        self.assertReads("क्या तुम्हें भूख लगी है?", "tum", "unmarked")
        self.assertReads("तुम्हारा काम क्या है?", "tum", "unmarked")
        self.assertReads("क्या तुमने खाना खा लिया है?", "tum", "unmarked")
        self.assertReads("आपसे मिलकर अच्छा लगा।", "aap", "unmarked")

    def test_imperatives(self) -> None:
        self.assertReads("यहाँ बैठिए।", "aap", "unmarked")
        self.assertReads("यहाँ बैठ जाओ।", "tum", "unmarked")

    def test_invariant_predicates(self) -> None:
        self.assertReads("क्या तुम खुश हो?", "tum", "unmarked")
        self.assertReads("आप शाकाहारी हैं?", "aap", "unmarked")
        self.assertReads("तुम इतने चुप क्यों हो?", "tum", "M")      # but the intensifier agrees

    def test_mixed_register(self) -> None:
        self.assertReads("आप बाद में समझोगे।", "mixed", "M")
        self.assertReads("आप जा सकते हो।", "mixed", "M")
        self.assertReads("आप कहाँ गई थी? तुम्हारी माँ पूछ रही थी।", "mixed", "F")

    def test_nothing_to_read(self) -> None:
        self.assertReads("धन्यवाद।", "unmarked", "unmarked")
        self.assertReads("", "unmarked", "unmarked")


class SpellingVariants(ExtractorCase):
    def test_variants_read_like_the_convention(self) -> None:
        self.assertReads("आप कब आयेंगी?", "aap", "F")
        self.assertReads("क्या तू गयी थी?", "tu", "F")
        self.assertReads("आप कब आएंगे?", "aap", "M")
        self.assertReads("आप किसका इंतज़ार कर रहे थें?", "aap", "M")


class FoundOnMTOutputs(ExtractorCase):
    """Constructions that real MT outputs use but the benchmark's references do not."""

    def test_ho_in_a_compound_verb_is_not_the_auxiliary(self) -> None:
        # हो गए = "became": हो is the verb stem here, not the तुम auxiliary हो.
        self.assertReads("क्या तुम गुम हो गए?", "tum", "M")
        self.assertReads("आप कितने बड़े हो गए!", "aap", "M")

    def test_pronoun_before_pehle_is_still_the_subject(self) -> None:
        # पहले / बाद can be postpositions, but after the pronoun they are adverbs ("already", "later").
        self.assertReads("जब फिल्म शुरू हुई थी तो क्या आप पहले ही अपनी सीट ले चुके थे?", "aap", "M")
        self.assertReads("क्या आप पहले कभी यहाँ आई हैं?", "aap", "F")

    def test_perfective_kie_is_not_a_polite_imperative(self) -> None:
        # किए / लिए / दिए end like बैठिए, but they are perfective participles.
        self.assertReads("आप मेरे लिए इंतज़ार क्यों नहीं किए?", "aap", "M")
        self.assertReads("तुम मेरे लिए इंतजार क्यों नहीं किए?", "tum", "M")

    def test_final_tag_question_is_not_a_clause(self) -> None:
        # The tag "है ना?" repeats the auxiliary; it says nothing about the addressee.
        self.assertReads("आपका वजन कम हो गया है, है ना?", "aap", "unmarked")
        self.assertReads("आपने वजन कम कर लिया है, है ना?", "aap", "unmarked")
        self.assertReads("तुम थक गई हो, है ना?", "tum", "F")

    def test_noun_subject_separated_from_its_verb_by_adverbs(self) -> None:
        # In समारोह पहले ही हो चुका था the ceremony is the subject, not the addressee.
        self.assertReads("जब तुम पहुँची, समारोह पहले ही हो चुका था।", "tum", "F")
        self.assertReads("जब आप पहुँचे, तो समारोह पहले ही हो चुका था।", "aap", "M")


class WholeBenchmark(unittest.TestCase):
    """Every reference, alternative and pro-drop variant of the benchmark (the development set)."""

    def test_benchmark_is_read_correctly(self) -> None:
        rows = ev.benchmark_rows(ROOT / "benchmark_v1_long.csv")
        ev.predict(rows)
        wrong = [f"{r['id']} {r['tier']}: {r['hindi']} -> {r['pred_register']}/{r['pred_gender']}, "
                 f"gold {r['gold_register']}/{r['gold_gender']}"
                 for r in rows if (r["pred_register"], r["pred_gender"]) != (r["gold_register"], r["gold_gender"])]
        self.assertEqual(wrong, [], "\n".join(wrong[:20]))


class Evaluation(unittest.TestCase):
    def test_label_normalisation(self) -> None:
        self.assertEqual(ev.canonical_register("AAP|tum"), "tum|aap")
        self.assertEqual(ev.canonical_register("tu|tum|aap"), "unmarked")
        self.assertEqual(ev.canonical_gender("f"), "F")
        self.assertEqual(ev.canonical_gender("Unmarked"), "unmarked")
        with self.assertRaises(ValueError):
            ev.canonical_register("formal")

    def test_wilson_interval(self) -> None:
        lo, hi = ev.wilson(90, 100)
        self.assertAlmostEqual(lo, 0.826, places=3)
        self.assertAlmostEqual(hi, 0.945, places=3)

    def test_sheet_then_labelled_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            mt, sheet = Path(tmp) / "mt.csv", Path(tmp) / "sheet.csv"
            with mt.open("w", encoding="utf-8", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["id", "system", "hindi"])
                w.writerow(["PST-05", "sysA", "तुम कहाँ गई थी?"])
                w.writerow(["PST-05", "sysB", "आप कहाँ गए थे?"])
            with redirect_stdout(StringIO()), redirect_stderr(StringIO()):
                self.assertEqual(ev.main(["--make-sheet", str(mt), "--out", str(sheet)]), 0)
                self.assertEqual(ev.main(["--make-sheet", str(mt), "--out", str(sheet)]), 1)  # never overwrite
            with sheet.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))
            self.assertEqual({r["english"] for r in rows}, {"Where did you go?"})
            self.assertNotIn("pred_register", rows[0])
            gold = {"तुम कहाँ गई थी?": ("tum", "F"), "आप कहाँ गए थे?": ("aap", "F")}  # second is a deliberate miss
            for r in rows:
                r["gold_register"], r["gold_gender"] = gold[r["hindi"]]
            with sheet.open("w", encoding="utf-8", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
            out = StringIO()
            with redirect_stdout(out):
                self.assertEqual(ev.main(["--labelled", str(sheet)]), 0)
            self.assertIn("errors (1)", out.getvalue())


if __name__ == "__main__":
    unittest.main()
