"""Unit tests for scripts/normalize.py.  Run from the dataset/ folder:  python -m unittest discover -s tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from normalize import normalize_hindi  # noqa: E402


class SpellingVariantsCollapse(unittest.TestCase):
    """Variants that differ only in spelling must normalise to the same string."""

    def assertSame(self, a: str, b: str) -> None:
        self.assertEqual(normalize_hindi(a), normalize_hindi(b), f"{a!r} vs {b!r}")

    def test_chandrabindu_vs_dot(self) -> None:
        self.assertSame("आप कब वापस आएँगे?", "आप कब वापस आएंगे?")
        self.assertSame("आप कब वापस आएँगी?", "आप कब वापस आएंगी?")
        self.assertSame("तुम कहाँ हो?", "तुम कहां हो?")
        self.assertSame("मैं आऊँगा।", "मैं आऊंगा।")

    def test_ya_spellings_of_verb_endings(self) -> None:
        self.assertSame("क्या तुम डाकघर गये थे?", "क्या तुम डाकघर गए थे?")
        self.assertSame("तू कहाँ गयी थी?", "तू कहाँ गई थी?")
        self.assertSame("आप कहाँ गयीं?", "आप कहाँ गईं?")
        self.assertSame("आप कब आयेंगे?", "आप कब आएँगे?")
        self.assertSame("तू कब जायेगा?", "तू कब जाएगा?")
        self.assertSame("तुम्हें जाना चाहिये।", "तुम्हें जाना चाहिए।")

    def test_nukta(self) -> None:
        self.assertSame("तू बहुत ज़्यादा मेहनत कर रहा है।", "तू बहुत ज्यादा मेहनत कर रहा है।")
        self.assertSame("फ़ोन", "फोन")
        self.assertSame("ज़", "ज")  # precomposed ज़ (U+095B)

    def test_nasal_consonant_clusters(self) -> None:
        self.assertSame("हिन्दी", "हिंदी")
        self.assertSame("सम्बन्ध", "संबंध")

    def test_punctuation_and_whitespace(self) -> None:
        self.assertSame("तुम कहाँ हो |", "तुम कहाँ हो।")
        self.assertSame("तुम  कहाँ ‍हो?", "तुम कहाँ हो?")


class MeaningIsPreserved(unittest.TestCase):
    """Normalisation must never erase a grammatical or lexical difference."""

    def test_register_and_gender_endings_stay_distinct(self) -> None:
        for a, b in [("गई", "गईं"), ("है", "हैं"), ("हो", "हैं"), ("थी", "थीं"), ("जाओगे", "जाओगी"),
                     ("आएगा", "आएँगे"), ("होगी", "होंगी"), ("जा रही हो", "जा रही हैं")]:
            self.assertNotEqual(normalize_hindi(a), normalize_hindi(b), f"{a} / {b} must stay distinct")

    def test_pronoun_ye_untouched(self) -> None:
        self.assertEqual(normalize_hindi("ये मेरा घर है।"), "ये मेरा घर है।")

    def test_retroflex_flap_kept(self) -> None:
        self.assertNotEqual(normalize_hindi("बड़ा"), normalize_hindi("बडा"))

    def test_non_homorganic_clusters_kept(self) -> None:
        self.assertIn("न्ह", normalize_hindi("उन्होंने"))
        self.assertIn("म्ह", normalize_hindi("तुम्हें"))


if __name__ == "__main__":
    unittest.main()
