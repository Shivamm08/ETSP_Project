"""Unit tests for the analysis helpers. Run from the repository root:
python -m unittest discover -s analysis/tests"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import auc, error_type, newcombe, register_group, sign_test  # noqa: E402


class Statistics(unittest.TestCase):
    def test_newcombe_matches_the_published_example(self) -> None:
        # Newcombe (1998), Statistics in Medicine 17: 56/70 vs 48/80, method 10 -> 0.0524 to 0.3339
        d, lo, hi = newcombe(56, 70, 48, 80)
        self.assertAlmostEqual(d, 0.2)
        self.assertAlmostEqual(lo, 0.0524, places=3)
        self.assertAlmostEqual(hi, 0.3339, places=3)

    def test_sign_test(self) -> None:
        self.assertAlmostEqual(sign_test(3, 0), 0.25)          # 2 * 0.5^3
        self.assertAlmostEqual(sign_test(10, 0), 2 / 1024)
        self.assertEqual(sign_test(0, 0), 1.0)
        self.assertEqual(sign_test(4, 4), 1.0)

    def test_auc(self) -> None:
        self.assertEqual(auc([3, 4], [1, 2]), 1.0)
        self.assertEqual(auc([1, 2], [1, 2]), 0.5)
        self.assertEqual(auc([1], [2]), 0.0)


class Labels(unittest.TestCase):
    def test_error_type(self) -> None:
        self.assertEqual(error_type("tu_m", "tu_f"), "gender")
        self.assertEqual(error_type("tu_m", "aap_m"), "register")
        self.assertEqual(error_type("tum_f", "aap_m"), "both")

    def test_register_group(self) -> None:
        self.assertEqual(register_group("aap"), "aap")
        self.assertEqual(register_group("tum|aap"), "ambiguous")
        self.assertEqual(register_group("mixed"), "mixed")


if __name__ == "__main__":
    unittest.main()
