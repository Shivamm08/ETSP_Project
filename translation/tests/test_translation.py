"""Unit tests for the translation helpers (no API keys or network needed). Run from the repository root:
python -m unittest discover -s translation/tests"""
import csv
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from config import OUTPUT_FIELDS, load_env, pending, write_outputs  # noqa: E402
from run_llm import build_jobs, clean, make_prompt  # noqa: E402

CORE = {"id": "PST-05", "split": "core", "english": "Where did you go?"}
PROBE = {"id": "H2-1F", "split": "h2_probe", "english": "Will you cook?"}


class Jobs(unittest.TestCase):
    def test_conditions_and_counts(self) -> None:
        jobs = build_jobs([CORE, PROBE], ["none", "register", "gender", "both"])
        by_condition = {c: [j for j in jobs if j[1] == c] for c in ("none", "register", "gender", "both")}
        self.assertEqual(len(by_condition["none"]), 2)        # every item, plain
        self.assertEqual(len(by_condition["register"]), 3)    # core only: तू, तुम, आप
        self.assertEqual(len(by_condition["gender"]), 2)      # core only: man, woman
        self.assertEqual(len(by_condition["both"]), 6)        # core only: all six cells
        self.assertTrue(all(j[0]["split"] == "core" for j in jobs if j[1] != "none"))

    def test_prompt(self) -> None:
        plain = make_prompt("Where did you go?", "", "")
        self.assertNotIn("Address", plain)
        both = make_prompt("Where did you go?", "tum", "F")
        self.assertIn("Address the person with the pronoun तुम.", both)
        self.assertIn("The person you are talking to is a woman.", both)
        self.assertTrue(both.endswith("Sentence: Where did you go?"))


class Clean(unittest.TestCase):
    def test_keeps_the_first_hindi_line(self) -> None:
        self.assertEqual(clean('Translation: "क्या तुम गए?"\nNote: informal register'), "क्या तुम गए?")
        self.assertEqual(clean("\nक्या आप गए?"), "क्या आप गए?")

    def test_reply_without_hindi_is_kept(self) -> None:
        self.assertEqual(clean("  sorry  "), "sorry")


class Resume(unittest.TestCase):
    def test_only_new_items_are_translated_and_old_rows_stay(self) -> None:
        rows = [{**CORE, "profile": "PAST", "construction": "perfective", "frame_id": "", "cue": "none"},
                {**PROBE, "profile": "FUT", "construction": "future", "frame_id": "fr1", "cue": "fem_stereotype"}]
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "system.csv"
            write_outputs(out, rows[:1], {}, {"PST-05": "आप कहाँ गए थे?"}, "sys", "model-1")
            done, todo = pending(rows, out)
            self.assertEqual(list(done), ["PST-05"])
            self.assertEqual([r["id"] for r in todo], ["H2-1F"])
            before = out.read_bytes()
            write_outputs(out, rows, done, {"H2-1F": "क्या तुम खाना बनाओगी?"}, "sys", "model-1")
            with out.open(encoding="utf-8", newline="") as fh:
                written = list(csv.DictReader(fh))
            self.assertEqual([r["id"] for r in written], ["PST-05", "H2-1F"])   # benchmark order
            self.assertEqual(list(written[0]), OUTPUT_FIELDS)
            self.assertTrue(out.read_bytes().startswith(before))               # the old row is unchanged


class Env(unittest.TestCase):
    def test_reads_keys_without_overriding_the_shell(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env = Path(tmp) / ".env"
            env.write_text('# comment\nETSP_TEST_A="from file"\nETSP_TEST_B=file\n\n', encoding="utf-8")
            os.environ["ETSP_TEST_B"] = "from shell"
            try:
                load_env(env)
                self.assertEqual(os.environ["ETSP_TEST_A"], "from file")
                self.assertEqual(os.environ["ETSP_TEST_B"], "from shell")
            finally:
                os.environ.pop("ETSP_TEST_A", None)
                os.environ.pop("ETSP_TEST_B", None)

    def test_missing_key_stops_with_a_message(self) -> None:
        os.environ.pop("ETSP_TEST_MISSING", None)
        with self.assertRaises(SystemExit) as ctx:
            config.api_key("ETSP_TEST_MISSING")
        self.assertIn(".env.example", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
