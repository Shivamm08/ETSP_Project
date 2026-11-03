"""Hindi spelling normalisation, so spelling variants never count as translation differences.

Apply `normalize_hindi` to BOTH the references and the system outputs before matching (extractor)
or scoring (chrF, BLEU, COMET). It removes only variation that carries no grammatical meaning:

  * chandrabindu -> anusvara             आएँगे -> आएंगे, हूँ -> हूं, कहाँ -> कहां
  * Perso-Arabic nukta dropped           ज़्यादा -> ज्यादा, फ़ोन -> फोन   (ड़ and ढ़ are real letters: kept)
  * य-spellings of verb endings          गये -> गए, गयी -> गई, गयीं -> गईं, जायेंगे -> जाएंगे, चाहिये -> चाहिए
  * nasal consonant before same-class stop -> anusvara     हिन्दी -> हिंदी, सम्बन्ध -> संबंध
  * Unicode NFC, ZWJ/ZWNJ removed, '|' -> '।', whitespace tidied

It never touches the endings that carry register or gender: the anusvara of गईं, हैं, थीं survives,
so गई / गईं and है / हैं stay different. build_dataset.py checks that normalisation never merges two
of the six cells of any item.

Usage:
    from normalize import normalize_hindi
    normalize_hindi("क्या आप कल आयेंगी?") == normalize_hindi("क्या आप कल आएँगी?")   # True
"""
from __future__ import annotations

import re
import unicodedata

CHANDRABINDU, ANUSVARA, NUKTA, VIRAMA = "ँ", "ं", "़", "्"
NUKTA_KEPT_AFTER = {"ड", "ढ"}  # ड़ (ṛ) and ढ़ (ṛh) are distinct letters, not loanword spellings

# Nasal consonant + virama before a stop of the same class is spelt with anusvara (पंचमाक्षर rule).
_HOMORGANIC = re.compile(
    "ङ्(?=[कखगघ])|ञ्(?=[चछजझ])|ण्(?=[टठडढ])|न्(?=[तथदध])|म्(?=[पफबभ])"
)
# Devanagari letters and signs only: excludes the danda । ॥ (U+0964-5) and digits (U+0966-F),
# so "चाहिये।" still counts as word-final.
_DEV_LETTER = r"[ऀ-ॣॱ-ॿ]"
# य-glide spellings: word-final ये/यी/यीं/यें, and ये before a future ending (जायेगा, जायेंगे).
_YA_FINAL = re.compile(rf"(?<={_DEV_LETTER})य(ीं|ें|े|ी)(?!{_DEV_LETTER})")
_YA_FUTURE = re.compile(rf"(?<={_DEV_LETTER})ये(?=ं?ग[ाीे])")
_YA_VOWEL = {"े": "ए", "ी": "ई", "ीं": "ईं", "ें": "एं"}


def _drop_nukta(text: str) -> str:
    out: list[str] = []
    for ch in text:
        if ch == NUKTA and (not out or out[-1] not in NUKTA_KEPT_AFTER):
            continue
        out.append(ch)
    return "".join(out)


def normalize_hindi(text: str) -> str:
    """Return a canonical spelling of a Hindi string (see module docstring for what changes)."""
    s = unicodedata.normalize("NFC", text)          # also splits precomposed nukta letters (U+0958-095F)
    s = s.replace("‌", "").replace("‍", "")
    s = s.replace(CHANDRABINDU, ANUSVARA)
    s = _drop_nukta(s)
    s = _HOMORGANIC.sub(ANUSVARA, s)
    s = _YA_FUTURE.sub("ए", s)
    s = _YA_FINAL.sub(lambda m: _YA_VOWEL[m.group(1)], s)
    s = s.replace("|", "।")
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"\s+([।?!,])", r"\1", s)
    return s
