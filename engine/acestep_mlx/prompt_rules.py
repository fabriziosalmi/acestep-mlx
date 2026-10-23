"""Deterministic reading of the user's prompt: vocals wanted, explicit BPM, explicit key.

No model is involved, so the same prompt always gives the same answer. Italian and
English. Values the LM's constrained decoding would reject are ignored, not clamped.
"""

import re
from typing import Optional

BPM_MIN, BPM_MAX = 30, 300  # upstream acestep/constants.py

# Phrases that ask for no voice win over any vocal word ("senza voce" contains "voce").
_NO_VOCALS = re.compile(
    r"\b(strumentale|instrumental|senza\s+(voce|voci|cantato|testo)|"
    r"no\s+(vocals?|voice|singing|lyrics)|without\s+(vocals?|voice|singing|lyrics))\b"
)
_VOCALS = re.compile(
    r"\b(voce|voci|vocale|vocali|cantato|cantata|cantante|cantanti|canta|cantare|testo|"
    r"coro|rap|rapper|rappato|duetto|a\s+cappella|vocals?|voice|voices|vocalist|singer|"
    r"singers|singing|sung|sings|lyrics|choir|duet)\b"
)
_BPM = re.compile(r"(?<![\d.,])(\d{2,3})(?:[.,]\d+)?\s*(?:bpm|b\.p\.m\.)", re.IGNORECASE)

_ACCIDENTALS = {"#": "#", "♯": "#", "sharp": "#", "diesis": "#",
                "b": "b", "♭": "b", "flat": "b", "bemolle": "b"}
_SOLFEGE = {"do": "C", "re": "D", "mi": "E", "fa": "F", "sol": "G", "la": "A", "si": "B"}
# Note letter must be a capital so the English word "am" is never read as A minor.
_KEY_EN = re.compile(r"(?<![\w#♯♭])([A-G])\s?(#|♯|b|♭|-sharp|-flat|\ssharp|\sflat)?"
                     r"(?:\s?((?i:major|minor|maj|min))\b|(m)\b)")
_KEY_IT = re.compile(
    r"\b(do|re|mi|fa|sol|la|si)\s?(#|♯|diesis|b|♭|bemolle)?\s+(maggiore|minore)\b",
    re.IGNORECASE,
)


def wants_vocals(prompt: str) -> bool:
    """True only if the prompt names a voice and does not ask for none."""
    text = prompt.lower()
    if _NO_VOCALS.search(text):
        return False
    return bool(_VOCALS.search(text))


def explicit_bpm(prompt: str) -> Optional[int]:
    """The BPM written in the prompt, if exactly one distinct value in range is given."""
    values = {int(m.group(1)) for m in _BPM.finditer(prompt)}
    values = {v for v in values if BPM_MIN <= v <= BPM_MAX}
    return values.pop() if len(values) == 1 else None


def _key(letter: str, accidental: Optional[str], minor: bool) -> str:
    acc = _ACCIDENTALS[accidental.strip(" -").lower()] if accidental else ""
    return f"{letter}{acc} {'minor' if minor else 'major'}"


def explicit_key(prompt: str) -> Optional[str]:
    """The key written in the prompt as upstream expects it ("F# minor"), if exactly one."""
    keys = set()
    for m in _KEY_EN.finditer(prompt):
        mode = (m.group(3) or m.group(4) or "").lower()
        keys.add(_key(m.group(1), m.group(2), mode in ("minor", "min", "m")))
    for m in _KEY_IT.finditer(prompt):
        keys.add(_key(_SOLFEGE[m.group(1).lower()], m.group(2), m.group(3).lower() == "minore"))
    return keys.pop() if len(keys) == 1 else None
