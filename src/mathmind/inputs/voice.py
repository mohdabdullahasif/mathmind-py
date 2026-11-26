"""Voice input normalizer for MathMind.

Takes a Whisper transcript and normalizes spoken math to an expression string.
No audio recording in this module – that's in voice.py which needs sounddevice.
"""
from __future__ import annotations

import re


_WORD_TO_NUM: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
    "million": "1000000",
}

_MATH_WORDS: list[tuple[re.Pattern, str]] = []


def _m(pat: str, rep: str) -> None:
    _MATH_WORDS.append((re.compile(pat, re.IGNORECASE), rep))


# exponents
_m(r"\bsquared\b", "**2")
_m(r"\bcubed\b", "**3")
_m(r"\bto\s+the\s+power\s+of\s+(\w+)", r"**\1")

# fractions
_m(r"\bover\b", "/")
_m(r"\bdivided\s+by\b", "/")
_m(r"\btimes\b", "*")
_m(r"\bmultiplied\s+by\b", "*")
_m(r"\bplus\b", "+")
_m(r"\bminus\b", "-")
_m(r"\bpoint\b", ".")

# variable common aliases
_m(r"\bx\s+equals?\b", "x =")
_m(r"\bequals?\b", "=")

# "square root of" -> sqrt(
_m(r"square\s+root\s+of\s+(\d+)", r"sqrt(\1)")


def normalize_transcript(transcript: str) -> str:
    """Convert spoken math to a canonical expression string."""
    t = transcript.strip()
    # replace word numbers
    for word, num in sorted(_WORD_TO_NUM.items(), key=lambda x: -len(x[0])):
        t = re.sub(rf"\b{word}\b", num, t, flags=re.IGNORECASE)
    # apply math word patterns
    for pattern, replacement in _MATH_WORDS:
        t = pattern.sub(replacement, t)
    # remove stray spaces around operators (preserve **)
    t = re.sub(r"\s*([+\-/=]|(?<!\*)\*(?!\*))\s*", r" \1 ", t)
    t = re.sub(r"\s*\*\*\s*", "**", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def transcribe_file(audio_path: str, model_name: str = "base") -> str:
    """Transcribe an audio file using Whisper. Requires openai-whisper installed."""
    try:
        import whisper  # type: ignore[import]
    except ImportError:
        raise ImportError(
            "openai-whisper not installed. Run: pip install openai-whisper"
        )
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_path)
    return result["text"].strip()


def voice_to_expression(audio_path: str, model_name: str = "base") -> tuple[str, str]:
    """Transcribe then normalize. Returns (transcript, expression)."""
    transcript = transcribe_file(audio_path, model_name)
    expression = normalize_transcript(transcript)
    return transcript, expression
