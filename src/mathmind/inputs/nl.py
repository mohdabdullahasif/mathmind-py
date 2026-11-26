"""Natural Language to Math expression translator.

MVP: rule/regex-based translator for common phrases.
Phrases handled: percentages, square root, word numbers, basic arithmetic words.
"""
from __future__ import annotations

import re

# ------------------------------------------------------------------ word numbers
_WORD_TO_NUM: dict[str, str] = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100", "thousand": "1000",
}


def _replace_word_numbers(text: str) -> str:
    for word, num in _WORD_TO_NUM.items():
        text = re.sub(rf"\b{word}\b", num, text, flags=re.IGNORECASE)
    return text


# ------------------------------------------------------------------ rule patterns
# Each rule: (pattern, replacement or callable)
_RULES: list[tuple[re.Pattern, str | object]] = []


def _add(pattern: str, replacement: str | object) -> None:
    _RULES.append((re.compile(pattern, re.IGNORECASE), replacement))


# "X% of Y" -> "Y * (X/100)"
_add(r"(\d+(?:\.\d+)?)\s*%\s*of\s+(\d+(?:\.\d+)?)", r"(\2 * (\1 / 100))")

# "square root of X" -> "sqrt(X)"
_add(r"square\s+root\s+of\s+(\d+(?:\.\d+)?)", r"sqrt(\1)")

# "cube root of X" -> "(\1)**(1/3)"
_add(r"cube\s+root\s+of\s+(\d+(?:\.\d+)?)", r"(\1**(1/3))")

# "X to the power of Y" / "X to the Y" -> "X**Y"
_add(r"(\d+(?:\.\d+)?)\s+to\s+the\s+power\s+of\s+(\d+(?:\.\d+)?)", r"\1**\2")
_add(r"(\d+(?:\.\d+)?)\s+to\s+the\s+(\d+(?:st|nd|rd|th)?)\s+power", r"\1**\2")

# "X squared" -> "X**2"
_add(r"(\d+(?:\.\d+)?)\s+squared", r"\1**2")

# "X cubed" -> "X**3"
_add(r"(\d+(?:\.\d+)?)\s+cubed", r"\1**3")

# "X more than Y" -> "Y + X"
_add(r"(\d+(?:\.\d+)?)\s+more\s+than\s+(\d+(?:\.\d+)?)", r"(\2 + \1)")

# "X less than Y" / "X fewer than Y" -> "Y - X"
_add(r"(\d+(?:\.\d+)?)\s+(?:less|fewer)\s+than\s+(\d+(?:\.\d+)?)", r"(\2 - \1)")

# "X times Y" -> "X * Y"
_add(r"(\d+(?:\.\d+)?)\s+times\s+(\d+(?:\.\d+)?)", r"(\1 * \2)")

# "X divided by Y" / "X over Y" -> "X / Y"
_add(r"(\d+(?:\.\d+)?)\s+(?:divided\s+by|over)\s+(\d+(?:\.\d+)?)", r"(\1 / \2)")

# "X plus Y" -> "X + Y"
_add(r"(\d+(?:\.\d+)?)\s+plus\s+(\d+(?:\.\d+)?)", r"(\1 + \2)")

# "X minus Y" -> "X - Y"
_add(r"(\d+(?:\.\d+)?)\s+minus\s+(\d+(?:\.\d+)?)", r"(\1 - \2)")

# word-problem: "X% off Y" -> "Y * (1 - X/100)"
_add(r"(\d+(?:\.\d+)?)\s*%\s*off\s+(\d+(?:\.\d+)?)", r"(\2 * (1 - \1/100))")

# "mean of X, Y, Z" -> "mean([X, Y, Z])"
# (placeholder – handled downstream in CLI)

# "simple interest P at R% for T years" -> "P * R/100 * T"
_add(
    r"simple\s+interest\s+(?:on\s+)?(\d+(?:\.\d+)?)\s+at\s+(\d+(?:\.\d+)?)%\s+for\s+(\d+(?:\.\d+)?)\s+years?",
    r"(\1 * \2/100 * \3)",
)

# tax: "X including Y% tax"  -> "X * (1 + Y/100)"
_add(r"(\d+(?:\.\d+)?)\s+including\s+(\d+(?:\.\d+)?)%\s+tax", r"(\1 * (1 + \2/100))")

# average: "average of X, Y, Z" handled downstream


def translate(text: str) -> tuple[str, bool]:
    """Translate natural language *text* to a math expression.

    Returns (expression_string, was_translated).
    """
    t = _replace_word_numbers(text.strip())
    original = t
    for pattern, replacement in _RULES:
        t = pattern.sub(replacement, t)  # type: ignore[arg-type]
    translated = t != original
    # clean up extra spaces
    t = re.sub(r"\s+", " ", t).strip()
    return t, translated
