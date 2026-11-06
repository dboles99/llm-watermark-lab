from __future__ import annotations

import unicodedata
from dataclasses import asdict, dataclass

SUSPICIOUS_CODEPOINTS = {
    0x00AD: "SOFT HYPHEN",
    0x061C: "ARABIC LETTER MARK",
    0x200B: "ZERO WIDTH SPACE",
    0x200C: "ZERO WIDTH NON-JOINER",
    0x200D: "ZERO WIDTH JOINER",
    0x200E: "LEFT-TO-RIGHT MARK",
    0x200F: "RIGHT-TO-LEFT MARK",
    0x202A: "LEFT-TO-RIGHT EMBEDDING",
    0x202B: "RIGHT-TO-LEFT EMBEDDING",
    0x202C: "POP DIRECTIONAL FORMATTING",
    0x202D: "LEFT-TO-RIGHT OVERRIDE",
    0x202E: "RIGHT-TO-LEFT OVERRIDE",
    0x2060: "WORD JOINER",
    0x2066: "LEFT-TO-RIGHT ISOLATE",
    0x2067: "RIGHT-TO-LEFT ISOLATE",
    0x2068: "FIRST STRONG ISOLATE",
    0x2069: "POP DIRECTIONAL ISOLATE",
    0xFEFF: "ZERO WIDTH NO-BREAK SPACE OR BOM",
}


@dataclass(frozen=True)
class CharacterFinding:
    offset: int
    codepoint: str
    name: str
    category: str
    context: str


def scan_suspicious_characters(text: str, context_radius: int = 12) -> list[dict[str, object]]:
    findings: list[CharacterFinding] = []
    for offset, character in enumerate(text):
        codepoint = ord(character)
        category = unicodedata.category(character)
        private_use = category == "Co"
        tagged = 0xE0000 <= codepoint <= 0xE007F
        variation = 0xFE00 <= codepoint <= 0xFE0F or 0xE0100 <= codepoint <= 0xE01EF
        if codepoint in SUSPICIOUS_CODEPOINTS or private_use or tagged or variation:
            name = SUSPICIOUS_CODEPOINTS.get(codepoint, unicodedata.name(character, "UNNAMED"))
            start = max(0, offset - context_radius)
            end = min(len(text), offset + context_radius + 1)
            findings.append(
                CharacterFinding(
                    offset=offset,
                    codepoint=f"U+{codepoint:04X}",
                    name=name,
                    category=category,
                    context=text[start:end].replace(character, f"<{name}>"),
                )
            )
    return [asdict(finding) for finding in findings]


def normalization_report(text: str) -> dict[str, object]:
    forms = {form: unicodedata.normalize(form, text) for form in ("NFC", "NFD", "NFKC", "NFKD")}
    return {
        "length": len(text),
        "already_normalized": {form: value == text for form, value in forms.items()},
        "normalized_lengths": {form: len(value) for form, value in forms.items()},
    }

