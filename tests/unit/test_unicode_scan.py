from watermark_lab.forensics.unicode_scan import normalization_report, scan_suspicious_characters


def test_zero_width_character_is_localized() -> None:
    findings = scan_suspicious_characters("alpha\u200bbeta")
    assert findings[0]["offset"] == 5
    assert findings[0]["codepoint"] == "U+200B"


def test_normalization_report_detects_non_nfc_text() -> None:
    report = normalization_report("e\u0301")
    assert report["already_normalized"]["NFC"] is False

