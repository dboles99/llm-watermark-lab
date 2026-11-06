"""Regional-lexicon screening for mechanically converted Chinese text.

The headline test is ``test_converted_text_passes_a_script_check_but_fails_here``:
it demonstrates the exact blind spot this module exists to close.
"""

from __future__ import annotations

import pytest

from watermark_lab.corpora.tw_cn_discriminator import (
    REGIONAL_TERM_PAIRS,
    SIMPLIFIED_ONLY_CHARACTERS,
    screen,
    screen_corpus,
)

# Vocabulary that a character-level s2t conversion leaves untouched. Every glyph
# below is a valid Traditional form, so this text is "Traditional Chinese" to any
# script-level check — but the wording is mainland throughout.
CONVERTED_FROM_SIMPLIFIED = (
    "這個軟件的網絡功能需要更多的信息才能運行。"
    "用戶可以通過鼠標點擊屏幕上的菜單來打開文件夾，"
    "然後把視頻和音頻數據上傳到服務器的數據庫中。"
    "默認的內存設置可能會影響加載質量，建議進行優化。"
)

# The same content in genuine Taiwan usage.
TAIWAN_NATIVE = (
    "這個軟體的網路功能需要更多的資訊才能運行。"
    "使用者可以透過滑鼠點擊螢幕上的選單來開啟資料夾，"
    "然後把影片和音訊資料上傳到伺服器的資料庫中。"
    "預設的記憶體設定可能會影響載入品質，建議進行最佳化。"
)

UNCONVERTED_SIMPLIFIED = (
    "这个软件的网络功能需要更多的信息才能运行。"
    "用户可以通过鼠标点击屏幕上的菜单来打开文件夹。"
)


def _is_traditional_by_script(text: str) -> bool:
    """A naive script-level check, standing in for the common approach.

    Returns True when no Simplified-only character is present. This is the check
    that mechanically converted text defeats.
    """

    return not any(ch in SIMPLIFIED_ONLY_CHARACTERS for ch in text)


class TestTheBlindSpot:
    def test_converted_text_passes_a_script_check_but_fails_here(self) -> None:
        """The whole point of the module, in one assertion pair."""

        assert _is_traditional_by_script(CONVERTED_FROM_SIMPLIFIED) is True

        result = screen(CONVERTED_FROM_SIMPLIFIED)
        assert result.verdict == "mainland-indicators"
        assert result.mainland_total > 5
        assert result.taiwan_total == 0
        assert "would pass this text" in result.note

    def test_native_taiwan_text_is_consistent(self) -> None:
        result = screen(TAIWAN_NATIVE)
        assert result.verdict == "taiwan-consistent"
        assert result.taiwan_total > 5
        assert result.mainland_total == 0

    def test_the_two_texts_are_the_same_length_but_score_oppositely(self) -> None:
        """Length is not what separates them; vocabulary is."""

        converted = screen(CONVERTED_FROM_SIMPLIFIED)
        native = screen(TAIWAN_NATIVE)
        assert abs(converted.cjk_characters - native.cjk_characters) < 15
        assert converted.mainland_ratio == pytest.approx(1.0)
        assert native.mainland_ratio == pytest.approx(0.0)


class TestUnconvertedSimplified:
    def test_simplified_characters_are_reported_separately(self) -> None:
        """Never converted at all is a script failure, not a vocabulary one."""

        result = screen(UNCONVERTED_SIMPLIFIED, minimum_cjk=10)
        assert result.simplified_characters
        assert any("not converted to Traditional at all" in w for w in result.warnings)

    def test_script_check_catches_what_this_also_catches(self) -> None:
        assert _is_traditional_by_script(UNCONVERTED_SIMPLIFIED) is False


class TestAbsenceIsNotEvidence:
    def test_short_text_reports_insufficient_evidence(self) -> None:
        result = screen("今天天氣很好。")
        assert result.verdict == "insufficient-evidence"
        assert "NOT evidence of Taiwan origin" in result.note

    def test_long_text_without_discriminating_terms_is_still_insufficient(self) -> None:
        text = "今天天氣很好，我們去公園散步，看到很多小鳥在樹上唱歌。" * 6
        result = screen(text)
        assert result.cjk_characters >= 50
        assert result.verdict == "insufficient-evidence"

    def test_mainland_ratio_is_none_not_zero_without_evidence(self) -> None:
        """0.0 would wrongly read as 'no mainland usage detected and measured'."""

        result = screen("今天天氣很好。" * 20)
        assert result.mainland_ratio is None

    def test_empty_text_does_not_crash(self) -> None:
        assert screen("").verdict == "insufficient-evidence"


class TestMixedText:
    def test_both_forms_present_gives_mixed(self) -> None:
        mixed = CONVERTED_FROM_SIMPLIFIED[:60] + TAIWAN_NATIVE[:60]
        result = screen(mixed)
        assert result.verdict == "mixed"
        assert result.mainland_total > 0
        assert result.taiwan_total > 0
        assert 0.0 < (result.mainland_ratio or 0.0) < 1.0


class TestEvidenceIsAttached:
    def test_hits_name_the_taiwan_equivalent_and_gloss(self) -> None:
        result = screen(CONVERTED_FROM_SIMPLIFIED)
        by_term = {hit.term: hit for hit in result.mainland_hits}
        assert "軟件" in by_term
        assert by_term["軟件"].taiwan_equivalent == "軟體"
        assert by_term["軟件"].gloss == "software"

    def test_hits_are_ordered_by_count(self) -> None:
        text = "軟件軟件軟件網絡" * 10
        result = screen(text, minimum_cjk=10)
        counts = [hit.count for hit in result.mainland_hits]
        assert counts == sorted(counts, reverse=True)

    def test_result_serialises(self) -> None:
        payload = screen(CONVERTED_FROM_SIMPLIFIED).as_dict()
        assert payload["verdict"] == "mainland-indicators"
        assert payload["mainland_hits"]
        assert payload["mainland_ratio"] == pytest.approx(1.0)


class TestLexiconIntegrity:
    def test_pairs_are_distinct(self) -> None:
        for mainland, taiwan, gloss in REGIONAL_TERM_PAIRS:
            if "shared" in gloss:
                continue
            assert mainland != taiwan, f"{mainland} is not discriminating"

    def test_no_pair_contains_simplified_only_characters(self) -> None:
        """Both sides must be valid Traditional, or the test is trivial."""

        for mainland, taiwan, _gloss in REGIONAL_TERM_PAIRS:
            for term in (mainland, taiwan):
                assert not any(ch in SIMPLIFIED_ONLY_CHARACTERS for ch in term), term


class TestCorpusScreening:
    def test_flags_only_the_contaminated_documents(self) -> None:
        report = screen_corpus(
            {
                "doc-native": TAIWAN_NATIVE,
                "doc-converted": CONVERTED_FROM_SIMPLIFIED,
                "doc-short": "很好。",
            }
        )
        assert report["document_count"] == 3
        assert report["recommended_exclusions"] == ["doc-converted"]
        assert "doc-native" in report["by_verdict"]["taiwan-consistent"]
        assert "doc-short" in report["by_verdict"]["insufficient-evidence"]

    def test_rejects_empty_corpus(self) -> None:
        with pytest.raises(ValueError):
            screen_corpus({})
