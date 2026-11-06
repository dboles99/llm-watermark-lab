"""OCR intake, degradation, and print-scan round-trip survival measurement.

OCR is treated here as a scientific instrument and an attack channel, not as an
intake convenience:

- **Instrument.** Engine identity, model bytes, language, and rendering
  parameters all change the measurement, so each is pinned and fingerprinted the
  way a tokenizer is. Two engines are supported because a survival result that
  holds across independently built engines is far stronger than one that holds
  for a single model family.
- **Attack channel.** A print-scan-OCR round trip destroys every container-bound
  provenance channel while preserving meaning exactly, which makes it a rare
  attack that is legitimate under the project's own quality constraints.

Text recovered by OCR is a reconstruction, never a copy. Detector thresholds
calibrated on clean text do not transfer to it, and the governance rule in
``docs`` and in ``result.OCR_RECONSTRUCTION_NOTE`` says so explicitly.
"""

from watermark_lab.ocr.calibration import (
    MINIMUM_PILOT_CLUSTERS,
    PILOT_STATUS_NOTE,
    ClusterScore,
    OcrCalibrationError,
    OcrNullResult,
    build_calibration_manifest,
    measure_ocr_null,
    write_calibration_artifact,
)
from watermark_lab.ocr.degrade import (
    DEFAULT_SEVERITIES,
    DEGRADATIONS,
    LADDER,
    Degradation,
    DegradationError,
    apply_chain,
    apply_degradation,
    build_ladder,
)
from watermark_lab.ocr.engine import (
    OcrEngine,
    available_engine_ids,
    get_engine,
    register_engine,
)
from watermark_lab.ocr.locale_comparison import (
    DEFAULT_RATIO_MARGIN,
    LocaleReading,
    LocaleSummary,
    LocaleVerdict,
    compare_locales,
    evaluate_locale_prediction,
    screen_recovered_text,
)
from watermark_lab.ocr.profile import (
    AssetFingerprint,
    OcrProfile,
    OcrProfileError,
    available_engines,
    build_profile,
    fingerprint_assets,
    rapidocr_available,
    tesseract_available,
)
from watermark_lab.ocr.render import (
    RenderError,
    RenderSettings,
    build_render_settings,
    render_fingerprint,
    render_text_to_image,
    select_font,
)
from watermark_lab.ocr.result import (
    OCR_RECONSTRUCTION_NOTE,
    PROVENANCE_CHANNEL,
    OcrResult,
    TextBlock,
    sort_reading_order,
)
from watermark_lab.ocr.survival import (
    CONTAINER_BOUND_CHANNELS,
    CONTENT_BOUND_CHANNELS,
    DEFAULT_COVERAGE_FLOOR,
    DETECTION_DROPOUT_NOTE,
    ChannelSurvival,
    RoundTripResult,
    character_error_rate,
    context_corruption_span,
    levenshtein,
    measure_channel_survival,
    round_trip,
    summarise_surface,
    word_error_rate,
)

__all__ = [
    "CONTAINER_BOUND_CHANNELS",
    "CONTENT_BOUND_CHANNELS",
    "DEFAULT_COVERAGE_FLOOR",
    "DEFAULT_RATIO_MARGIN",
    "DEFAULT_SEVERITIES",
    "DEGRADATIONS",
    "DETECTION_DROPOUT_NOTE",
    "LADDER",
    "MINIMUM_PILOT_CLUSTERS",
    "OCR_RECONSTRUCTION_NOTE",
    "PILOT_STATUS_NOTE",
    "PROVENANCE_CHANNEL",
    "AssetFingerprint",
    "ChannelSurvival",
    "ClusterScore",
    "Degradation",
    "DegradationError",
    "LocaleReading",
    "LocaleSummary",
    "LocaleVerdict",
    "OcrCalibrationError",
    "OcrEngine",
    "OcrNullResult",
    "OcrProfile",
    "OcrProfileError",
    "OcrResult",
    "RenderError",
    "RenderSettings",
    "RoundTripResult",
    "TextBlock",
    "apply_chain",
    "apply_degradation",
    "available_engine_ids",
    "available_engines",
    "build_calibration_manifest",
    "build_ladder",
    "build_profile",
    "build_render_settings",
    "character_error_rate",
    "compare_locales",
    "context_corruption_span",
    "evaluate_locale_prediction",
    "fingerprint_assets",
    "get_engine",
    "levenshtein",
    "measure_channel_survival",
    "measure_ocr_null",
    "rapidocr_available",
    "register_engine",
    "render_fingerprint",
    "render_text_to_image",
    "round_trip",
    "screen_recovered_text",
    "select_font",
    "sort_reading_order",
    "summarise_surface",
    "tesseract_available",
    "word_error_rate",
    "write_calibration_artifact",
]
