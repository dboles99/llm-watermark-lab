# DeepL Watermark and Provenance Research Note

## Current finding as of August 14, 2026

DeepL publicly documents an optional **visible document watermark** for file translation. The API parameter `enable_watermark=true` adds diagonal text reading “Translated by DeepL” to DOCX and PDF output. This is a page-level document mark, not evidence of an invisible statistical watermark embedded in translated word choices.

Primary documentation:

- https://developers.deepl.com/api-reference/document/upload-and-translate-a-document
- https://support.deepl.com/hc/en-us/articles/28874150283164-Add-a-watermark-to-your-file-translation

No primary source located in the current review documents a secret DeepL token-distribution watermark, a public DeepL text-verification detector, C2PA assertions for translated text, or a method that attributes pasted plain text to DeepL. Absence of public documentation is not proof that no internal markers exist. It means the application must classify invisible DeepL attribution as unsupported.

## Application treatment

- Detect the visible “Translated by DeepL” document watermark and its page-layer object when present.
- Record the `enable_watermark` API option in authorized DeepL translation manifests.
- Preserve DeepL document metadata and translation-memory identifiers when available and permitted.
- Treat DeepL translation and back translation as named transformations in watermark stress tests.
- Build a blinded DeepL-versus-control corpus only for keyless research and stylometry, with explicit false-positive warnings.
- Do not call a style classifier a DeepL watermark detector.

## Research queue

1. Monitor DeepL developer documentation, security material, release notes, patents, and academic publications.
2. Test whether visible watermarks survive DOCX-to-PDF, PDF-to-DOCX, copy-paste, print-to-PDF, and OOXML cleanup.
3. Inspect authorized DeepL outputs for metadata and document-object changes before analyzing prose.
4. Compare DeepL translation with other translation systems under the same source, language pair, glossary, formality, and document settings.
5. Evaluate how DeepL translation transforms known KGW, MCmark-inspired, and MirrorMark-inspired marks.

