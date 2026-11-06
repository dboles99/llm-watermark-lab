# Platform-Neutral Unknown-Pattern Forensics

## Purpose

The unknown-pattern scanner searches for hidden or unexplained structure that is not tied to Claude, DeepL, OpenAI, or another provider. It is a forensic discovery layer. A finding means “this feature is unusual under the selected controls,” not “this text is AI-generated.”

## Analysis families

### Unicode and character-level checks

- zero-width space, joiner, non-joiner, word joiner, byte-order mark, soft hyphen, variation selectors, tag characters, and private-use characters;
- bidirectional controls and isolate markers;
- homoglyph and mixed-script confusables;
- noncanonical Unicode normalization and normalization-sensitive offsets;
- unusual nonbreaking spaces, thin spaces, ideographic spaces, tabs, trailing spaces, and line endings;
- repeated code-point or byte patterns and encoded binary channels.

### Document-container checks

- Word hidden-text properties and white-on-white text;
- comments, tracked changes, fields, content controls, bookmarks, custom XML, macros, relationships, embedded files, and external links;
- document properties, custom properties, author/software metadata, timestamps, and revision identifiers;
- header/footer VML watermarks and page-layer objects;
- PDF metadata, annotations, optional content groups, invisible text layers, JavaScript, attachments, and OCR overlays.

### Linguistic and structural checks

- acrostics, telestics, first-letter and last-letter channels;
- suspicious sentence-length, paragraph-length, punctuation, capitalization, and synonym schedules;
- periodic lexical classes, alternating word choices, unusual function-word distributions, and constrained paraphrase patterns;
- repeated n-gram contexts, low-entropy spans, and discontinuities between adjacent passages;
- language-model tokenizer anomalies under multiple public tokenizer families;
- stylometric shifts that may indicate mixed sources, translation, editing, or templating.

### Statistical discovery

- autocorrelation and spectral analysis of token or character features;
- runs tests, permutation tests, entropy, compression ratios, and change-point detection;
- multiple tokenizer views and sliding-window localization;
- multiple-testing correction and holdout confirmation;
- matched baselines by language, domain, author group, file type, and editing history.

## Finding classes

| Finding | Meaning | Permitted conclusion |
| --- | --- | --- |
| Deterministic hidden character | Exact code point or property is present | Report location |
| Container anomaly | Hidden or unusual file structure is present | Report object, relationship, or metadata |
| Statistical anomaly | Feature is unlikely under the matched control | Open a hypothesis and holdout test |
| Reproduced covert channel | Encoding and decoding are demonstrated on controlled data | Report compatible channel, not original author |
| Unknown unexplained pattern | Pattern persists but mechanism is unverified | Preserve evidence and state uncertainty |

## Research-agent workflow

1. Detect and localize the anomaly.
2. Reproduce it on a byte and document-object level.
3. Test common benign explanations.
4. Build matched controls.
5. Propose measurable mechanisms.
6. Confirm on holdout samples.
7. Record the result as deterministic, statistical, reproduced, or unresolved.

