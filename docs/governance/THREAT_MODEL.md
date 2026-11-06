# Threat Model and Responsible-Use Boundary

## Protected assets

- Secret watermark keys and rotation history
- Original and normalized third-party text
- Project/payload mappings
- Calibration corpora and threshold policies
- Reproducibility records and exported findings

## Adversaries and failure modes

- An editor accidentally adds a watermark while proofreading human text.
- An attacker removes a watermark by paraphrase, translation, mixing, or token substitution.
- An attacker spoofs a known public watermark to frame a human author.
- A user overinterprets a weak or uncalibrated score.
- A research agent hallucinates compatibility with a proprietary provider.
- Keys leak through logs, notebooks, prompts, screenshots, or Git.

## Controls

- Encrypt keys at rest and refer to them only by opaque key ID.
- Separate creator and verifier roles.
- Require explicit disclosure before watermarking user-facing output.
- Maintain a tamper-evident run ledger with input and artifact hashes.
- Use matched negative controls and language/domain-specific calibration.
- Display “inconclusive” as a first-class outcome.
- Prohibit automated misconduct, hiring, grading, or disciplinary decisions.
- Sandbox research agents and deny direct key-store access.

## Forensic analysis boundary

Forensic analysis may use public papers, public model outputs, project-owned watermarks, and authorized black-box experiments. It must not extract provider secrets, bypass access controls, impersonate a provider verifier, or make unsupported attribution claims.

