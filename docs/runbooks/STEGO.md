# Steganographic Watermark Runbook

## Purpose

This runbook describes how to hide, recover, and verify secret messages inside
ordinary text using the `watermark_lab.stego` package, and how to use the
iterative stochastic tinker workflow to probe channel robustness.

Steganographic watermarks are dual-purpose: they (a) hide a payload so that
an observer cannot tell that anything is hidden, and (b) bind the payload
with an HMAC integrity tag so that any tampering or injection attack is
detected by the holder of the shared secret. They are *not* a substitute
for encryption — the cover text is public and the payload is recoverable by
anyone who holds the secret — but they defeat unnoticed tampering.

## Channels

Three pluggable channels are implemented under `src/watermark_lab/stego/channels/`:

| Channel | Carrier | Capacity | Stealth | Robustness |
| --- | --- | --- | --- | --- |
| `whitespace` | space/tab pattern after each word | 2 bits / word | Low (visible in monospace) | High (survives copy/paste) |
| `unicode` | `U+200B` / `U+200C` zero-width chars | 1 bit / character | High (invisible) | Low (stripped by many editors) |
| `token_choice` | KGW green/red-list token selection | 1 bit / token | Medium (numeric token covers) | Medium (KGW z-score survives) |

Pick the channel that matches your threat model:

- For long cover text where you can afford visible whitespace artefacts,
  `whitespace` is the most robust against reformatting.
- For short cover text or where the user will see the rendered output,
  `unicode` is invisible but should not be sent through any tool that
  normalizes Unicode.
- For already-tokenized numeric text (or as a research bridge into the KGW
  pipeline), `token_choice` reuses the KGW green-list key as the stego key,
  giving a watermark + steganography dual.

## Integrity envelope

Every payload is wrapped in one of two frames before embedding:

- **Full frame** (default): `magic(4) | version(1) | len(4) | payload | nonce(4) | tag(4)`
  with a PBKDF2-derived key and a 32-bit truncated HMAC-SHA256 tag.
- **Short frame** (auto-selected when the channel cannot fit the full frame):
  `magic(4) | version(1) | len(2) | payload | tag(4)` with the caller's
  shared secret used directly as the HMAC key.

Both frames carry the magic bytes `UEST` so the decoder can locate the
envelope inside a longer bit string. The decoder scans backward for the
magic so trailing channel noise is ignored.

## CLI usage

Activate the environment and install the package:

```powershell
.venv\Scripts\Activate.ps1
uv pip install -e .
```

### Encode

```powershell
watermark-lab stego-encode `
  --text "A long cover text with enough words to hold the short envelope safely" `
  --payload "hi" `
  --secret "shared-secret" `
  --channel whitespace
```

For the `token_choice` channel, pass the KGW parameters:

```powershell
watermark-lab stego-encode `
  --text "0 1 2 3 4 5 ... 119" `
  --payload "ok" `
  --secret "shared-secret" `
  --channel token_choice `
  --key-id test-kgw-v1 `
  --vocab-size 50257 `
  --gamma 0.25 `
  --hash-key 15485863
```

### Decode

```powershell
watermark-lab stego-decode `
  --text "<stego text>" `
  --secret "shared-secret" `
  --channel whitespace
```

The decoded payload is printed as UTF-8 text. If the HMAC tag does not
verify, `status` is `error` and `warnings` explains the failure.

### Verify without revealing the payload

```powershell
watermark-lab stego-verify `
  --text "<stego text>" `
  --secret "shared-secret" `
  --channel whitespace
```

Returns `verified`, `forged`, or `inconclusive` without printing the
payload bytes. Use this when you only need to prove integrity.

### Tinker workflow

```powershell
watermark-lab stego-tinker `
  --text "<cover>" `
  --payload "ok" `
  --secret "shared-secret" `
  --channel whitespace `
  --generations 3 `
  --population-size 6 `
  --mutation-rate 0.15
```

This runs a small evolutionary search: each generation encodes the payload
into a candidate cover, perturbs the stego text, decodes it, and scores
the variant by recoverability (45%), integrity (30%), stealth (15%), and
channel-specific robustness (10%). The top half of each generation seeds
the next. The output is a JSON summary with the overall best variant and
every variant tried.

## Library usage

```python
from watermark_lab.stego import StegoDecoder, StegoEncoder, StegoTinkerer

encoder = StegoEncoder(channel_id="whitespace", secret=b"shared-secret")
result = encoder.encode(cover_text, "hidden payload")
if result.status != "ok":
    raise RuntimeError(result.warnings)

stego_text = result.text

decoder = StegoDecoder(channel_id="whitespace", secret=b"shared-secret")
decoded = decoder.decode(stego_text)
assert decoded.status == "ok"
assert decoded.payload == b"hidden payload"

# Verify only — payload is not returned.
verify = decoder.verify(stego_text)
assert verify.status == "verified"
```

For `token_choice`, production callers should pass the raw key directly so
the key material does not flow through the test key vault:

```python
encoder = StegoEncoder(
    channel_id="token_choice",
    secret=b"hmac-secret",
    key=b"kgw-watermark-key",
    key_id="production",
)
```

## Anti-injection guarantees

The HMAC tag binds the payload to the shared secret. The decoder fails
closed on any of the following:

1. **Wrong secret** — `Integrity check failed: forged or tampered message`.
2. **Truncated envelope** — `Envelope parse failed: Envelope truncated`.
3. **Wrong magic** — `Envelope parse failed: Invalid envelope magic`.
4. **Bit flip in a carrier** — usually surfaces as a truncated envelope or
   a failed HMAC tag depending on where the flip lands.
5. **Appended attacker stego text** — the decoder scans backward for the
   magic, finds the attacker's envelope, and tries to verify it under the
   victim's secret; the tag check fails.

If an attacker can replace the entire stego text with their own envelope
under their own secret, the decoder still rejects it under the victim's
secret. There is no scenario in which an attacker without the shared
secret can produce text that verifies under the victim's secret short of
an HMAC forgery, which is computationally infeasible for SHA-256.

## Governance

- All cover text is synthetic, self-owned, or explicitly authorized.
- The shared secret is a bearer credential: anyone who holds it can read
  and forge stego text under that secret. Move secrets to a real key
  store before production use.
- The `token_choice` channel reuses the KGW key as the stego key. If the
  KGW key is disclosed, the stego channel is also broken. Keep these keys
  separate in production by passing `key=` explicitly.
- Do not claim that steganographic watermarks are encryption. They are a
  tamper-evidence mechanism, not a confidentiality mechanism.

## Tests

- `tests/stego/test_stego.py` — 13 tests covering roundtrips for all
  three channels, wrong-secret rejection, capacity errors, tamper
  detection, injection-attack rejection, the tinker workflow, and
  raw-key support for `token_choice`.