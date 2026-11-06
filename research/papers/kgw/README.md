# KGW Paper Workspace

This directory holds the pinned primary papers for the KGW-style watermark adapter.

## Papers

1. Kirchenbauer et al., *A Watermark for Large Language Models*, ICML 2023, arXiv:2301.10226.
2. Kirchenbauer et al., *On the Reliability of Watermarks for Large Language Models*, ICLR 2024, arXiv:2306.04634.

The reliability paper is admitted as `KGW_RELIABILITY_2024` in the canonical
ledger, with the final ICLR PDF and DBLP record. The author implementation is
pinned separately at commit `82922516930c02f8aa322765defdb5863d07a00e` under
Apache-2.0. Repository code is implementation evidence, not a substitute for a
scholarly paper.

## Do not commit

- Copyrighted PDFs without permission.
- Watermark keys or API keys.

## How to update the manifest

After adding a PDF, compute its SHA-256 and update `manifest.json`:

```powershell
Get-FileHash -Path "research/papers/kgw/kirchenbauer-2023-watermark-arxiv-2301.10226.pdf" -Algorithm SHA256
```

Then copy the hash into `manifest.json` and change `status` to `papers-pinned`.
