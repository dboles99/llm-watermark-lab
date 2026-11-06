# Research Workspace

`frontier_evidence.json` is the canonical admitted scholarly ledger and must validate against `schemas/literature-evidence.schema.json`. Every admitted record requires a clear persistent citation, a scholarly index record, and a directly downloadable PDF under `docs/research/EVIDENCE_ADMISSION_POLICY.md`.

`annotated_bibliography.csv` and `ANNOTATED_BIBLIOGRAPHY.md` are the broader legacy discovery bibliography. A record appearing there is not automatically admitted. `provider_sources.csv` separately records official provider statements that are not scholarly literature.

Public paper PDFs may be placed in `research/papers/` for local authorized use. Record their version and SHA-256 in the paper-ingestion task. Do not commit copyrighted PDFs to a public repository unless redistribution is permitted.
