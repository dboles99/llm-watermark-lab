# Scholarly Evidence Admission Policy

## Rule

A publication may support a scientific claim, method choice, baseline, prior-art boundary, or experiment only when all of the following are present:

1. A clear bibliographic identity: exact title, authors, year, venue/status, and at least one stable identifier such as a DOI, versioned arXiv identifier, or archival venue identifier.
2. A directly downloadable full-text PDF whose title and authors match the bibliographic record.
3. A resolvable record in at least one scholarly index: Crossref, OpenAlex, Semantic Scholar, Google Scholar, DBLP, PubMed, or another named scholarly index approved by the research gate.
4. A recorded evidence tier, version/status, verification date, and permitted use.

The machine-readable admitted set is `research/frontier_evidence.json`, validated by `schemas/literature-evidence.schema.json`.

## Tiers

- `core`: peer-reviewed archival work that may support primary claims, baselines, and preregistered hypotheses.
- `adjacent`: peer-reviewed work from statistics, cryptography, provenance, privacy, or another directly relevant field. It may support construction choices, but not a claim about text-watermark performance unless it evaluates that setting.
- `watchlist`: an indexed preprint with a stable version and direct PDF. It may identify possible overlap or motivate a clearly labeled exploratory hypothesis. It may not establish state of the art, performance superiority, or novelty.

Unindexed submissions, search-result-only manuscripts, blogs, videos, social posts, inaccessible abstracts, and records without a direct PDF are excluded rather than cited with a caveat.

## Claim discipline

- Separate author claims, reproduced project findings, project inferences, and hypotheses.
- Never infer novelty from a targeted search. Use: `No directly equivalent system was found in the admitted corpus as of YYYY-MM-DD.`
- Never claim `first`, `unique`, `unprecedented`, or `no one has done this` without a separate systematic prior-art and patent search.
- A standard or provider document is authoritative only for its own specification or product behavior. It is not scholarly literature and is stored outside the literature ledger.
- Every experimental method must cite the exact paper version and author-code revision used.

## Admission workflow

1. The literature curator creates a candidate record and verifies the stable identifier, direct PDF, and index record.
2. The curator hashes the acquired PDF in the canonical evidence bundle and records the retrieval date and license.
3. A blind bibliography auditor independently resolves the identifier and compares the PDF title/authors/version.
4. The root orchestrator validates the ledger and records the gate. Failed candidates are omitted from evidence outputs; they are not silently retained as informal citations.
5. Version changes create a superseding record. Published or promotion-consumed evidence is never amended in place.

## Automated minimum checks

- unique `record_id`;
- no duplicate DOI or versioned arXiv identifier;
- every record is schema-valid;
- every PDF URL resolves as a PDF;
- every index URL resolves and describes the same work;
- `watchlist` records are not cited as conclusive evidence;
- retrieval and source checks use no `zh-CN` fallback or mechanically converted `zh-CN` corpus.
