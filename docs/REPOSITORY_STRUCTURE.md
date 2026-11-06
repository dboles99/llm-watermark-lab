# Repository Structure

```text
llm-watermark-lab/
├── src/watermark_lab/
│   ├── core/                 typed contracts, run engine, registry, statistics
│   ├── detectors/            known keyed and keyless detector plugins
│   ├── watermarkers/         controlled generation-time watermark plugins
│   ├── attacks/              controlled transformations for survival measurement
│   ├── forensics/            Unicode, document-container, and unknown-pattern scans
│   ├── agents/               bounded research-agent integrations
│   ├── services/             model gateway, extraction, reports, background jobs
│   ├── storage/              database models and artifact repositories
│   └── gui/                  PySide6 desktop application
├── crates/watermark-core/    optional Rust acceleration and secure primitives
├── mock-server/              simulated provider API in Docker
├── project/artifacts/        measured evidence
├── research/                 papers, canonical bibliography, provider research
├── notebooks/                bibliography, calibration, robustness, and reports
├── schemas/                  versioned JSON schemas
├── database/migrations/      PostgreSQL schema changes
├── data/                     ignored local corpora and keys
├── artifacts/                ignored run outputs, exports, and plots
├── docs/                     architecture, governance, research, and runbooks
├── tests/                    unit, integration, and fixture tests
├── config/                   safe example configuration
├── scripts/                  build, validation, packaging, and installation tools
├── docker-compose.yml
├── pyproject.toml
└── Cargo.toml
```



