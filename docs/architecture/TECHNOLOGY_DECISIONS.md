# Technology Decisions

## Decision summary

Use **Python as the application and research language**, **Rust as a narrow performance and security core**, **PySide6 for the Windows GUI**, **PostgreSQL for authoritative records**, **Polars/Arrow for analytical data**, and **Jupyter for evaluation notebooks**. Do not write custom C++ in the MVP.

## Recommended stack

| Layer | Recommendation | Why | Alternative or later use |
| --- | --- | --- | --- |
| Desktop GUI | Python 3.12 + PySide6 | Official Qt bindings, strong Windows widgets, file dialogs, tables, plots, and background jobs | Tauri after the engine stabilizes if a web-based UI becomes important |
| Application orchestration | Python, Pydantic, Typer | Fast research iteration, typed configs, good CLI and ML ecosystem | Rust service only if profiling proves Python orchestration is a bottleneck |
| Statistical core | NumPy + SciPy initially | Transparent equations, mature statistical tests, easy paper reproduction | Rust for hot loops after golden-vector parity |
| High-throughput core | Rust + PyO3/maturin | Memory safety, deterministic concurrency, distributable Python extension | C++ only for unavoidable library integration |
| Tokenization | Hugging Face Tokenizers and Transformers fast tokenizers | Rust-backed speed plus token-to-source alignment | `tiktoken` for OpenAI encodings; SentencePiece for models that publish `.model` files |
| Tabular analysis | Polars + Apache Arrow/Parquet | Rust execution engine, parallelism, efficient reproducible run analysis | Pandas retained for notebook interoperability only |
| Local classifier inference | ONNX Runtime | Cross-platform, hardware-aware inference without coupling to PyTorch at runtime | PyTorch for research training and export |
| API clients | OpenAI Python SDK + Azure Identity; generic HTTPX adapter | Supports Azure/OpenAI-compatible APIs while keeping providers outside core logic | Provider-specific adapters only at the boundary |
| Mock API | FastAPI in Docker | Easy typed fixtures, fault injection, streaming later, both API envelope shapes | Rust Axum only if the mock becomes a heavy load simulator |
| Database | PostgreSQL 17 + SQLAlchemy + Alembic | JSONB, transactions, indexing, audit queries, mature migrations | SQLite only for explicitly portable demo mode |
| Secrets | Azure Key Vault in Azure; Windows Credential Manager or DPAPI locally | Keys stay out of TOML, logs, notebooks, and Git | A dedicated Vault service if deployment expands |
| Packaging | PyInstaller one-folder build first | Easier diagnostics and dependency handling than one-file bundles | MSIX or WiX MSI after stable release builds |
| Test/quality | pytest, Hypothesis, Ruff, mypy, cargo test, clippy | Statistical property tests plus language-specific checks | Add mutation testing after baseline coverage |
| Observability | Python logging with JSON formatter and OpenTelemetry interfaces | Local audit trail and later Azure integration | Azure Monitor exporter only in deployed environments |

## Why not all Rust now?

Watermark research changes quickly. Python makes equations, token inspection, notebooks, plotting, model clients, and paper reproduction much faster to iterate. Rewriting every experimental method in Rust early would slow scientific validation and increase the chance of implementing the wrong algorithm efficiently.

Rust should begin with narrow, measurable components:

- keyed pseudorandom generation
- token-class partitioning
- rolling/context hash computation
- dynamic-programming alignment
- passage-window scoring
- secure key-buffer handling
- batch detection over large corpora

Each Rust function must first have a Python reference implementation and golden parity tests.

## Why no custom C++?

The useful C++ already arrives inside Qt, ONNX Runtime, PyTorch, and some tokenizer libraries. Custom C++ would add build complexity, undefined-behavior risk, and a third implementation language without a clear product advantage. If a paper releases only a C++ reference implementation, wrap it behind a subprocess or stable C ABI after security review rather than spreading C++ through the application.

## Tokenizer policy

Watermark detection is tokenizer-sensitive, so “best tokenizer” is the wrong abstraction. Store the exact tokenizer family, files, revision, normalization policy, special-token settings, and hash with each run.

- Use Hugging Face fast tokenizers when matching open model tokenizers.
- Use `tiktoken` only for documented OpenAI encodings.
- Use SentencePiece for models that publish SentencePiece assets.
- Add proprietary tokenizers only through authorized provider adapters.
- Never silently substitute one tokenizer for another.

## API design

The core engine should use local typed interfaces, not HTTP. Add a headless FastAPI layer around it for automation and future web clients:

- `POST /v1/intake`
- `POST /v1/runs/detect`
- `POST /v1/runs/create`
- `POST /v1/runs/transform`
- `POST /v1/runs/compare`
- `GET /v1/runs/{run_id}`
- `GET /v1/runs/{run_id}/artifacts`
- `GET /v1/methods`

Long runs should use a job ID and polling or server-sent events. Do not make detector plugins depend on FastAPI objects.

## Non-Anthropic model ensemble

Azure OpenAI, GPT-5.6 Luna, Kimi 2.7 Code, GLM 5.2, and later models are provider profiles behind one gateway. They can propose hypotheses, generate attacks, or supply experimental classifier features. Store model name, deployment, version, sampling settings, prompt hash, and raw response hash for reproducibility.

Their prose judgments are not detector scores. Only deterministic features validated on held-out data may enter the calibrated decision layer.

## Build order

1. Python reference algorithms, run manifests, tests, GUI intake, and Docker mock.
2. KGW end-to-end run and calibration report.
3. Rust core for profiled bottlenecks with parity tests.
4. Provider-neutral ensemble and Azure adapter.
5. MCmark-inspired reproduction.
6. MirrorMark-inspired reproduction.
7. Windows executable and installer hardening.
8. Optional real Anthropic adapter only after the app is stable and an authorized detection interface exists.

## Current primary documentation checked

- Qt for Python: https://doc.qt.io/qtforpython-6/
- Hugging Face Tokenizers: https://huggingface.co/docs/tokenizers/en/index
- PyO3: https://pyo3.rs/
- ONNX Runtime: https://onnxruntime.ai/docs/
- Polars: https://docs.pola.rs/
- SQLAlchemy PostgreSQL: https://docs.sqlalchemy.org/en/latest/dialects/postgresql.html
- Azure OpenAI Responses API: https://learn.microsoft.com/en-us/azure/foundry/openai/how-to/responses
- PyInstaller: https://pyinstaller.org/

