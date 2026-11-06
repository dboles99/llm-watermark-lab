CREATE TABLE projects (
    id UUID PRIMARY KEY,
    name TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE samples (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    source_path TEXT,
    media_type TEXT NOT NULL,
    input_sha256 CHAR(64) NOT NULL,
    normalized_sha256 CHAR(64),
    language TEXT,
    token_count INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE runs (
    id UUID PRIMARY KEY,
    parent_run_id UUID REFERENCES runs(id),
    sample_id UUID REFERENCES samples(id),
    run_type TEXT NOT NULL,
    method_id TEXT NOT NULL,
    compatibility TEXT NOT NULL,
    key_id TEXT,
    config JSONB NOT NULL,
    status TEXT NOT NULL,
    software_version TEXT NOT NULL,
    git_commit TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE measurements (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id),
    metric TEXT NOT NULL,
    value DOUBLE PRECISION,
    unit TEXT,
    passage_start INTEGER,
    passage_end INTEGER,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE artifacts (
    id UUID PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES runs(id),
    artifact_type TEXT NOT NULL,
    path TEXT NOT NULL,
    sha256 CHAR(64) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE TABLE research_hypotheses (
    id UUID PRIMARY KEY,
    project_id UUID REFERENCES projects(id),
    observation TEXT NOT NULL,
    proposed_mechanism TEXT NOT NULL,
    test_plan TEXT NOT NULL,
    result TEXT,
    alternative_explanation TEXT,
    evidence_strength TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

