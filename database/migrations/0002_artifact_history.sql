-- Add reproducible provenance and governed artifact history.
-- PostgreSQL remains the deployment target; the Python storage package also provides a
-- dependency-free SQLite adapter for local use.

ALTER TABLE runs
    ADD COLUMN IF NOT EXISTS environment JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS model_provenance JSONB,
    ADD COLUMN IF NOT EXISTS rosetta_id TEXT,
    ADD COLUMN IF NOT EXISTS rosetta_sha256 CHAR(64),
    ADD COLUMN IF NOT EXISTS executed_commands JSONB NOT NULL DEFAULT '[]'::jsonb;

ALTER TABLE artifacts
    ADD COLUMN IF NOT EXISTS lineage_id UUID,
    ADD COLUMN IF NOT EXISTS supersedes_artifact_id UUID REFERENCES artifacts(id),
    ADD COLUMN IF NOT EXISTS version INTEGER NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS byte_size BIGINT NOT NULL DEFAULT 0,
    ADD COLUMN IF NOT EXISTS change_reason TEXT,
    ADD COLUMN IF NOT EXISTS frozen BOOLEAN NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS created_at TIMESTAMPTZ NOT NULL DEFAULT now();

UPDATE artifacts SET lineage_id = id WHERE lineage_id IS NULL;
ALTER TABLE artifacts ALTER COLUMN lineage_id SET NOT NULL;

ALTER TABLE artifacts
    ADD CONSTRAINT artifacts_version_positive CHECK (version >= 1),
    ADD CONSTRAINT artifacts_byte_size_nonnegative CHECK (byte_size >= 0),
    ADD CONSTRAINT artifacts_sha256_format CHECK (sha256 ~ '^[a-f0-9]{64}$'),
    ADD CONSTRAINT artifacts_canonical_path CHECK (
        path LIKE 'project/artifacts/' || run_id::text || '/%'
    ),
    ADD CONSTRAINT artifacts_supersession_shape CHECK (
        (version = 1 AND supersedes_artifact_id IS NULL)
        OR (version > 1 AND supersedes_artifact_id IS NOT NULL)
    ),
    ADD CONSTRAINT artifacts_not_self_superseding CHECK (supersedes_artifact_id IS DISTINCT FROM id),
    ADD CONSTRAINT artifacts_unique_path UNIQUE (path),
    ADD CONSTRAINT artifacts_unique_predecessor UNIQUE (supersedes_artifact_id);

CREATE INDEX artifacts_lineage_version_idx ON artifacts (lineage_id, version);

CREATE TABLE artifact_amendments (
    id UUID PRIMARY KEY,
    artifact_id UUID NOT NULL REFERENCES artifacts(id),
    sequence INTEGER NOT NULL CHECK (sequence >= 1),
    before_sha256 CHAR(64) NOT NULL CHECK (before_sha256 ~ '^[a-f0-9]{64}$'),
    after_sha256 CHAR(64) NOT NULL CHECK (after_sha256 ~ '^[a-f0-9]{64}$'),
    before_byte_size BIGINT NOT NULL CHECK (before_byte_size >= 0),
    after_byte_size BIGINT NOT NULL CHECK (after_byte_size >= 0),
    reason TEXT NOT NULL CHECK (length(btrim(reason)) > 0),
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT artifact_amendments_hash_changed CHECK (before_sha256 <> after_sha256),
    CONSTRAINT artifact_amendments_sequence_unique UNIQUE (artifact_id, sequence)
);

CREATE OR REPLACE FUNCTION prevent_run_lineage_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.parent_run_id IS DISTINCT FROM OLD.parent_run_id THEN
        RAISE EXCEPTION 'run parent lineage is immutable';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER runs_immutable_lineage
BEFORE UPDATE ON runs
FOR EACH ROW EXECUTE FUNCTION prevent_run_lineage_mutation();

CREATE OR REPLACE FUNCTION enforce_artifact_immutability()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.run_id IS DISTINCT FROM OLD.run_id
       OR NEW.lineage_id IS DISTINCT FROM OLD.lineage_id
       OR NEW.supersedes_artifact_id IS DISTINCT FROM OLD.supersedes_artifact_id
       OR NEW.version IS DISTINCT FROM OLD.version
       OR NEW.path IS DISTINCT FROM OLD.path THEN
        RAISE EXCEPTION 'artifact lineage and path are immutable';
    END IF;
    IF OLD.frozen AND (
        NEW.sha256 IS DISTINCT FROM OLD.sha256
        OR NEW.byte_size IS DISTINCT FROM OLD.byte_size
        OR NEW.metadata IS DISTINCT FROM OLD.metadata
    ) THEN
        RAISE EXCEPTION 'frozen artifacts may only be superseded';
    END IF;
    IF OLD.frozen AND NOT NEW.frozen THEN
        RAISE EXCEPTION 'an artifact cannot be unfrozen';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifacts_immutable_lineage
BEFORE UPDATE ON artifacts
FOR EACH ROW EXECUTE FUNCTION enforce_artifact_immutability();

CREATE OR REPLACE FUNCTION validate_artifact_amendment()
RETURNS trigger LANGUAGE plpgsql AS $$
DECLARE
    current_artifact artifacts%ROWTYPE;
BEGIN
    SELECT * INTO current_artifact FROM artifacts WHERE id = NEW.artifact_id FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'artifact does not exist';
    END IF;
    IF current_artifact.frozen THEN
        RAISE EXCEPTION 'frozen artifacts may only be superseded';
    END IF;
    IF NEW.before_sha256 IS DISTINCT FROM current_artifact.sha256
       OR NEW.before_byte_size IS DISTINCT FROM current_artifact.byte_size THEN
        RAISE EXCEPTION 'amendment before-state does not match current artifact';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER artifact_amendments_validate_insert
BEFORE INSERT ON artifact_amendments
FOR EACH ROW EXECUTE FUNCTION validate_artifact_amendment();

CREATE OR REPLACE FUNCTION prevent_amendment_mutation()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    RAISE EXCEPTION 'artifact amendment entries are append-only';
END;
$$;

CREATE TRIGGER artifact_amendments_append_only
BEFORE UPDATE OR DELETE ON artifact_amendments
FOR EACH ROW EXECUTE FUNCTION prevent_amendment_mutation();
