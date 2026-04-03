-- =============================================================================
-- CouchHire — Supabase (PostgreSQL) Schema
-- Run this in the Supabase SQL Editor to create the applications table.
-- =============================================================================

-- Enable UUID generation if not already enabled
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ---------------------------------------------------------------------------
-- applications — tracks every job application through the pipeline
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS applications (
    -- Primary key
    id              UUID            DEFAULT uuid_generate_v4() PRIMARY KEY,

    -- Input: at least one of jd_text or jd_url should be present.
    -- jd_text is nullable because the user may only provide a link + role title;
    -- the pipeline scrapes the JD later and backfills this column.
    jd_text         TEXT,

    -- Source URL (e.g. Greenhouse, Lever, LinkedIn job link)
    jd_url          TEXT,

    -- Raw role/title the user typed when no full JD was provided
    -- (distinct from `role` which is extracted by jd_parser after parsing)
    role_input      TEXT,

    -- Parsed requirements (stored as JSONB for flexible querying)
    -- Keys: role, company, apply_method, apply_target, subject_line_format,
    --        email_instructions, cover_letter_required, skills, etc.
    requirements    JSONB           NOT NULL DEFAULT '{}'::jsonb,

    -- Company and role extracted from requirements (denormalised for easy querying)
    company         TEXT,
    role            TEXT,

    -- Match scoring
    match_score     NUMERIC(5, 1)   CHECK (match_score >= 0 AND match_score <= 100),

    -- Apply route determined by apply_router agent
    route           TEXT            CHECK (route IN ('email', 'form', 'manual')),

    -- Generated artefacts
    resume_pdf_path TEXT,
    resume_content  TEXT,
    cover_letter    TEXT,
    email_subject   TEXT,
    email_body      TEXT,

    -- Gmail draft
    draft_url       TEXT,

    -- Pipeline status tracking
    status          TEXT            NOT NULL DEFAULT 'pending'
                                   CHECK (status IN (
                                       'pending',         -- just created, pipeline not started
                                       'scraping',        -- scraping JD from jd_url
                                       'parsing',         -- JD being parsed
                                       'scoring',         -- match score being computed
                                       'below_threshold', -- score below MATCH_THRESHOLD, skipped
                                       'tailoring',       -- resume/cover letter being generated
                                       'drafting',        -- email draft being created
                                       'awaiting_review', -- draft ready, waiting for user
                                       'applied',         -- user sent the application
                                       'error'            -- pipeline failed
                                   )),

    -- Outcome label — set by user via Telegram /outcome command
    -- Used by nlp/retrain.py to retrain the match scorer
    outcome         TEXT            CHECK (outcome IN (
                                       'interview',     -- got an interview
                                       'rejected',      -- got a rejection
                                       'no_response',   -- no response after reasonable time
                                       'offer',         -- received an offer
                                       'withdrawn'      -- user withdrew application
                                   )),

    -- Error details (populated when status = 'error')
    error_message   TEXT,

    -- Timestamps
    created_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMPTZ     NOT NULL DEFAULT NOW(),

    -- Constraint: at least one of jd_text, jd_url, or role_input must be present
    -- so we never have a completely empty application row
    CONSTRAINT chk_has_input CHECK (
        jd_text IS NOT NULL OR jd_url IS NOT NULL OR role_input IS NOT NULL
    )
);

-- ---------------------------------------------------------------------------
-- Indexes for common query patterns
-- ---------------------------------------------------------------------------

-- Dashboard: list applications sorted by date
CREATE INDEX IF NOT EXISTS idx_applications_created_at
    ON applications (created_at DESC);

-- Dashboard: filter by status
CREATE INDEX IF NOT EXISTS idx_applications_status
    ON applications (status);

-- Retrain: fetch only labeled outcomes
CREATE INDEX IF NOT EXISTS idx_applications_outcome
    ON applications (outcome)
    WHERE outcome IS NOT NULL;

-- Dashboard: filter by company
CREATE INDEX IF NOT EXISTS idx_applications_company
    ON applications (company);

-- ---------------------------------------------------------------------------
-- Trigger: auto-update updated_at on row modification
-- ---------------------------------------------------------------------------
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Use DROP + CREATE to make the script idempotent
DROP TRIGGER IF EXISTS trg_applications_updated_at ON applications;
CREATE TRIGGER trg_applications_updated_at
    BEFORE UPDATE ON applications
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- ---------------------------------------------------------------------------
-- Row Level Security (RLS)
-- Enable RLS so Supabase anon key can only access via policies.
-- Adjust policies based on your auth setup.
-- ---------------------------------------------------------------------------
ALTER TABLE applications ENABLE ROW LEVEL SECURITY;

-- Allow all operations for authenticated and anon users (single-user app).
-- Tighten this if you add multi-user support later.
DROP POLICY IF EXISTS "Allow all access for anon" ON applications;
CREATE POLICY "Allow all access for anon"
    ON applications
    FOR ALL
    USING (true)
    WITH CHECK (true);
