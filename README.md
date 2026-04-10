# CouchHire

> Job applications, automated. You stay on the couch.

CouchHire is a fully agentic job application pipeline. Paste a job description, approve the generated application via Telegram, and it fires — tailored resume, cover letter, email, and ATS form filling all handled automatically. A self-improving NLP match scorer learns from your personal outcomes over time.

---

## Table of Contents

1. [How It Works](#how-it-works)
2. [Tech Stack](#tech-stack)
3. [Project Structure](#project-structure)
4. [Agent Responsibilities & Interfaces](#agent-responsibilities--interfaces)
5. [Pipeline Flow](#pipeline-flow)
6. [Prerequisites](#prerequisites)
7. [Getting Your API Keys](#getting-your-api-keys)
8. [Installation](#installation)
9. [Configuration](#configuration)
10. [Database Setup (Supabase)](#database-setup-supabase)
11. [CV Setup](#cv-setup)
12. [Running the Project](#running-the-project)
13. [Docker (Alternative)](#docker-alternative)
14. [Self-Improving Loop](#self-improving-loop)
15. [Streamlit Dashboard](#streamlit-dashboard)
16. [Roadmap](#roadmap)
17. [Contributing](#contributing)

---

## How It Works

1. A job description comes in — either pasted manually, discovered via job board search (JobSpy), or scraped from a URL.
2. The JD Parser extracts structured requirements: skills, apply method, cover letter needed, subject line format.
3. The CV RAG agent retrieves the most relevant sections of your master CV from ChromaDB.
4. The Match Scorer computes a fit percentage using sentence-transformers. If below the configured threshold, the job is skipped or flagged.
5. If the match passes, three agents run in parallel: Resume Tailor (compiles a tailored PDF via pdflatex), Cover Letter (if required), and Email Drafter.
6. A Telegram notification arrives with a summary card — company, role, match %, apply route — and Approve / Edit buttons.
7. On approval, the Apply Router detects the method: Gmail MCP for email drafts, semi-autonomous browser agent for ATS forms (with LLM-assisted field mapping and human-in-the-loop interrupts), or a manual-apply alert.
8. The application is logged in Supabase. You later tap an outcome button in Telegram (No Reply / Screening / Interview / Rejected / Offer).
9. Every 10 new outcome labels, the match scorer automatically retrains on your personal history.

---

## Tech Stack

| Layer | Tool | Notes |
|---|---|---|
| Agent orchestration | LangGraph | Multi-agent workflows, graph-based state machine |
| LLM abstraction | LiteLLM | Swap Groq / Gemini / Anthropic / OpenAI with one line in config.py |
| CV vector store | ChromaDB (local) | Master CV chunked by section and embedded locally |
| NLP | spaCy + sentence-transformers | NER for skill extraction, cosine similarity for match scoring |
| Database | Supabase (PostgreSQL) | Applications table, outcome labels, retraining data |
| Job search | JobSpy (python-jobspy) | Multi-board job scraping: Indeed, LinkedIn, Glassdoor, Google, ZipRecruiter |
| Email drafts | Gmail MCP ([google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp)) | Creates Gmail draft via `draft_gmail_message` tool with resume PDF attachment (never auto-sends) |
| Browser automation | Playwright | Semi-autonomous ATS form filling via CDP (`connect_over_cdp()`), LLM-assisted field mapping, human-in-the-loop interrupt system via Telegram |
| Notifications + review | Telegram bot (python-telegram-bot) | Approve / edit / label outcomes from your phone |
| Dashboard | Streamlit | Application tracker, analytics, retrain controls, settings |
| Resume compiler | pdflatex (local) | Compiles tailored .tex → PDF on the user's machine |

---

## Project Structure

```
couchhire/
├── agents/           # LangGraph agents + resume generation helpers
│   ├── jd_parser.py          # JD parsing + requirements extraction → requirements dict
│   ├── cv_rag.py             # ChromaDB retrieval from master CV → ranked CV sections
│   ├── match_scorer.py       # NLP match scoring → float 0–100
│   ├── resume_tailor.py      # LaTeX resume tailoring → compiled PDF path
│   ├── llm_selector.py       # LLM-driven content selection + instruction enforcement (internal helper)
│   ├── resume_assembler.py   # LaTeX block extraction + itemize wrapper utilities (internal helper)
│   ├── cv_content_helpers.py # Section → named block extraction + content inventory builder (internal helper)
│   ├── cover_letter.py       # Cover letter generation (conditional, receives resume_content) → string
│   ├── email_drafter.py      # Email draft generation (receives resume_content for project-specific context) → subject + body strings
│   └── apply_router.py       # Detects email / form / manual route → route string
├── apply/            # Gmail draft creator (MCP client), Playwright browser agent, session handoff
│   ├── gmail_sender.py       # MCP client: create_draft() + send_draft() via draft_gmail_message / send_gmail_message tools (google_workspace_mcp)
│   ├── browser_agent.py      # Semi-autonomous ATS form filler (LLM field mapping + blocker detection + Telegram interrupts)
│   └── session_handoff.py    # CDP session manager (launches Chromium with --remote-debugging-port for Playwright + manual takeover)
├── bot/              # Telegram bot
│   └── telegram_bot.py       # Notifications, /outcome, /search, /apply, gate handlers
├── jobs/             # JobSpy job search + CV-based filtering
│   ├── job_search.py         # JobSpy wrapper: multi-board concurrent search (Indeed, LinkedIn, Google, ZipRecruiter)
│   └── job_filter.py         # Score + filter results against user's CV
├── dashboard/        # Streamlit app (Step 26 — not yet built)
│   └── app.py                # Streamlit dashboard (reads from Supabase)
├── db/               # Supabase client + schema
│   ├── supabase_client.py    # CRUD helpers for applications table
│   ├── schema.sql            # Full CREATE TABLE (idempotent) — run this first in Supabase
│   └── create_tables.py      # One-time table creation script (direct Postgres or Management API fallback)
├── nlp/              # spaCy NER model + retrain loop
│   ├── ner_model.py          # spaCy NER for JD skill extraction
│   └── retrain.py            # Fine-tune match scorer on outcome labels (auto-triggers every N labels)
├── llm/              # LiteLLM wrapper (all LLM calls go through here)
│   └── client.py             # complete() with 9-model fallback chain across 5 providers
├── cv/
│   ├── uploads/              # User's personal files — gitignored
│   │   ├── master_cv.*       # .tex, .pdf, or .docx — any format accepted
│   │   ├── resume_template.tex       # Optional custom LaTeX template
│   │   ├── cover_letter_template.tex # Optional custom cover letter template
│   │   └── instructions.md  # Optional tailoring preferences
│   ├── chroma_store/         # ChromaDB embeddings — gitignored
│   ├── output/               # Compiled PDFs + .tex intermediates — gitignored
│   ├── embed_cv.py           # Orchestrates parse → embed pipeline
│   ├── cv_parser.py          # Parses .tex/.pdf/.docx into named sections
│   └── defaults/             # Fallback templates and instructions — committed
│       ├── resume_template.tex
│       ├── cover_letter_template.tex  # XeLaTeX template with {{PLACEHOLDER}} markers, %%IF_<KEY>%% conditional blocks, %%FONT_PLACEHOLDER%% marker
│       └── instructions.md
├── pipeline.py               # LangGraph orchestrator — main entry point (19 nodes, 6 conditional edges)
├── generate_resume.py        # Standalone resume generator CLI (skips email/Telegram)
├── config.py                 # Reads .env, validates all keys on startup, exposes config
├── .env                      # Your secrets — never committed to git
├── .env.example              # Template with all variable names — committed to git
├── .gitignore                # Excludes .env, cv/uploads/, cv/chroma_store/, cv/output/, compiled PDFs
├── requirements.txt          # Pinned Python dependencies
├── tests/                    # pytest suite (test_step25, test_llm_selector, test_skills_assembly, etc.)
├── test_jds/                 # Sample JDs for testing (7 role types)
└── docker-compose.yml        # (Step 28 — not yet built)
```

---

## Agent Responsibilities & Interfaces

This section is intentionally precise so that each module has a clear contract. Every agent reads from and writes to a shared `state` dict that LangGraph passes through the pipeline.

### `agents/jd_parser.py`
- **Input:** raw JD text (string)
- **Output:** `requirements` dict with keys:
  - `company` (str), `role` (str), `skills` (list[str])
  - `apply_method` (str): `"email"` | `"url"` | `"unknown"`
  - `apply_target` (str): email address or URL
  - `cover_letter_required` (bool)
  - `subject_line_format` (str | None)
  - `github_requested` (bool)
  - `form_fields` (list[str])
- **LLM call:** yes — uses `llm/client.py`
- **Stores to Supabase:** yes — `requirements` column as JSONB

### `agents/cv_rag.py`
- **Input:** `requirements` dict
- **Output:** `cv_sections` — list of strings, ranked by relevance to JD
- **ChromaDB:** reads from `cv/chroma_store/`
- **No LLM call** — pure embedding similarity retrieval

### `agents/match_scorer.py`
- **Input:** JD text (string), `cv_sections` (list[str])
- **Output:** `match_score` — float between 0 and 100
- **Model:** `sentence-transformers` (all-MiniLM-L6-v2 or equivalent)
- **Threshold:** read from `config.MATCH_THRESHOLD` (default: 60)
- **If score < threshold:** pipeline sets `state["skip"] = True` and halts

### `agents/resume_tailor.py`

- **Reads:** `state["cv_sections"]`, `state["requirements"]`
- **Also retrieves from ChromaDB:** resume template (`type=template`), tailoring instructions (`type=instructions`)
- **Process:** Uses `llm_selector.py` for LLM-driven content selection (which projects, skills, certs, leadership items to include), then `resume_assembler.py` for LaTeX block extraction, then injects tailored content into template's `%%INJECT:<SECTION>%%` markers, compiles via `pdflatex`
- **3-tier content resolution per section:** TIER 1 = exact template block, TIER 2 = LLM-reformatted master CV content, TIER 3 = raw master CV fallback
- **Instruction enforcement:** Role-conditional includes/excludes (e.g. "exclude CouchHire for Quant roles", "include Paper Presentation for Quant roles")
- **Intermediate file:** `cv/output/tailored_<timestamp>.tex` (gitignored)
- **Output paths:** `get_output_dir(company, role)` from `config.py` organises output as `<OUTPUT_BASE_DIR>/<Company>/<Role>/Resume.pdf`
- **`resume_content`:** generated deterministically by parsing the tailored LaTeX (no LLM call) — zero hallucination in the summary passed to `cover_letter.py` and `email_drafter.py`

### `agents/cover_letter.py`
- **Input:** `requirements`, `cv_sections`, `resume_content` (structured summary from resume_tailor — what was emphasised)
- **Output:** `cover_letter_text` (str) — complements resume, never repeats it
- **Only runs if:** `requirements["cover_letter_required"] == True`
- **Constraint:** prompt explicitly instructs "ONLY reference what the resume covers" and "DO NOT repeat bullet points" — the cover letter adds depth, narrative, and motivation
- **Header:** displays `Role - Company` (e.g. `AI/ML Engineer - CouchHire`) via `{{TARGET_ROLE}}` placeholder in the template
- **Paragraph 2 framing:** sells fit and positioning — answers "why this person for this role", not what they built
- **Template features:** `%%IF_<KEY>%%...%%ENDIF_<KEY>%%` conditional blocks for optional fields (phone, LinkedIn, portfolio); `%%FONT_PLACEHOLDER%%` for dynamic font selection (Montserrat if available, Helvetica fallback)
- **LLM call:** yes — uses `llm/client.py`

### `agents/email_drafter.py`
- **Input:** `requirements`, `resume_content` (structured summary of tailored resume — what projects/skills were emphasised), `cover_letter_text` (may be None), `resume_pdf_path`
- **Output:** `email_subject` (str), `email_body` (str, ≤200 words)
- **Generates human-quality emails** that reference specific projects from the tailored resume by name, mention education briefly, and include a proper signature (name, email, phone)
- **Tone:** casual-professional — modeled after real application emails, not corporate-speak. Uses a real example email in the prompt as a voice/tone reference
- **Always includes:** GitHub link woven naturally into the body (not as a standalone line)
- **LLM call:** yes — uses `llm/client.py`

### `agents/apply_router.py`
- **Input:** `requirements["apply_method"]`, `requirements["apply_target"]`
- **Output:** `route` (str): `"email"` | `"form"` | `"manual"`
- **No LLM call** — deterministic routing logic only

### `apply/gmail_sender.py`
- **Input:** `email_subject`, `email_body`, `resume_pdf_path`, `apply_target`, optional `cover_letter_pdf_path`, `user_google_email` (from `APPLICANT_EMAIL` in config)
- **`create_draft()`:** calls `draft_gmail_message` tool on the Gmail MCP server (`google_workspace_mcp`) with resume PDF (and cover letter PDF if present) attached — returns `(draft_url, draft_id)`
- **`send_email()`:** calls `send_gmail_message` tool on the MCP server — returns the message ID (str) for constructing the sent URL. Only called after Gate 2 approval
- **Draft-only by default** — the pipeline creates a draft, user reviews in Gmail, Gate 2 approval triggers send
- **No LLM call**

### `apply/browser_agent.py`
- **Input:** `apply_target` (URL), all generated documents (resume PDF, cover letter, applicant data)
- **Action:** Semi-autonomous ATS form filler with LLM-assisted field mapping
- For each form page: extracts visible text + field labels → sends to LLM → gets field mapping → fills fields
- **On any blocker** (missing field, validation error, CAPTCHA, unknown UI):
  - Takes screenshot → sends to Telegram with context
  - **Simple blockers:** user replies with text → agent fills and continues (answer stored in `form_answers.json` for future reuse)
  - **Complex blockers:** user takes over browser via CDP (`chrome://inspect` → `localhost:9222`), taps `[Done]` → agent resumes
- Uses `threading.Event()` for pause/resume — no Redis or external state store
- **No LLM call for form filling logic** — LLM is used only for field mapping (via `llm/client.py`)

### `apply/session_handoff.py`
- **Action:** CDP session manager — launches Chromium with `--remote-debugging-port=9222`
- `launch_browser(session_id)` — starts Chromium with CDP enabled
- `get_cdp_url()` — returns the CDP WebSocket URL for Playwright's `connect_over_cdp()`
- `get_takeover_instructions()` — returns human-readable instructions for user to connect via `chrome://inspect`
- `close_browser()` — cleanup
- Does NOT depend on Telegram bot — pure browser lifecycle management

### `bot/telegram_bot.py`
- **Outbound messages:** new job card, manual takeover alert, send confirmation, daily digest
- **Inbound buttons:** Approve, Edit, Done (takeover), No Reply, Screening, Interview, Rejected, Offer
- **On Edit:** re-triggers generator agents; on Approve: triggers apply route
- **On outcome label:** writes status to Supabase, checks if retrain threshold is hit

### `db/supabase_client.py`
- Thin wrapper around `supabase-py`
- Exposes: `insert_application()`, `update_status()`, `get_all_applications()`, `get_labeled_outcomes()`

### `nlp/retrain.py`
- **Trigger:** called automatically (via background thread in Telegram bot) when `should_retrain()` returns True — every `RETRAIN_EVERY` new outcome labels once `MIN_RETRAIN_LABELS` threshold is met
- **Input:** fetches labeled rows from Supabase (`jd_text`, `resume_content`, `outcome`); filters out `withdrawn` and rows missing either text field
- **Training:** builds `InputExample` pairs with `CosineSimilarityLoss` — outcome → label: `offer=1.0`, `interview=0.8`, `no_response=0.25`, `rejected=0.1`; oversamples positives if negative:positive ratio exceeds 3:1; adaptive epochs (8/5/3/2) based on dataset size
- **Model:** always retrains from `0xnbk/nbk-ats-semantic-v1-en` base (never incremental fine-tuning); saved to `nlp/models/match_scorer_finetuned/` (gitignored)
- **Output:** `match_scorer.py` auto-loads the fine-tuned model on next call via `reload_model()`
- **Public API:** `retrain(force=False) -> dict`, `should_retrain() -> bool`

### `llm/client.py`
- Single function: `complete(prompt, system_prompt=None, max_tokens=None) → str`
- Reads `LLM_PROVIDER` from config, routes to correct LiteLLM model via 9-model fallback chain across 5 providers
- `max_tokens` parameter for calls needing long structured output (e.g. JSON content selection)
- **Strips `<think>...</think>` blocks globally** from all LLM responses — handles reasoning models (Qwen3, DeepSeek-R1) that leak internal thinking tokens. No agent needs to handle this individually
- All agents call this — never call LiteLLM directly from an agent

### `config.py`
- Loads `.env` via `python-dotenv`
- **Validates all required keys on import** — raises a clear error listing every missing variable before anything else runs
- Exposes constants: `LLM_PROVIDER`, `MATCH_THRESHOLD`, `SUPABASE_URL`, `SUPABASE_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GITHUB_URL`, `MIN_RETRAIN_LABELS`, `RETRAIN_EVERY`, `FINETUNED_MODEL_DIR`

---

## Pipeline Flow

```
Input (JobSpy search / python pipeline.py --jd "..." / URL)
        │
        ▼
jd_parser.py
  → extracts: company, role, skills, apply_method, apply_target,
    cover_letter_required, subject_line_format, github_requested
        │
        ▼
cv_rag.py  +  match_scorer.py
  → ChromaDB retrieves top-k CV sections relevant to JD
  → sentence-transformers scores cosine similarity → match_score (float)
        │
        ▼
match_score >= MATCH_THRESHOLD?
  No  → log to Supabase as skipped, Telegram notification, stop
  Yes → continue
        │
        ▼
Generator agents (sequential via LangGraph)
  ├─ resume_tailor.py   → LLM content selection → tailored .tex → pdflatex → PDF path
  ├─ cover_letter.py    → only if cover_letter_required == True (receives resume_content)
  └─ email_drafter.py   → subject line + human-quality body referencing resume projects (receives resume_content)
        │
        ▼
telegram_bot.py — notification card
  → company, role, match %, apply route
  → [Approve] [Edit] buttons
        │
   ┌────┴────┐
[Edit]    [Approve]
  │            │
  ▼            ▼
Re-run     apply_router.py
generator  → route = "email"   → gmail_sender.py (Gmail MCP)
agents     → route = "form"    → browser_agent.py (LLM-assisted form fill + human-in-the-loop)
           → route = "manual"  → Telegram alert (apply manually)
                                 → you take over → tap [Done] → pipeline resumes
        │
        ▼
supabase_client.insert_application()
  → logs: company, role, jd_raw, requirements, match_score,
    resume_version, resume_latex, cover_letter, email_draft,
    apply_route, status="sent", applied_at
        │
        ├──▶ Streamlit dashboard reads from Supabase in real time
        │
        ▼
Telegram: outcome label buttons
  [No Reply] [Screening] [Interview] [Rejected] [Offer]
        │
        ▼
supabase_client.update_status()
  → every 10 new labels → nlp/retrain.py fires automatically
  → match scorer retrains on personal outcome history
  → new model replaces old if validation score improves
```

---

## Prerequisites

Install these before anything else.

**Python 3.11+**
```bash
python --version   # must be 3.11 or higher
```

**pdflatex** — required to compile tailored resumes
```bash
# macOS
brew install --cask mactex

# Ubuntu / Debian
sudo apt-get install texlive-full

# Windows
# Download and install MiKTeX from https://miktex.org/download
# Verify:
pdflatex --version
```

> **💡 Auto-install:** CouchHire automatically installs missing LaTeX packages via `tlmgr` when compiling your resume template. This requires `tlmgr` to be available (included with TeX Live). If auto-install fails due to permissions, run the suggested `sudo tlmgr install ...` command manually.

**Playwright browsers** — installed after pip dependencies, see Installation step 4.

**A Telegram account** — to create your bot and get your chat ID.

**A Supabase account** — free tier is sufficient. Sign up at https://supabase.com.

---

## Getting Your API Keys

You need at least one LLM key. All others are required for full functionality.

### LLM Provider (pick one to start)

**Groq** (recommended — fast and free tier available)
1. Go to https://console.groq.com
2. Sign in → API Keys → Create API Key
3. Copy the key → `GROQ_API_KEY` in your `.env`

**Google Gemini**
1. Go to https://aistudio.google.com/app/apikey
2. Create API Key
3. Copy → `GEMINI_API_KEY`

**Anthropic**
1. Go to https://console.anthropic.com
2. API Keys → Create Key
3. Copy → `ANTHROPIC_API_KEY`

**OpenAI** (paid — no meaningful free tier)
1. Go to https://platform.openai.com/api-keys
2. Create new secret key
3. Copy → `OPENAI_API_KEY`

**Mistral** (free — phone verification only)
1. Go to https://console.mistral.ai
2. Sign up with phone verification
3. API Keys → Create Key
4. Copy → `MISTRAL_API_KEY`

**OpenRouter** (free — email only, auto-routes across 29+ free models)
1. Go to https://openrouter.ai
2. Sign up with email
3. Keys → Create Key
4. Copy → `OPENROUTER_API_KEY`

---

### Supabase
1. Go to https://supabase.com and create a new project
2. In your project: Settings → API
3. Copy **Project URL** → `SUPABASE_URL`
4. Copy **anon / public key** → `SUPABASE_KEY`

---

### Telegram Bot
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow the prompts (choose a name and username)
3. BotFather gives you a token → `TELEGRAM_BOT_TOKEN`
4. To get your Chat ID:
   - Start your bot (send it `/start`)
   - Go to `https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates`
   - Find `"chat": {"id": ...}` in the response → that number is `TELEGRAM_CHAT_ID`

---

### Gmail MCP (google_workspace_mcp)

CouchHire uses [`google_workspace_mcp`](https://github.com/taylorwilsdon/google_workspace_mcp) as its Gmail MCP server. CouchHire is an MCP client — it connects to this server over Streamable HTTP and calls `draft_gmail_message` / `send_gmail_message` tools.

**Google Cloud setup:**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a new project (or use an existing one)
3. Enable the **Gmail API** (APIs & Services → Library → Gmail API → Enable)
4. Configure the **OAuth consent screen** (APIs & Services → OAuth consent screen):
   - Choose External
   - Add scopes: `gmail.compose`, `gmail.modify`
   - Add your Gmail address as a **test user** (required while app is in Testing status)
5. Create **OAuth credentials** (APIs & Services → Credentials → Create → OAuth client ID):
   - Application type: **Desktop application**
   - Copy the **Client ID** and **Client Secret**

**MCP server setup:**
1. Install `uv` (system-level, not inside your project venv):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create a launcher script (e.g. `~/start-gmail-mcp.sh`):
   ```bash
   #!/bin/bash
   export GOOGLE_OAUTH_CLIENT_ID="your-client-id.apps.googleusercontent.com"
   export GOOGLE_OAUTH_CLIENT_SECRET="your-client-secret"
   export OAUTHLIB_INSECURE_TRANSPORT=1
   export MCP_SINGLE_USER_MODE=true
   export USER_GOOGLE_EMAIL="your-email@gmail.com"

   uvx workspace-mcp --tools gmail --transport streamable-http
   ```
3. `chmod +x ~/start-gmail-mcp.sh` and run it — server starts at `http://localhost:8000/mcp`
4. On first use, it opens your browser for Google OAuth — sign in and authorize
5. Set `GMAIL_MCP_URL=http://localhost:8000/mcp` in your `.env`

**Important:** The MCP server runs via `uvx` (system-level) — completely separate from your project venv. Start it in one terminal before running the pipeline.

---

### Job Search (JobSpy)
No API keys required! JobSpy scrapes job boards directly.

Optional configuration in `.env`:
- `JOBSPY_SITES` — Which boards to search (default: `indeed,linkedin,google`)
- `JOBSPY_COUNTRY` — Country for Indeed/Glassdoor (default: `USA`)
- `JOBSPY_PROXIES` — Proxy list to avoid rate limiting (optional)

⚠️  LinkedIn is rate-limit aggressive (~10 pages per IP). Use proxies for heavy LinkedIn scraping.

---

## Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/your-username/couchhire.git
cd couchhire
```

### Step 2 — Create and activate a virtual environment
```bash
python -m venv venv

# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### Step 3 — Install Python dependencies
```bash
pip install -r requirements.txt
```

### Step 4 — Install Playwright browsers
```bash
playwright install chromium
```

### Step 5 — Install the spaCy English model
```bash
python -m spacy download en_core_web_sm
```

### Step 6 — Set up your environment variables
```bash
cp .env.example .env
```
Open `.env` in any text editor and fill in every value. See [Getting Your API Keys](#getting-your-api-keys) above.

---

## Configuration

All runtime configuration lives in `config.py`, which reads from `.env`. You do not edit `config.py` directly — only `.env`.

**Choosing your LLM provider:**
```
LLM_PROVIDER=groq    # groq | gemini | anthropic | openai
```
Only the API key for the chosen provider needs to be set. The others can be left blank.

**Match threshold:**
```
MATCH_THRESHOLD=60   # integer 0–100, default 60
```
Jobs with a match score below this value are skipped automatically. Lower it to apply more broadly, raise it to be more selective.

**Your GitHub URL:**
```
GITHUB_URL=https://github.com/your-username
```
This is automatically included in every email draft.

**Your email address:**
```
APPLICANT_EMAIL=you@example.com
```
Used in cover letters, ATS form fields, and email drafts.

**Browser agent variables (optional):**
```
APPLICANT_PHONE=+91 9800155779    # for ATS form fields
APPLICANT_LINKEDIN=https://linkedin.com/in/your-profile
FORM_ANSWERS_PATH=cv/uploads/form_answers.json  # default, override if needed
CDP_PORT=9222                     # Chrome DevTools Protocol port, default 9222
BROWSER_HEADLESS=false            # run browser agent headless (default false)
```

**Output directory (optional):**
```
OUTPUT_BASE_DIR=/path/to/your/output   # defaults to cv/output/
```
Output is organised as `OUTPUT_BASE_DIR/<Company>/<Role>/Resume.pdf` and `Cover Letter.pdf`.

**NLP retraining variables (optional):**
```
MIN_RETRAIN_LABELS=10              # minimum labeled outcomes before retraining is allowed
RETRAIN_EVERY=10                   # auto-retrain every N new labels (0 = manual only)
```
These control the auto-retrain trigger.

**Job search variables (optional):**
```
MIN_MATCH_SCORE=60.0               # minimum match % to show in job search results (default 60.0)
MAX_SEARCH_RESULTS=10              # max jobs to show after filtering in search (default 10)
``` With defaults, retraining fires when you have at least 10 labels and every 10 new labels after that. Set `RETRAIN_EVERY=0` to disable auto-retrain and use the dashboard Retrain button instead.

The full list of variables is in `.env.example` with a comment explaining each one.

---

## Database Setup (Supabase)

### Step 1 — Run the schema
1. In your Supabase project, click **SQL Editor** in the left sidebar
2. Click **New query**
3. Open `db/schema.sql` from this repo, copy the entire contents, paste into the editor
4. Click **Run**

You should see the `applications` table created with no errors.

### Step 2 — Verify the connection
```bash
python -c "from db.supabase_client import get_all_applications; print('Supabase connection OK')"
```

The schema creates one table:

```sql
applications (
  id              uuid primary key default uuid_generate_v4(),
  jd_text         text,                    -- raw JD text (nullable — may only have URL initially)
  jd_url          text,                    -- source URL (Greenhouse, Lever, LinkedIn, etc.)
  role_input      text,                    -- raw role/title when no full JD provided
  requirements    jsonb not null default '{}',  -- structured output from jd_parser.py
  company         text,                    -- denormalised from requirements
  role            text,                    -- denormalised from requirements
  match_score     numeric(5,1),            -- 0–100
  route           text,                    -- 'email' | 'form' | 'manual'
  resume_pdf_path text,                    -- path to compiled tailored PDF
  resume_content  text,                    -- plain-text summary of what was emphasised
  cover_letter    text,
  email_subject   text,
  email_body      text,
  draft_url       text,                    -- Gmail draft deeplink
  status          text not null default 'pending',  -- 10-state: pending → scraping → parsing → scoring → below_threshold/tailoring → drafting → awaiting_review → applied → error
  outcome         text,                    -- interview | rejected | no_response | offer | withdrawn
  error_message   text,                    -- populated when status = 'error'
  source          text default 'cli',      -- 'cli' | 'telegram' | 'jobspy'
  applied_at      timestamptz default now(),
  updated_at      timestamptz default now()  -- auto-updated via trigger
)
```

---

## CV Setup

CouchHire accepts your master CV in any format — LaTeX, PDF, or Word.

### Step 1 — Place your files in cv/uploads/

| File | Required | Description |
|---|---|---|
| `master_cv.tex` / `master_cv.pdf` / `master_cv.docx` | Yes | Your full master CV |
| `resume_template.tex` | No | Your preferred LaTeX resume layout |
| `cover_letter_template.tex` | No | Your preferred cover letter layout (XeLaTeX with `{{PLACEHOLDER}}` markers, `%%IF_<KEY>%%` conditional blocks, `%%FONT_PLACEHOLDER%%` font marker) |
| `instructions.md` | No | Your tailoring preferences |

If you do not provide a template or instructions, CouchHire uses its own defaults automatically.

#### Tailoring instructions examples

Add plain English preferences to `cv/uploads/instructions.md`:

- "Always keep to 1 page"
- "Lead with projects for ML roles, lead with experience for quant roles"
- "Never include GPA"
- "Use a two-column skills section"

The default `cv/defaults/instructions.md` also includes a **Positioning Philosophy** — lead with outcome not stack, show 2–3 projects max, use impact verbs only. Your custom instructions override or extend these defaults.

### Step 2 — Embed your CV

```bash
python cv/embed_cv.py
```

This parses your CV into sections, embeds each section using sentence-transformers, and stores everything in ChromaDB locally. Re-run this command any time you update your CV or template.

### Step 3 — Verify

```bash
python -c "
import chromadb
client = chromadb.PersistentClient(path='cv/chroma_store')
col = client.get_collection('master_cv')
print(f'Chunks embedded: {col.count()}')
"
```

---

## Running the Project

Run each of these in a separate terminal window. All three need to be running simultaneously for the full pipeline to work.

### Terminal 1 — Telegram bot
```bash
python bot/telegram_bot.py
```
Keep this running. It listens for your Approve / Edit / label taps and triggers the appropriate pipeline steps.

### Terminal 2 — Streamlit dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501. Shows all applications, analytics, and retrain controls.

### Terminal 3 — Run the pipeline

**Paste a JD manually:**
```bash
python pipeline.py --jd "Paste the full job description text here"
```

**Pass a URL to scrape:**
```bash
python pipeline.py --url "https://jobs.example.com/job/12345"
```

**Search job boards:**
```bash
python pipeline.py --search --query "machine learning engineer" --location "London"
```

After running, check your Telegram — you should receive a notification card for each job that passes the match threshold.

---

## Standalone Resume Generation

If you just want to generate a tailored resume without the full pipeline (no Telegram, no email, no apply routing):

```bash
# With a JD pasted inline
python generate_resume.py --jd "Paste the full job description text here"

# From a file
python generate_resume.py --file path/to/jd.txt

# Uses a built-in default JD for quick testing
python generate_resume.py
```

The output PDF is saved to `cv/output/` and the path is printed to the console.

---

## Docker (Alternative)

If you prefer not to manage Python, pdflatex, and Playwright installations manually, Docker handles all of it.

### Step 1 — Make sure Docker Desktop is installed
Download from https://www.docker.com/products/docker-desktop

### Step 2 — Fill in your `.env` file
Same as the manual setup — copy `.env.example` to `.env` and fill in your keys.

### Step 3 — Start everything
```bash
docker compose up --build
```

This starts the Telegram bot, Streamlit dashboard, and pipeline together. The Streamlit dashboard is available at http://localhost:8501.

To stop:
```bash
docker compose down
```

---

## Self-Improving Loop

CouchHire gets better at predicting which jobs are worth applying for the more you use it.

1. An application is sent → logged in Supabase with `status = 'sent'`
2. You receive a response from the employer → open Telegram → tap the outcome button for that application
3. Available labels: **No Reply**, **Screening**, **Interview**, **Rejected**, **Offer**
4. The label is written to Supabase
5. Every `RETRAIN_EVERY` new labels (default: 10), `nlp/retrain.py` fires automatically in a background thread
6. The scorer fine-tunes on your personal `(jd_text, resume_content, outcome)` pairs using `CosineSimilarityLoss` — outcomes map to similarity targets (offer=1.0, interview=0.8, no_response=0.25, rejected=0.1)
7. Class imbalance is handled by oversampling positives to keep the negative:positive ratio at most 3:1
8. The fine-tuned model is saved to `nlp/models/match_scorer_finetuned/` and loaded automatically on the next pipeline run
9. From that point forward, the pipeline uses your personalised model

The more you label, the more personalised the scoring becomes to your actual experience and the roles that respond to you.

---

## Streamlit Dashboard

Available at http://localhost:8501 once running.

| Tab | What it shows |
|---|---|
| Tracker | All applications, sortable by date / match score / status. Click any row to see full details. |
| Analytics | Response rate by role type, resume version performance, match score distribution over time |
| Retrain | Outcome labels table, manual retrain button, model accuracy history |
| Settings | LLM provider selector, match threshold slider, ChromaDB status, Supabase connection test |

---

## Roadmap

- [x] Core pipeline (JD parser → resume tailor → email drafter)
- [x] Telegram bot (notifications + approval buttons + /outcome command + /search + /apply + gate handlers + auto-retrain hook)
- [x] Gmail MCP integration (draft_gmail_message + send_gmail_message via google_workspace_mcp over Streamable HTTP)
- [x] Semi-autonomous browser agent (LLM-assisted ATS form filling + human-in-the-loop)
- [x] CDP session management for browser takeover
- [x] Supabase logging
- [x] NLP match scorer + self-improving retraining loop (CosineSimilarityLoss, outcome labels, class balancing)
- [x] Multi-board job search (JobSpy — Indeed, LinkedIn, Google, Glassdoor, ZipRecruiter)
- [x] LiteLLM 9-model fallback chain across 5 providers
- [x] Cover letter PDF compilation (XeLaTeX template with placeholder markers)
- [x] LangGraph pipeline orchestrator (19 nodes, 6 conditional edges, 2 Telegram approval gates, CLI with --jd/--url/--file/--search modes)
- [x] Integration tests (55/55 passing)
- [x] Pipeline hardening (error routing on agent failure, `<think>` tag stripping in LLM client)
- [x] Human-quality application emails (email drafter receives resume_content, references specific projects)
- [x] Post-send confirmation (Telegram shows 'Application Sent' with sent message URL after Gate 2)
- [ ] Streamlit dashboard (Step 26 — next)
- [ ] Full pytest suite (Step 27)
- [ ] Docker support (Step 28)
- [ ] Open source release

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Test that `python pipeline.py --jd "test"` runs without errors
5. Open a pull request with a clear description of what you changed and why

For bugs, open an issue with the error message, your OS, Python version, and which LLM provider you are using.

**Never commit your `.env` file, `cv/uploads/` directory, or the `cv/chroma_store/` directory.**

---

## License

MIT — use it, fork it, improve it.
