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

1. A job description comes in — either pasted manually, pulled automatically via Indeed MCP, or scraped from a URL.
2. The JD Parser extracts structured requirements: skills, apply method, cover letter needed, subject line format.
3. The CV RAG agent retrieves the most relevant sections of your master CV from ChromaDB.
4. The Match Scorer computes a fit percentage using sentence-transformers. If below the configured threshold, the job is skipped or flagged.
5. If the match passes, three agents run in parallel: Resume Tailor (compiles a tailored PDF via pdflatex), Cover Letter (if required), and Email Drafter.
6. A Telegram notification arrives with a summary card — company, role, match %, apply route — and Approve / Edit buttons.
7. On approval, the Apply Router detects the method: Gmail MCP for email, Playwright for ATS forms, or a live handoff alert for CAPTCHAs.
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
| Job search | Indeed MCP | Proactive job pulls into the pipeline |
| Email send | Gmail MCP | Sends email + resume PDF attachment |
| Browser automation | Playwright | ATS form filling, Chrome DevTools Protocol for live session handoff |
| Notifications + review | Telegram bot (python-telegram-bot) | Approve / edit / label outcomes from your phone |
| Dashboard | Streamlit | Application tracker, analytics, retrain controls, settings |
| Resume compiler | pdflatex (local) | Compiles tailored .tex → PDF on the user's machine |

---

## Project Structure

```
couchhire/
├── agents/
│   ├── jd_parser.py          # JD parsing + requirements extraction → requirements dict
│   ├── cv_rag.py             # ChromaDB retrieval from master CV → ranked CV sections
│   ├── match_scorer.py       # NLP match scoring → float 0–100
│   ├── resume_tailor.py      # LaTeX resume tailoring → compiled PDF path
│   ├── cover_letter.py       # Cover letter generation (conditional) → string
│   ├── email_drafter.py      # Email draft generation → subject + body strings
│   └── apply_router.py       # Detects email / form / manual route → route string
├── apply/
│   ├── gmail_sender.py       # Gmail MCP integration → sends email with PDF attachment
│   ├── browser_agent.py      # Playwright ATS form filling
│   └── session_handoff.py    # Live browser session handoff via Chrome DevTools Protocol
├── bot/
│   └── telegram_bot.py       # Inbound/outbound Telegram notifications and buttons
├── dashboard/
│   └── app.py                # Streamlit dashboard (reads from Supabase)
├── db/
│   ├── supabase_client.py    # Supabase read/write helpers
│   └── schema.sql            # Full CREATE TABLE statement — run this first in Supabase
├── nlp/
│   ├── ner_model.py          # spaCy NER for JD skill extraction
│   └── retrain.py            # Auto-retraining loop (fires every 10 new outcome labels)
├── llm/
│   └── client.py             # LiteLLM wrapper — all LLM calls go through here
├── cv/
│   ├── uploads/              # User's personal files — gitignored
│   │   ├── master_cv.*       # .tex, .pdf, or .docx — any format accepted
│   │   ├── resume_template.tex  # Optional custom LaTeX template
│   │   └── instructions.md   # Optional tailoring preferences
│   ├── chroma_store/         # ChromaDB embeddings — gitignored
│   ├── output/               # Compiled PDFs — gitignored
│   ├── embed_cv.py           # Orchestrates parse → embed pipeline
│   ├── cv_parser.py          # Parses .tex/.pdf/.docx into named sections
│   └── defaults/             # Fallback template and instructions — committed
│       ├── resume_template.tex
│       └── instructions.md
├── pipeline.py               # LangGraph orchestrator — main entry point
├── config.py                 # Reads .env, validates all keys on startup, exposes config
├── .env                      # Your secrets — never committed to git
├── .env.example              # Template with all variable names — committed to git
├── .gitignore                # Excludes .env, cv/uploads/, cv/chroma_store/, cv/output/, compiled PDFs
├── docker-compose.yml        # One-command alternative to manual setup
├── Dockerfile                # Container definition for the main app
├── requirements.txt          # Pinned Python dependencies
└── README.md
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
- **Process:** injects tailored content into template's `%%INJECT:<SECTION>%%` markers, compiles via `pdflatex`
- **Intermediate file:** `cv/output/tailored_<timestamp>.tex` (gitignored)

### `agents/cover_letter.py`
- **Input:** `requirements`, `cv_sections`
- **Output:** `cover_letter_text` (str) — 3 paragraphs
- **Only runs if:** `requirements["cover_letter_required"] == True`
- **LLM call:** yes — uses `llm/client.py`

### `agents/email_drafter.py`
- **Input:** `requirements`, `cover_letter_text` (may be None), `resume_pdf_path`
- **Output:** `email_subject` (str), `email_body` (str, ≤200 words)
- **Always includes:** GitHub link (read from config)
- **LLM call:** yes — uses `llm/client.py`

### `agents/apply_router.py`
- **Input:** `requirements["apply_method"]`, `requirements["apply_target"]`
- **Output:** `route` (str): `"email"` | `"form"` | `"manual"`
- **No LLM call** — deterministic routing logic only

### `apply/gmail_sender.py`
- **Input:** `email_subject`, `email_body`, `resume_pdf_path`, `apply_target`
- **Action:** sends email via Gmail MCP with PDF attached
- **No LLM call**

### `apply/browser_agent.py`
- **Input:** `apply_target` (URL), all generated documents
- **Action:** Playwright fills ATS form fields (name, email, resume upload, cover letter, LinkedIn, GitHub)
- **On CAPTCHA:** pauses, calls `session_handoff.py`, sends Telegram alert
- **Runs non-headless** so manual takeover is possible

### `apply/session_handoff.py`
- **Action:** exposes existing Playwright browser session via Chrome DevTools Protocol for manual control
- **Telegram:** sends alert with [Done] button; on tap, pipeline resumes logging

### `bot/telegram_bot.py`
- **Outbound messages:** new job card, manual takeover alert, send confirmation, daily digest
- **Inbound buttons:** Approve, Edit, Done (takeover), No Reply, Screening, Interview, Rejected, Offer
- **On Edit:** re-triggers generator agents; on Approve: triggers apply route
- **On outcome label:** writes status to Supabase, checks if retrain threshold is hit

### `db/supabase_client.py`
- Thin wrapper around `supabase-py`
- Exposes: `insert_application()`, `update_status()`, `get_all_applications()`, `get_labeled_outcomes()`

### `nlp/retrain.py`
- **Trigger:** called automatically when count of new outcome labels since last retrain reaches 10
- **Input:** fetches `(jd_embedding, cv_embedding, outcome)` rows from Supabase
- **Output:** new match scorer model replaces old if validation accuracy improves

### `llm/client.py`
- Single function: `complete(prompt, system_prompt=None) → str`
- Reads `LLM_PROVIDER` from config, routes to correct LiteLLM model
- All agents call this — never call LiteLLM directly from an agent

### `config.py`
- Loads `.env` via `python-dotenv`
- **Validates all required keys on import** — raises a clear error listing every missing variable before anything else runs
- Exposes constants: `LLM_PROVIDER`, `MATCH_THRESHOLD`, `SUPABASE_URL`, `SUPABASE_KEY`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `GITHUB_URL`

---

## Pipeline Flow

```
Input (Indeed MCP / python pipeline.py --jd "..." / URL)
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
Generator agents (run in parallel via LangGraph)
  ├─ resume_tailor.py   → tailored .tex → pdflatex → PDF path
  ├─ cover_letter.py    → only if cover_letter_required == True
  └─ email_drafter.py   → subject line + body (≤200 words)
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
agents     → route = "form"    → browser_agent.py (Playwright)
           → route = "manual"  → session_handoff.py → Telegram alert
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

### Gmail MCP
1. Go to https://console.cloud.google.com
2. Create a new project (or use an existing one)
3. Enable the **Gmail API** for the project
4. OAuth consent screen → configure as External
5. Credentials → Create OAuth 2.0 Client ID (Desktop app)
6. Download the credentials JSON
7. Run the Gmail MCP auth flow (see the Gmail MCP docs) to exchange for a token
8. The resulting token → `GMAIL_MCP_TOKEN`

---

### Indeed MCP
1. Go to https://developer.indeed.com
2. Create an application to get API access
3. Copy your token → `INDEED_MCP_TOKEN`

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
  id              uuid primary key default gen_random_uuid(),
  company         text,
  role            text,
  jd_raw          text,
  requirements    jsonb,         -- structured output from jd_parser.py
  match_score     float,
  resume_version  text,
  resume_latex    text,          -- full .tex source of the tailored resume
  cover_letter    text,
  email_draft     text,
  apply_route     text,          -- 'email' | 'form' | 'manual'
  status          text,          -- 'sent' | 'no_reply' | 'screening' | 'interview' | 'rejected' | 'offer'
  applied_at      timestamptz default now(),
  updated_at      timestamptz default now()
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
| `instructions.md` | No | Your tailoring preferences |

If you do not provide a template or instructions, CouchHire uses its own defaults automatically.

#### Tailoring instructions examples

Add plain English preferences to `cv/uploads/instructions.md`:

- "Always keep to 1 page"
- "Lead with projects for ML roles, lead with experience for quant roles"
- "Never include GPA"
- "Use a two-column skills section"

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

**Pull jobs automatically from Indeed MCP:**
```bash
python pipeline.py --indeed --query "machine learning engineer" --location "London"
```

After running, check your Telegram — you should receive a notification card for each job that passes the match threshold.

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
5. Every 10 new labels, `nlp/retrain.py` fires automatically
6. The match scorer retrains on your personal `(jd_embedding, cv_embedding, outcome)` history
7. The new model replaces the old one only if its validation accuracy is higher
8. From that point forward, the pipeline uses the improved model

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

- [ ] Core pipeline (JD parser → resume tailor → email drafter)
- [ ] Telegram bot (notifications + approval buttons)
- [ ] Gmail MCP integration
- [ ] Playwright form filling
- [ ] Manual takeover session handoff
- [x] Supabase logging
- [ ] Streamlit dashboard
- [ ] NLP match scorer + retraining loop
- [ ] Indeed MCP job pulls
- [x] LiteLLM multi-provider support
- [ ] Docker support
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