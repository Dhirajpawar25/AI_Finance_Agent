# AI Financial Assistant — Telegram Bot

An AI-powered financial assistant that lives inside Telegram. It behaves like an experienced financial analyst + executive assistant — conversational, proactive, and personalized. No slash commands, no menus, no buttons. Just text, voice, images, and documents.

Built with **Python + FastAPI**.

---

## ✨ What It Does

| Capability | Examples |
|---|---|
| **Onboarding** | Natural conversational onboarding — role, interests, watchlist, insights, briefing time. Users can skip anything. |
| **Company & Market Research** | `Compare Microsoft and Google from an investment perspective` · `What's the market doing today?` |
| **Live Stock Info** | `How is Nvidia trading?` — price, %, open/high/low, volume |
| **News with Context** | `Any Tesla news this week?` — headlines with "why it matters" |
| **SEC Filings** | `Recent SEC filings for Meta?` — 8-K / 10-K / 10-Q / insider Form 4 |
| **Earnings** | `What earnings are coming up?` |
| **Price History** | `How has AAPL performed over 6 months?` |
| **Custom Alerts** | `Alert me if NVDA moves more than 5% in a day` · `Track Tesla and notify me on major news` · `Notify me on new SEC filings for Apple` |
| **Daily Briefing** | Personalized morning briefing delivered at the user's chosen time with markets, watchlist, and news. |
| **Document Intelligence** | Upload a PDF (annual report, earnings deck, 10-K, investment memo) → instant executive summary + Q&A. |
| **Voice Messages** | Send a voice note → transcribed → answered (needs OpenAI or Gemini key). |
| **Images** | Send a chart or screenshot → vision model explains what it shows. |
| **Memory** | Learns role, watchlist, interests over time and personalizes responses. |
| **Google Integrations** | Optional OAuth for Gmail / Calendar / Sheets / Drive — search emails, prep for meetings, schedule events, analyze spreadsheets, find documents. Tokens stored per user. |

### No Telegram-specific UI
The assistant uses **only** natural conversation — no slash commands, inline buttons, quick replies, or menus. The only exception is the internal `/start` command to trigger onboarding (standard Telegram convention).

---

## 🧠 AI Strategy

- **Provider-agnostic**: OpenAI, Anthropic (Claude), or Gemini — just add the matching API key(s).
- **Rule-based fallback**: if no AI key is configured, the assistant still answers price/filings/news using deterministic templates so the demo always works.
- **Tool grounding**: live data from yfinance/SEC/Finnhub is retrieved first, then the LLM synthesizes a concise analyst-style answer. The model is instructed to *never invent numbers*.
- **Context + memory**: conversation history (last 10 turns) and long-term memories (role, watchlist, interests, learned facts) are injected into the system prompt.

---

## 🏗 Architecture

```
AI_Financial_Agent/
├── app/
│   ├── main.py                    # FastAPI app (webhook, OAuth callback, demo routes, lifespan)
│   ├── config.py                  # Settings from .env (pydantic-settings)
│   ├── database.py                # SQLAlchemy engine + session
│   ├── models.py                  # User, Conversation, Message, Memory, Alert, Document, Integration, BriefingLog
│   ├── bot.py                     # Telegram handlers — text, voice, image, PDF, TXT
│   └── services/
│       ├── ai.py                  # AI providers (OpenAI/Anthropic/Gemini) + rule-based fallback + transcription glue
│       ├── market_data.py         # yfinance prices, history, overview, news, earnings calendar
│       ├── sec_data.py            # SEC EDGAR company filings
│       ├── document_service.py    # PDF text extraction, summarization, doc persistence
│       ├── integrations.py        # Google OAuth plumbing (Gmail/Calendar/Sheets/Drive)
│       ├── tools.py               # Tool registry used by the assistant
│       ├── assistant.py           # Orchestration — intent detection, tool calls, memory, onboarding
│       └── scheduler.py           # APScheduler — daily briefings + alert monitoring
├── run.py                         # Dev entry: FastAPI + polling bot in one process
├── setup_webhook.py               # Register webhook with Telegram
├── requirements.txt
└── .env.example
```

### Key Flows

1. **Message → Reply**
   `Telegram update → bot.py handler → assistant.generate_reply() → detect intent + extract tickers → run tools (yfinance/SEC) → LLM synthesizes answer with user profile + conversation history → reply persisted`

2. **Daily Briefing**
   `APScheduler (every minute) → find users whose briefing time == now → build personalized briefing from market data → send via Telegram`

3. **Alerts**
   `APScheduler (every 5 min) → evaluate price/news/filing alerts → fire notification → log AlertEvent (6h cooldown)`

4. **Document**
   `PDF upload → text extraction → persist to DB → summarize via LLM → user can ask follow-up questions`

---

## 🚀 Quick Start

### 1. Create a Telegram bot

1. Message [@BotFather](https://t.me/BotFather) → `/newbot` → copy the token.

### 2. Environment

```bash
copy .env.example .env
```

Edit `.env`:

| Variable | Purpose |
|---|---|
| `TELEGRAM_BOT_TOKEN` | **Required** — from BotFather |
| `OPENAI_API_KEY` | Optional — enables conversational AI + voice + vision (recommended) |
| `ANTHROPIC_API_KEY` | Optional — alternative AI provider |
| `GEMINI_API_KEY` | Optional — alternative AI provider + transcription |
| `FINNHUB_API_KEY` | Optional — extra company fundamentals/news |
| `GOOGLE_CLIENT_ID` / `GOOGLE_CLIENT_SECRET` | Optional — Gmail/Calendar/Sheets OAuth |
| `DATABASE_URL` | Optional — defaults to local SQLite |

### 3. Install & run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt

python run.py                     # FastAPI + polling bot in one process
```

Your bot is now live in **polling mode** — no public URL needed.

### 4. Optional: Google integrations (Gmail / Calendar / Sheets / Drive, ~10 min)

These let the assistant do things like *"search my emails about Apple"*, *"what's on my calendar this week?"*, *"schedule a meeting tomorrow at 2 PM"*, *"analyze my Q3 financial model in Sheets"*, and *"find the acquisition memo in my Drive"*.

1. Go to [Google Cloud Console](https://console.cloud.google.com) → create a project (or pick one).
2. **Enable APIs** (APIs & Services → Library):
   - Gmail API
   - Google Calendar API
   - Google Sheets API
   - Google Drive API
3. **Configure OAuth consent screen** (APIs & Services → OAuth consent screen):
   - App name: `AI Financial Agent`
   - User type: **External** (for demo; add your own Google account under *Test users*)
4. **Create OAuth Client ID** (APIs & Services → Credentials → Create Credentials → OAuth client ID):
   - Application type: **Web application**
   - Authorized redirect URI → `http://localhost:8000/oauth/google/callback` (or your public HTTPS URL for a live demo)
5. Copy the Client ID + Secret into `.env`:
   ```
   GOOGLE_CLIENT_ID=xxxx.apps.googleusercontent.com
   GOOGLE_CLIENT_SECRET=xxxx
   GOOGLE_REDIRECT_URI=http://localhost:8000/oauth/google/callback
   ```
6. **Restart** the bot, then in Telegram say **"connect my Gmail"** (or *Calendar* / *Sheets* / *Drive*) — the bot sends a link, you authorize, and it confirms back in chat.

> ⚠️ For a **live judge demo** over the internet, use ngrok (`ngrok http 8000`) or a deployed URL, and set `GOOGLE_REDIRECT_URI` + the Google Cloud redirect URI to match that public URL.

---

## 🚀 Deploy to Fly.io (recommended — always-on, free tier)

Fly.io keeps the bot **always-on** within its free allowance, so the daily briefing scheduler fires reliably and judges can message the bot live from any device. Data persists on a free volume (SQLite survives redeploys).

> Requires a credit/debit card on file for account verification (free tier usage stays free). If you have no card, use the [Render fallback](#deploy-to-render-fallback--no-card-required).

### 1. Install flyctl

```bash
# Windows (PowerShell)
irm https://fly.io/install.ps1 | iex
# macOS / Linux
curl -L https://fly.io/install.sh | sh
```

### 2. Prepare & launch

```bash
# From the project root
fly auth login
fly launch --no-deploy          # creates fly.toml from the existing one; pick your app name
```

### 3. Create persistent storage for SQLite

```bash
fly volumes create sqlite_data --size 1 --region bom   # or your nearest region
```

### 4. Set secrets (Bot token, AI key, Google OAuth)

```bash
fly secrets set TELEGRAM_BOT_TOKEN="YOUR_TOKEN"
fly secrets set GEMINI_API_KEY="YOUR_GEMINI_KEY"

# Google integrations (skip if you don't need Gmail/Calendar/Sheets/Drive)
fly secrets set GOOGLE_CLIENT_ID="YOUR_CLIENT_ID"
fly secrets set GOOGLE_CLIENT_SECRET="YOUR_CLIENT_SECRET"
fly secrets set GOOGLE_REDIRECT_URI="https://YOUR-APP.fly.dev/oauth/google/callback"
```

The `fly.toml` already sets `USE_WEBHOOK=true`, `DATA_DIR=/data`, and `DATABASE_URL=sqlite:////data/financial_agent.db`.

### 5. Deploy

```bash
fly deploy
```

### 6. Register the Telegram webhook

```bash
# Run once from your machine (uses TELEGRAM_BOT_TOKEN from .env)
python setup_webhook.py https://YOUR-APP.fly.dev
```

### 7. Google OAuth redirect URI (if using integrations)

In [Google Cloud Console](https://console.cloud.google.com) → Credentials → your OAuth Client → **Authorized redirect URIs**, add:

```
https://YOUR-APP.fly.dev/oauth/google/callback
```

(Replace the `http://localhost:8000/...` entry.)

### 8. Verify

- Open the bot in Telegram from your phone → `/start` → onboarding works.
- Say **"connect my Gmail"** → you get a link like `https://YOUR-APP.fly.dev/oauth/google/connect?...` → click, authorize, bot confirms.
- Check `https://YOUR-APP.fly.dev/health` → `{"status":"ok", ...}`.

> **Redeploys persist data** because SQLite lives on the `sqlite_data` volume.

---

## 🚀 Deploy to Render (fallback — no card required)

Render's free tier **sleeps after ~15 min idle**, which can delay the first response and **skip scheduled briefings** while asleep. If you use Render, add a free [cron-job.org](https://cron-job.org) ping to `https://YOUR-APP.onrender.com/health` every 10 minutes to keep it warm.

### Steps

1. Push this repo to GitHub.
2. On Render → **New → Web Service** → connect the repo.
3. Set:
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn app.main:app --host 0.0.0.0 --port $PORT`
   - **Health check path:** `/health`
4. Add env vars in Render dashboard:
   ```
   USE_WEBHOOK=true
   TELEGRAM_BOT_TOKEN=YOUR_TOKEN
   GEMINI_API_KEY=YOUR_GEMINI_KEY
   GOOGLE_CLIENT_ID=YOUR_CLIENT_ID        # optional
   GOOGLE_CLIENT_SECRET=YOUR_CLIENT_SECRET # optional
   GOOGLE_REDIRECT_URI=https://YOUR-APP.onrender.com/oauth/google/callback
   ```
5. Deploy, then register the webhook:
   ```bash
   python setup_webhook.py https://YOUR-APP.onrender.com
   ```
6. Add `https://YOUR-APP.onrender.com/oauth/google/callback` to Google Console **Authorized redirect URIs**.
7. Create a cron-job.org job hitting `/health` every 10 min so the schedule/briefings work.

> ⚠️ Free Render uses an ephemeral disk — user data resets on redeploy. For persistence, add a Render **PostgreSQL** instance and set `DATABASE_URL=postgres://...` (add `psycopg2-binary` to `requirements.txt`).

---

## 💬 Demo Script (for your demo video)

1. **Start**: Open the bot → send `/start` → complete the 5-step onboarding naturally (role, interests, watchlist, insights, briefing time — or skip).
2. **Market overview**: *"What should I know before markets open today?"*
3. **Company research**: *"Compare Microsoft and Google from an investment perspective."*
4. **Live price**: *"How is Nvidia trading right now?"*
5. **News with context**: *"Why did Apple move today?"* (or ask about any news)
6. **Filings**: *"Recent SEC filings for Meta?"*
7. **Alert**: *"Alert me if NVDA moves more than 5% in a day."* (also survives as a background job)
8. **Document**: Upload a PDF → *"Give me a 5-point executive summary"* → follow-up: *"What are the biggest risks?"*
9. **Voice**: Send a voice note asking the same.
10. **Google integrations** (if configured): *"connect my Gmail"* → click the link → authorize → bot confirms → *"search my emails for anything about Nvidia"* → get a summarized inbox. Then *"what's on my calendar this week?"* or *"schedule a meeting with my team tomorrow at 2 PM"*.
11. **Briefing**: Set a time → see the confirmation; optional — show the scheduler firing (set time to now+1 min).

---

## 🛠 Tech Stack

- **Python 3.10+** · **FastAPI** · **Uvicorn**
- **python-telegram-bot v21** (conversational handlers)
- **SQLAlchemy 2.0** + **SQLite** (swap to Postgres/MySQL via `DATABASE_URL`)
- **APScheduler** (daily briefings, alert monitoring)
- **yfinance** (prices, history, news, earnings) · **SEC EDGAR** (filings) · **Finnhub** (optional fundamentals)
- **pypdf** (PDF extraction)
- **OpenAI / Anthropic / Gemini** SDKs (provider-agnostic AI)

---

## 📌 Design Notes

- **Quality over frequency** — briefings only send when users opt in at their chosen time.
- **Silence by default** — no spam; alerts only fire when conditions actually trigger.
- **Personalization** — watchlist/role/interests learned through conversation, not forms.
- **Honest uncertainty** — if live data can't be retrieved, the assistant says so instead of guessing.
- **Extensible** — the tool registry (`tools.py`) makes adding new data sources or actions a one-line change.