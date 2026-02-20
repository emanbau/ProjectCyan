# ProjectCyan
Agentic workflows + crypto
## Table of Contents
1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [File Purpose Reference](#file-purpose-reference)
4. [How the Agent and App Connect](#how-the-agent-and-app-connect)
5. [How to Run the Agent](#how-to-run-the-agent)

---

## Project Overview

This project is an autonomous cryptocurrency trading research system. At its core, a Claude Opus 4.6 AI agent acts as a senior quantitative researcher — forming hypotheses about market patterns, building strategies, running backtests, learning from results, and repeating the cycle indefinitely. Human oversight is maintained through an OpenClaw interface (Telegram/Discord), and live trade signals can flow in from TradingView via webhooks.

The system is split into five Python packages inside a monorepo, each with a distinct responsibility, all orchestrated together via Docker Compose.

---

## Repository Structure

```
projectcyan/
├── app/
│   ├── core/                          # Package 1 — Shared foundation
│   │   ├── config.py
│   │   └── models.py
│   │
│   ├── trading-engine/                # Package 2 — Backtester & data
│   │   ├── data.py
│   │   ├── features.py
│   │   └── backtester.py
│   │
│   ├── agent/                         # Package 3 — Claude AI agent
│   │   ├── tools.py
│   │   ├── prompts.py
│   │   ├── memory.py
│   │   └── graph.py
│   │
│   ├── api/                           # Package 4 — HTTP server
│   │   └── main.py
│   │
│   └── interface/                     # Package 5 — OpenClaw skill
│       └── openclaw_skill/
│           └── trading_agent_skill.py
│
├── docker-compose.yml
├── pyproject.toml
└── .env
```

---

## File Purpose Reference

### Package 1: `core` — Shared Foundation

Every other package imports from `core`. It has no dependencies on other packages in the monorepo — it is the base layer.

| File | Purpose |
|------|---------|
| `core/config.py` | Loads all environment variables (API keys, database URLs, risk thresholds, model name) into a single validated `Settings` object using Pydantic. Every package imports `settings` from here rather than reading `os.environ` directly. This is the single source of truth for all configuration. |
| `core/models.py` | Defines all shared Pydantic data models used across packages: `StrategyConfig` (what Claude sends to the backtester), `BacktestResult` (what the backtester returns to Claude), `ResearchNote` (what gets saved to memory after each cycle), and `TradeSignal` (a live trade instruction routed to the execution layer). These models enforce type safety across package boundaries. |

---

### Package 2: `trading-engine` — Backtester & Data Pipeline

This package is the quantitative sandbox. It has no knowledge of Claude or LangGraph — it is a pure data and computation layer that exposes a clean function interface the agent calls as tools.

| File | Purpose |
|------|---------|
| `engine/data.py` | Responsible for all market data ingestion. Connects to exchanges via `ccxt`, paginates through OHLCV history automatically to overcome exchange candle limits, merges in perpetual futures funding rates (fetched separately at 8h intervals), and merges in open interest history. Also provides `fetch_latest_price()` for live signal use and `validate_ohlcv()` to catch data quality issues before they corrupt feature calculations. |
| `engine/features.py` | A self-contained feature library. Every technical indicator the agent can use is registered in a `FEATURE_REGISTRY` dict using the `@register_feature` decorator. The `compute_features()` function takes a raw OHLCV DataFrame and a list of feature names, computes only the requested features, and returns the enriched DataFrame with no lookahead bias. The `get_available_features()` function returns a catalogue of all features with descriptions — this is what Claude reads to understand what signals are available to it. Adding new features to the system means adding one decorated function here. |
| `engine/backtester.py` | The core backtesting engine. Takes a `StrategyConfig`, fetches data, computes features, applies triple-barrier labeling to generate robust trade labels, trains a LightGBM model in a walk-forward split (70/30 train/test), runs the out-of-sample signals through VectorBT for realistic portfolio simulation with fees and slippage, computes SHAP feature importances, checks for overfitting by comparing in-sample vs out-of-sample Sharpe, and returns a fully populated `BacktestResult`. This is the function Claude calls most frequently during the research loop. |

---

### Package 3: `agent` — Claude AI Agent (LangGraph)

This is the brain of the system. It orchestrates Claude Opus 4.6 through a LangGraph state machine, gives it tools to interact with the trading engine and memory, and runs the autonomous research loop.

| File | Purpose |
|------|---------|
| `agent/prompts.py` | Contains `ORCHESTRATOR_SYSTEM_PROMPT` — the detailed instruction set that shapes Claude's behaviour as a quantitative researcher. It explains the research process step by step (form hypothesis → design strategy → backtest → interpret → rule features in/out → save → repeat), defines the decision thresholds for promoting strategies to paper trading, describes what good and bad research looks like, and specifies the output format Claude must follow so its responses can be parsed programmatically. This file is the most important configuration for the quality of the agent's research. |
| `agent/tools.py` | Defines all LangChain tools Claude can call during a research cycle. Each tool is a typed, documented Python function decorated with `@tool`. The tools are: `run_backtest_tool` (calls the backtester and auto-saves results to memory), `get_feature_catalogue` (returns available indicators), `query_research_history` (semantic search over prior research), `save_research_note_tool` (persists cycle learnings to ChromaDB), `get_current_market_regime` (returns live regime classification), and `generate_pine_script` (produces TradingView Pine Script for a strategy). The `RESEARCH_TOOLS` list at the bottom is what gets bound to the Claude model. |
| `agent/memory.py` | The persistent knowledge base. Uses ChromaDB as a local vector database with two collections: `research_notes` (one document per research cycle) and `backtest_results` (one document per backtest run). Provides `save_research_note()` and `save_backtest_result()` for writing, and `query_knowledge_base()` for semantic search retrieval. Also provides aggregation helpers: `get_ruled_out_features()` counts features consistently ruled out across cycles, `get_ruled_in_features()` does the same for ruled-in features, and `get_research_summary()` produces a snapshot of all accumulated knowledge — used to bootstrap Claude's context at the start of a new session. |
| `agent/graph.py` | The LangGraph state machine that governs the research loop. Defines `ResearchState` (the typed state that flows through the graph, including message history, feature rulebook, promoted strategies, and current hypothesis), three graph nodes (`orchestrator_node` where Claude reasons, `tool_node` where tool calls execute, `update_state_from_messages` which parses Claude's output to update structured state), and the conditional routing logic that decides whether to execute tools, continue to the next cycle, or end. The `run_research_loop()` function at the bottom is the entry point — it builds the graph, initialises state, and streams events in real time. |

---

### Package 4: `api` — FastAPI HTTP Server

The API layer exposes the agent and trading engine to the outside world. It is the integration point for TradingView webhooks, the OpenClaw interface, and any manual operator commands.

| File | Purpose |
|------|---------|
| `api/main.py` | Defines the FastAPI application with three primary endpoints. `POST /webhook/tradingview` receives inbound signals from TradingView alerts — it verifies the source IP against TradingView's published IP ranges, checks the passphrase, constructs a `TradeSignal`, and routes it to the agent for reasoning before execution. `POST /research/start` allows a human operator (or OpenClaw) to manually trigger a new research cycle with an optional seed hypothesis. `GET /research/status` returns the current state of the research loop. On startup, APScheduler is configured to run `run_research_loop()` automatically on the interval defined in config (default: every 60 minutes). |

---

### Package 5: `interface` — OpenClaw Skill

The human-in-the-loop layer. This is an OpenClaw skill that exposes the trading agent's capabilities to Telegram or Discord so a human can monitor, approve, and override the system via chat.

| File | Purpose |
|------|---------|
| `interface/trading_agent_skill.py` | Defines five async functions that map to OpenClaw commands: `get_research_status()` polls the API for current cycle status, `start_research()` triggers a new research cycle with an optional hypothesis, `approve_trade()` approves a pending signal for execution, `reject_trade()` rejects a signal with a reason, and `kill_switch()` halts all trading immediately. These functions are called by OpenClaw in response to messages in your Telegram or Discord channel. |

---

### Infrastructure Files

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Defines and wires together all infrastructure services: TimescaleDB (PostgreSQL with time-series extension for trade/market data storage), Redis (in-memory cache for real-time feature vectors and latest prices), ChromaDB (vector database for agent memory), the FastAPI server, the agent worker process, and Grafana (monitoring dashboards). All services share a Docker network and mount persistent volumes so data survives container restarts. |
| `pyproject.toml` | Monorepo package manifest. Declares all five packages, their dependencies, and shared dev tooling (pytest, ruff, mypy). Using a monorepo means all packages share one virtual environment and can import from each other directly. |
| `.env.example` | Template for the `.env` file. Contains all required environment variables with placeholder values. Copy this to `.env` and fill in real values before running. Never commit `.env` to version control. |

---

## How the Agent and App Connect

The system has two distinct operational flows: the **autonomous research loop** and the **live signal flow**. Here is how every component participates in each.

### Flow 1: Autonomous Research Loop

This is the primary function of the system. It runs on a schedule and requires no human input.

```
APScheduler (in api/main.py)
    │
    │  fires run_research_loop() every N minutes
    ▼
agent/graph.py — build_research_graph()
    │
    │  initialises ResearchState, enters graph
    ▼
orchestrator_node (Claude Opus 4.6)
    │
    │  reads system prompt from prompts.py
    │  reads current state (features ruled in/out, prior hypothesis)
    │  reasons about what to do next
    │  emits tool_calls
    ▼
tool_node (LangGraph ToolNode)
    │
    ├─▶ get_feature_catalogue()
    │       └── reads FEATURE_REGISTRY from engine/features.py
    │
    ├─▶ query_research_history("relevant query")
    │       └── semantic search on ChromaDB via memory.py
    │
    ├─▶ get_current_market_regime()
    │       └── returns live regime data (TODO: wire to live classifier)
    │
    ├─▶ run_backtest_tool(strategy_config)
    │       ├── calls engine/backtester.py → run_backtest()
    │       │       ├── calls engine/data.py → fetch_ohlcv()
    │       │       │       └── ccxt → Binance API → paginated OHLCV + funding + OI
    │       │       ├── calls engine/features.py → compute_features()
    │       │       ├── triple-barrier labeling
    │       │       ├── LightGBM walk-forward train/test
    │       │       ├── VectorBT portfolio simulation
    │       │       └── SHAP feature importances
    │       └── auto-saves result to ChromaDB via memory.py
    │
    ├─▶ save_research_note_tool(learnings)
    │       └── persists cycle summary to ChromaDB via memory.py
    │
    └─▶ generate_pine_script(strategy_name, result)
            └── returns Pine Script ready to paste into TradingView
    │
    ▼
update_state_from_messages
    │  parses Claude's output
    │  updates features_ruled_in, features_ruled_out
    │  extracts next_hypothesis
    ▼
should_continue_research (routing function)
    │
    ├─▶ "tools"    → back to tool_node (Claude called more tools)
    ├─▶ "continue" → back to orchestrator_node (next cycle)
    └─▶ "end"      → END (max cycles reached or agent chose to stop)
```

After each research cycle, the agent's learnings are stored in ChromaDB. The next cycle starts by querying that knowledge base, so the agent compounds knowledge over time rather than starting from scratch.

---

### Flow 2: Live Signal Flow (TradingView → Execution)

This flow handles inbound trade signals from TradingView alerts.

```
TradingView Alert fires
    │
    │  HTTP POST to https://your-server:8080/webhook/tradingview
    ▼
api/main.py — tradingview_webhook()
    │
    ├── verify source IP against TradingView's IP whitelist
    ├── verify passphrase from payload
    ├── construct TradeSignal (from core/models.py)
    │
    │  route to agent for contextual reasoning
    ▼
Claude Opus 4.6 — evaluate_signal()
    │
    ├── cross-checks ML signal confidence
    ├── checks current market regime
    ├── checks portfolio state (open positions, drawdown)
    ├── applies risk rules (max concurrent positions, kill switch)
    │
    ├─▶ APPROVED → ccxt order execution → order log → notify OpenClaw
    └─▶ REJECTED → log reason → notify OpenClaw
    │
    ▼
interface/trading_agent_skill.py (OpenClaw)
    │
    │  sends Telegram/Discord message to operator:
    │  "SIGNAL: Long BTC/USDT at $95,420 — Agent approved — Executing"
    │  or
    │  "SIGNAL: Long BTC/USDT — Agent REJECTED — Reason: elevated funding rate"
    ▼
Operator can reply:
    "approve <signal_id>"  → approve_trade()
    "reject <signal_id>"   → reject_trade()
    "kill"                 → kill_switch()
```

---

### Data Flow Summary

```
Binance API
    ↓ ccxt (data.py)
TimescaleDB ←──────────────────── stores all OHLCV history
    ↓
Redis ←──────────────────────────── caches latest feature vectors
    ↓
Feature Engine (features.py)
    ↓
Backtester (backtester.py)
    ↓
BacktestResult
    ↓
Claude Opus 4.6 (graph.py) ←──── reads/writes ChromaDB (memory.py)
    ↓
TradeSignal
    ↓
FastAPI (main.py) ←─────────────── also receives from TradingView
    ↓
ccxt order execution
    ↓
OpenClaw notification (trading_agent_skill.py)
```

---

## How to Run the Agent

### Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- A Binance account with API keys (Futures enabled)
- An Anthropic API key with Claude Opus 4.6 access
- A TradingView Pro+ subscription (for webhooks)
- An OpenClaw account with a bot connected to Telegram or Discord

---

### Step 1: Clone and Configure

```bash
# Clone the repo
git clone https://github.com/emanbau/ProjectCyan.git
cd ProjectCyan

# Copy environment template
cp .env.example .env
```

Open `.env` and fill in every value:

```env
# Anthropic
ANTHROPIC_API_KEY=sk-ant-...
CLAUDE_MODEL=claude-opus-4-6

# Database
POSTGRES_URL=postgresql://agent:yourpassword@localhost:5432/crypto_agent
POSTGRES_PASSWORD=yourpassword
REDIS_URL=redis://localhost:6379

# ChromaDB
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Exchange
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET=your_binance_secret

# TradingView
TV_WEBHOOK_PASSPHRASE=choose_a_strong_random_passphrase

# Risk parameters
MAX_DRAWDOWN_KILL=0.15
MAX_CONCURRENT_POSITIONS=5
KELLY_FRACTION=0.25

# Research loop
RESEARCH_CYCLE_INTERVAL_MINUTES=60
MIN_BACKTEST_SHARPE=1.5
MAX_BACKTEST_DRAWDOWN=0.12
```

---

### Step 2: Install Python Dependencies

```bash
# Install all packages in development mode from monorepo root
pip install -e app/core \
            -e app/trading-engine \
            -e app/agent \
            -e app/api \
            -e app/interface
```

Or install everything in one command using the root `pyproject.toml`:

```bash
pip install -e ".[all]"
```

---

### Step 3: Start Infrastructure Services

```bash
# Start PostgreSQL (TimescaleDB), Redis, ChromaDB, and Grafana
docker-compose up -d postgres redis chromadb grafana

# Verify all services are healthy
docker-compose ps
```

Expected output — all services should show `healthy` or `running`:
```
NAME        STATUS          PORTS
postgres    running         0.0.0.0:5432->5432/tcp
redis       running         0.0.0.0:6379->6379/tcp
chromadb    running         0.0.0.0:8000->8000/tcp
grafana     running         0.0.0.0:3000->3000/tcp
```

---

### Step 4: Initialise the Database

```bash
# Run database migrations to create TimescaleDB tables
python -m crypto_agent.engine.db_init
```

---

### Step 5: Validate the Data Pipeline

Before running the agent, confirm data ingestion works:

```bash
python - <<EOF
from crypto_agent.engine.data import fetch_ohlcv, validate_ohlcv

df = fetch_ohlcv("BTC/USDT", timeframe="1h", lookback_days=30)
print(df.tail())
print(f"Rows: {len(df)}, Columns: {list(df.columns)}")
validate_ohlcv(df, "BTC/USDT")
print("Data pipeline OK")
EOF
```

---

### Step 6: Validate the Backtester

Run a quick sanity-check backtest before connecting Claude:

```bash
python - <<EOF
from crypto_agent.core.models import StrategyConfig
from crypto_agent.engine.backtester import run_backtest

config = StrategyConfig(
    name="sanity_check",
    description="Quick validation test",
    features=["rsi_14", "volume_zscore"],
    entry_logic="RSI oversold with volume confirmation",
    exit_logic="RSI overbought or stop loss",
    assets=["BTC/USDT"],
    timeframe="1h",
    lookback_days=180,
)

result = run_backtest(config)
print(result.model_dump_json(indent=2))
EOF
```

A result with any Sharpe ratio and a verdict means the backtester is working correctly. The specific numbers don't matter at this stage.

---

### Step 7: Start the API Server

```bash
# Start the FastAPI server
uvicorn app.api.main:app --host 0.0.0.0 --port 8080 --reload

# Confirm it's running
curl http://localhost:8080/research/status
# Expected: {"cycle_count": 0, "last_cycle": null}
```

---

### Step 8: Trigger the First Research Cycle Manually

```bash
# Kick off the research loop with a seed hypothesis
curl -X POST "http://localhost:8080/research/start" \
     -H "Content-Type: application/json" \
     -d '{"hypothesis": "RSI mean reversion combined with volume confirmation may produce consistent edge on BTC and ETH in ranging markets"}'
```

Watch the logs — you should see Claude reasoning through the hypothesis, calling tools, running backtests, and saving research notes.

---

### Step 9: Configure TradingView Webhook

1. In TradingView, open the Pine Script strategy the agent generated
2. Click **Alerts** → **Create Alert**
3. Set **Webhook URL** to: `https://your-server-domain:8080/webhook/tradingview`
4. Set the **Message** body to:
```json
{
  "ticker": "{{ticker}}",
  "exchange": "{{exchange}}",
  "action": "{{strategy.order.action}}",
  "price": "{{close}}",
  "passphrase": "your_tv_webhook_passphrase_from_env"
}
```
5. Save the alert

---

### Step 10: Install the OpenClaw Skill

1. Copy `app/interface/openclaw_skill/trading_agent_skill.py` to your OpenClaw skills directory
2. Update `AGENT_API_URL` in the skill file to point to your server
3. Restart OpenClaw
4. In Telegram/Discord, type: `get research status` to confirm the connection

---

### Running in Production (Full Docker)

Once everything is validated, run the complete stack in Docker:

```bash
# Build and start everything
docker-compose up -d --build

# Follow logs from the agent
docker-compose logs -f agent

# Follow logs from the API server
docker-compose logs -f api
```

---

### Monitoring

- **Grafana dashboard**: `http://your-server:3000` (default login: admin/admin)
- **API docs**: `http://your-server:8080/docs` (FastAPI auto-generated Swagger UI)
- **Research status**: `GET http://your-server:8080/research/status`
- **ChromaDB**: Research notes and backtests are browsable at `http://your-server:8000`

---

### Troubleshooting

| Problem | Likely Cause | Fix |
|---------|-------------|-----|
| `ccxt.ExchangeError` on data fetch | Binance API key not configured for Futures | Enable Futures trading on your Binance API key settings |
| `chromadb.errors.InvalidDimensionException` | ChromaDB collection dimension mismatch | Delete the ChromaDB volume and restart: `docker-compose down -v chromadb && docker-compose up -d chromadb` |
| Claude not calling tools | System prompt not loading correctly | Check `ANTHROPIC_API_KEY` in `.env` and confirm model string is `claude-opus-4-6` |
| TradingView webhook returning 403 | IP or passphrase mismatch | Confirm `TV_WEBHOOK_PASSPHRASE` in `.env` matches the value in your TradingView alert message |
| Research loop not auto-starting | APScheduler not initialising | Check FastAPI startup logs for scheduler errors; confirm `RESEARCH_CYCLE_INTERVAL_MINUTES` is set |