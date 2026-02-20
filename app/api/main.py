from fastapi import FastAPI, BackgroundTasks, HTTPException, Request
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from app.agent.graph import run_research_loop
from app.core.models import TradeSignal
from app.core.config import settings

app = FastAPI(title="Crypto AI Agent API")
scheduler = AsyncIOScheduler()

# ── Research Loop Scheduling ──────────────────────────────────────
@app.on_event("startup")
async def start_scheduler():
    scheduler.add_job(
        run_research_loop,
        "interval",
        minutes=settings.research_cycle_interval_minutes,
        id="research_loop"
    )
    scheduler.start()

# ── TradingView Webhook ───────────────────────────────────────────
TRADINGVIEW_IPS = {"52.89.214.238", "34.212.75.30", "54.218.53.128"}

@app.post("/webhook/tradingview")
async def tradingview_webhook(payload: dict, request: Request):
    # Verify source
    if request.client.host not in TRADINGVIEW_IPS:
        raise HTTPException(403, "Unauthorized IP")
    if payload.get("passphrase") != settings.tv_webhook_passphrase:
        raise HTTPException(403, "Invalid passphrase")

    signal = TradeSignal(
        asset=payload["ticker"],
        direction=payload["action"],
        entry_price=float(payload["price"]),
        stop_loss=float(payload["price"]) * 0.97,
        take_profit=float(payload["price"]) * 1.04,
        position_size_pct=0.05,
        confidence=0.75,
        strategy_name=payload.get("strategy", "tradingview"),
        reasoning="TradingView alert triggered",
        source="tradingview"
    )

    # Route to agent for approval reasoning
    # In full implementation: call agent evaluate_signal()
    return {"status": "received", "signal_id": "abc123"}

# ── Manual Research Trigger ───────────────────────────────────────
@app.post("/research/start")
async def start_research(
    hypothesis: str = "",
    background_tasks: BackgroundTasks = None
):
    background_tasks.add_task(run_research_loop, hypothesis)
    return {"status": "research loop started", "hypothesis": hypothesis}

@app.get("/research/status")
async def research_status():
    # TODO: return current cycle status from DB
    return {"cycle_count": 0, "last_cycle": None}