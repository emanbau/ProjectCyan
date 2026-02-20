# packages/interface/openclaw_skill/trading_agent_skill.py
"""
OpenClaw skill that connects to the trading agent API.
Install this skill in OpenClaw to receive trade signals
and approve/reject trades via Telegram or Discord.
"""
import httpx
from typing import Optional

AGENT_API_URL = "http://your-server:8000"

async def get_research_status() -> str:
    """Get the current status of the AI research loop"""
    async with httpx.AsyncClient() as client:
        resp = await client.get(f"{AGENT_API_URL}/research/status")
        data = resp.json()
    return f"Research cycle #{data['cycle_count']} | Last: {data['last_cycle']}"

async def start_research(hypothesis: Optional[str] = None) -> str:
    """Trigger a new research cycle with an optional hypothesis"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AGENT_API_URL}/research/start",
            params={"hypothesis": hypothesis or ""}
        )
    return f"Research loop started. {resp.json()}"

async def approve_trade(signal_id: str) -> str:
    """Approve a pending trade signal for execution"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AGENT_API_URL}/trades/approve/{signal_id}"
        )
    return f"Trade {signal_id} approved: {resp.json()}"

async def reject_trade(signal_id: str, reason: str) -> str:
    """Reject a pending trade signal"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(
            f"{AGENT_API_URL}/trades/reject/{signal_id}",
            json={"reason": reason}
        )
    return f"Trade rejected: {resp.json()}"

async def kill_switch() -> str:
    """Emergency: halt all trading immediately"""
    async with httpx.AsyncClient() as client:
        resp = await client.post(f"{AGENT_API_URL}/trading/halt")
    return f"KILL SWITCH ACTIVATED: {resp.json()}"