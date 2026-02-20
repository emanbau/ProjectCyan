# packages/core/crypto_agent/core/models.py
from pydantic import BaseModel
from typing import Optional, Literal
from datetime import datetime

class StrategyConfig(BaseModel):
    """Schema Claude uses to define a strategy"""
    name: str
    description: str
    features: list[str]
    entry_logic: str           # Human readable description
    exit_logic: str
    assets: list[str]
    timeframe: Literal["5m", "15m", "1h", "4h", "1d"]
    lookback_days: int = 730
    stop_loss_pct: float = 0.03
    take_profit_pct: float = 0.06
    position_sizing: Literal["fixed", "half_kelly", "quarter_kelly"] = "quarter_kelly"

class BacktestResult(BaseModel):
    strategy_name: str
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_return: float
    n_trades: int
    avg_trade_duration_hours: float
    feature_importances: dict[str, float]
    regime_breakdown: dict[str, float]   # sharpe by regime
    overfitting_score: float             # in-sample vs out-of-sample gap
    verdict: Literal["promote", "iterate", "discard"]
    verdict_reason: str

class ResearchNote(BaseModel):
    cycle_id: str
    timestamp: datetime
    hypothesis: str
    strategies_tested: list[str]
    features_ruled_in: list[str]
    features_ruled_out: list[str]
    best_result: Optional[BacktestResult]
    learnings: str
    next_hypothesis: str

class TradeSignal(BaseModel):
    asset: str
    direction: Literal["long", "short"]
    entry_price: float
    stop_loss: float
    take_profit: float
    position_size_pct: float
    confidence: float
    strategy_name: str
    reasoning: str
    source: Literal["agent", "tradingview", "ml_model"]