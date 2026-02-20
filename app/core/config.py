# packages/core/crypto_agent/core/config.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # Anthropic
    anthropic_api_key: str
    claude_model: str = "claude-opus-4-6"

    # Database
    postgres_url: str
    redis_url: str

    # ChromaDB
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Exchange
    binance_api_key: str
    binance_secret: str

    # TradingView
    tv_webhook_passphrase: str

    # Risk
    max_drawdown_kill: float = 0.15
    max_concurrent_positions: int = 5
    kelly_fraction: float = 0.25

    # Research loop
    research_cycle_interval_minutes: int = 60
    min_backtest_sharpe: float = 1.5
    max_backtest_drawdown: float = 0.12

    class Config:
        env_file = ".env"

settings = Settings()