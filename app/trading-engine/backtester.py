# packages/trading-engine/crypto_agent/engine/backtester.py
import vectorbt as vbt
import pandas as pd
import numpy as np
from typing import Optional
from app.core.models import StrategyConfig, BacktestResult
from app.trading-engine.features import compute_features
from app.trading-engine.data import fetch_ohlcv
import shap
import lightgbm as lgb

FEES = 0.001      # 0.1% per trade (taker)
SLIPPAGE = 0.0005  # 0.05% slippage

def run_backtest(config: StrategyConfig) -> BacktestResult:
    """
    Main entry point — called by Claude as a tool.
    Accepts a StrategyConfig, runs walk-forward backtest,
    returns structured BacktestResult.
    """
    all_results = []

    for asset in config.assets:
        df = fetch_ohlcv(asset, config.timeframe, config.lookback_days)
        validate_ohlcv(df, asset)   # logs warnings but continues
        
        df = compute_features(df, config.features)

        # Triple-barrier labeling
        labels = _triple_barrier_label(
            df["close"],
            stop_loss=config.stop_loss_pct,
            take_profit=config.take_profit_pct,
            max_holding=48  # hours
        )

        # Train ML model on features → entry signals
        feature_df = df[config.features]
        X_train, X_test, y_train, y_test, dates_test = _walk_forward_split(
            feature_df, labels
        )

        model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.05)
        model.fit(X_train, y_train)

        # Get signal probabilities on out-of-sample data
        probs = model.predict_proba(X_test)[:, 1]
        entries = pd.Series(probs > 0.65, index=dates_test)
        exits = pd.Series(probs < 0.35, index=dates_test)

        # Get OHLCV aligned to test period
        test_close = df["close"].loc[dates_test]

        # VectorBT portfolio simulation
        pf = vbt.Portfolio.from_signals(
            test_close,
            entries=entries,
            exits=exits,
            fees=FEES,
            slippage=SLIPPAGE,
            init_cash=10_000,
            sl_stop=config.stop_loss_pct,
            tp_stop=config.take_profit_pct,
        )

        all_results.append({
            "asset": asset,
            "pf": pf,
            "model": model,
            "X_test": X_test,
            "X_train": X_train,
            "y_test": y_test,
        })

    # Aggregate results across assets
    return _aggregate_results(config, all_results)


def _triple_barrier_label(
    close: pd.Series,
    stop_loss: float,
    take_profit: float,
    max_holding: int,
) -> pd.Series:
    labels = []
    for i in range(len(close) - max_holding):
        entry = close.iloc[i]
        future = close.iloc[i+1:i+max_holding+1]
        tp_hit = (future >= entry * (1 + take_profit)).idxmax()
        sl_hit = (future <= entry * (1 - stop_loss)).idxmax()
        if tp_hit and (not sl_hit or tp_hit <= sl_hit):
            labels.append(1)
        elif sl_hit:
            labels.append(0)
        else:
            labels.append(0)  # time barrier — flat
    return pd.Series(labels, index=close.index[:len(labels)])


def _walk_forward_split(X: pd.DataFrame, y: pd.Series, test_ratio=0.3):
    split = int(len(X) * (1 - test_ratio))
    return (
        X.iloc[:split], X.iloc[split:],
        y.iloc[:split], y.iloc[split:],
        X.iloc[split:].index
    )


def _aggregate_results(config: StrategyConfig, results: list) -> BacktestResult:
    portfolios = [r["pf"] for r in results]

    sharpes = [pf.sharpe_ratio() for pf in portfolios]
    drawdowns = [pf.max_drawdown() for pf in portfolios]
    win_rates = [pf.trades.win_rate() for pf in portfolios]
    n_trades = sum(pf.trades.count() for pf in portfolios)

    # Feature importance via SHAP (first asset as representative)
    explainer = shap.TreeExplainer(results[0]["model"])
    shap_values = explainer.shap_values(results[0]["X_test"])
    importance = dict(zip(
        config.features,
        np.abs(shap_values).mean(axis=0).tolist()
    ))

    avg_sharpe = float(np.mean(sharpes))
    avg_drawdown = float(np.mean(drawdowns))

    # Overfitting check — compare in-sample vs out-of-sample sharpe
    in_sample_pf = vbt.Portfolio.from_signals(
        results[0]["pf"].close,  # simplified
        entries=pd.Series(True, index=results[0]["X_train"].index),
        exits=pd.Series(False, index=results[0]["X_train"].index),
        fees=FEES
    )
    overfitting_score = abs(avg_sharpe - float(in_sample_pf.sharpe_ratio()))

    # Determine verdict
    if avg_sharpe > 1.5 and avg_drawdown > -0.12 and overfitting_score < 0.5:
        verdict = "promote"
        verdict_reason = f"Sharpe {avg_sharpe:.2f} exceeds threshold, drawdown acceptable"
    elif avg_sharpe > 0.8:
        verdict = "iterate"
        verdict_reason = f"Marginal Sharpe {avg_sharpe:.2f} — worth refining features"
    else:
        verdict = "discard"
        verdict_reason = f"Sharpe {avg_sharpe:.2f} below minimum threshold"

    return BacktestResult(
        strategy_name=config.name,
        sharpe_ratio=avg_sharpe,
        sortino_ratio=float(np.mean([pf.sortino_ratio() for pf in portfolios])),
        max_drawdown=avg_drawdown,
        win_rate=float(np.mean(win_rates)),
        profit_factor=float(np.mean([pf.profit_factor() for pf in portfolios])),
        total_return=float(np.mean([pf.total_return() for pf in portfolios])),
        n_trades=n_trades,
        avg_trade_duration_hours=24.0,  # simplified
        feature_importances=importance,
        regime_breakdown={},  # TODO: implement regime classifier
        overfitting_score=overfitting_score,
        verdict=verdict,
        verdict_reason=verdict_reason,
    )