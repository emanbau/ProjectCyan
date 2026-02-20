import uuid
import json
from datetime import datetime
from typing import Optional
import chromadb
from chromadb.config import Settings
from langchain_anthropic import ChatAnthropic
from app.core.config import settings

# ── ChromaDB Client Setup ─────────────────────────────────────────
_client = chromadb.HttpClient(
    host=settings.chroma_host,
    port=settings.chroma_port,
    settings=Settings(anonymized_telemetry=False)
)

# Two collections — one for research notes, one for backtest results
_research_collection = _client.get_or_create_collection(
    name="research_notes",
    metadata={"hnsw:space": "cosine"}
)

_backtest_collection = _client.get_or_create_collection(
    name="backtest_results",
    metadata={"hnsw:space": "cosine"}
)


# ── Embedding Helper ──────────────────────────────────────────────
def _embed(text: str) -> list[float]:
    """
    Use ChromaDB's default embedding function (all-MiniLM-L6-v2).
    Runs locally — no API call needed for embeddings.
    ChromaDB handles this automatically when you don't pass embeddings manually.
    """
    # ChromaDB auto-embeds when embeddings=None, so we just return the text
    # and let the collection handle it. This function is a placeholder
    # in case you want to swap to a custom embedder later.
    return None  # signals ChromaDB to use its default embedder


# ── Save Research Note ────────────────────────────────────────────
def save_research_note(
    hypothesis: str,
    strategies_tested: list[str],
    features_ruled_in: list[str],
    features_ruled_out: list[str],
    learnings: str,
    next_hypothesis: str,
    best_sharpe: Optional[float] = None,
    best_strategy_name: Optional[str] = None,
) -> str:
    """
    Persist a research cycle note to ChromaDB.
    The document text is a rich natural language summary so semantic
    search works well. Metadata fields allow structured filtering.
    """
    note_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    # Build a rich text document for semantic search
    document = f"""
RESEARCH CYCLE — {timestamp}

HYPOTHESIS: {hypothesis}

STRATEGIES TESTED: {', '.join(strategies_tested)}

FEATURES RULED IN: {', '.join(features_ruled_in) if features_ruled_in else 'None'}
FEATURES RULED OUT: {', '.join(features_ruled_out) if features_ruled_out else 'None'}

LEARNINGS:
{learnings}

NEXT HYPOTHESIS: {next_hypothesis}
    """.strip()

    # Structured metadata for filtered queries
    metadata = {
        "type": "research_note",
        "timestamp": timestamp,
        "hypothesis": hypothesis[:200],  # ChromaDB metadata has size limits
        "features_ruled_in": json.dumps(features_ruled_in),
        "features_ruled_out": json.dumps(features_ruled_out),
        "strategies_tested": json.dumps(strategies_tested),
        "best_sharpe": best_sharpe or 0.0,
        "best_strategy_name": best_strategy_name or "",
        "next_hypothesis": next_hypothesis[:200],
    }

    _research_collection.add(
        documents=[document],
        metadatas=[metadata],
        ids=[note_id]
    )

    return note_id


# ── Save Backtest Result ──────────────────────────────────────────
def save_backtest_result(
    strategy_name: str,
    strategy_config: dict,
    result: dict,
) -> str:
    """
    Persist a backtest result to ChromaDB so the agent can
    retrieve similar past experiments during future research cycles.
    """
    result_id = str(uuid.uuid4())
    timestamp = datetime.utcnow().isoformat()

    document = f"""
BACKTEST RESULT — {timestamp}

STRATEGY: {strategy_name}
DESCRIPTION: {strategy_config.get('description', '')}

FEATURES USED: {', '.join(strategy_config.get('features', []))}
ENTRY LOGIC: {strategy_config.get('entry_logic', '')}
EXIT LOGIC: {strategy_config.get('exit_logic', '')}
ASSETS: {', '.join(strategy_config.get('assets', []))}
TIMEFRAME: {strategy_config.get('timeframe', '')}

RESULTS:
- Sharpe Ratio: {result.get('sharpe_ratio', 'N/A')}
- Max Drawdown: {result.get('max_drawdown', 'N/A')}
- Win Rate: {result.get('win_rate', 'N/A')}
- Profit Factor: {result.get('profit_factor', 'N/A')}
- Total Return: {result.get('total_return', 'N/A')}
- N Trades: {result.get('n_trades', 'N/A')}
- Overfitting Score: {result.get('overfitting_score', 'N/A')}

FEATURE IMPORTANCES:
{json.dumps(result.get('feature_importances', {}), indent=2)}

VERDICT: {result.get('verdict', 'N/A')}
VERDICT REASON: {result.get('verdict_reason', 'N/A')}
    """.strip()

    metadata = {
        "type": "backtest_result",
        "timestamp": timestamp,
        "strategy_name": strategy_name,
        "features": json.dumps(strategy_config.get("features", [])),
        "assets": json.dumps(strategy_config.get("assets", [])),
        "timeframe": strategy_config.get("timeframe", ""),
        "sharpe_ratio": float(result.get("sharpe_ratio", 0.0)),
        "max_drawdown": float(result.get("max_drawdown", 0.0)),
        "win_rate": float(result.get("win_rate", 0.0)),
        "verdict": result.get("verdict", "discard"),
    }

    _backtest_collection.add(
        documents=[document],
        metadatas=[metadata],
        ids=[result_id]
    )

    return result_id


# ── Query Knowledge Base ──────────────────────────────────────────
def query_knowledge_base(
    query: str,
    top_k: int = 5,
    collection: str = "both",
    filters: Optional[dict] = None,
) -> list[dict]:
    """
    Semantic search across research notes and/or backtest results.
    Returns a list of relevant documents with their metadata.

    Args:
        query:      Natural language search query
        top_k:      Number of results to return
        collection: 'research_notes', 'backtest_results', or 'both'
        filters:    Optional ChromaDB where clause e.g.
                    {"sharpe_ratio": {"$gte": 1.5}}
                    {"verdict": {"$eq": "promote"}}
    """
    results = []

    def _query_collection(col, n):
        kwargs = dict(query_texts=[query], n_results=min(n, col.count() or 1))
        if filters:
            kwargs["where"] = filters
        return col.query(**kwargs)

    if collection in ("research_notes", "both"):
        try:
            res = _query_collection(_research_collection, top_k)
            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "source": "research_note",
                    "document": doc,
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i],
                })
        except Exception as e:
            results.append({"source": "research_note", "error": str(e)})

    if collection in ("backtest_results", "both"):
        try:
            res = _query_collection(_backtest_collection, top_k)
            for i, doc in enumerate(res["documents"][0]):
                results.append({
                    "source": "backtest_result",
                    "document": doc,
                    "metadata": res["metadatas"][0][i],
                    "distance": res["distances"][0][i],
                })
        except Exception as e:
            results.append({"source": "backtest_result", "error": str(e)})

    # Sort by semantic similarity (lower distance = more relevant)
    results.sort(key=lambda x: x.get("distance", 999))
    return results[:top_k]


# ── Convenience Queries ───────────────────────────────────────────
def get_best_strategies(min_sharpe: float = 1.5, limit: int = 10) -> list[dict]:
    """Fetch all promoted strategies above a Sharpe threshold"""
    try:
        res = _backtest_collection.query(
            query_texts=["high sharpe promoted strategy"],
            n_results=min(limit, _backtest_collection.count() or 1),
            where={
                "$and": [
                    {"sharpe_ratio": {"$gte": min_sharpe}},
                    {"verdict": {"$eq": "promote"}},
                ]
            }
        )
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(res["documents"][0], res["metadatas"][0])
        ]
    except Exception:
        return []


def get_ruled_out_features() -> list[str]:
    """
    Aggregate all features ruled out across research history.
    Used to prevent the agent re-testing known dead ends.
    """
    try:
        # Fetch all research notes (no semantic filter, just get everything)
        res = _research_collection.get(
            where={"type": {"$eq": "research_note"}},
            include=["metadatas"]
        )
        all_ruled_out = []
        for meta in res["metadatas"]:
            ruled_out = json.loads(meta.get("features_ruled_out", "[]"))
            all_ruled_out.extend(ruled_out)

        # Return features ruled out more than once (consensus)
        from collections import Counter
        counts = Counter(all_ruled_out)
        return [f for f, count in counts.items() if count >= 2]
    except Exception:
        return []


def get_ruled_in_features() -> list[str]:
    """
    Aggregate all features consistently ruled in across research history.
    """
    try:
        res = _research_collection.get(
            where={"type": {"$eq": "research_note"}},
            include=["metadatas"]
        )
        all_ruled_in = []
        for meta in res["metadatas"]:
            ruled_in = json.loads(meta.get("features_ruled_in", "[]"))
            all_ruled_in.extend(ruled_in)

        from collections import Counter
        counts = Counter(all_ruled_in)
        return [f for f, count in counts.items() if count >= 2]
    except Exception:
        return []


def get_research_summary() -> dict:
    """
    High-level summary of all research conducted so far.
    Useful for injecting into Claude's context at the start of a session.
    """
    return {
        "total_research_cycles": _research_collection.count(),
        "total_backtests": _backtest_collection.count(),
        "features_ruled_in": get_ruled_in_features(),
        "features_ruled_out": get_ruled_out_features(),
        "best_strategies": get_best_strategies(min_sharpe=1.5, limit=5),
    }