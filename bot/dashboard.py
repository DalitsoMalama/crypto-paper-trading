"""Write docs/data.json: everything the web dashboard (docs/index.html) shows."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd


def _num(x):
    return None if x is None or (isinstance(x, float) and (math.isinf(x) or math.isnan(x))) else x


def _max_drawdown(history: list[list]) -> float:
    peak, worst = None, 0.0
    for _, eq in history:
        peak = eq if peak is None else max(peak, eq)
        worst = min(worst, eq / peak - 1)
    return worst * 100


def bot_summary(r) -> dict:
    b, br = r.bot, r.broker
    trades = br.trades
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    positions = []
    for sym, p in br.positions.items():
        price = r.prices.get(sym, p.entry_price)
        cost = p.qty * p.entry_price + p.entry_fee
        positions.append({
            "symbol": sym, "qty": p.qty, "entry": p.entry_price, "price": price, "value": p.qty * price,
            "pnl": p.qty * price - cost, "pnl_pct": (p.qty * price - cost) / cost * 100,
            "stop": p.stop, "take": _num(p.take), "opened_at": p.opened_at, "note": p.note,
        })
    history = r.history[-2000:]
    return {
        "id": b["id"], "name": b["name"], "description": b["description"], "strategy": b["strategy"],
        "timeframe": b["timeframe"], "symbols": b["symbols"],
        "starting_cash": b["starting_cash"], "equity": r.equity, "cash": br.cash,
        "return_pct": (r.equity / b["starting_cash"] - 1) * 100,
        "max_drawdown_pct": _max_drawdown(r.history),
        "realized_pnl": sum(t["pnl"] for t in trades),
        "trades_count": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else None,
        "profit_factor": _num(sum(wins) / -sum(losses)) if losses and sum(losses) < 0 else None,
        "positions": positions,
        "trades": list(reversed(trades[-100:])),
        "history": history,
        "signals": r.signals,
        "halted": r.risk.halted_reason,
        "daily_blocked": r.risk.daily_blocked,
        "started_at": r.started_at, "updated_at": r.updated_at, "last_bar": r.last_bar, "error": r.error,
    }


def write_dashboard(cfg: dict, runners) -> Path:
    path = Path(cfg["paper"]["dashboard_file"])
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "updated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "exchange": cfg["exchange"],
        "mode": "paper",
        "bots": [bot_summary(r) for r in runners],
    }
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, default=str, separators=(",", ":")))
    tmp.replace(path)
    return path
