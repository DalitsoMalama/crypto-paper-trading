"""Event-driven backtester, shared by every strategy.

Per bar: (1) fill orders queued at the previous close at this bar's open (sells first, then buys in priority
order), (2) check stops/targets against this bar's low/high, (3) mark equity at the close and ratchet trailing
stops, (4) read entry/exit signals at the close (plus the time limit) and queue orders for the next bar.

`start`/`end` are fractions of the tradable bars (after warmup), so you can test on one slice of history and check
the result on a later slice the settings were never tuned on. The Claude agent is not used in backtests.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from .broker import PaperBroker
from .data import timeframe_seconds
from .risk import RiskManager
from .strategies import atr, build_strategy


def prepare_bot(bot: dict, data: dict[str, pd.DataFrame]) -> tuple:
    strategy = build_strategy(bot)
    frames = strategy.prepare_all(data)
    for f in frames.values():
        f["atr"] = atr(f, bot["exits"]["atr_period"])
        if "priority" not in f:
            f["priority"] = 0.0
    index = sorted(set.intersection(*(set(f.index) for f in frames.values())))
    frames = {sym: f.loc[index] for sym, f in frames.items()}
    warmup = max(strategy.warmup(), bot["exits"]["atr_period"] * 3)
    return strategy, frames, index, warmup


def run_backtest(bot: dict, data: dict[str, pd.DataFrame], start: float = 0.0, end: float = 1.0,
                 prepared: tuple | None = None) -> dict:
    strategy, frames, index, warmup = prepared or prepare_bot(bot, data)
    tradable = len(index) - warmup
    if tradable < 50:
        raise ValueError(f"{bot['id']}: only {len(index)} bars, {warmup} needed for warmup. Use more --days.")
    i0, i1 = warmup + int(tradable * start), warmup + int(tradable * end)
    ex = bot["exits"]

    broker = PaperBroker(bot["starting_cash"], bot["fee_rate"], bot["slippage_bps"])
    risk = RiskManager(bot)
    syms = list(frames)
    cols = ["open", "high", "low", "close", "atr", "entry", "exit", "priority"]
    arr = {s: frames[s][cols].to_numpy(dtype=float) for s in syms}
    O, H, L, C, A, EN, EX, PR = range(8)
    pending_sell: dict[str, str] = {}
    pending_buy: dict[str, float] = {}  # symbol -> ATR at signal bar
    entry_bar: dict[str, int] = {}
    last_exit: dict[str, int] = {}
    equity_curve, bars_in_market = [], 0

    def close_pos(sym, px, ts, reason, i):
        broker.sell(sym, px, ts, reason)
        last_exit[sym] = i
        entry_bar.pop(sym, None)

    for i in range(i0, i1):
        ts = index[i]
        row = {s: arr[s][i] for s in syms}
        opens = {s: row[s][O] for s in syms}

        # 1. fill queued orders at the open
        for sym, reason in pending_sell.items():
            close_pos(sym, opens[sym], ts, reason, i)
        for sym in sorted(pending_buy, key=lambda s: arr[s][i - 1][PR]):
            ok, _ = risk.can_open(len(broker.positions), check_kill_file=False)
            if not ok:
                break
            stop, take = risk.exit_levels(opens[sym], pending_buy[sym])
            qty = risk.size(broker.equity(opens), broker.cash, opens[sym], stop)
            if broker.buy(sym, qty, opens[sym], ts, stop, take, strategy.name):
                entry_bar[sym] = i
        pending_sell.clear()
        pending_buy.clear()

        # 2. stops and targets inside the bar (stop assumed first if both are touched: conservative)
        for sym, pos in list(broker.positions.items()):
            r = row[sym]
            if r[L] <= pos.stop:
                close_pos(sym, min(pos.stop, r[O]), ts, "trailing stop" if pos.stop >= pos.entry_price else "stop loss", i)
            elif r[H] >= pos.take:
                close_pos(sym, max(pos.take, r[O]), ts, "take profit", i)

        # 3. mark to market, then ratchet trailing stops using this closed bar
        equity = broker.equity({s: row[s][C] for s in syms})
        risk.update(equity, ts)
        equity_curve.append((ts, equity))
        bars_in_market += bool(broker.positions)
        for sym in broker.positions:
            broker.trail(sym, row[sym][H], row[sym][A], ex["trail_atr"])

        # 4. signals at the close -> orders for the next open
        if i == i1 - 1:
            continue
        for sym in syms:
            r = row[sym]
            if sym in broker.positions:
                if r[EX]:
                    pending_sell[sym] = "signal exit"
                elif ex["max_hold_bars"] and i - entry_bar.get(sym, i) >= ex["max_hold_bars"]:
                    pending_sell[sym] = "time limit"
            elif r[EN] and not np.isnan(r[A]) and i - last_exit.get(sym, -10**9) > ex["cooldown_bars"]:
                pending_buy[sym] = r[A]

    for sym in list(broker.positions):
        broker.sell(sym, arr[sym][i1 - 1][C], index[i1 - 1], "end of test")

    curve = pd.Series(dict(equity_curve), name="equity")
    curve.iloc[-1] = broker.cash
    hold = {s: arr[s][i1 - 1][C] / arr[s][i0][O] - 1 for s in syms}
    m = metrics(curve, broker.trades, bot["timeframe"])
    m["exposure_pct"] = bars_in_market / max(1, i1 - i0) * 100
    return {
        "equity": curve,
        "trades": pd.DataFrame(broker.trades),
        "metrics": m,
        "buy_and_hold": hold,
        "halted": risk.halted_reason,
        "period": (index[i0], index[i1 - 1]),
    }


def metrics(curve: pd.Series, trades: list[dict], timeframe: str) -> dict:
    rets = curve.pct_change().dropna()
    bars_per_year = 365 * 86400 / timeframe_seconds(timeframe)
    sharpe = rets.mean() / rets.std() * math.sqrt(bars_per_year) if rets.std() > 0 else 0.0
    wins = [t["pnl"] for t in trades if t["pnl"] > 0]
    losses = [t["pnl"] for t in trades if t["pnl"] <= 0]
    if losses and sum(losses) < 0:
        pf = sum(wins) / -sum(losses)
    else:
        pf = float("inf") if wins else 0.0
    return {
        "start_equity": curve.iloc[0],
        "end_equity": curve.iloc[-1],
        "total_return_pct": (curve.iloc[-1] / curve.iloc[0] - 1) * 100,
        "max_drawdown_pct": (curve / curve.cummax() - 1).min() * 100,
        "sharpe": sharpe,
        "trades": len(trades),
        "win_rate_pct": len(wins) / len(trades) * 100 if trades else 0.0,
        "profit_factor": pf,
        "avg_trade_pct": sum(t["return_pct"] for t in trades) / len(trades) if trades else 0.0,
    }


def save_results(res: dict, name: str, out_dir: str = "results") -> Path:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    res["equity"].to_csv(out / f"{name}_equity.csv")
    res["trades"].to_csv(out / f"{name}_trades.csv", index=False)
    return out


def print_report(res: dict, title: str) -> None:
    m = res["metrics"]
    a, b = res["period"]
    print(f"\n=== {title}: {a:%Y-%m-%d} to {b:%Y-%m-%d} (simulated, after fees and slippage) ===")
    print(f"Equity:        {m['start_equity']:,.2f} -> {m['end_equity']:,.2f}  ({m['total_return_pct']:+.2f}%)")
    print(f"Max drawdown:  {m['max_drawdown_pct']:.2f}%   Sharpe {m['sharpe']:.2f}   in market {m['exposure_pct']:.0f}% of the time")
    print(f"Trades:        {m['trades']}  win rate {m['win_rate_pct']:.1f}%  avg {m['avg_trade_pct']:+.2f}%  profit factor {m['profit_factor']:.2f}")
    print("Buy & hold:    " + ", ".join(f"{s} {r * 100:+.1f}%" for s, r in res["buy_and_hold"].items()))
    if res["halted"]:
        print(f"KILL SWITCH:   {res['halted']}")
    if len(res["trades"]):
        print("Exit reasons:  " + ", ".join(f"{k} {v}" for k, v in res["trades"]["reason"].value_counts().items()))
