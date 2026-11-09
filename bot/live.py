"""Live PAPER trading for several bots at once.

`run_tick` does one pass over every bot: fetch the latest closed candles and prices, check stops, act on any newly
closed candle, save each bot's state to state/<id>.json and refresh the dashboard data. The local `paper` command
calls it in a loop; the cloud job calls it once every few minutes. All state lives in files, so either works.

There is deliberately no real-order code in this project. Moving to real money should be a separate, deliberate
change made by a human after a long paper-trading record.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pandas as pd

from .broker import PaperBroker
from .data import fetch_recent, make_exchange, timeframe_seconds
from .risk import RiskManager
from .strategies import atr, build_strategy

log = logging.getLogger("paper")
HISTORY_EVERY_S = 30 * 60  # equity snapshot spacing for the dashboard chart


class BotRunner:
    def __init__(self, bot: dict, state_dir: Path):
        self.bot = bot
        self.strategy = build_strategy(bot)
        self.tf_s = timeframe_seconds(bot["timeframe"])
        self.bars_needed = max(self.strategy.warmup(), bot["exits"]["atr_period"] * 3) + 50
        self.broker = PaperBroker(bot["starting_cash"], bot["fee_rate"], bot["slippage_bps"])
        self.risk = RiskManager(bot)
        self.path = state_dir / f"{bot['id']}.json"
        self.last_bar: str | None = None
        self.last_exit: dict[str, str] = {}
        self.history: list[list] = []  # [iso time, equity]
        self.signals: dict[str, dict] = {}
        self.started_at = pd.Timestamp.now(tz="UTC").isoformat()
        self.updated_at: str | None = None
        self.error: str | None = None
        self.equity = bot["starting_cash"]
        self.prices: dict[str, float] = {}
        self._load()
        self.agent = None
        if bot["agent"].get("enabled"):
            from .agent import ClaudeAgent

            self.agent = ClaudeAgent(bot["agent"]["model"])

    # ---- persistence
    def _load(self) -> None:
        if not self.path.exists():
            return
        s = json.loads(self.path.read_text())
        self.broker.load_dict(s["broker"])
        self.risk.load_dict(s["risk"])
        for k in ("last_bar", "last_exit", "history", "signals", "started_at", "updated_at", "equity", "prices"):
            if k in s:
                setattr(self, k, s[k])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "bot_id": self.bot["id"], "broker": self.broker.to_dict(), "risk": self.risk.to_dict(),
            "last_bar": self.last_bar, "last_exit": self.last_exit, "history": self.history,
            "signals": self.signals, "started_at": self.started_at, "updated_at": self.updated_at,
            "equity": self.equity, "prices": self.prices, "error": self.error,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=1, default=str))
        tmp.replace(self.path)

    # ---- trading
    def _close(self, sym: str, price: float, now: pd.Timestamp, reason: str) -> None:
        t = self.broker.sell(sym, price, now, reason)
        self.last_exit[sym] = now.isoformat()
        log.info("[%s] SELL %s %s @ %.6g pnl %+.2f", self.bot["id"], sym, reason, t["exit"], t["pnl"])

    def tick(self, get_frame, prices: dict[str, float], now: pd.Timestamp) -> None:
        b, ex = self.bot, self.bot["exits"]
        tag = b["id"]
        try:
            frames = {s: get_frame(s, b["timeframe"], self.bars_needed) for s in b["symbols"]}
            px = {s: prices[s] for s in b["symbols"]}
        except Exception as e:  # network hiccups must not kill the other bots
            self.error = f"{now:%Y-%m-%d %H:%M} data error: {e}"
            log.warning("[%s] %s", tag, self.error)
            self.save()
            return
        self.error = None
        self.prices = px

        # stops and targets against the live price
        for sym, pos in list(self.broker.positions.items()):
            if px[sym] <= pos.stop:
                self._close(sym, px[sym], now, "trailing stop" if pos.stop >= pos.entry_price else "stop loss")
            elif px[sym] >= pos.take:
                self._close(sym, px[sym], now, "take profit")

        bar_ts = min(f.index[-1] for f in frames.values())  # latest candle every symbol has closed
        if str(bar_ts) != self.last_bar:
            self._on_new_bar(frames, px, bar_ts, now)
            self.last_bar = str(bar_ts)

        self.equity = self.broker.equity(px)
        self.risk.update(self.equity, now)
        if not self.history or (now - pd.Timestamp(self.history[-1][0])).total_seconds() >= HISTORY_EVERY_S:
            self.history.append([now.isoformat(), round(self.equity, 2)])
        self.updated_at = now.isoformat()
        self.save()

    def _on_new_bar(self, frames, px, bar_ts, now) -> None:
        b, ex, tag = self.bot, self.bot["exits"], self.bot["id"]
        prepared = self.strategy.prepare_all(frames)
        rows = {}
        for sym, f in prepared.items():
            f["atr"] = atr(f, ex["atr_period"])
            if bar_ts in f.index:
                rows[sym] = f.loc[bar_ts]
        log.info("[%s] new %s candle %s", tag, b["timeframe"], bar_ts)

        for sym, row in rows.items():
            self.signals[sym] = {"bar": str(bar_ts), "close": float(row["close"]), "entry": bool(row["entry"]),
                                 "exit": bool(row["exit"]), "text": self.strategy.describe(row)}

        for sym, pos in list(self.broker.positions.items()):
            row = rows.get(sym)
            if row is None:
                continue
            self.broker.trail(sym, row["high"], row["atr"], ex["trail_atr"])
            held = (bar_ts - pd.Timestamp(pos.opened_at)).total_seconds() / self.tf_s
            if row["exit"]:
                self._close(sym, px[sym], now, "signal exit")
            elif ex["max_hold_bars"] and held >= ex["max_hold_bars"]:
                self._close(sym, px[sym], now, "time limit")
            elif px[sym] <= self.broker.positions[sym].stop:
                self._close(sym, px[sym], now, "trailing stop")

        def cooled(sym):
            if sym not in self.last_exit:
                return True
            return (now - pd.Timestamp(self.last_exit[sym])).total_seconds() > ex["cooldown_bars"] * self.tf_s

        candidates = [s for s, r in rows.items()
                      if r["entry"] and s not in self.broker.positions and r["atr"] == r["atr"] and cooled(s)]
        candidates.sort(key=lambda s: rows[s].get("priority", 0) if "priority" in rows[s] else 0)
        for sym in candidates:
            ok, why = self.risk.can_open(len(self.broker.positions))
            if not ok:
                log.info("[%s] SKIP %s entry: %s", tag, sym, why)
                continue
            price, row = px[sym], rows[sym]
            stop, take = self.risk.exit_levels(price, row["atr"])
            equity = self.broker.equity({**{s: p.entry_price for s, p in self.broker.positions.items()}, **px})
            qty = self.risk.size(equity, self.broker.cash, price, stop)
            note = self.strategy.describe(row)
            if self.agent:
                portfolio = f"equity {equity:.2f}, cash {self.broker.cash:.2f}, open {list(self.broker.positions)}"
                d = self.agent.review(sym, prepared[sym], [note], portfolio)
                log.info("[%s] AGENT %s approve=%s size=%.2f | %s", tag, sym, d.approve, d.size_multiplier, d.reasoning)
                qty *= d.size_multiplier
                note += f" | agent x{d.size_multiplier:.2f}: {d.reasoning}"
            if qty * price < 10:
                log.info("[%s] SKIP %s entry: size too small", tag, sym)
                continue
            p = self.broker.buy(sym, qty, price, now, stop, take, note)
            if p:
                log.info("[%s] BUY %s qty %.6g @ %.6g stop %.6g", tag, sym, p.qty, p.entry_price, p.stop)


def run_tick(cfg: dict, runners: list[BotRunner], ex) -> None:
    now = pd.Timestamp.now(tz="UTC")
    cache: dict[tuple, pd.DataFrame] = {}

    def get_frame(sym, tf, bars):
        key = (sym, tf)
        if key not in cache or len(cache[key]) < bars:
            cache[key] = fetch_recent(ex, sym, tf, bars)
        return cache[key].tail(bars)

    symbols = sorted({s for r in runners for s in r.bot["symbols"]})
    prices: dict[str, float] = {}
    try:
        for sym, t in ex.fetch_tickers(symbols).items():
            if t.get("last"):
                prices[sym] = float(t["last"])
    except Exception as e:
        log.warning("ticker fetch failed (%s); falling back to one at a time", e)
        for sym in symbols:
            try:
                prices[sym] = float(ex.fetch_ticker(sym)["last"])
            except Exception as e2:
                log.warning("%s ticker error %s", sym, e2)

    for r in runners:
        r.tick(get_frame, prices, now)
        log.info("[%s] equity %.2f | cash %.2f | open %s%s", r.bot["id"], r.equity, r.broker.cash,
                 list(r.broker.positions), f" | HALTED: {r.risk.halted_reason}" if r.risk.halted_reason else "")

    from .dashboard import write_dashboard

    write_dashboard(cfg, runners)


def run_paper(cfg: dict, bots: list[dict], once: bool = False) -> None:
    state_dir = Path(cfg["paper"]["state_dir"])
    runners = [BotRunner(b, state_dir) for b in bots]
    ex = make_exchange(cfg["exchange"])
    log.info("Paper trading %d bots on %s: %s", len(runners), cfg["exchange"], ", ".join(r.bot["id"] for r in runners))
    while True:
        run_tick(cfg, runners, ex)
        if once:
            return
        time.sleep(cfg["paper"]["poll_seconds"])
