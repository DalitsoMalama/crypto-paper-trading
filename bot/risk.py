"""Hard risk limits. This layer has the final say on every entry; nothing upstream can override it."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

KILL_FILE = Path("KILL")  # create this file to stop all new entries immediately


class RiskManager:
    def __init__(self, bot: dict):
        r, ex = bot["risk"], bot["exits"]
        self.risk_per_trade = r["risk_per_trade"]
        self.max_position_pct = r["max_position_pct"]
        self.max_open_positions = r["max_open_positions"]
        self.max_daily_loss_pct = r["max_daily_loss_pct"]
        self.max_drawdown_pct = r["max_drawdown_pct"]
        self.stop_atr = ex["stop_atr"]
        self.take_profit_atr = ex["take_profit_atr"]
        self.fee_rate = bot["fee_rate"]
        self.peak: float | None = None
        self.day: str | None = None
        self.day_start_equity: float | None = None
        self.daily_blocked = False
        self.halted_reason: str | None = None  # permanent until state is reset by a human

    def update(self, equity: float, ts: pd.Timestamp) -> None:
        day = ts.strftime("%Y-%m-%d")
        if day != self.day:
            self.day, self.day_start_equity, self.daily_blocked = day, equity, False
        self.peak = equity if self.peak is None else max(self.peak, equity)
        drawdown = 1 - equity / self.peak
        if self.halted_reason is None and drawdown >= self.max_drawdown_pct:
            self.halted_reason = f"max drawdown {drawdown:.1%} reached at {ts}"
        if equity <= self.day_start_equity * (1 - self.max_daily_loss_pct):
            self.daily_blocked = True

    def can_open(self, open_positions: int, check_kill_file: bool = True) -> tuple[bool, str]:
        if check_kill_file and KILL_FILE.exists():
            return False, "KILL file present"
        if self.halted_reason:
            return False, f"halted: {self.halted_reason}"
        if self.daily_blocked:
            return False, "daily loss limit hit"
        if open_positions >= self.max_open_positions:
            return False, "max open positions"
        return True, "ok"

    def exit_levels(self, price: float, atr: float) -> tuple[float, float]:
        """Initial stop and take-profit from volatility. A take_profit_atr of 0 means no fixed target."""
        stop = price - self.stop_atr * atr
        take = price + self.take_profit_atr * atr if self.take_profit_atr > 0 else float("inf")
        return stop, take

    def size(self, equity: float, cash: float, price: float, stop: float) -> float:
        """Quantity such that hitting the stop loses ~risk_per_trade of equity, within position and cash caps."""
        if not stop < price:
            return 0.0
        # risk_per_trade >= 1 means "size by max_position_pct only" (equal-weight bots and the benchmark)
        by_risk = float("inf") if self.risk_per_trade >= 1 else equity * self.risk_per_trade / (price - stop)
        by_cap = equity * self.max_position_pct / price
        by_cash = cash / (price * (1 + self.fee_rate)) * 0.999
        return max(0.0, min(by_risk, by_cap, by_cash))

    def to_dict(self) -> dict:
        keys = ("peak", "day", "day_start_equity", "daily_blocked", "halted_reason")
        return {k: getattr(self, k) for k in keys}

    def load_dict(self, d: dict) -> None:
        for k, v in d.items():
            setattr(self, k, v)
