"""Simulated (paper) broker. Spot, long-only, with fees and slippage. It never talks to an exchange."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class Position:
    symbol: str
    qty: float
    entry_price: float
    entry_fee: float
    stop: float
    take: float
    opened_at: str
    note: str = ""
    highest: float = 0.0  # highest high since entry, for the trailing stop


@dataclass
class PaperBroker:
    cash: float
    fee_rate: float
    slippage_bps: float
    positions: dict[str, Position] = field(default_factory=dict)
    trades: list[dict] = field(default_factory=list)

    @property
    def slip(self) -> float:
        return self.slippage_bps / 10_000

    def buy(self, symbol: str, qty: float, price: float, ts, stop: float, take: float, note: str = "") -> Position | None:
        fill = price * (1 + self.slip)
        qty = min(qty, self.cash / (fill * (1 + self.fee_rate)))
        if qty <= 0 or symbol in self.positions or stop >= fill:
            return None
        cost = qty * fill
        fee = cost * self.fee_rate
        self.cash -= cost + fee
        pos = Position(symbol, qty, fill, fee, stop, take, str(ts), note, highest=fill)
        self.positions[symbol] = pos
        return pos

    def trail(self, symbol: str, bar_high: float, atr: float, trail_atr: float) -> None:
        """Chandelier trailing stop: highest high since entry minus trail_atr x ATR. The stop only ever moves up."""
        pos = self.positions.get(symbol)
        if pos is None or trail_atr <= 0 or not atr == atr:  # atr == atr is False for NaN
            return
        pos.highest = max(pos.highest, bar_high)
        pos.stop = max(pos.stop, pos.highest - trail_atr * atr)

    def sell(self, symbol: str, price: float, ts, reason: str) -> dict | None:
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return None
        fill = price * (1 - self.slip)
        proceeds = pos.qty * fill
        fee = proceeds * self.fee_rate
        self.cash += proceeds - fee
        cost_basis = pos.qty * pos.entry_price + pos.entry_fee
        pnl = proceeds - fee - cost_basis
        trade = {
            "symbol": symbol,
            "opened_at": pos.opened_at,
            "closed_at": str(ts),
            "qty": pos.qty,
            "entry": pos.entry_price,
            "exit": fill,
            "pnl": pnl,
            "return_pct": pnl / cost_basis * 100,
            "reason": reason,
            "note": pos.note,
        }
        self.trades.append(trade)
        return trade

    def equity(self, prices: dict[str, float]) -> float:
        return self.cash + sum(p.qty * prices.get(s, p.entry_price) for s, p in self.positions.items())

    def to_dict(self) -> dict:
        return {
            "cash": self.cash,
            "positions": {s: asdict(p) for s, p in self.positions.items()},
            "trades": self.trades,
        }

    def load_dict(self, d: dict) -> None:
        self.cash = d["cash"]
        self.positions = {s: Position(**p) for s, p in d["positions"].items()}
        self.trades = d["trades"]
