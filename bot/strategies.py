"""Trading strategies.

Each strategy turns candles into two boolean columns per bar, `entry` and `exit`, computed only from data up to
and including that bar. The engine acts on them at the next bar's open (backtest) or right away (live), so there is
no look-ahead. Stops, trailing stops, time limits and sizing are handled by the engine from the bot's `exits:`
settings, so every strategy gets the same risk controls.

Strategies:
  trend_ensemble     buy uptrends when trend/breakout/pullback signals agree; ride them with a trailing stop
  mean_reversion     buy sharp dips below the lower Bollinger band; sell on the bounce back to the average
  momentum_rotation  each day hold the strongest coins by recent return; rotate out of the ones that weaken
  squeeze_breakout   wait for volatility to compress, buy the breakout on high volume, trail it
  buy_and_hold       benchmark only: buy once, never sell
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------------- indicators

def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    out = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    return out.where(loss != 0, 100.0)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def bollinger(close: pd.Series, n: int, k: float) -> tuple[pd.Series, pd.Series, pd.Series]:
    mid = sma(close, n)
    sd = close.rolling(n).std()
    return mid, mid + k * sd, mid - k * sd


def uptrend(close: pd.Series, ema_len: int, slope_bars: int) -> pd.Series:
    """True when price is above a long EMA that is itself rising."""
    if ema_len <= 0:
        return pd.Series(True, index=close.index)
    e = ema(close, ema_len)
    return (close > e) & (e > e.shift(slope_bars))


# ----------------------------------------------------------------------------- base

class Strategy:
    name = "base"
    DEFAULTS: dict = {}
    GRID: dict = {}  # parameter grid for `main.py sweep`, as "params.x" / "exits.y" paths

    def __init__(self, **params):
        unknown = set(params) - set(self.DEFAULTS)
        if unknown:
            raise ValueError(f"{self.name}: unknown params {sorted(unknown)}")
        self.p = {**self.DEFAULTS, **params}

    def warmup(self) -> int:
        raise NotImplementedError

    def prepare(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return df plus boolean `entry` and `exit` columns and any features describe() reads."""
        raise NotImplementedError

    def prepare_all(self, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
        return {sym: self.prepare(df) for sym, df in frames.items()}

    def describe(self, row: pd.Series) -> str:
        raise NotImplementedError


def _finish(out: pd.DataFrame, entry, exit_) -> pd.DataFrame:
    out["entry"] = pd.Series(entry, index=out.index).fillna(False).astype(bool)
    out["exit"] = pd.Series(exit_, index=out.index).fillna(False).astype(bool)
    return out


# ----------------------------------------------------------------------------- 1. trend ensemble

class TrendEnsemble(Strategy):
    """The original bot: EMA trend, RSI pullback and Donchian breakout vote; only trades in an uptrend."""
    name = "trend_ensemble"
    DEFAULTS = {
        "ema_fast": 12, "ema_slow": 26,
        "rsi_period": 14, "rsi_oversold": 40, "rsi_overbought": 80,
        "donchian_entry": 20, "donchian_exit": 10,
        "regime_ema": 200, "regime_slope_bars": 10,
        "min_votes": 1, "exit_votes": 0, "exit_on_regime_off": True,
    }
    GRID = {
        "exits.stop_atr": [2.0, 2.5, 3.5],
        "exits.trail_atr": [2.5, 3.5, 5.0],
        "params.regime_ema": [100, 200],
        "params.min_votes": [1, 2],
    }

    def warmup(self):
        return max(self.p["ema_slow"] * 3, self.p["regime_ema"] * 2, self.p["donchian_entry"] + 1)

    def prepare(self, df):
        p, c = self.p, df["close"]
        out = df.copy()
        fast, slow = ema(c, p["ema_fast"]), ema(c, p["ema_slow"])
        r = rsi(c, p["rsi_period"])
        upper = df["high"].rolling(p["donchian_entry"]).max().shift(1)
        lower = df["low"].rolling(p["donchian_exit"]).min().shift(1)
        out["v_ema"] = np.where(fast > slow, 1, -1)
        out["v_rsi"] = np.select([r < p["rsi_oversold"], r > p["rsi_overbought"]], [1, -1], 0)
        out["v_donchian"] = np.select([c > upper, c < lower], [1, -1], 0)
        out["regime"] = uptrend(c, p["regime_ema"], p["regime_slope_bars"])
        out["rsi"] = r
        votes = out[["v_ema", "v_rsi", "v_donchian"]]
        bulls, bears = (votes == 1).sum(axis=1), (votes == -1).sum(axis=1)
        entry = out["regime"] & (bulls >= p["min_votes"]) & (bears == 0)
        exit_ = pd.Series(False, index=df.index)
        if p["exit_votes"] > 0:
            exit_ |= bears >= p["exit_votes"]
        if p["exit_on_regime_off"]:
            exit_ |= ~out["regime"]
        return _finish(out, entry, exit_)

    def describe(self, row):
        names = {"v_ema": "EMA trend", "v_rsi": "RSI pullback", "v_donchian": "breakout"}
        bulls = [n for k, n in names.items() if row[k] == 1]
        bears = [n for k, n in names.items() if row[k] == -1]
        return (f"uptrend={'yes' if row['regime'] else 'no'} | bullish: {', '.join(bulls) or '-'} | "
                f"bearish: {', '.join(bears) or '-'} | RSI {row['rsi']:.0f}")


# ----------------------------------------------------------------------------- 2. mean reversion

class MeanReversion(Strategy):
    """Buy a close below the lower Bollinger band with oversold RSI; sell when price gets back to the middle band.

    Opposite logic to the trend bot: it bets that sharp drops partly snap back. The `crash_filter` skips dips while
    price is far below its long average (a falling knife rather than a dip).
    """
    name = "mean_reversion"
    DEFAULTS = {"bb_len": 20, "bb_k": 2.0, "rsi_period": 14, "rsi_entry": 30, "trend_ema": 200, "crash_filter": 0.10}
    GRID = {
        "params.bb_k": [2.0, 2.5],
        "params.rsi_entry": [25, 30, 35],
        "params.crash_filter": [0.05, 0.10, 1.0],
        "exits.stop_atr": [2.0, 3.0],
    }

    def warmup(self):
        return max(self.p["bb_len"] * 2, self.p["trend_ema"] * 2, self.p["rsi_period"] * 3)

    def prepare(self, df):
        p, c = self.p, df["close"]
        out = df.copy()
        mid, upper, lower = bollinger(c, p["bb_len"], p["bb_k"])
        r = rsi(c, p["rsi_period"])
        long_avg = ema(c, p["trend_ema"])
        out["bb_mid"], out["bb_lower"], out["rsi"], out["long_avg"] = mid, lower, r, long_avg
        not_crashing = c > long_avg * (1 - p["crash_filter"])
        entry = (c < lower) & (r < p["rsi_entry"]) & not_crashing
        exit_ = c >= mid
        return _finish(out, entry, exit_)

    def describe(self, row):
        return (f"close {row['close']:.4g} vs lower band {row['bb_lower']:.4g} / middle {row['bb_mid']:.4g} | "
                f"RSI {row['rsi']:.0f} | {row['close'] / row['long_avg'] * 100 - 100:+.1f}% vs long average")


# ----------------------------------------------------------------------------- 3. momentum rotation

class MomentumRotation(Strategy):
    """Rank every coin by its return over `lookback` bars. Hold the top `top_n` while they're rising and above their
    average; sell a coin when it falls out of the top `hold_rank` or below its average. Needs a wide coin list."""
    name = "momentum_rotation"
    DEFAULTS = {"lookback": 30, "top_n": 3, "hold_rank": 5, "trend_sma": 50,
                "market_symbol": "BTC/USD", "market_sma": 0}
    GRID = {
        "params.lookback": [14, 30, 60],
        "params.top_n": [2, 3],
        "params.market_sma": [0, 50, 100],
        "exits.stop_atr": [3.0, 5.0],
    }

    def warmup(self):
        return max(self.p["lookback"], self.p["trend_sma"], self.p["market_sma"]) + 5

    def prepare_all(self, frames):
        p = self.p
        mom = pd.DataFrame({s: df["close"] / df["close"].shift(p["lookback"]) - 1 for s, df in frames.items()})
        rank = mom.rank(axis=1, ascending=False, method="first")
        # Market filter: when BTC is below its long average the whole market tends to fall together, so hold cash.
        if p["market_sma"] and p["market_symbol"] in frames:
            m = frames[p["market_symbol"]]["close"]
            market_ok = m > sma(m, p["market_sma"])
        else:
            market_ok = None
        out = {}
        for s, df in frames.items():
            o = df.copy()
            avg = sma(df["close"], p["trend_sma"])
            o["momentum"], o["rank"], o["avg"] = mom[s], rank[s], avg
            o["market_ok"] = True if market_ok is None else market_ok.reindex(df.index).fillna(False)
            o["priority"] = rank[s]  # engine buys the strongest coins first when slots are limited
            entry = (o["rank"] <= p["top_n"]) & (o["momentum"] > 0) & (df["close"] > avg) & o["market_ok"]
            exit_ = (o["rank"] > p["hold_rank"]) | (o["momentum"] < 0) | (df["close"] < avg) | ~o["market_ok"].astype(bool)
            out[s] = _finish(o, entry, exit_)
        return out

    def describe(self, row):
        rank = "-" if pd.isna(row["rank"]) else f"#{int(row['rank'])}"
        market = "" if not self.p["market_sma"] else f" | BTC above {self.p['market_sma']}-day average: {'yes' if row['market_ok'] else 'no'}"
        return (f"rank {rank} | {self.p['lookback']}-bar return {row['momentum'] * 100:+.1f}% | "
                f"above average: {'yes' if row['close'] > row['avg'] else 'no'}{market}")


# ----------------------------------------------------------------------------- 4. squeeze breakout

class SqueezeBreakout(Strategy):
    """When the Bollinger bands have been unusually narrow (volatility squeeze), buy a close above the upper band on
    above-average volume. Exit when price falls back under the middle band, or by the trailing stop."""
    name = "squeeze_breakout"
    DEFAULTS = {"bb_len": 20, "bb_k": 2.0, "squeeze_lookback": 120, "squeeze_pct": 0.20,
                "squeeze_recent": 6, "volume_mult": 1.5, "trend_ema": 0}
    GRID = {
        "params.squeeze_pct": [0.1, 0.2, 0.3],
        "params.volume_mult": [1.0, 1.5],
        "params.trend_ema": [0, 200],
        "exits.trail_atr": [2.5, 4.0],
    }

    def warmup(self):
        return max(self.p["squeeze_lookback"] + self.p["bb_len"], self.p["trend_ema"] * 2) + 5

    def prepare(self, df):
        p, c = self.p, df["close"]
        out = df.copy()
        mid, upper, lower = bollinger(c, p["bb_len"], p["bb_k"])
        width = (upper - lower) / mid
        width_rank = width.rolling(p["squeeze_lookback"]).rank(pct=True)
        squeezed = (width_rank <= p["squeeze_pct"]).astype(float)
        recent_squeeze = squeezed.shift(1).rolling(p["squeeze_recent"]).max() == 1
        vol_ok = df["volume"] > sma(df["volume"], 20) * p["volume_mult"]
        out["bb_mid"], out["bb_upper"], out["width_rank"], out["vol_ratio"] = mid, upper, width_rank, df["volume"] / sma(df["volume"], 20)
        entry = recent_squeeze & (c > upper) & vol_ok & uptrend(c, p["trend_ema"], 10)
        exit_ = c < mid
        return _finish(out, entry, exit_)

    def describe(self, row):
        return (f"band width percentile {row['width_rank'] * 100:.0f}% | close {row['close']:.4g} vs upper {row['bb_upper']:.4g} | "
                f"volume {row['vol_ratio']:.1f}x average")


# ----------------------------------------------------------------------------- benchmark

class BuyAndHold(Strategy):
    """Not a strategy: buys once and holds. Any bot worth real money should beat this on return or drawdown."""
    name = "buy_and_hold"

    def warmup(self):
        return 1

    def prepare(self, df):
        return _finish(df.copy(), True, False)

    def describe(self, row):
        return "benchmark: buy once and hold"


REGISTRY = {cls.name: cls for cls in (TrendEnsemble, MeanReversion, MomentumRotation, SqueezeBreakout, BuyAndHold)}


def build_strategy(bot: dict) -> Strategy:
    return REGISTRY[bot["strategy"]](**bot["params"])
