"""Market data: public exchange candles via ccxt, a CSV cache, and synthetic data for offline tests."""
from __future__ import annotations

import logging
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)
CACHE_DIR = Path("data")
COLUMNS = ["ts", "open", "high", "low", "close", "volume"]


def timeframe_seconds(timeframe: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400, "w": 604800}
    return int(timeframe[:-1]) * units[timeframe[-1]]


def to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS).drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.set_index("ts").sort_index().astype(float)


def make_exchange(exchange_id: str):
    import ccxt  # imported lazily so backtests on synthetic data work without network

    return getattr(ccxt, exchange_id)({"enableRateLimit": True})


def _page(ex, symbol: str, timeframe: str, since: int) -> list[list]:
    tf_ms = timeframe_seconds(timeframe) * 1000
    rows: list[list] = []
    while True:
        batch = ex.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        nxt = batch[-1][0] + tf_ms
        if nxt <= since or nxt >= ex.milliseconds():
            break
        since = nxt
    return rows


def fetch_ohlcv(exchange_id: str, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """Page through public candle history. No API key required."""
    ex = make_exchange(exchange_id)
    rows = _page(ex, symbol, timeframe, ex.milliseconds() - days * 86_400_000)
    if not rows:
        raise RuntimeError(f"{exchange_id} returned no candles for {symbol}")
    return to_frame(rows)


def fetch_recent(ex, symbol: str, timeframe: str, bars: int) -> pd.DataFrame:
    """The most recent `bars` closed candles (the still-forming candle is dropped)."""
    since = ex.milliseconds() - (bars + 2) * timeframe_seconds(timeframe) * 1000
    rows = _page(ex, symbol, timeframe, since)
    if not rows:
        raise RuntimeError(f"no candles for {symbol} {timeframe}")
    df = to_frame(rows)
    now = pd.Timestamp.now(tz="UTC")
    closed = df.index + pd.Timedelta(seconds=timeframe_seconds(timeframe)) <= now
    return df[closed]


def synthetic_ohlcv(symbol: str, timeframe: str, days: int, seed: int | None = None) -> pd.DataFrame:
    """Regime-switching random walk. Useful for testing the plumbing offline — not for judging a strategy."""
    n = int(days * 86400 / timeframe_seconds(timeframe))
    rng = np.random.default_rng(seed if seed is not None else zlib.crc32(symbol.encode()))
    drift = np.zeros(n)
    regime = 0.0
    for i in range(n):
        if rng.random() < 0.01:
            regime = rng.choice([-0.0006, 0.0, 0.0008])
        drift[i] = regime
    rets = drift + rng.normal(0, 0.008, n)
    close = 100 * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[100.0], close[:-1]])
    wick = np.abs(rng.normal(0, 0.004, n))
    high = np.maximum(open_, close) * (1 + wick)
    low = np.minimum(open_, close) * (1 - wick)
    end = pd.Timestamp.now(tz="UTC").floor("h")
    idx = pd.date_range(end=end, periods=n, freq=pd.Timedelta(seconds=timeframe_seconds(timeframe)))
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": rng.uniform(100, 1000, n)},
        index=idx.rename("ts"),
    )


def backtest_symbol(cfg: dict, symbol: str) -> str:
    """Live trading uses the live exchange's pairs (e.g. Kraken BTC/USD); long backtests use an exchange with deep
    history (Binance BTC/USDT). Prices of USD and USDT pairs track each other closely."""
    base, quote = symbol.split("/")
    return f"{base}/{cfg['backtest'].get('quote', quote)}"


def load_ohlcv(cfg: dict, symbol: str, timeframe: str, days: int, source: str = "exchange", refresh: bool = False) -> pd.DataFrame:
    if source == "synthetic":
        return synthetic_ohlcv(symbol, timeframe, days)
    exchange = cfg["backtest"]["exchange"]
    sym = backtest_symbol(cfg, symbol)
    CACHE_DIR.mkdir(exist_ok=True)
    path = CACHE_DIR / f"{exchange}_{sym.replace('/', '-')}_{timeframe}_{days}d.csv"
    if path.exists() and not refresh:
        return pd.read_csv(path, index_col="ts", parse_dates=["ts"])
    log.info("Downloading %s %s (%d days) from %s", sym, timeframe, days, exchange)
    df = fetch_ohlcv(exchange, sym, timeframe, days)
    df.to_csv(path)
    return df
