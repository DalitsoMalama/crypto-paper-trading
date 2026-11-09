"""Parameter sweep with an out-of-sample check.

Every combination is run on the first part of history ("in-sample", where you would tune) and on the later part
("out-of-sample", never used for choosing). Settings are ranked by in-sample results only, and the out-of-sample
columns show whether that ranking held up. If the best in-sample settings fall apart out-of-sample, they were
curve-fit to noise. Each strategy defines its own GRID in bot/strategies.py.
"""
from __future__ import annotations

import copy
import itertools

import pandas as pd

from .backtest import prepare_bot, run_backtest
from .strategies import REGISTRY


def set_path(bot: dict, path: str, value) -> None:
    *parents, key = path.split(".")
    node = bot
    for p in parents:
        node = node[p]
    node[key] = value


def score(m: dict) -> float:
    """Return / drawdown: rewards making money without deep losses. Too few trades scores nothing."""
    if m["trades"] < 8:
        return float("-inf")
    return m["total_return_pct"] / max(1.0, -m["max_drawdown_pct"])


def run_sweep(bot: dict, data: dict, split: float = 0.6) -> pd.DataFrame:
    grid = REGISTRY[bot["strategy"]].GRID
    rows = []
    for values in itertools.product(*grid.values()):
        b = copy.deepcopy(bot)
        for k, v in zip(grid, values):
            set_path(b, k, v)
        prepared = prepare_bot(b, data)
        ins = run_backtest(b, data, 0.0, split, prepared)["metrics"]
        oos = run_backtest(b, data, split, 1.0, prepared)["metrics"]
        row = {k.split(".")[-1]: v for k, v in zip(grid, values)}
        row.update({
            "IS_ret%": ins["total_return_pct"], "IS_dd%": ins["max_drawdown_pct"], "IS_trades": ins["trades"],
            "IS_score": score(ins),
            "OOS_ret%": oos["total_return_pct"], "OOS_dd%": oos["max_drawdown_pct"], "OOS_trades": oos["trades"],
            "OOS_score": score(oos),
        })
        rows.append(row)
    return pd.DataFrame(rows).sort_values("IS_score", ascending=False).reset_index(drop=True)
