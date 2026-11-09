"""Claude review layer.

Claude sees each entry the strategies propose and returns a structured decision: approve or veto, and a
size multiplier between 0 and 1. It can only make trades smaller or skip them. It cannot open trades the
strategies did not propose, enlarge them, move stops, or block exits. Any error or refusal = veto (fail closed).
"""
from __future__ import annotations

import logging

import pandas as pd
from pydantic import BaseModel

log = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are the risk-review step of a paper-trading crypto bot. Rule-based strategies have proposed
opening a long spot position. Your job is to decide whether the setup is sound enough to take, and how much of the
proposed size to use.

Approve when the strategy signals agree with the recent price action. Veto or reduce when you see conflicting
evidence: the move is already extended, volatility is unusually high relative to the stop distance, price is
running into a recent high, or the portfolio is already concentrated. A hard risk manager runs after you, so you do
not need to enforce position limits yourself; focus on whether this particular setup is worth taking.

size_multiplier is a fraction from 0.0 to 1.0 of the proposed size. Keep reasoning to two or three sentences that a
human reviewing the log can follow."""


class Decision(BaseModel):
    approve: bool
    size_multiplier: float
    reasoning: str


class ClaudeAgent:
    def __init__(self, model: str = "claude-opus-5-5"):
        import anthropic

        self.anthropic = anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def review(self, symbol: str, df: pd.DataFrame, strategy_notes: list[str], portfolio: str) -> Decision:
        recent = df[["open", "high", "low", "close", "volume"]].tail(48).round(6).to_csv()
        prompt = (
            f"Proposed entry: LONG {symbol}\n\n"
            f"Strategy readings at the last closed bar:\n- " + "\n- ".join(strategy_notes) + "\n\n"
            f"Portfolio: {portfolio}\n\n"
            f"Last 48 candles (CSV):\n{recent}"
        )
        try:
            resp = self.client.messages.parse(
                model=self.model,
                max_tokens=16000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                output_format=Decision,
            )
        except self.anthropic.APIError as e:
            log.warning("Claude review failed for %s (%s) — vetoing", symbol, e)
            return Decision(approve=False, size_multiplier=0.0, reasoning=f"API error: {e}")

        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            return Decision(approve=False, size_multiplier=0.0, reasoning=f"no usable answer (stop_reason={resp.stop_reason})")
        d = resp.parsed_output
        d.size_multiplier = min(1.0, max(0.0, d.size_multiplier))
        if not d.approve:
            d.size_multiplier = 0.0
        return d
