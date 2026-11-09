"""Load config.yaml and expand each entry under `bots:` into a self-contained settings dict."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULT_EXITS = {
    "atr_period": 14,
    "stop_atr": 3.0,
    "trail_atr": 0.0,
    "take_profit_atr": 0.0,
    "max_hold_bars": 0,
    "cooldown_bars": 0,
}


def load_config(path: str = "config.yaml") -> dict:
    return yaml.safe_load(Path(path).read_text())


def bot_settings(cfg: dict, bot: dict) -> dict:
    b = copy.deepcopy(bot)
    b["risk"] = {**cfg["risk"], **bot.get("risk", {})}
    b["exits"] = {**DEFAULT_EXITS, **bot.get("exits", {})}
    b["params"] = bot.get("params", {})
    b.setdefault("description", "")
    for key in ("starting_cash", "fee_rate", "slippage_bps"):
        b.setdefault(key, cfg[key])
    b["agent"] = {**cfg.get("agent", {"enabled": False}), **bot.get("agent", {})}
    return b


def all_bots(cfg: dict, only: list[str] | None = None) -> list[dict]:
    bots = [bot_settings(cfg, b) for b in cfg["bots"] if b.get("enabled", True)]
    if only:
        bots = [b for b in bots if b["id"] in only]
        if not bots:
            raise SystemExit(f"No enabled bot matches {only}. Ids: {[b['id'] for b in cfg['bots']]}")
    return bots
