# Agentic Crypto Trading Bots (paper trading)

Four strategies trade side by side, each with its own simulated $10,000 account, using live public prices from
Kraken. A web dashboard shows how each is doing. A free GitHub Actions job runs them every 15 minutes, so they keep
going with your PC off. **Every order is simulated.** There's no code anywhere in this project that places real
orders or uses exchange API keys.

## The bots

| Bot | Strategy | Candles | Idea |
|---|---|---|---|
| Trend Follower (`trend`) | `trend_ensemble` | 4h | Buy uptrends when trend/breakout/pullback signals fire; trailing stop |
| Dip Buyer (`meanrev`) | `mean_reversion` | 1h | Buy sharp drops below the lower Bollinger band; sell on the bounce |
| Momentum Rotation (`rotation`) | `momentum_rotation` | 1d | Hold the 3 strongest of 9 coins; cash when BTC is below its average |
| Squeeze Breakout (`squeeze`) | `squeeze_breakout` | 4h | Buy high-volume breakouts after unusually quiet periods; trailing stop |

Every bot gets the same hard risk limits from `config.yaml`: sizing from the stop distance, max position size, max
open positions, daily loss limit, a 15% max-drawdown kill switch, and the `KILL` file. Strategies live in
`bot/strategies.py`, and each one only produces `entry`/`exit` signals. Stops, sizing and execution are shared, so
the comparison between bots is fair.

## Commands

```bash
.venv\Scripts\python main.py backtest
```
```bash
.venv\Scripts\python main.py sweep --bot rotation --no-kill-switch
```
```bash
.venv\Scripts\python main.py tick
```
```bash
.venv\Scripts\python main.py dashboard
```

- `backtest` tests every bot on about 4 years of Binance history and prints the early period, the later period and a
  summary table. Add `--bot trend` for one bot.
- `sweep` tries the parameter combinations in that strategy's `GRID`. It ranks them on the early part of history and
  shows how each did on the later part. **Trust the later-period ("OOS") columns.**
- `tick` runs one live paper pass for every bot. `paper` does the same in a loop on your PC.
- `dashboard` serves `docs/` at http://localhost:8000.

## Running in the cloud (free, PC can be off)

`.github/workflows/paper.yml` runs `main.py tick` about every 15 minutes on GitHub's servers. It then commits the
bots' accounts (`state/`) and the dashboard data (`docs/data.json`) back to the repository. GitHub Pages serves
`docs/` as the dashboard website.

- GitHub runs scheduled jobs on a best-effort basis, so gaps of 15–30+ minutes happen. That's fine for these
  strategies. Stops are checked at each run, not continuously.
- Don't run `paper` on your PC while the cloud job is running: two copies would each trade their own copy of the
  accounts.
- To pause the cloud bots, open the repository's Actions tab, choose "Paper trading", then "Disable workflow".
- If the repository has no commits for 60 days, GitHub disables scheduled jobs. The bot's own commits count as
  activity.

## Turning on the Claude agent

Add your Anthropic API key as a repository secret named `ANTHROPIC_API_KEY` (cloud), or in `.env` (local). Then set
`agent.enabled: true` in `config.yaml`. Claude then reviews every proposed entry and can veto or shrink it, never
enlarge it. Each entry costs one API call.

## Before real money

- Two months of paper trading is a short sample: a bot might make only 5–20 trades in that time. Judge each bot on
  return **and** worst drawdown, and compare its paper results with what its backtest led you to expect.
- The paper broker assumes 0.1% fees and 0.05% slippage. Kraken's real fees at low volume are higher (around 0.25%
  for maker orders and 0.40% for taker orders). The `fee_rate` in `config.yaml` should match the exchange and account
  you'll actually use.
- Going live would need a separate real-order broker module, exchange API keys with withdrawals disabled, and a
  small starting size. That's a deliberate step for you to take; this project doesn't include it.
