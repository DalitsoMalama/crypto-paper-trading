"""Command line entry point.

  python main.py backtest                       # all bots, 4 years of history, early/late split
  python main.py backtest --bot trend           # one bot
  python main.py sweep --bot meanrev            # try parameter combinations with an out-of-sample check
  python main.py paper                          # live paper trading loop for every bot (Ctrl+C to stop)
  python main.py tick                           # one pass for every bot, then exit (used by the cloud job)
  python main.py dashboard                      # view the dashboard at http://localhost:8000
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

from bot.config import all_bots, load_config


def setup_logging(log_file: str | None = None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)
    for noisy in ("httpx", "urllib3", "ccxt"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def cmd_backtest(cfg, args):
    from bot.backtest import print_report, run_backtest, save_results
    from bot.data import load_ohlcv

    summary = []
    for bot in all_bots(cfg, args.bot):
        if args.no_kill_switch:
            bot["risk"]["max_drawdown_pct"] = 1.0
        data = {s: load_ohlcv(cfg, s, bot["timeframe"], args.days, args.source, args.refresh) for s in bot["symbols"]}
        print(f"\n################ {bot['name']} ({bot['id']}, {bot['strategy']}, {bot['timeframe']}) ################")
        full = run_backtest(bot, data)
        print_report(full, "Full period")
        row = {"bot": bot["id"], "full_ret%": full["metrics"]["total_return_pct"], "full_dd%": full["metrics"]["max_drawdown_pct"]}
        if args.split:
            ins, oos = run_backtest(bot, data, 0.0, args.split), run_backtest(bot, data, args.split, 1.0)
            print_report(ins, "In-sample")
            print_report(oos, "Out-of-sample")
            row.update({"IS_ret%": ins["metrics"]["total_return_pct"], "OOS_ret%": oos["metrics"]["total_return_pct"],
                        "OOS_dd%": oos["metrics"]["max_drawdown_pct"], "OOS_pf": oos["metrics"]["profit_factor"]})
        row["trades"] = full["metrics"]["trades"]
        summary.append(row)
        save_results(full, bot["id"])
    if len(summary) > 1:
        import pandas as pd

        with pd.option_context("display.float_format", "{:.2f}".format, "display.width", 200):
            print("\n================ Summary ================")
            print(pd.DataFrame(summary).to_string(index=False))
    if args.source == "synthetic":
        print("(Synthetic data: this only proves the plumbing works. It says nothing about real performance.)")


def cmd_sweep(cfg, args):
    import pandas as pd

    from bot.data import load_ohlcv
    from bot.sweep import run_sweep

    for bot in all_bots(cfg, args.bot):
        if args.no_kill_switch:
            bot["risk"]["max_drawdown_pct"] = 1.0
        data = {s: load_ohlcv(cfg, s, bot["timeframe"], args.days) for s in bot["symbols"]}
        table = run_sweep(bot, data, args.split)
        Path("results").mkdir(exist_ok=True)
        table.to_csv(f"results/sweep_{bot['id']}.csv", index=False)
        print(f"\n################ sweep: {bot['name']} ({bot['id']}) ################")
        with pd.option_context("display.width", 220, "display.max_columns", 20, "display.float_format", "{:.2f}".format):
            print(table.head(12).to_string())
        top = table.head(5)
        print(f"Top 5 by in-sample score: median out-of-sample return {top['OOS_ret%'].median():+.2f}%, "
              f"{(top['OOS_ret%'] > 0).sum()}/5 profitable out-of-sample. "
              f"All {len(table)} combos: {(table['OOS_ret%'] > 0).mean() * 100:.0f}% profitable out-of-sample.")


def cmd_dashboard(cfg, args):
    import functools
    import http.server

    root = Path(cfg["paper"]["dashboard_file"]).parent
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    print(f"Dashboard: http://localhost:{args.port}  (Ctrl+C to stop)")
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()


def main() -> None:
    load_dotenv()
    p = argparse.ArgumentParser(description="Agentic crypto trading bots (paper trading only)")
    p.add_argument("--config", default="config.yaml")
    sub = p.add_subparsers(dest="cmd", required=True)

    bt = sub.add_parser("backtest")
    bt.add_argument("--days", type=int, default=1500)
    bt.add_argument("--split", type=float, default=0.6, help="also report before/after this fraction (0 = off)")
    bt.add_argument("--source", choices=["exchange", "synthetic"], default="exchange")
    bt.add_argument("--refresh", action="store_true", help="re-download candles instead of using the cache")

    sw = sub.add_parser("sweep", help="try parameter combinations; rank on early data, check on later data")
    sw.add_argument("--days", type=int, default=1500)
    sw.add_argument("--split", type=float, default=0.6)

    for sp in (bt, sw):
        sp.add_argument("--bot", nargs="*", help="bot ids (default: all)")
        sp.add_argument("--no-kill-switch", action="store_true",
                        help="research only: don't halt forever after max drawdown, so long tests run to the end")

    pp = sub.add_parser("paper", help="live paper trading loop")
    pp.add_argument("--bot", nargs="*")
    tk = sub.add_parser("tick", help="one live paper pass for every bot, then exit")
    tk.add_argument("--bot", nargs="*")
    db = sub.add_parser("dashboard", help="serve the dashboard locally")
    db.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    cfg = load_config(args.config)
    if args.cmd == "backtest":
        setup_logging()
        cmd_backtest(cfg, args)
    elif args.cmd == "sweep":
        setup_logging()
        cmd_sweep(cfg, args)
    elif args.cmd == "dashboard":
        cmd_dashboard(cfg, args)
    else:
        setup_logging(cfg["paper"]["log_file"])
        from bot.live import run_paper

        try:
            run_paper(cfg, all_bots(cfg, args.bot), once=args.cmd == "tick")
        except KeyboardInterrupt:
            logging.info("Stopped by user. State is saved in %s/", cfg["paper"]["state_dir"])


if __name__ == "__main__":
    main()
