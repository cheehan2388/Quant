import argparse
import logging
import sys

from .volarb_ivrv import main as ivrv_main
from .volarb_calendar import main as calendar_main


def setup_logger(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger("livetrade")


def main():
    parser = argparse.ArgumentParser(description="Realtime live-trade strategies")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ivrv", help="Run implied-vs-realized vol arbitrage")
    sub.add_parser("calendar", help="Run ATM calendar spread")

    args, unknown = parser.parse_known_args()

    if args.command == "ivrv":
        # Strip the subcommand token before delegating to submodule parser
        sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:] if x != "ivrv"]
        ivrv_main()
    elif args.command == "calendar":
        sys.argv = [sys.argv[0]] + [x for x in sys.argv[1:] if x != "calendar"]
        calendar_main()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


