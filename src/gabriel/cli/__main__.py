"""Command line interface for GABRIEL."""
import argparse
from ..tasks import SimpleRating, EloRater, Identification



def main() -> None:
    parser = argparse.ArgumentParser(prog="gabriel")
    parser.add_argument("command", choices=["rate", "elo", "identify"])
    args = parser.parse_args()

    if args.command == "rate":
        print("Simple rating not yet implemented")
    elif args.command == "elo":
        print("Elo rating not yet implemented")
    elif args.command == "identify":
        print("Identification not yet implemented")


if __name__ == "__main__":
    main()
