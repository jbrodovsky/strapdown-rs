from argparse import ArgumentParser

from analysis.preprocess import preprocess_data

__version__ = "0.1.0"


def main() -> None:
    # print("Hello from analysis!")

    parser = ArgumentParser(
        description="Data analysis and simulation orchestration tools for use with Strapdown-sim."
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"strapdown-analysis package version {__version__}",
    )

    commands = parser.add_subparsers(title="command", dest="command")

    preprocess = commands.add_parser(
        "preprocess", help="Preprocess raw trajectory data."
    )

    preprocess.add_argument(
        "-i",
        "--input_dir",
        type=str,
        default="data/raw",
        help="Base directory for the sensor logger app data.",
    )
    preprocess.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="data",
        help="Output directory for the cleaned data.",
    )
    preprocess.add_argument(
        "-b",
        "--buffer",
        type=float,
        default=0.1,
        help="Buffer amount to inflate the bounding box by (as a percentage). Default is 0.1 (10 percent).",
    )
    preprocess.add_argument(
        "--getmaps",
        action="store_true",
        help="Download geophysical maps for each trajectory.",
    )

    args = parser.parse_args()

    if args.command == "preprocess":
        preprocess_data(args)


if __name__ == "__main__":
    main()
