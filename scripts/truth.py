import os
from argparse import ArgumentParser

import pandas as pd


def run_truth_mechanization(input_dir: str, output_dir: str):
    """
    Create truth mechanization data sets from all .csv files in input_dir.
    """
    print(f"Reading data from: {input_dir}")
    print(f"Writing results to: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith(".csv"):
                input_file = os.path.join(root, file)
                base = os.path.splitext(os.path.basename(input_file))[0]
                output_file = os.path.join(output_dir, f"{base}_truth.csv")
                print(f"Processing: {input_file}")
                try:
                    os.system(
                        f"strapdown --mode closed-loop --input {input_file} --output {output_file}"
                    )
                except Exception as err:
                    print(f"Skipping {input_file} due to error: {err}")
    print("Truth mechanization data sets created.")


def main():
    parser = ArgumentParser(description="Create truth mechanization data sets.")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Input directory containing .csv files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for truth data sets.",
    )
    args = parser.parse_args()
    run_truth_mechanization(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
