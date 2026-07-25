"""Regenerate the spectra removal-vs-preservation figure from a saved results.csv.

Avoids re-running the full sweep — just re-plots. Run from this directory:

    python plot_frontier.py [--results ivae_sweep_results/results.csv]
"""

import argparse

import pandas as pd
from ivae_sweep import TARGETS, make_tradeoff_plot


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="ivae_sweep_results/results.csv")
    parser.add_argument("--out", default="ivae_sweep_results/tradeoff.png")
    parser.add_argument("--spender", default="spender_I")
    args = parser.parse_args()

    df = pd.read_csv(args.results)
    make_tradeoff_plot(df, TARGETS, args.spender, args.out)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
