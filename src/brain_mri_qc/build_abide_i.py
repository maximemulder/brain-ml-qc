#!/usr/bin/env python
from importlib import resources

import pandas as pd


SITES = [
    ("KUL", "LEUVEN"),
]

def main():
    reference = resources.files('mriqc_learn.datasets') / 'abide.tsv'

    # Read the TSV file directly without needing a Path
    with resources.as_file(reference) as path:
        dataframe = pd.read_csv(path, sep='\t')

    # Get unique values from the 'site' column
    unique_sites = dataframe['site'].unique()

    # Sort for cleaner output
    unique_sites = sorted(unique_sites)

    # Display results
    print(f"Total unique sites: {len(unique_sites)}")
    print("\nUnique site values:")
    print("-" * 30)
    for site in unique_sites:
        count = dataframe[dataframe['site'] == site].shape[0]
        print(f"{site:12} : {count:5} subjects")


if __name__ == '__main__':
    main()
