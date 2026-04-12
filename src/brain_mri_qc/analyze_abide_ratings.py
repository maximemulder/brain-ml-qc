#!/usr/bin/env python
from collections import Counter

import pandas as pd
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from brain_mri_qc.abide import get_abide_labels


def get_rating_distributions(dataframe: pd.DataFrame):
    """
    Returns the distribution of rating combinations across all subjects.

    Returns:
        dict: Maps rating combination tuples to their count
    """

    distributions = Counter()

    for _, row in dataframe.iterrows():
        # Collect valid ratings (ignore 'n/a' and NaN)
        ratings = []
        for rater in ['rater_1', 'rater_2', 'rater_3']:
            val = row[rater]
            if val != 'n/a' and pd.notna(val):
                try:
                    ratings.append(float(val))
                except (ValueError, TypeError):
                    pass

        # Sort ratings to ignore reviewer order
        ratings_tuple = tuple(sorted(ratings))
        distributions[ratings_tuple] += 1

    return distributions

def print_rating_distributions(distributions):
    """Pretty print the rating distributions."""

    print("=" * 60)
    print("RATING COMBINATIONS DISTRIBUTION")
    print("=" * 60)
    print(f"{'Combination':<25} {'Count':<10} {'Interpretation':<30}")
    print("-" * 60)

    # Define interpretation for common patterns
    interpretations = {
        (): "No ratings",
        (-1.0,): "Single: Bad",
        (0.0,): "Single: Average",
        (1.0,): "Single: Good",
        (-1.0, 0.0): "Disagreement (Bad vs Avg)",
        (-1.0, 1.0): "Large disagreement (Bad vs Good)",
        (0.0, 1.0): "Disagreement (Avg vs Good)",
        (-1.0, -1.0): "2x Bad",
        (0.0, 0.0): "2x Average",
        (1.0, 1.0): "2x Good",
        (-1.0, -1.0, -1.0): "3x Bad (Unanimous)",
        (0.0, 0.0, 0.0): "3x Average (Unanimous)",
        (1.0, 1.0, 1.0): "3x Good (Unanimous)",
        (-1.0, -1.0, 0.0): "2x Bad, 1x Avg",
        (-1.0, -1.0, 1.0): "2x Bad, 1x Good (Large disagreement)",
        (-1.0, 0.0, 0.0): "1x Bad, 2x Avg",
        (-1.0, 0.0, 1.0): "All three different (Controversial)",
        (0.0, 0.0, 1.0): "2x Avg, 1x Good",
        (-1.0, 1.0, 1.0): "1x Bad, 2x Good (Large disagreement)",
    }

    # Sort by count (descending)
    for combo, count in sorted(distributions.items(), key=lambda x: x[1], reverse=True):
        combo_str = str(combo).replace('.0', '')
        interpretation = interpretations.get(combo, "Other combination")

        # Highlight large disagreements
        if combo in [(-1.0, 1.0), (-1.0, -1.0, 1.0), (-1.0, 1.0, 1.0), (-1.0, 0.0, 1.0)]:
            interpretation += " ⚠️"

        print(f"{combo_str:<25} {count:<10} {interpretation:<30}")

    print("=" * 60)

    # Summary statistics
    total_subjects = sum(distributions.values())
    print("\nSUMMARY:")
    print(f"Total subjects: {total_subjects}")

    # Count by category
    unanimous = sum(count for combo, count in distributions.items() if len(combo) == 3 and len(set(combo)) == 1)
    partial_agree = sum(count for combo, count in distributions.items() if len(combo) >= 2 and len(set(combo)) == 1)
    disagreement = sum(count for combo, count in distributions.items() if len(combo) >= 2 and len(set(combo)) > 1)
    single_rater = sum(count for combo, count in distributions.items() if len(combo) == 1)
    no_ratings = distributions.get((), 0)

    print(f"  - Unanimous (3x same): {unanimous}")
    print(f"  - Partial agreement (2x same): {partial_agree - unanimous}")
    print(f"  - Any disagreement: {disagreement}")
    print(f"  - Single rater only: {single_rater}")
    print(f"  - No ratings: {no_ratings}")

def main():
    dataframe = get_abide_labels()

    distributions = get_rating_distributions(dataframe)

    print_rating_distributions(distributions)


if __name__ == '__main__':
    main()
