#!/usr/bin/env python
import sys
import argparse
from collections import Counter
from typing import Literal, TypedDict

import pandas as pd

from brain_mri_qc.abide import collect_available_ratings, get_abide_labels
from brain_mri_qc.utils import normal_variance


class Rating(TypedDict):
    score: int | None
    confidence: Literal['high', 'medium', 'low', 'exclude']
    n_raters: int
    n_majority: int
    agreement_level: float


def compute_scan_rating(row: pd.Series) -> Rating:
    """
    Compute the quality of a scan based on the available manual assessements.
    """

    ratings = collect_available_ratings(row)
    n_raters = len(ratings)

    # No raters
    if n_raters == 0:
        return {
            'score': None,
            'confidence': 'exclude',
            'n_raters': 0,
            'n_majority': 0,
            'agreement_level': 0.0,
        }

    # Large disagreement (both 1 and -1 present)
    if 1 in ratings and -1 in ratings:
        return {
            'score': None,
            'confidence': 'exclude',
            'n_raters': n_raters,
            'n_majority': 0,
            'agreement_level': 0.0,
        }

    # Single rater
    if n_raters == 1:
        score = ratings[0]
        return {
            'score': score,
            'confidence': 'low',
            'n_raters': 1,
            'n_majority': 1,
            'agreement_level': 1.0,
        }

    agreement = normal_variance(ratings, -1, 1)

    counts = Counter(ratings)
    score = counts.most_common(1)[0][0]
    if score > 0:
        score = 1
    if score < 0:
        score = -1

    n_majority = counts[score]

    # Determine confidence based on agreement and number of raters
    if n_raters == n_majority:
        confidence = 'high'
    elif n_raters == 3 and n_majority == 2:
        confidence = 'medium'
    else:
        confidence = 'low'

    return {
        'score': score,
        'confidence': confidence,
        'n_raters': n_raters,
        'n_majority': n_majority,
        'agreement_level': round(agreement, 2),
    }


def sort_rating_infos(results: pd.DataFrame, sort_arg: str) -> pd.DataFrame:
    """
    Sort the rating information based on the specified columns.
    """

    # Define custom sort order for confidence
    confidence_order = {'exclude': 0, 'low': 1, 'medium': 2, 'high': 3}
    results['confidence_sort'] = results['confidence'].map(confidence_order)

    sort_specs = sort_arg.split(',')
    sort_columns = []
    sort_orders = []
    for spec in sort_specs:
        spec = spec.strip()
        if spec.startswith('+'):
            col = spec[1:]
            order = 'asc'
        elif spec.startswith('-'):
            col = spec[1:]
            order = 'desc'
        else:
            col = spec
            order = 'asc'  # default
        if col == 'confidence':
            col = 'confidence_sort'
        sort_columns.append(col)
        sort_orders.append(order == 'asc')
    if sort_columns:
        results = results.sort_values(by=sort_columns, ascending=sort_orders)
        # Drop the temporary column
        results = results.drop(columns=['confidence_sort'])

    return results


def main():
    parser = argparse.ArgumentParser(description='Synthesize ABIDE ratings.')

    parser.add_argument('--sort',
        help='Sort by columns with + for ascending, - for descending (e.g., +score,-confidence).')

    args = parser.parse_args()

    dataframe = get_abide_labels()

    # Get the rating information of each row.
    rating_infos = dataframe.apply(compute_scan_rating, axis=1)

    # Expand the rating dict into separate columns.
    rating_df = pd.DataFrame(rating_infos.tolist())

    # Combine with original dataframe.
    result_df = pd.concat([dataframe[['subject_id', 'site']], rating_df], axis=1)

    # Apply sorting if specified
    if args.sort:
        result_df = sort_rating_infos(result_df, args.sort)

    # Print only the TSV to console
    result_df.to_csv(sys.stdout, sep='\t', index=False)


if __name__ == '__main__':
    main()
