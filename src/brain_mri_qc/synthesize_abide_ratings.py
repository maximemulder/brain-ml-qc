#!/usr/bin/env python
import sys
from collections import Counter
from typing import Literal, TypedDict

import pandas as pd

from brain_mri_qc.abide import get_abide_labels


class Rating(TypedDict):
    score: int | None
    confidence: Literal['high', 'medium', 'low', 'exclude']
    reason: str
    n_raters: int
    agreement_level: float


def compute_scan_rating(row: pd.Series) -> Rating:
    """
    Compute the quality of a scan based on the available manual assessements.
    """

    # Collect valid ratings
    ratings = []
    for rater in ['rater_1', 'rater_2', 'rater_3']:
        val = row[rater]
        if val != 'n/a' and pd.notna(val):
            try:
                ratings.append(float(val))
            except (ValueError, TypeError):
                pass

    n_raters = len(ratings)

    # No assessments
    if n_raters == 0:
        return {
            'score': None,
            'confidence': 'exclude',
            'reason': 'No ratings available',
            'n_raters': 0,
            'agreement_level': 0.0
        }

    # Single rater
    if n_raters == 1:
        score = ratings[0]
        return {
            'score': score,
            'confidence': 'low',
            'reason': f'Single rater only (score: {score})',
            'n_raters': 1,
            'agreement_level': 1.0
        }

    # Large disagreement (both 1 and -1 present)
    if 1 in ratings and -1 in ratings:
        return {
            'score': None,
            'confidence': 'exclude',
            'reason': f'Large disagreement: Good vs Bad conflict {ratings}',
            'n_raters': n_raters,
            'agreement_level': 0.0
        }

    counts = Counter(ratings)
    majority_score = counts.most_common(1)[0][0]
    majority_count = counts[majority_score]
    agreement = majority_count / n_raters

    # Determine confidence based on agreement and number of raters
    if n_raters == 3 and majority_count == 3:
        confidence = 'high'
        reason = 'Unanimous agreement: 3/3 raters'
    elif n_raters == 3 and majority_count == 2:
        confidence = 'medium'
        minority_score = [s for s in ratings if s != majority_score][0]
        reason = f'Majority (2/3): {majority_score} with 1 dissenter ({minority_score})'
    elif n_raters == 2 and majority_count == 2:
        confidence = 'high'
        reason = 'Complete agreement: 2/2 raters'
    elif n_raters == 2 and majority_count == 1:
        confidence = 'low'
        reason = f'Disagreement between 2 raters: {ratings}'
    else:
        confidence = 'low'
        reason = f'Mixed ratings: {ratings}'

    return {
        'score': majority_score,
        'confidence': confidence,
        'reason': reason,
        'n_raters': n_raters,
        'agreement_level': agreement
    }

class Rating(TypedDict):
    score: int | None
    confidence: Literal['high', 'medium', 'low', 'exclude']
    reason: str
    n_raters: int
    agreement_level: float


def main():
    dataframe = get_abide_labels()

    # Apply compute_row_rating to each row
    rating_results = dataframe.apply(compute_scan_rating, axis=1)

    # Expand the rating dict into separate columns
    rating_df = pd.DataFrame(rating_results.tolist())

    # Combine with original dataframe
    result_df = pd.concat([dataframe[['subject_id', 'site']], rating_df], axis=1)

    # Print only the TSV to console
    result_df.to_csv(sys.stdout, sep='\t', index=False)


if __name__ == '__main__':
    main()
