#!/usr/bin/env python
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torchio as tio


@dataclass
class Transform:
    name: str
    probability: float
    transform: tio.Transform


TRANSFORMS = [
    Transform('motion',     0.4, tio.RandomMotion(degrees=10, translation=5)),
    Transform('bias_field', 0.3, tio.RandomBiasField(coefficients=0.5)),
    Transform('spike',      0.2, tio.RandomSpike(num_spikes=1)),
    Transform('noise',      0.1, tio.RandomNoise(std=0.1)),
]

TRANSFORM_NAMES         = [transform.name        for transform in TRANSFORMS]
TRANSFORM_PROBABILITIES = [transform.probability for transform in TRANSFORMS]


def main():
    parser = argparse.ArgumentParser(description=(
        "Read a clean NIfTIs directory and create a training dataset by adding synthesized artifacts"
        " and a label file."
    ))

    parser.add_argument(
        '--num-samples',
        type=int,
        help="Number of images to process.",
    )

    parser.add_argument(
        'input',
        type=Path,
        help="The input directory that contains clean NIfTIs.",
    )

    parser.add_argument(
        'output',
        type=Path,
        help="The output directory where the training dataset will be copied.",
    )

    args = parser.parse_args()

    create_dataset(args.input, args.output, args.num_samples)

def create_dataset(input_path: Path, output_path: Path, num_samples: int | None):
    # Check that the output directory is empty, creating it if necessary.
    output_path.mkdir(exist_ok=True)
    if not is_empty_directory(output_path):
        raise Exception(f"Output directory '{output_path}' is not empty.")

    # Get the images directory and labels file paths.
    images_path = output_path / 'images'
    labels_path = output_path / 'labels.csv'

    # Create the output images directory.
    images_path.mkdir()

    # Get the subjects from the input directory.
    subjects = get_subjects(input_path, num_samples)

    # Write the header to the labels CSV file.
    with open(labels_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'artifact', *TRANSFORM_NAMES])

    for subject in subjects:
        clean_path     = images_path / f'{subject.name}_clean.nii.gz'
        corrupted_path = images_path / f'{subject.name}_corrupted.nii.gz'

        # Save the clean image.
        print(f"Writing clean image '{clean_path}'.")
        subject.image.save(clean_path)

        write_label_row(labels_path, clean_path.name, None)

        # Generate a corrupted version by applying the artifact transform
        # We apply the transform to a copy of the subject to keep the original intact
        image_copy = subject.copy()['image']

        # Chooose a transform randomly based on their probabilities.
        transform_id = np.random.choice(len(TRANSFORMS), p=TRANSFORM_PROBABILITIES)
        transform = TRANSFORMS[transform_id]

        print(f"Applying random transform '{transform.name}'")
        corrupted_image = transform.transform(image_copy)

        print(f"Writing corrupted image '{clean_path}'")
        corrupted_image.save(images_path / f'{subject.name}_corrupted.nii.gz')

        write_label_row(labels_path, corrupted_path.name, transform.name)


def get_subjects(input_path: Path, num_samples: int | None) -> list[tio.Subject]:
    """
    Get the list of subjects from the input directory.
    """

    subjects: list[tio.Subject] = []
    for i, file_path in enumerate(input_path.iterdir()):
        if num_samples is not None and i >= num_samples:
            break

        [stem, extension] = file_path.name.split('.', maxsplit=1)

        if extension != 'nii.gz':
            print(f"Non-NIfTI file found, skipping '{file_path}'")
            continue

        print(f"Found NIfTI file '{file_path}'")

        subjects.append(tio.Subject(
            name=stem,
            path=file_path,
            image=tio.ScalarImage(file_path),
        ))

    return subjects


def write_label_row(labels_path: Path, filename: str, transform_name: str | None):
    """
    Write am image row to the labels CSV file.
    """

    # Determine if the image is clean or corrupted.
    has_artifact = 1 if transform_name is not None else 0

    # Create a one-hot encoding for the transforms.
    transform_flags = [1 if transform_name == column else 0 for column in TRANSFORM_NAMES]

    # Append the image row to the CSV.
    with open(labels_path, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([filename, has_artifact, *transform_flags])


def is_empty_directory(path: Path):
    """
    Check if a directory is empty.
    """

    return not any(path.iterdir())


if __name__ == '__main__':
    main()
