# Brain MRI QC

Automated neurological MRI quality control using AI.

## Install

In the project root directory:

```sh
# Create a Python virtual environment for the project.
python3 -m venv .venv
# Active the Python virtual environment.
source .venv/bin/activate
# Install the project in editable mode as well as its dependencies.
pip install -e .
```

## Training dataset

To create the training dataset with synthesized artifacts, use the `create-dataset` script on a clean dataset:

```sh
create-dataset /path/to/ixi/dir /path/to/brain/mri/qc/dir
```

## Validation dataset

To download the validation dataset, ABIDE-I, use the `download-abide` script:

```sh
download-abide abide-i /path/to/dest/dir/
```

Note that this script can also be used to download the ABIDE-II dataset. Since the ABIDE datasets are quite large, the script can be run several times to download only the missing files in the destination directory.

The two following scripts are also available:

- `analyze-abide-ratings`: Analyze the distribution of manual ratings available in the ABIDE dataset.
- `synthesize-abide-ratings`: Synthesize the distribution of manual ratings available in the ABIDE dataset.
