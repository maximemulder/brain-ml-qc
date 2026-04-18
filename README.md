# Brain MRI QC

Automated neurological MRI quality control using AI.

## Install

To install Brain MRI QC, use the following commands in the project root directory:

```sh
# Create a Python virtual environment for the project.
python3 -m venv .venv
# Active the Python virtual environment.
# This command only works in Bash-compatible shells, use appropriate script for other shells.
source .venv/bin/activate
# Install the project in editable mode as well as its dependencies.
pip install -e .
```

Once the project is installed, its various tools should be available directly as shell commands. These tools are described invidually below.

## Datasets

This project mainly uses two datasets: IXI and ABIDE I.

## Dataset Pipelines

### Download ABIDE dataset

The `download-abide` tool can be used to automatically download the ABIDE I or ABIDE II dataset:

```sh
download-abide abide-i /path/to/dest/dir/
```

Since the ABIDE datasets are quite large, the tool can be run several times to download only the missing files in the destination directory.

Note that the well behavior of this script partially depends on the dataset provider that may change and is out of our control.

### Synthesize artifacts

The `synthesize-artifacts` tool can be used to create a copy of the IXI dataset that contains images with random synthesized MRI artifacts using TorchIO:

```sh
synthesize-artifacts /path/to/input/ixi /path/to/output/modified/ixi
```

A `labels.tsv` file will be created in the output dataset with information about the artifacts created for each scan.

### Summarize ABIDE manual assessements

The `summarize-abide-ratings` tool can be used to get a summary of the ABIDE I manual assessements distribution. The input rating file is obtained from the `mriqc_learn` Python package and does not need to be supplied manually.

```sh
summarize-abide-ratings
```

### Labelize ABIDE manual assessements

The `labelize-abide-ratings` tool can be used to combine the ABIDE I manual assessements into labels to use for training. The input rating file is obtained from the `mriqc_learn` Python package and does not need to be supplied manually.

```sh
labelize-abide-ratings --dataset /path/to/abide/i
```

A `labels.tsv` file will be created in the dataset with information about the combined ratings for each scan.

## Training pipelines

The training pipelines can be used through shell scripts present in the `shell` directory, to be ran from the root project directory (the one with this `README.md` file !).

## MRI-QC

To run the [MRI-QC](https://mriqc.readthedocs.io/en/latest/) tool on the ABIDE dataset, use the following steps.

First, organize the ABIDE dataset into a BIDS dataset using the `bidsify-abide` tool:

```sh
bidsify-abide /path/to/input/abide/dir /path/to/output/abide/bids --sites=SITE1,SITE2...
```

Then, use the `run-mri-qc` wrapper on the generated BIDS dataset:

```sh
run-mri-qc /path/to/input/abide/bids /path/to/output/mriqc/dir
```

Note that running MRI-QC requires to have Docker installed on your machine.
