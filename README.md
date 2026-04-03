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
