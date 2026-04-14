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

## ABIDE I

### Download

To download the validation dataset, ABIDE-I, use the `download-abide` script:

```sh
download-abide abide-i /path/to/dest/dir/
```

Note that this script can also be used to download the ABIDE-II dataset. Since the ABIDE datasets are quite large, the script can be run several times to download only the missing files in the destination directory.

### Manual assessements

The ABIDE I dataset comes with expert manual assessements on some scans used in the MRIQC paper. The manual assessements are done by three raters (`rater_1`, `rater_2`, `rater_3`) and categorize scans in three categories (`-1` = bad, `0` = average, `1` = good).

To get a short analysis of the manual assessements distribution, use the `analyze-abide-ratings` script :

```sh
analyze-abide-ratings
```

To annotate the ABIDE I dataset with the manual ratings, use the `annotate-abide-ratings` script:

```sh
annotate-abide-ratings /path/to/abide/1/dir
```

The ABIDE I directory must contain the scan files from the different ABIDE I sites. The script will emit a warning if a scan cannot be found. The script will save the labels to a `labels.tsv` file inside of the dataset. The script also be used without a dataset argument to get a summary of the ABIDE I manual assessements in the console.
