# Tools

## TorchIO

> TorchIO is a Python package containing a set of tools to efficiently read, preprocess, sample, augment, and write 3D medical images in deep learning applications written in PyTorch, including intensity and spatial transforms for data augmentation and preprocessing.

Link: https://github.com/TorchIO-project/torchio

The transform features provided by the library can notably be used to simulate MRI artifacts. More information at the following link:
https://docs.torchio.org/transforms/augmentation/#intensity

## MRIQC

> MRIQC extracts no-reference IQMs (image quality metrics) from structural (T1w and T2w), functional and diffusion MRI (magnetic resonance imaging) data.

Link: https://github.com/nipreps/mriqc/

## MRIQC-learn

This seems to be the library used to train the MRIQC tool. It notably seems to contain the manual ratings of the ABIDE and DS030 datasets that were used to train the tool as TSV files in the `mriqc/datasets` directory.

Link: https://github.com/nipreps/mriqc-learn
