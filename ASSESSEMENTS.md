# Assessements

## Useful links

- [Gibbs artifact](https://radiopaedia.org/articles/gibbs-and-truncation-artifacts): Normal and common artifact.

## Methodology

Maxime's web-based brain viewer ([link](https://maximemulder.github.io/brain-render/)) was used to visualize corrupted scans and compare them against clean scans.

## IXI (synthesized artifacts)

Visual assessements of the synthesized artifacts on the IXI dataset.

Assessements:
- Motion: The image is blurry.
- Bias field: The intensity of the image is uneven, lighter on one side and darker on the other. The corrupted image achieves higher peak intensity, but automatic color calibration makes it hard to see in terms of absolute values.
- Spike: An array of white and dark lines cross the image. Extremely noticeable.
- Noise: Do not see the difference (maybe the noise is too weak to notice ?).

## ABIDE II

Student manual assessements for ABIDE I. The files to assess are based on the expert assessements from the MRIQC paper, which can be obtained using the `synthesize-abide-assessements` command.

### MAX_MUN 0051334

Path: `MaxMun_c/0051334/session_1/anat_1/mprage.nii.gz`
Assessement:
- The contrast seems a little low, but outside of that it looks normal.

### MAX_MUN 0051348

Path: `MaxMun_c/0051348/session_1/anat_1/mprage.nii.gz`
Assessment:
- Prevalent Gibbs artifact.

### MAX_MUN 0051350

Path: `MaxMun_d/0051350/session_1/anat_1/mprage.nii.gz`
Assessment:
- The head seems somewhat tilted.
- Prevalent Gibbs artifact.

### KKI 50779

Path: `KKI/0050779/session_1/anat_1/mprage.nii.gz`
Assessement:
- The image seems unnormally bright compared to clean images. Although adjusting brightness and contrast seems to fix the issue.


## SDU 50182

Path: `SDSU/0050182/session_1/anat_1/mprage.nii.gz`
Assessement:
- Cannot find a difference with clean images.
