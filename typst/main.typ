#set text(
  // font: "New Computer Modern"
  size: 7pt,
)

#show table: set block(breakable: false)

== Automatic quality control of brain T1-weighted magnetic resonance images for a clinical data warehouse

- Publication: January 2022
- Journal: Medical Image Analysis
- Tool name: Conv5_FC3
- Architecture: Convolutional Neural Network
- Data type: 3D T1-weighted brain MRI
- Dataset: Greater Paris area hospitals data warehouse
- Training: 5000 images
- Testing: 500 images
- QC goals:
  1. identify images which are not proper T1-weighted brain MRIs
  2. identify acquisitions for which gadolinium was injected
  3. rate the overall image quality
- Results:
  - \1. and 2. balanced accuracy and F1-score >90%
  - balanced accuracy and F1-score >80%, substantially lower than that of human raters

== MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites

- Publication: January 2017
- Journal/Conference: PLOS One
- Name: MRIQC
- Architecture: Binary classifier (accept/exclude) — specific architecture not detailed in abstract
- Data type: MRI (general, not limited to T1-weighted, but neuroimaging focus)
- Dataset: Multi‑site (17 sites, N=1102 for training; held‑out: 2 sites, N=265)
- Training: 1102 images (17 sites)
- Testing: 265 images (2 held‑out sites) + leave‑one‑site‑out cross‑validation on 17 sites
- QC goals:
  1. Extract quality measures
  2. Fit a binary (accept/exclude) classifier
- Results:
  - Leave‑one‑site‑out (new sites): Accuracy = 76% ± 13%
  - Held‑out dataset (2 sites): Accuracy = 76%
  - Intra‑site prediction: High accuracy (exact value not given, but noted as high)
  - Performance on unseen sites: Statistically above chance, but susceptible to site effects and unable to account for artifacts specific to new sites
  - Overall assessment: Good for intra‑site, limited generalization to unseen sites; requires more labeled data and new approaches to between‑site variability

== QC-Automator: Deep Learning-Based Automated Quality Control for Diffusion MR Images

- Publication: January 2020
- Journal/Conference: Frontiers in Neuroscience
- Name: QC-Automator
- Architecture: Convolutional Neural Networks (CNN) with transfer learning
- Data type: Diffusion MRI (dMRI)
- Dataset: 155 unique subjects, 5 scanners with different dMRI acquisitions
- Training: ~332,000 slices
- Testing: [To be filled] (replicability demonstrated on other datasets with different acquisition parameters)
- QC goals:
  1. Detect multiple artifact types: motion, multiband interleaving, ghosting, susceptibility, herringbone, chemical shifts
- Results:
  - Artifact detection accuracy: 98%
  - Fast processing time
  - Replicable on other datasets with different acquisition parameters

== Automatic brain MRI motion artifact detection based on end-to-end deep learning

- Publication: January 2022
- Journal/Conference: Medical Image Analysis
- Name: 3D-CNN-v1/2
- Architecture: Lightweight 3D Convolutional Neural Network (end-to-end, no elaborate pre-processing)
- Data type: 3D T1-weighted brain MRI
- Dataset: Publicly available T1-weighted images + locally scanned subjects under conventional and active head motion conditions
- Training: N = 1661, UK Biobank
- Testing: N = 411, UK Biobank (same sites)
- QC goals:
  - Classify brain scans into clinically usable or unusable categories (motion artifact detection)
- Results:
  - End-to-end 3D CNN balanced accuracy: 94.41%
  - Support Vector Machine (SVM) on image quality metrics balanced accuracy: 88.44%
  - Statistical comparison: No significant difference between the two models (confusion matrices, error rates, ROC curves)
- Conclusion: Both methods similarly effective in identifying severe motion artifacts
- Important note: This neural network was tested on same site as it was trained (not cross-site generalization)

#pagebreak()

== Systematic comparisons of different quality control approaches applied to three large pediatric neuroimaging datasets

- Publication: July 2023
- Journal: NeuroImage

=== Low overlap between QC approaches
- Different methods (visual, metric, automated) exclude different participants
- Visual vs. automated: only moderate overlap (rtet = 0.52–0.59)

=== QC exclusion introduces sampling bias
Excluded participants tend to be: younger, lower IQ, lower adaptive functioning, lower cortical thickness/subcortical volume
→ Exclusion is NOT random

=== Minimal impact on clinical/brain metrics in large datasets
- Different QC approaches exclude different subsets, but variation in clinical/brain metrics is minimal
- Effect on brain structure does not differ by QC approach

=== Automated QC advantages
- Standardized, reduces between-study differences
- Supported for large datasets despite moderate overlap with visual QC

=== Dataset specificity matters
- Results are dataset-specific; generalization cannot be assumed

#pagebreak()

== Publications

#table(
  columns: 4,
  inset: 4pt,
  align: horizon,
  table.header(
    [*Tool*],[*Date*],[*Journal*],[*Paper*],
  ),
  [MRIQC],[January 2017],[PLOS One],[MRIQC: Advancing the automatic prediction of image quality in MRI from unseen sites],
  [QC-Automator],[January 2020],[Frontiers in Neuroscience],[QC-Automator: Deep Learning-Based Automated Quality Control for Diffusion MR Images],
  [Conv5_FC3],[January 2022],[Medical Image Analysis],[Automatic quality control of brain T1-weighted magnetic resonance images for a clinical data warehouse],
  [3D-CNN-v1/2],[January 2022],[Medical Image Analysis],[Automatic brain MRI motion artifact detection based on end-to-end deep learning is similarly effective as traditional machine learning trained on image quality metrics],
)

== Models

#table(
  columns: 4,
  inset: 4pt,
  align: horizon,
  table.header(
    [*Tool*],[*Data modality*],[*Data type*],[*Architecture*],
  ),
  [MRIQC],[T1-weighted brain MRI],[3D],[IQMs + classifier],
  [QC-Automator],[Diffusion MRI],[2D (axial and sagittal slices)],[CNN],
  [Conv5_FC3],[T1-weighted],[3D],[CNN],
  [3D-CNN-v1/2],[T1-weighted],[3D],[CNN],
)

== Datasets

#table(
  columns: 7,
  inset: 4pt,
  align: horizon,
  table.header(
    [*Tool*],[*Training Dataset*],[*Training sample*],[*Testing dataset*],[*Testing sample*],[*External validation*],[*Open data*]
  ),
  [MRIQC],[ABIDE],[1102 (17 sites)],[DS030],[265 (2 sites)],[#sym.checkmark],[#sym.checkmark],
  [QC-Automator],[3 unspecified datasets],[79 subjects],[2 unspecified different datasets],[88 subjects],[#sym.checkmark],[#sym.crossmark],
  [Conv5_FC3],[Greater Paris area hospitals],[5000],[held-out data],[500],[#sym.crossmark],[#sym.crossmark],
  [3D-CNN-v1/2],[UK Biobank + MR-ART (local)],[1661],[held-out data],[411],[#sym.crossmark],[#sym.checkmark],
)

== Labels

#table(
  columns: 4,
  inset: 4pt,
  align: horizon,
  table.header(
    [*Tool*],[*Classes*],[*Manual QC*],[*Other*],
  ),
  [MRIQC],[exclude /doubtful / accept],[3 raters],[-],
  [QC-Automator],[artifact-free / artifact],[2 raters],[Data augmentation:
    - rotating
    - cropping
    - zooming
    - shearing
  ],
  [Conv5_FC3],[
    - rejection (not T1w MRI): 0 / 1
    - gadolinium: 0 / 1 /2
    - contrast: 0 / 1 /2
    - noise: 0 / 1 /2
  ],[2 trained raters],[-],
  [3D-CNN-v1/2],[Motion: 1 / 2 /3],[5 raters: 2 senior, 3 junior],[-],
)

== Results

#table(
  columns: 10,
  inset: 4pt,
  align: horizon,
  table.header(
    [*Tool*], [*Task*], [*Accuracy*], [*Balanced Accuracy*], [*Sensitivity*], [*Specificity*], [*AUC*], [*F1*], [*Human*], [*Notes*],
  ),
  [MRIQC], [Accept/Exclude], [76%], [—], [28%], [—], [0.707], [—], [—], [Overfitting; ghost artifact],
  [QC-Automator], [Axial artifact], [98%], [—], [91%], [—], [—], [0.94], [—], [Slice-level],
  [QC-Automator], [Sagittal artifact], [98%], [—], [91%], [—], [—], [0.92], [—], [Slice-level],
  [Conv5_FC3], [SR detection], [—], [93.8%], [91.8%], [95.7%], [93.8%], [94.9%], [97.1%], [Below human],
  [Conv5_FC3], [Gadolinium], [—], [97.1%], [96.5%], [97.8%], [97.1%], [97.0%], [96.1%], [Matches human],
  [Conv5_FC3], [Tier3 vs 2-1], [—], [83.5%], [79.9%], [87.1%], [83.5%], [84.1%], [91.6%], [Below human],
  [Conv5_FC3], [Tier2 vs 1], [—], [71.7%], [77.4%], [65.9%], [71.7%], [74.1%], [88.3%], [Below human],
  [3D-CNN-v1/2], [Usable/Unusable], [92.9%], [94.4%], [96.4%], [92.4%], [0.973], [—], [—], [Best overall],
)

== Differences

#table(
  columns: 13,
  inset: 4pt,
  align: horizon + center,
  table.header(
    [*Paper*], [*Unit*], [*Input*], [*Task*], [*Balance*], [*Human*], [*External*], [*Overfit*], [*Best Metric*], [*Modality*], [*Inter-rater*], [*Augmentation*], [*CV Method*],
  ),
  [MRIQC], [3D], [IQMs], [Binary], [Imbal.], [No], [Yes], [LoSo], [76% acc], [T1w], [—], [No], [LoSo],
  [QC-Automator], [2D], [Raw], [Binary], [—], [No], [Yes], [Simple], [98% acc], [dMRI], [—], [Yes], [Simple],
  [Conv5_FC3], [3D], [Raw], [Multi], [Reported], [Yes], [No], [5-fold], [97.1% bal], [T1w], [0.68-0.89], [No], [5-fold],
  [3D-CNN-v1/2], [3D], [Raw], [Binary], [Balanced], [No], [Yes], [Median], [94.4% bal], [T1w], [Reported], [No], [Nested],
)
