# Uncertainty Aware Deep Learning Framework for Classification-Guided Medical Image Segmentation

PyTorch implementation of a unified framework for **medical image segmentation, explainability, and uncertainty estimation**.

The framework integrates:

- **Classification Head (CH)** for image-level supervision
- **Class Activation Mapping (CAM)** for spatial explanation
- **Anatomy-aware uncertainty estimation**
- **Average Calibration Error (ACE)** loss for prediction calibration

## Architectures

- U-Net
- U-Net++
- Attention U-Net
<img width="1313" height="800" alt="image" src="https://github.com/user-attachments/assets/1de732db-e00e-4d0f-a72c-beddaa1b0ef6" />


## Datasets

- ACDC
- COVID-19 CT
- BUSI
- PU2756 Pulmonary Ultrasound

All datasets are formulated as **binary segmentation** tasks with images resized to **256 × 256** and normalized to **[0, 1]**.

## Framework

### Segmentation

The segmentation network produces a pixel-wise prediction:

$$
\hat{y} = f_{\theta}(x)
$$

The segmentation loss combines Dice loss and Binary Cross-Entropy:

$$
\mathcal{L}_{seg}
=
\frac{1}{2}\mathcal{L}_{Dice}
+
\frac{1}{2}\mathcal{L}_{BCE}
$$

### Classification Head

The Classification Head (CH) is attached to the encoder bottleneck to provide image-level supervision:

$$
\mathcal{L}_{cls}
=
BCE(\hat{c},c)
$$

where $c$ represents the image-level target derived from the segmentation mask.

### Class Activation Mapping

Class Activation Mapping (CAM) provides spatial information about the regions contributing to the classification prediction. It uses the deep encoder features and classification weights to provide additional spatial guidance to the segmentation network.

### Anatomy-Aware Uncertainty

Anatomical priors are used to estimate spatial prediction uncertainty:

$$
U =
\frac{1}{M}
\sum_{i=1}^{M}
w_i
\left|
\bar{y}_i-y^*
\right|
$$

where $\bar{y}_i$ represents the retrieved anatomical prior segmentations, $y^*$ is the predicted segmentation, and $w_i$ is the similarity-based weight of each prior.

Higher uncertainty indicates greater disagreement between the prediction and anatomically plausible segmentations.

### Average Calibration Error

ACE measures the difference between predicted confidence and observed accuracy:

$$
ACE =
\frac{1}{CM}
\sum_{c=1}^{C}
\sum_{m=1}^{M}
|o_{cm}-e_{cm}|
$$

The regional calibration loss is:

$$
\mathcal{L}_{ACE}
=
\frac{1}{N}
\sum_{i=1}^{N}
|u_i-e_i|
$$

where $u_i$ is the predicted uncertainty and $e_i$ is the corresponding segmentation error.

## Overall Objective

The complete framework combines segmentation, classification, CAM, and calibration objectives:

$$
\mathcal{L}_{total}
=
\mathcal{L}_{seg}
+
\lambda_{cls}\mathcal{L}_{cls}
+
\lambda_{CAM}\mathcal{L}_{CAM}
+
\lambda_{ACE}\mathcal{L}_{ACE}
$$

The experimental loss weights are:

$$
\lambda_{cls}=0.2,\qquad
\lambda_{CAM}=0.3,\qquad
\lambda_{ACE}=0.5
$$

Different ablation configurations selectively add or remove **CH, CAM, and ACE** to study their individual and combined contributions.

## Evaluation

The framework evaluates segmentation using:

- Dice
- IoU
- Precision
- Recall
- F1-score

It also evaluates prediction uncertainty and its correlation with segmentation errors.

Qualitative outputs include:

- Segmentation masks
- CAM maps
- Uncertainty maps

