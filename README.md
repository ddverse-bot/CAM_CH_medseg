# Uncertainty-Aware Deep Learning Framework for Classification-Guided Medical Image Segmentation

PyTorch implementation of a unified framework for **medical image segmentation, explainability, and uncertainty estimation**.

The framework integrates:

- **Classification Head (CH)** for image-level supervision
- **Class Activation Mapping (CAM)** for spatial explanation
- **Anatomy-aware uncertainty estimation**
- **Average Calibration Error (ACE)** loss for prediction calibration

---

## Architectures

<img width="1313" height="800" alt="Framework architecture diagram" src="https://github.com/user-attachments/assets/4da27f58-6067-40ac-9fb8-7c02b0627f08" />

The framework is evaluated using:

- U-Net
- U-Net++
- Attention U-Net

---

## Datasets

Experiments are conducted on four publicly available medical imaging datasets:

| Dataset | Modality |
|---|---|
| ACDC | Cardiac MRI |
| COVID-19 CT | Chest CT |
| BUSI | Breast Ultrasound |
| PU2756 | Pulmonary Ultrasound |

All datasets are formulated as **binary segmentation** tasks. Images are resized to **256 × 256** and normalized to **[0, 1]**.

---

## Framework

### 1. Segmentation

The segmentation network produces a pixel-wise prediction:

$$
\hat{y} = f_{\theta}(x)
$$

The segmentation loss combines Dice loss and Binary Cross-Entropy:

$$
\mathcal{L}_{seg} = \frac{1}{2}\mathcal{L}_{Dice} + \frac{1}{2}\mathcal{L}_{BCE}
$$

### 2. Classification Head

The Classification Head (CH) is attached to the encoder bottleneck to provide additional image-level supervision and strengthen the learned feature representation.

The classification loss is:

$$
\mathcal{L}_{cls} = BCE(\hat{c}, c)
$$

where $c$ represents the image-level target derived from the segmentation mask.

### 3. Class Activation Mapping

Class Activation Mapping (CAM) provides spatial information about the regions contributing to the classification prediction.

CAM is generated from the deep encoder features and classification weights, providing additional spatial guidance to the segmentation network.

### 4. Anatomy-Aware Uncertainty

Anatomical priors are used to estimate spatial prediction uncertainty.

The uncertainty map is defined as:

$$
U = \frac{1}{M}\sum_{i=1}^{M} w_i \left| \bar{y}_i - y^* \right|
$$

where:

- $\bar{y}_i$ — the retrieved anatomical prior segmentations
- $y^*$ — the predicted segmentation
- $w_i$ — the similarity-based weight assigned to the $i$-th anatomical prior
- $M$ — the number of retrieved anatomical priors

Higher uncertainty indicates greater disagreement between the predicted segmentation and anatomically plausible segmentations.

### 5. Average Calibration Error

Average Calibration Error (ACE) measures the difference between predicted confidence and observed accuracy:

$$
ACE = \frac{1}{CM}\sum_{c=1}^{C}\sum_{m=1}^{M} \left| o_{cm} - e_{cm} \right|
$$

where $o_{cm}$ represents the observed accuracy and $e_{cm}$ represents the expected confidence for class $c$ and confidence bin $m$.

The regional calibration loss is:

$$
\mathcal{L}_{ACE} = \frac{1}{N}\sum_{i=1}^{N} \left| u_i - e_i \right|
$$

where $u_i$ is the predicted uncertainty and $e_i$ is the corresponding segmentation error.

---

## Overall Objective

The complete framework combines segmentation, classification, CAM, and calibration objectives:

$$
\mathcal{L}_{total} = \mathcal{L}_{seg} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{CAM}\mathcal{L}_{CAM} + \lambda_{ACE}\mathcal{L}_{ACE}
$$

The experimental loss weights are:

$$
\lambda_{cls} = 0.2, \qquad \lambda_{CAM} = 0.3, \qquad \lambda_{ACE} = 0.5
$$

Different ablation configurations selectively add or remove **CH, CAM, and ACE** to study their individual and combined contributions.

---

## Evaluation

The framework evaluates segmentation performance using:

- Dice
- IoU
- Precision
- Recall
- F1-score

Uncertainty is evaluated using the magnitude of the spatial uncertainty and its correlation with segmentation errors.

Qualitative analysis includes:

- Segmentation predictions
- Class Activation Maps
- Uncertainty maps

---

## Project Structure

```text
CAM_CH_medseg/
├── models/
│   ├── unet.py
│   ├── unetplusplus.py
│   ├── attunet.py
│   └── classification_head.py
│
├── losses/
│   ├── segmentation_loss.py
│   ├── classification_loss.py
│   ├── cam_loss.py
│   ├── ace_loss.py
│   └── total_loss.py
│
├── datasets/
│   ├── busi.py
│   ├── acdc.py
│   ├── pulmonary.py
│   └── covid.py
│
├── uncertainty/
│   └── anatomical_prior.py
│
├── configs/
│   ├── busi.yaml
│   ├── acdc.yaml
│   ├── pulmonary.yaml
│   └── covid.yaml
│
├── train.py
├── evaluate.py
├── make_visuals.py
├── run_all.py
├── requirements.txt
└── README.md
```
