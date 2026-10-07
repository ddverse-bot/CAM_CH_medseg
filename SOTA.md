# State-of-the-Art Comparison Methods

This document describes the additional methods included in the
state-of-the-art (SOTA) comparison experiments reported in the manuscript.

## Primary Framework

The proposed framework is primarily evaluated using:

- U-Net
- U-Net++
- Attention U-Net

The Classification Head (CH), Class Activation Mapping (CAM), and Average
Calibration Error (ACE) components are systematically evaluated using these
primary backbones.

## SOTA Comparison Methods

### nnU-Net

nnU-Net is a self-configuring framework for biomedical image segmentation
that automatically adapts preprocessing, network configuration, training,
and inference procedures to the characteristics of a given dataset.

Reference:

Isensee, F., Jaeger, P. F., Kohl, S. A. A., Petersen, J., and Maier-Hein,
K. H. "nnU-Net: a self-configuring method for deep learning-based
biomedical image segmentation." Nature Methods, 18, 203--211, 2021.

BibTeX key:

`isensee2021nnunet`

---

### nnU-Net (ResEnc)

nnU-Net (ResEnc) denotes the residual-encoder configuration of nnU-Net.
This configuration is included as an additional comparison baseline to
evaluate the proposed framework against a stronger modern nnU-Net variant.

Reference:

Isensee, F. et al. "nnU-Net Revisited: A Call for Rigorous Validation in
3D Medical Image Segmentation." arXiv preprint arXiv:2404.09556, 2024.

BibTeX key:

`isensee2024nnunetrevisited`

---

### MedNeXt

MedNeXt is a ConvNeXt-inspired architecture designed specifically for
medical image segmentation. It provides an additional convolutional
baseline with modern architectural components.

Reference:

Roy, S. et al. "MedNeXt: Transformer-driven Scaling of ConvNets for
Medical Image Segmentation." Medical Image Computing and Computer Assisted
Intervention (MICCAI), 2023, pp. 405--415.

BibTeX key:

`roy2023mednext`

---

### RWKV-UNet

RWKV-UNet incorporates long-range contextual modeling into a U-Net-style
medical image segmentation framework.

Reference:

Jiang, Y. et al. "RWKV-UNet: Improving UNet with Long-Range Cooperation
for Effective Medical Image Segmentation." arXiv:2501.08458, 2025.

BibTeX key:

`jiang2025rwkvunet`

---

### SAM-Mix

SAM-Mix is included as a recent Segment Anything Model (SAM)-based
comparison method for medical image segmentation.

Reference:

Ward, M. and Imran, A. "Annotation-Efficient Task Guidance for Medical
Segment Anything." IEEE International Symposium on Biomedical Imaging
(ISBI), 2025.

BibTeX key:

`ward2025sammix`

---

### NACL

NACL refers to Neighbor-aware Calibration of segmentation networks with
penalty-based constraints. It is included as a comparison method with
particular relevance to segmentation calibration.

Reference:

Murugesan, B., Vasudeva, A., Liu, J., Lombaert, H., Ben Ayed, I., and Dolz,
J. "Neighbor-aware calibration of segmentation networks with penalty-based
constraints." Medical Image Analysis, 101, 103501, 2025.

BibTeX key:

`murugesan2025nacl`

---

## Experimental Role

The primary U-Net-based architectures are used for the systematic CH, CAM,
and ACE ablation experiments.

The additional SOTA methods are included to provide broader comparisons
with recent medical image segmentation approaches. Each method is evaluated
on the datasets for which results are reported in the manuscript.

Exact implementation details, configurations, pretrained weights, and
software versions should be recorded for each SOTA method to facilitate
reproducibility.
