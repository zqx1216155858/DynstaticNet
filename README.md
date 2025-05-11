# DynStaticNet: Dynamic-Static Dual-Branch Network for Multi-Weather Video Restoration

![License](https://img.shields.io/badge/license-MIT-green)

> Official PyTorch implementation of **"DynStaticNet: Dynamic-Static Dual-Branch Network for Multi-Weather Video Restoration"**.  
> [[Paper]](TBD) | [[Project Page]](TBD) | [[Video Demo]](TBD)

## 🌟 Overview

**DynStaticNet** is a biologically-inspired dual-branch video restoration framework that decouples spatial and temporal features, enabling unified modeling of dynamic degradations (e.g., rain, snow) and static degradations (e.g., haze). Our method introduces:

- ✅ A **temporal branch** with 3D adaptive self-attention (3DASA) to model long-range motion.
- ✅ A **spatial branch** enhanced with Multi-Gradient Aggregation Convolution (**MGAConv**) to extract fine-grained details.
- ✅ An **implicit task-adaptive degradation query** to generalize across unknown weather conditions.
- ✅ A highly **efficient design** that reduces computational cost without sacrificing performance.

## 🐍 Requirements

- Python 3.8+
- PyTorch >= 1.10
- CUDA >= 11.3
- torchvision, numpy, tqdm, scikit-image

Install dependencies:
```bash
pip install -r requirements.txt
