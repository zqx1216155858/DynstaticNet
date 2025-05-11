# DynStaticNet: Dynamic-Static Dual-Branch Network for Multi-Weather Video Restoration

![License](https://img.shields.io/badge/license-MIT-green)

> 🔬 Official PyTorch implementation of **"DynStaticNet: Dynamic-Static Dual-Branch Network for Multi-Weather Video Restoration"**  
> [[Paper (Coming Soon)]](#) | [[Project Page]](#) | [[Video Demo]](#)

---

## 🌟 Overview

**DynStaticNet** is a biologically-inspired dual-branch framework for video restoration under complex and mixed weather conditions, such as rain, snow, and haze. It decouples spatial and temporal modeling to achieve efficient and accurate multi-weather restoration.

Key contributions include:

- 🧠 **Temporal Branch** with 3D Adaptive Self-Attention (3DASA) for modeling long-range motion.
- 👁️ **Spatial Branch** with Multi-Gradient Aggregation Convolution (MGAConv) to enhance spatial detail extraction using directional gradients.
- 🧩 **Task-Adaptive Degradation Query** for generalization to unknown or mixed degradations without task labels.
- ⚡ **Efficient Design** reducing >70% FLOPs vs traditional attention-based models.

---

## 📁 Project Structure

- `configs/` — YAML configs for training/testing  
- `data/` — Dataset loader and preprocessing scripts  
- `models/` — Model architecture: DynStaticNet, 3DASA, MGAConv  
- `scripts/` — Shell scripts for automation  
- `utils/` — Helper functions and tools  
- `pretrained/` — Pretrained checkpoints (download separately)  
- `main_train.py` — Training entry point  
- `main_eval.py` — Evaluation entry point  
- `README.md` — This file

---

## 🐍 Requirements

- Python 3.8+
- PyTorch ≥ 1.10
- CUDA ≥ 11.3
- torchvision
- numpy
- scikit-image
- tqdm
- pyyaml


---

## 📦 Dataset Preparation

We reorganize three publicly available datasets into a unified **Multi-Weather Video Dataset (MWVD)**:

- 🌧️ [RainSynAll100](https://github.com/lsy17096535/PRN)
- 🌫️ [HazeWorld](https://github.com/volcanoscout/HazeWorld)
- ❄️ [RVSD (Snow)](https://github.com/chenyanglei/SnowFormer)

All videos are split into **10-frame clips**, and we ensure **no overlap** between training and test sets.

Expected directory structure:

- `data/MWVD/train/input/` — 1350 clips(haze:450, rain:450, snow:450)
- `data/MWVD/train/gt/` — 1350 clips(haze:450, rain:450, snow:450)
- `data/MWVD/test/input/` — 150 clips(haze:50, rain:50, snow:50)
- `data/MWVD/test/gt/` — 150 clips(haze:50, rain:50, snow:50)  

To process and organize data:

```bash
bash scripts/prepare_dataset.sh
```

---

## 🚀 Training

To train DynStaticNet:

```bash
python main_train.py --config configs/train_dynstatic.yaml
```

To resume training from checkpoint:

```bash
python main_train.py --resume --checkpoint pretrained/latest.pth
```

---

## 📈 Evaluation

To evaluate a pretrained model:

```bash
python main_eval.py --config configs/eval.yaml --checkpoint pretrained/dynstaticnet.pth
```

---

## 🧠 Model Details

- **Embedding dimension**: 24  
- **3DASA attention heads**: [1, 2, 4]  
- **Blocks per stage**: [4, 6, 8]; refinement stage contains 4 blocks  
- **Batch size**: 1 
- **Each input**: 10 consecutive frames  
- **Input shape**: `[1, 10, 3, H, W]`  
- **Patch size**: 128 × 128  
- **Augmentation**: Random rotation, horizontal flipping  
- **Optimizer**: Adam  
- **Learning rate**: Initial `2e-4`, decays to `1e-7` using cosine annealing

---

## 📊 Pretrained Models

| Model        | Dataset | PSNR (dB) |  SSIM  | LPIPS | FLOPs (G) | Download |
|--------------|---------|-----------|--------|-------|-----------|----------|
| DynStaticNet | MWVD    | 28.19     | 0.9166 |0.0582 | 48.7      | [Coming Soon](#) |
---


## 💻 Code Availability

We have released all source code, pretrained models, and dataset generation tools to support reproducibility.

👉 **GitHub Repository**:  
[https://github.com/zqx1216155858/DynstaticNet](https://github.com/zqx1216155858/DynstaticNet)

---

## 📜 Citation

If you find our work helpful, please consider citing:

```bibtex
@article{your2024dynstaticnet,
  title={DynStaticNet: Dynamic-Static Dual-Branch Network for Multi-Weather Video Restoration},
  author={Tao Gao, Qianxi Zhang, Ting Chen, Yuanbo Wen, Ziqi Li and Tao Lei},
  journal={TBD},
  year={2025}
}
```

---

## 📬 Contact

- 📧 Email: zqx1216@chd.due.cn 
- 🐛 Issues: Please open an issue in this repository

© 2024 DynStaticNet Authors. Released under the [MIT License](LICENSE).
