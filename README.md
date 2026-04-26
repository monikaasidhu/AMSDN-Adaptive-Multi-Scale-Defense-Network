# AMSDN: Adaptive Multi-Scale Defense Network

A research-grade PyTorch implementation of a unified framework for defending against patch and sparse adversarial attacks.

## 🎯 Overview

AMSDN combines multiple defense mechanisms in a single, end-to-end trainable architecture:

1. **Multi-Scale Feature Extraction** (ConvNeXt-Tiny + FPN)
2. **Adaptive Attention** (Spatial + Channel + Multi-Scale Pyramid)
3. **Selective Purification** (Feature-space denoising)
4. **Prediction Consistency Verification**
5. **Self-Supervised Robustness Training** (SSRT)
6. **Randomized Smoothing Certification**

## 📊 Architecture

```
Input Image (3×32×32)
    ↓
┌─────────────────────────────────────┐
│ Stage 1: ConvNeXt-Tiny + FPN        │
│ Output: Multi-scale features        │
│ [P2, P3, P4, P5] @ 256 channels     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 2: Adaptive Attention         │
│ • Spatial Attention                 │
│ • Channel Attention                 │
│ • Multi-Scale Pyramid Attention     │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 3: Selective Purification     │
│ • Anomaly Detection                 │
│ • Feature Denoising                 │
│ • Selective Fusion                  │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Stage 4: Prediction Consistency     │
│ (Optional, expensive)               │
└─────────────────────────────────────┘
    ↓
┌─────────────────────────────────────┐
│ Classification Head                 │
│ Output: Logits (10 classes)         │
│         Anomaly Score               │
│         Detection Decision          │
└─────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
# CUDA-capable GPU (recommended)

pip install -r requirements.txt
```

### Training Pipeline

```bash
# 1. Self-Supervised Pretraining (Optional, ~1 hour)
python training/pretrain_ssrt.py

# 2. Adversarial Training (~2 hours)
python training/adversarial_train.py

# 3A. Fast Robust Tuning (recommended on limited hardware, ~15-25 minutes)
python training/fast_robust_tune.py

# 3A-low. Demo / low-resource mode
python training/fast_robust_tune.py --low-resource

# 3B. Multi-Attack Fine-tuning (~30 minutes+, expensive)
python training/finetune_attacks.py

# 4. Evaluation (~20 minutes)
python evaluation/evaluate.py

# 5. Certification (~30 minutes, use small sample size)
python evaluation/certification.py
```

### Google Colab

Open `notebooks/AMSDN_Colab.ipynb` in Google Colab for a complete interactive tutorial.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_USERNAME/AMSDN/blob/main/notebooks/AMSDN_Colab.ipynb)

## 📁 Repository Structure

```
AMSDN/
│
├── data/
│   └── cifar10.py                 # CIFAR-10 data loading
│
├── models/
│   ├── backbone/
│   │   └── convnext_fpn.py        # ConvNeXt + FPN
│   ├── attention/
│   │   └── adaptive_attention.py  # Multi-scale attention
│   ├── purification/
│   │   └── selective_purifier.py  # Adversarial purification
│   └── amsdn.py                   # Main AMSDN model
│
├── training/
│   ├── pretrain_ssrt.py           # Self-supervised pretraining
│   ├── adversarial_train.py       # Adversarial training
│   ├── fast_robust_tune.py        # Lightweight robust-tuning alternative
│   └── finetune_attacks.py        # Multi-attack fine-tuning
│
├── attacks/
│   ├── patch_attacks.py           # Patch attacks (AdvPatch, BPDA)
│   └── pixel_attacks.py           # Sparse pixel attacks
│
├── evaluation/
│   ├── evaluate.py                # Comprehensive evaluation
│   └── certification.py           # Randomized smoothing
│
├── utils/
│   └── helpers.py                 # Visualization & utilities
│
├── notebooks/
│   └── AMSDN_Colab.ipynb          # Interactive Colab notebook
│
├── requirements.txt
└── README.md
```

## 🔬 Implemented Attacks

- **PGD** (Projected Gradient Descent): ε=8/255, 16/255
- **C&W** (Carlini-Wagner): L2 attack
- **Patch Attacks**: Localized perturbations (4×4, 8×8 pixels)
- **Pixel Attacks**: Sparse perturbations (5, 10 pixels)
- **Adaptive BPDA**: Gradient obfuscation circumvention

## 📈 Expected Results (CIFAR-10)

After full training (~3-4 hours on T4 GPU):

| Metric | Value |
|--------|-------|
| Clean Accuracy | ~85-90% |
| PGD-8 Robust Accuracy | ~60-70% |
| Patch-4 Robust Accuracy | ~65-75% |
| Pixel-5 Robust Accuracy | ~70-80% |
| Detection Rate | ~75-85% |
| Certified Accuracy (r=0.25) | ~50-60% |

*Note: Results depend on training hyperparameters and random initialization.*

## 🛠️ Customization

### Change Dataset

```python
# In data/cifar10.py, replace CIFAR10 with your dataset
# Modify image size in models accordingly
```

### Modify Architecture

```python
# models/amsdn.py
model = AMSDN(
    num_classes=100,              # Change number of classes
    pretrained=True,              # Use pretrained backbone
    purification_threshold=0.5,   # Detection threshold
    consistency_samples=5         # Verification samples
)
```

### Attack Strength

```python
# training/adversarial_train.py
attack = PGDAttack(
    epsilon=16/255,    # Increase perturbation budget
    alpha=2/255,
    num_steps=20
)
```

## 📊 Monitoring Training

```bash
# Launch TensorBoard
tensorboard --logdir=checkpoints/
```

## 🧪 Testing Individual Components

```python
# Test backbone
python models/backbone/convnext_fpn.py

# Test attention
python models/attention/adaptive_attention.py

# Test purification
python models/purification/selective_purifier.py

# Test full model
python models/amsdn.py
```

## ⚙️ Configuration

### Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | T4 (16GB) | A100 (40GB) |
| RAM | 12GB | 32GB |
| Storage | 5GB | 10GB |

### Training Time

| Stage | Time (T4) | Time (A100) |
|-------|-----------|-------------|
| SSRT Pretraining | ~1 hour | ~20 minutes |
| Adversarial Training | ~2 hours | ~40 minutes |
| Fine-tuning | ~30 minutes | ~10 minutes |
| Evaluation | ~20 minutes | ~5 minutes |
| Certification | ~30 minutes | ~10 minutes |

## 🐛 Troubleshooting

### Out of Memory (OOM)

```python
# Reduce batch size in training scripts
batch_size = 64  # or 32
```

### Slow Training

```python
# Reduce number of epochs for demo
num_epochs = 10  # instead of 100
```

### Fine-tuning Too Slow

If `training/finetune_attacks.py` is too slow to complete, use:

```bash
python training/fast_robust_tune.py
```

For very limited GPU or Colab sessions, use:

```bash
python training/fast_robust_tune.py --low-resource
```

This keeps the project goal aligned because it still:

- starts from the adversarially trained AMSDN checkpoint
- trains against gradient, localized, and sparse perturbation patterns
- keeps full attack evaluation separate, so robustness is still measured properly
- reduces epochs, batch size, PGD evaluation steps, and keeps the backbone frozen longer

Current recommended pipeline:

```bash
python training/adversarial_train.py
python training/fast_robust_tune.py --low-resource
python evaluation/evaluate.py
```

### ImportError

```bash
# Ensure you're in the AMSDN root directory
cd AMSDN/
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

## 📝 Citation

If you use this code for research, please cite:

```bibtex
@misc{amsdn2024,
  title={AMSDN: Adaptive Multi-Scale Defense Network},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/YOUR_USERNAME/AMSDN}}
}
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Open a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- ConvNeXt architecture from [timm](https://github.com/rwightman/pytorch-image-models)
- Inspired by adversarial defense research from Cohen et al., Brown et al., and others
- Built with PyTorch

## 📧 Contact

For questions or issues, please open a GitHub issue or contact: your.email@example.com

---

**Note:** This is a research implementation. For production use, additional testing and optimization are required.

## 🎓 Related Work

- **Randomized Smoothing:** Cohen et al., "Certified Adversarial Robustness via Randomized Smoothing" (ICML 2019)
- **Adversarial Patch:** Brown et al., "Adversarial Patch" (NIPS 2017 Workshop)
- **PGD Attack:** Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks" (ICLR 2018)
- **FPN:** Lin et al., "Feature Pyramid Networks for Object Detection" (CVPR 2017)
- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s" (CVPR 2022)

## ⚠️ Limitations

1. **Computational Cost:** Randomized smoothing certification is expensive (~10-100x slower than inference)
2. **Trade-offs:** Improved robustness may reduce clean accuracy
3. **Dataset:** Currently tested only on CIFAR-10; ImageNet support requires modifications
4. **Adaptive Attacks:** Defense may be vulnerable to more sophisticated adaptive attacks
5. **Hyperparameter Sensitivity:** Performance depends on careful tuning

## 🔮 Future Work

- [ ] ImageNet support
- [ ] Multi-GPU training
- [ ] More attack types (FGSM, DeepFool, etc.)
- [ ] Adversarial training with stronger attacks
- [ ] Model compression for deployment
- [ ] Uncertainty quantification
- [ ] Real-world evaluation on physical adversarial examples

---

**Status:** ✅ Fully implemented and tested on CIFAR-10

**Last Updated:** December 2024
