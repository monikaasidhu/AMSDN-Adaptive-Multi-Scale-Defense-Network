# AMSDN: Adaptive Multi-Scale Defense Network

AMSDN is a research-oriented PyTorch project for defending image classifiers against patch-based, sparse-pixel, and gradient-based adversarial attacks. The repository combines multi-scale representation learning, attention, selective purification, anomaly detection, optional consistency verification, self-supervised robustness pretraining, and randomized smoothing in a single framework.

## Overview

The project is built around a unified defense pipeline:

1. Multi-scale feature extraction with `ConvNeXt-Tiny + FPN`
2. Adaptive attention with spatial, channel, and pyramid attention
3. Selective feature purification guided by anomaly scores
4. Adversarial detection from aggregated multi-scale anomaly signals
5. Optional prediction consistency verification
6. Self-supervised robustness training (SSRT) for warm-starting


The current implementation is centered on `CIFAR-10` and is designed for experimentation, benchmarking, and research prototyping rather than production deployment.

## Architecture

```text
Input image (3 x 32 x 32)
    |
    v
+--------------------------------------+
| Stage 1: ConvNeXt-Tiny + FPN         |
| Multi-scale features: P2, P3, P4, P5 |
| 256 channels per level               |
+--------------------------------------+
    |
    v
+--------------------------------------+
| Stage 2: Adaptive Attention          |
| - Spatial attention                  |
| - Channel attention                  |
| - Multi-scale pyramid attention      |
+--------------------------------------+
    |
    v
+--------------------------------------+
| Stage 3: Selective Purification      |
| - Anomaly detection                  |
| - Feature denoising                  |
| - Selective fusion                   |
+--------------------------------------+
    |
    v
+--------------------------------------+
| Classification + Detection Head      |
| - Class logits                       |
| - Anomaly score                      |
| - Adversarial decision               |
+--------------------------------------+
```

## Key Features

- `ConvNeXt-FPN backbone` for multi-scale feature extraction
- `Adaptive attention modules` at each pyramid level
- `Selective purification` in feature space rather than pixel space
- `Built-in adversarial detection` via anomaly scoring
- `SSRT pretraining` through reconstruction and contrastive robustness objectives
- `Multiple attack families` for training and evaluation
- `TensorBoard logging` during training

## Repository Structure

```text
AMSDN/
|-- attacks/
|   |-- patch_attack.py
|   |-- patch_attacks.py
|   |-- pixel_attacks.py
|-- data/
|   |-- cifar10.py
|-- evaluation/
|   |-- evaluate.py
|-- models/
|   |-- attention/
|   |   |-- adaptive_attention.py
|   |-- backbone/
|   |   |-- convnext_fpn.py
|   |-- purification/
|   |   |-- selective_purifier.py
|   |-- amsdn.py
|-- training/
|   |-- adversarial_train.py
|   |-- fast_robust_tune.py
|   |-- finetune_attacks.py
|   |-- pretrain_ssrt.py
|-- utils/
|   |-- helpers.py
|-- AMSDN_Colab.ipynb
|-- requirements.txt
|-- README.md
```

## Requirements

- Python `3.8+`
- PyTorch `2.0+`
- CUDA-capable GPU recommended for training

Install dependencies:

```bash
pip install -r requirements.txt
```

Core dependencies:

- `torch`
- `torchvision`
- `timm`
- `numpy`
- `matplotlib`
- `scikit-learn`
- `scipy`
- `tensorboard`
- `tqdm`
- `Pillow`

## Quick Start

### 1. Optional SSRT Pretraining

This stage produces an initialization that can be used as a warm-start for adversarial training.

```bash
python training/pretrain_ssrt.py
```

Checkpoint output:

```text
./checkpoints/ssrt/ssrt_best.pth
```

### 2. Adversarial Training

This is the main supervised defense training stage. It mixes clean samples with PGD, random patch, and sparse corruption examples while jointly learning classification and detection.

```bash
python training/adversarial_train.py
```

Checkpoint outputs:

```text
./checkpoints/adversarial/amsdn_best.pth
./checkpoints/adversarial/amsdn_last.pth
```

### 3. Robust Fine-Tuning Options

#### Fast robust tuning

Recommended when compute is limited or when you want a practical follow-up stage after adversarial training.

```bash
python training/fast_robust_tune.py
```

Low-resource preset:

```bash
python training/fast_robust_tune.py --low-resource
```

Useful overrides:

```bash
python training/fast_robust_tune.py --epochs 6 --batch-size 64 --eval-pgd-steps 5
```

Checkpoint outputs:

```text
./checkpoints/fast_robust/amsdn_fast_robust_best.pth
./checkpoints/fast_robust_low_resource/amsdn_fast_robust_best.pth
```

#### Full multi-attack fine-tuning

This is more expensive and uses stronger attack diversity, including C&W, patch, and few-pixel attacks.

```bash
python training/finetune_attacks.py
```

Checkpoint output:

```text
./checkpoints/finetuned/amsdn_finetuned_best.pth
```

### 4. Evaluation

Runs clean evaluation, calibrates a detection threshold, and reports performance under multiple attacks including adaptive BPDA-style patch attacks.

```bash
python evaluation/evaluate.py
```

Results output:

```text
./results/evaluation_results.json
```

## Recommended Training Pipelines

### Full pipeline

```bash
python training/pretrain_ssrt.py
python training/adversarial_train.py
python training/fast_robust_tune.py
python evaluation/evaluate.py
```

### Practical low-resource pipeline

```bash
python training/adversarial_train.py
python training/fast_robust_tune.py --low-resource
python evaluation/evaluate.py
```

### Expensive research pipeline

```bash
python training/pretrain_ssrt.py
python training/adversarial_train.py
python training/finetune_attacks.py
python evaluation/evaluate.py
```

## Implemented Attacks

The repository includes or evaluates the following attack families:

- `PGD` with common epsilon settings such as `8/255` and `16/255`
- `C&W` L2 attack
- `Adversarial Patch` attacks with localized perturbations
- `Few-Pixel / Sparse Pixel` attacks
- `Adaptive BPDA Patch` attack for stronger defense-aware evaluation
- `Random patch` and `random sparse` corruptions used as lightweight training proxies
- `FGSM-style` fast robustness tuning perturbations

## Evaluation Outputs

The main evaluation script reports:

- Clean accuracy
- Robust accuracy per attack
- Attack success rate
- Detection true positive rate
- False positive rate on clean data
- Inference throughput
- Detection threshold selected from clean anomaly calibration


## Limitations

1. Robustness improvements can reduce clean accuracy.
2. The current implementation is focused on `CIFAR-10`.
3. Sophisticated adaptive attacks may still weaken the defense.
4. Performance is sensitive to training configuration and checkpoint choice.

## Future Work

- ImageNet-scale adaptation
- multi-GPU training support
- stronger adaptive evaluations
- broader attack coverage such as FGSM and DeepFool
- compression or deployment-oriented variants
- more extensive physical-world adversarial testing


## Acknowledgments

- [PyTorch](https://pytorch.org/)
- [timm](https://github.com/rwightman/pytorch-image-models) for ConvNeXt implementations
- prior work on adversarial robustness, adversarial patches, sparse attacks, FPNs.
