# FA-MoE: Improving Medical Image Generation through Frequency-Aware Mixture of Experts

**CVPR 2026 Findings**

> Official PyTorch implementation of FA-MoE, a frequency-aware diffusion model for medical image generation using Expert-Choice Mixture of Experts routing in the DCT domain.

---

## Overview

FA-MoE improves medical image generation by operating in the frequency domain (DCT) and routing tokens through specialized experts based on their frequency characteristics. Key contributions:

- **Frequency-Aware (FA) Tokenization**: Block-wise DCT coefficients are organized by frequency band, with entropy-based loss reweighting to emphasize informative frequencies.
- **Expert-Choice MoE (EC-MoE)**: Each expert independently selects the top-C tokens it processes, avoiding load imbalance issues in standard token-choice routing.
- **DCT-domain Diffusion**: The diffusion process operates on DCT coefficients rather than pixel values, enabling fine-grained frequency control.

Evaluated on two cardiac imaging benchmarks: **ACDC** (cardiac MRI) and **EchoNet-Dynamic** (cardiac ultrasound).

---

## Requirements

Install dependencies:

```bash
pip install torch torchvision accelerate ml_collections einops numpy scipy opencv-python pillow wandb absl-py ptflops lpips
```

Optional (for memory-efficient attention):
```bash
pip install xformers
```

---

## Data Preparation

### ACDC (Cardiac MRI)

Download the [ACDC dataset](https://www.creatis.insa-lyon.fr/Challenge/acdc/) and organize as expected by `datasets.py`. Extract frequency features using:

```bash
python scripts/extract_empty_feature.py
```

### EchoNet-Dynamic (Cardiac Ultrasound)

Download [EchoNet-Dynamic](https://echonet.github.io/dynamic/) and configure the dataset path in your config file.

---

## Training

### Baseline (no MoE)

```bash
accelerate launch train_greyscale.py --config configs/acdc_wholeheart_uncond_uvit_greyscale_mid_4by4.py
```

### FA-MoE (Frequency-Aware Expert-Choice MoE)

```bash
accelerate launch train_greyscale_FA_MoE.py --config configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r16.py
```

Key config parameters:

| Parameter | Description |
|---|---|
| `num_fa_length` | Number of frequency-aware coefficient lengths (default: 8) |
| `num_fa_repeats` | Repeats per frequency length (default: 16) |
| `low_freqs` | Number of low-frequency DCT coefficients (default: 16) |
| `block_sz` | DCT block size (default: 4) |
| `depth` | Transformer depth (default: 16) |
| `embed_dim` | Embedding dimension (default: 768) |

Multiple expert configurations are provided under `configs/` (e.g., `l6r16` = 6 MoE layers, rank 16).

---

## Evaluation

Evaluate a trained checkpoint:

```bash
python eval.py --config configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r16.py
```

For frequency-aware evaluation metrics:

```bash
python eval_FA.py --config configs/acdc_wholeheart_uncond_uvit_FA_EC_MoE_greyscale_mid_4by4_l8r16.py
```

Metrics reported: FID, Inception Score (IS), LPIPS, CMMD.

---

## Model Architecture

FA-MoE builds on a **U-shaped Vision Transformer (UViT)** backbone operating on DCT tokens:

```
Input Image
    │
    ▼
Block-wise 4×4 DCT → Zigzag ordering → Frequency-Aware Tokens
    │
    ▼
Patch Embedding (DCT coefficients → embed_dim)
    │
    ▼
UViT Transformer (In-blocks → Mid-block → Out-blocks with skip connections)
    │         │
    │    [MoE Layers: Expert-Choice Routing]
    │         └── Each expert selects top-C tokens
    │             Routing: Linear(dim → num_experts)
    │             Experts: FFN with GELU activation
    ▼
Noise Prediction Head
    │
    ▼
DPM-Solver Sampling → Generated Image
```

---

## Configuration Variants

| Config | Description |
|---|---|
| `*_greyscale_mid_4by4.py` | Baseline UViT, no MoE |
| `*_FA_MoE_*_l8r16.py` | FA-MoE, 8 MoE layers, rank 16 |
| `*_FA_EC_MoE_*_l6r16.py` | Expert-Choice MoE, 6 layers |
| `*_FA_EC_MoE_*_l8r16.py` | Expert-Choice MoE, 8 layers |
| `configs_shift/` | Ablation variants |

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{famoe2026cvpr,
  title     = {FA-MoE: Improving Medical Image Generation through Frequency-Aware Mixture of Experts},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
  year      = {2026},
}
```

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
