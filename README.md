# CrossFusion: Annotation-Driven Asymmetric Cross-Modal Attention

[![Paper](https://img.shields.io/badge/Paper-Information%20Fusion-blue)](https://www.sciencedirect.com/journal/information-fusion)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-orange)](https://pytorch.org)

Official implementation of **"Annotation-Driven Asymmetric Cross-Modal Attention: Text-Anchored Fusion for Robust Multimodal Sentiment Analysis"**

> Dedi Irawan, Sudarmaji, Arif Hidayat, Guna Yanti Kemala Sari Siregar  
> Faculty of Computer Science, Universitas Muhammadiyah Metro  
> *Submitted to Information Fusion, Elsevier*

---

## Overview

CrossFusion encodes the empirical dominance of text in sentiment annotation directly into the cross-modal attention mechanism. Rather than assigning equal query (Q), key (K), and value (V) roles to all modalities — the symmetric assumption universally adopted in prior work — CrossFusion uses **text as the sole query anchor** while audio and video supply key-value representations.

This design is motivated by a concrete annotation statistic: 73% of MER 2024 sentiment labels are predictable from text alone at Fleiss κ ≥ 0.70, versus 55% for audio-only and 49% for video-only. The architecture encodes this asymmetry by construction rather than asking the optimizer to discover it.

**Key results:**
- WAF = 0.863 on MER 2024 (+2.3 points over best symmetric baseline, p = 0.012)
- WAF = 0.880 on MER 2023 (+5.0 points over best symmetric baseline)
- MMRI = 0.94 under 40% audio missingness (vs. 0.71 for symmetric baseline)
- The WAF gain scales with the text-audio inter-annotator agreement gap (Δκ), not dataset size

---

## Architecture

```
Text (BERT-base) ──────────────────────────► Q
                                              │
Audio (wav2vec 2.0) ──┐                      ▼
                       ├──► [K, V] ──► Asymmetric Cross-Modal Attention ──► z_attn
Video (ViT-base/16) ──┘                      │
                                              ▼
                                         Gating (g_T, g_A, g_V)
                                              │
                                              ▼
                                         z_fused ──► Prediction Head
```

The gating mechanism `g = σ(W_g · [h_T; h_A; h_V] + b_g)` adapts per-sample modality contributions at inference time **without requiring explicit missingness flags**.

---

## Installation

```bash
git clone https://github.com/crossfusion-msa/crossfusion-msa.git
cd crossfusion-msa
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, CUDA 11.8, GPU with ≥80GB VRAM (A100 SXM4 recommended for full fine-tuning). For reduced VRAM, see [LoRA variant](#lora-variant).

---

## Datasets

CrossFusion is evaluated on three datasets:

| Dataset | Utterances | Languages | κ_text | κ_audio | Δκ |
|---------|-----------|-----------|--------|---------|-----|
| MER 2023 | 4,642 train | 8 (Mandarin-dominant) | 0.74 | 0.61 | 0.13 |
| MER 2024 | 9,241 train | 8 (Mandarin-dominant) | 0.74 | 0.63 | 0.11 |
| CMU-MOSEI | 16,326 train | English | 0.79 | 0.74 | 0.05 |

**Download MER 2023/2024:** https://doi.org/10.1145/3689092.3689959  
**Download CMU-MOSEI:** https://www.dropbox.com/sh/hyzpgx1hp9nj37s/AAB7FhBqJOFDw2hEX2R5uJfa

After downloading, update dataset paths in `configs/config.yaml`.

---

## Quick Start

### Training

```bash
# MER 2024 (full fine-tuning, 5-fold CV)
python scripts/train.py \
    --config configs/config.yaml \
    --dataset mer2024 \
    --output_dir checkpoints/mer2024

# MER 2023
python scripts/train.py \
    --config configs/config.yaml \
    --dataset mer2023 \
    --output_dir checkpoints/mer2023

# CMU-MOSEI
python scripts/train.py \
    --config configs/config.yaml \
    --dataset cmu_mosei \
    --output_dir checkpoints/cmu_mosei
```

### Evaluation

```bash
# Full-modality evaluation
python scripts/evaluate.py \
    --checkpoint checkpoints/mer2024/best_fold_0.pt \
    --dataset mer2024 \
    --split test

# Missing-modality robustness (MMRI)
python evaluation/mmri.py \
    --checkpoint checkpoints/mer2024/best_fold_0.pt \
    --dataset mer2024 \
    --modality audio \
    --p_miss 0.2 0.4 0.6 0.8
```

### Reproducing Main Results (Table 3)

```bash
python scripts/reproduce_table3.py --config configs/config.yaml
```

This runs 5-fold cross-validation on all three datasets and reports WAF, UAR, and MMRI with 95% CIs. Expected runtime: ~70 hours on 1× A100 80GB.

---

## Missing-Modality Robustness Index (MMRI)

CrossFusion introduces MMRI as a normalized, cross-model-comparable robustness metric:

```
MMRI = WAF(p_miss) / WAF(0)
```

where `WAF(p_miss)` is performance at missingness rate `p_miss` and `WAF(0)` is full-modality performance. MMRI ∈ (0, 1], where 1.0 = no degradation.

```python
from evaluation.mmri import compute_mmri

mmri, ci_lower, ci_upper = compute_mmri(
    model=model,
    dataloader=test_loader,
    modality='audio',
    p_miss=0.40,
    n_folds=5,
    seed=42
)
print(f"MMRI = {mmri:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]")
```

CIs are computed via the delta method applied to the ratio WAF(p_miss)/WAF(0).

---

## High-Agreement Subset Split

The high-agreement subset (κ ≥ 0.70, 61% of MER 2024) used in Table 8 is provided at:

```
data/splits/mer2024_high_agreement_split.json
```

Format:
```json
{
  "high_agreement": ["utt_id_1", "utt_id_2", ...],
  "low_agreement": ["utt_id_3", ...],
  "threshold": 0.70,
  "n_high": 5637,
  "n_low": 3604
}
```

---

## LoRA Variant

For researchers without A100 80GB access, a LoRA-CrossFusion variant is available that freezes encoder weights and fine-tunes only low-rank adapter matrices (~4M trainable parameters), reducing VRAM requirements to ~32GB (V100 or A100 40GB):

```bash
python scripts/train.py \
    --config configs/config_lora.yaml \
    --dataset mer2024 \
    --output_dir checkpoints/mer2024_lora
```

Note: LoRA variant results are not reported in the paper. Performance comparison with full fine-tuning is ongoing.

---

## Reproducibility

All experiments use fixed seeds `{42, 43, 44, 45, 46}` for the five cross-validation folds, with deterministic mode enabled:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
```

Full environment:
- PyTorch 2.1.0 + CUDA 11.8
- HuggingFace Transformers 4.36.2
- Python 3.9.18

See `requirements.txt` for complete dependency list.

---

## Citation

```bibtex
@article{irawan2025crossfusion,
  title={Annotation-Driven Asymmetric Cross-Modal Attention: Text-Anchored Fusion for Robust Multimodal Sentiment Analysis},
  author={Irawan, Dedi and Sudarmaji and Hidayat, Arif and Siregar, Guna Yanti Kemala Sari},
  journal={Information Fusion},
  year={2025},
  publisher={Elsevier},
  note={Under review}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Contact

Dedi Irawan — dedimti@ummetro.ac.id  
ORCID: [0009-0007-4973-926X](https://orcid.org/0009-0007-4973-926X)  
Faculty of Computer Science, Universitas Muhammadiyah Metro, Indonesia
