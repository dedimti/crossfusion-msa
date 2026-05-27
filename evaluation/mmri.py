"""
Missing-Modality Robustness Index (MMRI) evaluation.

MMRI = WAF(p_miss) / WAF(0)

where WAF(p_miss) is weighted average F1 when modality m is absent
with probability p_miss, and WAF(0) is full-modality baseline.

Confidence intervals are computed via the delta method applied to
the ratio WAF(p_miss)/WAF(0), using empirical variance estimates
from cross-validation folds.
"""

import numpy as np
import torch
from sklearn.metrics import f1_score
from typing import List, Tuple


def compute_waf(preds: np.ndarray, labels: np.ndarray) -> float:
    """Compute Weighted Average F1."""
    return f1_score(labels, preds, average="weighted")


def compute_mmri(
    model,
    dataloader,
    modality: str,           # 'audio', 'video', or 'both'
    p_miss: float,           # Missingness probability [0, 1]
    n_runs: int = 5,         # Number of independent runs for CI estimation
    seed: int = 42,
    device: str = "cuda",
) -> Tuple[float, float, float]:
    """
    Compute MMRI with 95% confidence intervals via the delta method.

    Args:
        model:      CrossFusion model (eval mode).
        dataloader: DataLoader for the evaluation split.
        modality:   Which modality to drop ('audio', 'video', 'both').
        p_miss:     Probability of dropping the modality per utterance.
        n_runs:     Number of independent stochastic runs.
        seed:       Base random seed.
        device:     Device string.

    Returns:
        (mmri, ci_lower, ci_upper) — point estimate and 95% CI bounds.
    """
    model.eval()

    # --- Full modality baseline ---
    waf_full = _evaluate_full(model, dataloader, device)

    # --- Missing modality runs ---
    waf_miss_runs = []
    for run in range(n_runs):
        torch.manual_seed(seed + run)
        np.random.seed(seed + run)
        waf_miss = _evaluate_with_missing(model, dataloader, modality, p_miss, device)
        waf_miss_runs.append(waf_miss)

    waf_miss_mean = np.mean(waf_miss_runs)
    waf_miss_std = np.std(waf_miss_runs, ddof=1)

    # MMRI point estimate
    mmri = waf_miss_mean / waf_full if waf_full > 0 else 0.0

    # Delta method CI for ratio r = mu_miss / waf_full
    # Var(r) ≈ (1/waf_full)^2 * Var(waf_miss)
    # SE(r) ≈ waf_miss_std / (sqrt(n_runs) * waf_full)
    se_mmri = waf_miss_std / (np.sqrt(n_runs) * waf_full) if waf_full > 0 else 0.0
    z_95 = 1.96
    ci_lower = mmri - z_95 * se_mmri
    ci_upper = mmri + z_95 * se_mmri

    return float(mmri), float(ci_lower), float(ci_upper)


@torch.no_grad()
def _evaluate_full(model, dataloader, device: str) -> float:
    """Evaluate with all modalities present."""
    all_preds, all_labels = [], []
    for batch in dataloader:
        logits, _, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio_values=batch["audio_values"].to(device),
            audio_attention_mask=batch["audio_attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
        )
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["polarity"].numpy())
    return compute_waf(np.array(all_preds), np.array(all_labels))


@torch.no_grad()
def _evaluate_with_missing(
    model,
    dataloader,
    modality: str,
    p_miss: float,
    device: str,
) -> float:
    """Evaluate with stochastic modality dropping."""
    all_preds, all_labels = [], []
    for batch in dataloader:
        B = batch["input_ids"].shape[0]
        drop_mask = torch.rand(B) < p_miss  # which utterances lose the modality

        audio_values = batch["audio_values"].to(device)
        pixel_values = batch["pixel_values"].to(device)

        if modality in ("audio", "both"):
            audio_values = audio_values.clone()
            audio_values[drop_mask] = 0.0

        if modality in ("video", "both"):
            pixel_values = pixel_values.clone()
            pixel_values[drop_mask] = 0.0

        logits, _, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio_values=audio_values,
            audio_attention_mask=batch["audio_attention_mask"].to(device),
            pixel_values=pixel_values,
        )
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["polarity"].numpy())
    return compute_waf(np.array(all_preds), np.array(all_labels))


def build_mmri_curve(
    model,
    dataloader,
    modality: str,
    p_miss_values: List[float] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
    n_runs: int = 5,
    seed: int = 42,
    device: str = "cuda",
) -> dict:
    """
    Build full MMRI curve across a range of missingness probabilities.

    Returns a dict with keys:
        p_miss_values, mmri, ci_lower, ci_upper, waf_full
    """
    waf_full = _evaluate_full(model, dataloader, device)
    results = {
        "p_miss_values": p_miss_values,
        "mmri": [],
        "ci_lower": [],
        "ci_upper": [],
        "waf_full": waf_full,
        "modality": modality,
    }

    for p_miss in p_miss_values:
        mmri, ci_lo, ci_hi = compute_mmri(
            model, dataloader, modality, p_miss, n_runs, seed, device
        )
        results["mmri"].append(mmri)
        results["ci_lower"].append(ci_lo)
        results["ci_upper"].append(ci_hi)
        print(f"  p_miss={p_miss:.1f} | MMRI={mmri:.3f} [{ci_lo:.3f}, {ci_hi:.3f}]")

    return results
