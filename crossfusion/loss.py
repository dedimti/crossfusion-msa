"""
Confidence-weighted training objective for CrossFusion.

Loss = alpha * CrossEntropy(polarity) + (1 - alpha) * MSE(intensity)
Each sample is weighted by w_i = kappa_i / kappa_max,
where kappa_i is the per-utterance inter-annotator agreement (Fleiss kappa).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConfidenceWeightedLoss(nn.Module):
    """
    Combined polarity classification + intensity regression loss,
    weighted by per-sample annotator agreement.

    Args:
        alpha:       Weight for classification loss (default 0.7, tuned via grid search).
        kappa_max:   Maximum kappa in training split (used for normalization).
        label_smoothing: Label smoothing for CrossEntropy (default 0.1).
    """

    def __init__(self, alpha: float = 0.7, kappa_max: float = 1.0, label_smoothing: float = 0.1):
        super().__init__()
        self.alpha = alpha
        self.kappa_max = kappa_max
        self.ce_loss = nn.CrossEntropyLoss(
            label_smoothing=label_smoothing,
            reduction="none",
        )
        self.mse_loss = nn.MSELoss(reduction="none")

    def forward(
        self,
        polarity_logits: torch.Tensor,  # (B, num_classes)
        polarity_labels: torch.Tensor,  # (B,) long
        intensity_pred: torch.Tensor,   # (B, 1)
        intensity_labels: torch.Tensor, # (B, 1) float in [0, 1]
        kappa: torch.Tensor,            # (B,) per-sample IAA
    ) -> torch.Tensor:
        """Returns scalar weighted loss."""
        # Sample weights: w_i = kappa_i / kappa_max
        weights = (kappa / self.kappa_max).clamp(min=1e-6)  # (B,)

        # Classification loss (per sample)
        ce = self.ce_loss(polarity_logits, polarity_labels)  # (B,)

        # Intensity regression loss (per sample)
        mse = self.mse_loss(
            intensity_pred.squeeze(-1),
            intensity_labels.squeeze(-1),
        )  # (B,)

        # Combined per-sample loss
        per_sample = self.alpha * ce + (1 - self.alpha) * mse  # (B,)

        # Confidence-weighted mean
        loss = (weights * per_sample).sum() / weights.sum()

        return loss


class SymmetryLoss(nn.Module):
    """
    Computes the 'symmetry loss' — the WAF delta between CrossFusion and
    the symmetric-full baseline — for monitoring during training.

    This is a diagnostic metric, not a training objective.
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def compute_waf(logits: torch.Tensor, labels: torch.Tensor, num_classes: int = 3) -> float:
        """Compute Weighted Average F1 (WAF) from logits and labels."""
        preds = logits.argmax(dim=-1).cpu().numpy()
        labels_np = labels.cpu().numpy()

        from sklearn.metrics import f1_score
        return f1_score(labels_np, preds, average="weighted")
