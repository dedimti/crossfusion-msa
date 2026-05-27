"""
CrossFusion training script with 5-fold stratified cross-validation.

Usage:
    python scripts/train.py \
        --config configs/config.yaml \
        --dataset mer2024 \
        --output_dir checkpoints/mer2024

Reproduces the main results in Table 3 of the paper.
Hardware: 1x NVIDIA A100 80GB SXM4
Expected runtime: ~14 hours per fold, ~70 hours total (5 folds)
"""

import os
import sys
import json
import argparse
import random
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, recall_score, accuracy_score
import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossfusion.model import CrossFusion
from crossfusion.loss import ConfidenceWeightedLoss
from crossfusion.dataset import MultimodalSentimentDataset, build_dataloader


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def compute_metrics(preds: np.ndarray, labels: np.ndarray) -> dict:
    return {
        "waf": f1_score(labels, preds, average="weighted"),
        "uar": recall_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def train_one_epoch(model, loader, optimizer, criterion, device, config):
    model.train()
    total_loss = 0.0
    for batch in loader:
        optimizer.zero_grad()

        logits, intensity, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio_values=batch["audio_values"].to(device),
            audio_attention_mask=batch["audio_attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
        )
        loss = criterion(
            polarity_logits=logits,
            polarity_labels=batch["polarity"].to(device),
            intensity_pred=intensity,
            intensity_labels=batch["intensity"].unsqueeze(-1).to(device),
            kappa=batch["kappa"].to(device),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
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
    return compute_metrics(np.array(all_preds), np.array(all_labels))


def run_fold(fold_idx: int, train_manifest: str, val_manifest: str, config: dict, output_dir: str):
    seed = 42 + fold_idx
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\n{'='*60}")
    print(f"Fold {fold_idx + 1}/5  |  seed={seed}")
    print(f"{'='*60}")

    # Build dataloaders
    train_loader = build_dataloader(
        manifest_path=train_manifest,
        data_root=config["data_root"],
        split="train",
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
        p_miss_audio=0.0,
        p_miss_video=config["training"]["p_miss_video"],
    )
    val_loader = build_dataloader(
        manifest_path=val_manifest,
        data_root=config["data_root"],
        split="val",
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    # Build model
    model = CrossFusion(
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)

    # Optimizer with differential learning rates
    text_params = list(model.text_encoder.parameters())
    av_params = (
        list(model.audio_encoder.parameters()) +
        list(model.video_encoder.parameters())
    )
    other_params = (
        list(model.attention.parameters()) +
        list(model.gating.parameters()) +
        list(model.polarity_head.parameters()) +
        list(model.intensity_head.parameters()) +
        list(model.ln_text.parameters()) +
        list(model.ln_audio.parameters()) +
        list(model.ln_video.parameters()) +
        list(model.ln_fused.parameters())
    )
    optimizer = AdamW([
        {"params": text_params, "lr": config["training"]["lr_text"]},
        {"params": av_params,   "lr": config["training"]["lr_av"]},
        {"params": other_params, "lr": config["training"]["lr_av"] * 2},
    ], weight_decay=1e-2)

    # Scheduler
    total_steps = config["training"]["max_epochs"] * len(train_loader)
    warmup_steps = int(0.1 * total_steps)
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps - warmup_steps, eta_min=1e-7)

    # Loss
    criterion = ConfidenceWeightedLoss(
        alpha=config["training"]["alpha"],
        kappa_max=config["training"]["kappa_max"],
    )

    # Training loop with early stopping
    best_waf = 0.0
    patience = config["training"]["patience"]
    patience_counter = 0
    best_checkpoint = os.path.join(output_dir, f"best_fold_{fold_idx}.pt")
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, config)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()

        waf = val_metrics["waf"]
        print(f"Epoch {epoch:3d} | loss={train_loss:.4f} | "
              f"WAF={waf:.4f} | UAR={val_metrics['uar']:.4f} | "
              f"Acc={val_metrics['accuracy']:.4f}")

        if waf > best_waf:
            best_waf = waf
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
                "fold": fold_idx,
                "seed": seed,
            }, best_checkpoint)
            print(f"  ✓ New best WAF={best_waf:.4f} — checkpoint saved")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\nFold {fold_idx + 1} best WAF: {best_waf:.4f}")
    return best_waf


def main():
    parser = argparse.ArgumentParser(description="Train CrossFusion with 5-fold CV")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True, choices=["mer2023", "mer2024", "cmu_mosei"])
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    dataset_config = config["datasets"][args.dataset]
    fold_wafs = []

    for fold_idx in range(5):
        train_manifest = dataset_config["fold_manifests"][fold_idx]["train"]
        val_manifest = dataset_config["fold_manifests"][fold_idx]["val"]
        best_waf = run_fold(fold_idx, train_manifest, val_manifest, config, args.output_dir)
        fold_wafs.append(best_waf)

    mean_waf = np.mean(fold_wafs)
    std_waf = np.std(fold_wafs, ddof=1)

    print(f"\n{'='*60}")
    print(f"5-fold CV Results on {args.dataset.upper()}")
    print(f"WAF: {mean_waf:.4f} ± {std_waf:.4f}")
    for i, w in enumerate(fold_wafs):
        print(f"  Fold {i+1}: {w:.4f}")
    print(f"{'='*60}")

    results = {
        "dataset": args.dataset,
        "mean_waf": float(mean_waf),
        "std_waf": float(std_waf),
        "fold_wafs": [float(w) for w in fold_wafs],
    }
    with open(os.path.join(args.output_dir, "cv_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
