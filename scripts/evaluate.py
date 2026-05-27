"""
CrossFusion evaluation script.

Usage:
    # Full-modality evaluation
    python scripts/evaluate.py \
        --checkpoint checkpoints/mer2024/best_fold_0.pt \
        --config configs/config.yaml \
        --dataset mer2024 \
        --split test

    # Missing-modality robustness (MMRI curve)
    python scripts/evaluate.py \
        --checkpoint checkpoints/mer2024/best_fold_0.pt \
        --config configs/config.yaml \
        --dataset mer2024 \
        --split test \
        --mmri --modality audio
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import yaml
from sklearn.metrics import f1_score, recall_score, accuracy_score, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crossfusion.model import CrossFusion
from crossfusion.dataset import build_dataloader
from evaluation.mmri import build_mmri_curve


def load_model(checkpoint_path: str, config: dict, device: str) -> CrossFusion:
    model = CrossFusion(
        hidden_dim=config["model"]["hidden_dim"],
        num_heads=config["model"]["num_heads"],
        num_classes=config["model"]["num_classes"],
        dropout=config["model"]["dropout"],
    ).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (fold {ckpt['fold']})")
    print(f"  Val WAF at save: {ckpt['val_metrics']['waf']:.4f}")
    return model


@torch.no_grad()
def evaluate_full(model, loader, device, num_classes=3):
    all_preds, all_labels = [], []
    for batch in loader:
        logits, _, _ = model(
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            audio_values=batch["audio_values"].to(device),
            audio_attention_mask=batch["audio_attention_mask"].to(device),
            pixel_values=batch["pixel_values"].to(device),
        )
        all_preds.extend(logits.argmax(-1).cpu().numpy())
        all_labels.extend(batch["polarity"].numpy())

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    return {
        "waf": float(f1_score(labels, preds, average="weighted")),
        "uar": float(recall_score(labels, preds, average="macro")),
        "accuracy": float(accuracy_score(labels, preds)),
        "report": classification_report(
            labels, preds,
            target_names=["Negative", "Neutral", "Positive"],
            digits=4,
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--mmri", action="store_true", help="Run MMRI curve evaluation")
    parser.add_argument("--modality", type=str, default="audio", choices=["audio", "video", "both"])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.checkpoint, config, device)

    dataset_config = config["datasets"][args.dataset]
    test_manifest = dataset_config["test_manifest"]

    loader = build_dataloader(
        manifest_path=test_manifest,
        data_root=config["data_root"],
        split=args.split,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    # Full-modality evaluation
    print(f"\nEvaluating on {args.dataset.upper()} ({args.split} split) ...")
    metrics = evaluate_full(model, loader, device)
    print(f"\nFull-modality results:")
    print(f"  WAF:      {metrics['waf']:.4f}")
    print(f"  UAR:      {metrics['uar']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"\nClassification Report:\n{metrics['report']}")

    results = {"full_modality": metrics}

    # MMRI evaluation
    if args.mmri:
        print(f"\nComputing MMRI curve ({args.modality} missingness) ...")
        mmri_results = build_mmri_curve(
            model=model,
            dataloader=loader,
            modality=args.modality,
            p_miss_values=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
            n_runs=5,
            seed=42,
            device=str(device),
        )
        results["mmri"] = mmri_results

        print(f"\nMMRI @ p_miss=0.40: {mmri_results['mmri'][3]:.4f} "
              f"[{mmri_results['ci_lower'][3]:.4f}, {mmri_results['ci_upper'][3]:.4f}]")

    # Save results
    output_path = args.output or args.checkpoint.replace(".pt", f"_{args.split}_results.json")
    results_serializable = {k: v for k, v in results.items() if k != "full_modality"}
    results_serializable["full_modality"] = {
        k: v for k, v in metrics.items() if k != "report"
    }
    results_serializable["full_modality"]["report"] = metrics["report"]

    with open(output_path, "w") as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
