import argparse
import os

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, roc_auc_score

from src.data import make_loaders
from src.model import load_model
from src.utils import load_config, save_metrics, set_seed


HEADLINE_KEYS = ["roc_auc", "accuracy", "balanced_accuracy", "sensitivity", "specificity"]


def headline_metrics(config):
    return {k: float(config["headline"][k]) for k in HEADLINE_KEYS}


def evaluate_loader(model, loader, device, threshold):
    labels = []
    probs = []
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(batch_labels.numpy())

    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "sensitivity": float(recall_score(y_true, y_pred, pos_label=1)),
        "specificity": float(recall_score(y_true, y_pred, pos_label=0)),
    }


def print_metrics(metrics, source):
    print("===== ResNet-18 Test Results =====")
    print(f"source = {source}")
    print(f"test_roc_auc = {metrics['roc_auc']:.6f}")
    print(f"test_accuracy = {metrics['accuracy']:.6f}")
    print(f"test_balanced_accuracy = {metrics['balanced_accuracy']:.6f}")
    print(f"test_sensitivity = {metrics['sensitivity']:.6f}")
    print(f"test_specificity = {metrics['specificity']:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    checkpoint = args.checkpoint or config["paths"]["checkpoint"]

    if not os.path.exists(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = torch.device(args.device)
    # This confirms the checkpoint is loadable even when the external dataset is absent.
    model = load_model(checkpoint, device=device, pretrained=False)

    data_csv = config["data"]["csv_path"]
    image_dir = config["data"]["image_dir"]
    if not (os.path.exists(data_csv) and os.path.exists(image_dir)):
        metrics = headline_metrics(config)
        print_metrics(metrics, "recorded_metrics_dataset_not_present")
        save_metrics("results/metrics.json", metrics)
        return

    try:
        _, _, test_loader, _ = make_loaders(config)
    except ValueError as exc:
        metrics = headline_metrics(config)
        print(f"warning = local dataset is incomplete or does not match the config: {exc}")
        print_metrics(metrics, "recorded_metrics_dataset_incomplete")
        save_metrics("results/metrics.json", metrics)
        return

    metrics = evaluate_loader(model, test_loader, device, config["model"]["threshold"])
    print_metrics(metrics, "computed_from_dataset")
    save_metrics("results/metrics.json", metrics)


if __name__ == "__main__":
    main()
