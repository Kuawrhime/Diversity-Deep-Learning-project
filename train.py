import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from src.data import make_loaders
from src.model import ResNet18Binary
from src.utils import ensure_dir, load_config, set_seed


def compute_auc(loader, model, device):
    model.eval()
    labels = []
    probs = []
    with torch.no_grad():
        for images, batch_labels, _ in loader:
            images = images.to(device)
            outputs = model(images)
            probs.extend(torch.sigmoid(outputs).cpu().numpy())
            labels.extend(batch_labels.numpy())
    return roc_auc_score(labels, probs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    checkpoint_path = config["paths"]["checkpoint"]
    ensure_dir(os.path.dirname(checkpoint_path))
    ensure_dir(config["paths"]["results_dir"])

    train_loader, val_loader, _, class_counts = make_loaders(config)
    device = torch.device(args.device)
    model = ResNet18Binary(pretrained=config["model"]["pretrained"]).to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([class_counts[0] / class_counts[1]], device=device)
    )
    optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_val_auc = 0.0
    patience_counter = 0
    for epoch in range(config["training"]["num_epochs"]):
        model.train()
        train_loss = 0.0
        for images, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch + 1}"):
            images = images.to(device)
            labels = labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels, _ in val_loader:
                images = images.to(device)
                labels = labels.to(device).float()
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        val_auc = compute_auc(val_loader, model, device)
        scheduler.step(val_loss)
        print(f"epoch={epoch + 1} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_auc={val_auc:.4f}")

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            print(f"saved {checkpoint_path} with val_auc={val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stop_patience"]:
                print("early stopping")
                break

    print(f"best_val_auc = {best_val_auc:.6f}")


if __name__ == "__main__":
    main()
