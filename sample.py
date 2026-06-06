import argparse

import torch
from PIL import Image

from src.data import get_transforms
from src.model import load_model
from src.utils import load_config, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config["project"]["seed"])
    checkpoint = args.checkpoint or config["paths"]["checkpoint"]
    device = torch.device(args.device)
    model = load_model(checkpoint, device=device, pretrained=False)
    _, transform = get_transforms(config["model"]["image_size"])

    image = Image.open(args.image).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()
    label = "malignant" if prob >= config["model"]["threshold"] else "benign"
    print(f"prediction = {label}")
    print(f"malignant_probability = {prob:.6f}")


if __name__ == "__main__":
    main()
